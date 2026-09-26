"""Paper-guided operator-network adapters for masked field recovery.

These implementations are independent benchmark adapters, not copies of the
upstream repositories and not claims of exact paper reproduction.  The
architectural anchors are DeepONet's branch--trunk product, Transolver's
physics-slice attention, and GNOT's normalized linear cross-attention with
geometry-gated experts.  Every adapter keeps value, visibility, coordinates,
and geometry distinct so a physical zero never represents missing data.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .base import MethodCapabilities, register_method

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - optional neural dependency
    torch = None
    Tensor = Any
    nn = None
    F = None


_ADAPTATION_NOTE = (
    "Paper-guided variable-observation recovery adapter; independently implemented "
    "and not an exact reproduction of the cited forward-operator model."
)
_CAPABILITIES = MethodCapabilities(
    tasks=frozenset({"recovery", "forward", "inverse"}),
    trainable=True,
    requires_mask=True,
    reference_only=True,
    notes=_ADAPTATION_NOTE,
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for operator-network adapters")


if torch is not None:

    def _mask_tensor(x: Tensor, mask: Tensor | None) -> Tensor:
        if mask is None:
            return torch.ones((x.shape[0], 1, *x.shape[-2:]), device=x.device, dtype=x.dtype)
        if mask.ndim == 3:
            mask = mask[:, None]
        if mask.ndim != 4:
            raise ValueError("mask must have shape BHW or BCHW")
        if mask.shape[0] != x.shape[0] or mask.shape[-2:] != x.shape[-2:]:
            raise ValueError("mask and observations must share batch and spatial shapes")
        if mask.shape[1] != 1:
            mask = mask.amax(dim=1, keepdim=True)
        return mask.to(device=x.device, dtype=x.dtype)


    def _geometry_tensor(x: Tensor, geometry: Tensor | None, channels: int) -> Tensor:
        if channels < 0:
            raise ValueError("geometry_channels must be non-negative")
        if channels == 0:
            return x.new_zeros((x.shape[0], 0, *x.shape[-2:]))
        if geometry is None:
            raise ValueError(
                "geometry is required by this adapter; obstacle geometry cannot be omitted"
            )
        if geometry.ndim == 3:
            geometry = geometry[:, None]
        if geometry.ndim != 4:
            raise ValueError("geometry must have shape BHW or BCHW")
        if geometry.shape[0] != x.shape[0] or geometry.shape[-2:] != x.shape[-2:]:
            raise ValueError("geometry and observations must share batch and spatial shapes")
        if geometry.shape[1] != channels:
            raise ValueError(
                f"expected {channels} geometry channels, found {geometry.shape[1]}"
            )
        return geometry.to(device=x.device, dtype=x.dtype)


    def _coordinate_grid(x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        y = torch.linspace(0.0, 1.0, height, device=x.device, dtype=x.dtype)
        x_axis = torch.linspace(0.0, 1.0, width, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(y, x_axis, indexing="ij")
        return torch.stack((grid_y, grid_x), dim=0)[None].expand(x.shape[0], -1, -1, -1)


    def _flatten_channels(x: Tensor) -> Tensor:
        return x.flatten(2).transpose(1, 2).contiguous()


    class _PredictMixin:
        @staticmethod
        def _numpy(value: Any) -> Any:
            if isinstance(value, Tensor):
                return value.detach().cpu().numpy()
            if isinstance(value, Mapping):
                return {key: _PredictMixin._numpy(item) for key, item in value.items()}
            return np.asarray(value)

        def predict(self, observations: Any, mask: Any | None = None, **kwargs: Any) -> Any:
            parameter = next(self.parameters(), None)
            device = parameter.device if parameter is not None else torch.device("cpu")
            values = torch.as_tensor(observations, dtype=torch.float32, device=device)
            visible = (
                None
                if mask is None
                else torch.as_tensor(mask, dtype=torch.float32, device=device)
            )
            converted = {
                key: torch.as_tensor(value, dtype=torch.float32, device=device)
                if key == "geometry" and value is not None
                else value
                for key, value in kwargs.items()
            }
            self.eval()
            with torch.no_grad():
                return self._numpy(self.forward(values, mask=visible, **converted))


    def _mlp(widths: list[int], activation: type[nn.Module]) -> nn.Sequential:
        layers: list[nn.Module] = []
        for index, (in_width, out_width) in enumerate(
            zip(widths[:-1], widths[1:], strict=True)
        ):
            layers.append(nn.Linear(in_width, out_width))
            if index < len(widths) - 2:
                layers.append(activation())
        return nn.Sequential(*layers)


    class _UNetConvolution(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, values: Tensor) -> Tensor:
            return self.layers(values)


    @register_method("unet_paper_guided", aliases=("deep_mask_unet",))
    class PaperGuidedMaskUNet(_PredictMixin, nn.Module):
        """Four-level symmetric U-Net adapted from segmentation to masked fields."""

        name = "unet_paper_guided"
        capabilities = _CAPABILITIES

        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            width: int = 64,
            levels: int = 4,
            geometry_channels: int = 1,
        ) -> None:
            super().__init__()
            if width < 1 or levels < 1:
                raise ValueError("width and levels must be positive")
            self.geometry_channels = geometry_channels
            channels = [width * 2**level for level in range(levels)]
            encoder_inputs = [in_channels + 1 + geometry_channels, *channels[:-1]]
            self.encoders = nn.ModuleList(
                _UNetConvolution(source, target)
                for source, target in zip(encoder_inputs, channels, strict=True)
            )
            bottleneck_channels = channels[-1] * 2
            self.bottleneck = _UNetConvolution(channels[-1], bottleneck_channels)
            decoder_outputs = list(reversed(channels))
            decoder_inputs = [
                bottleneck_channels + decoder_outputs[0],
                *[
                    decoder_outputs[index - 1] + decoder_outputs[index]
                    for index in range(1, len(decoder_outputs))
                ],
            ]
            self.decoders = nn.ModuleList(
                _UNetConvolution(source, target)
                for source, target in zip(decoder_inputs, decoder_outputs, strict=True)
            )
            self.head = nn.Conv2d(width, out_channels, 1)

        def forward(
            self,
            observations: Tensor,
            mask: Tensor | None = None,
            geometry: Tensor | None = None,
        ) -> Tensor:
            if observations.ndim != 4:
                raise ValueError("PaperGuidedMaskUNet expects observations with shape BCHW")
            visible = _mask_tensor(observations, mask)
            geometry_value = _geometry_tensor(
                observations, geometry, self.geometry_channels
            )
            state = torch.cat((observations, visible, geometry_value), dim=1)
            skips: list[Tensor] = []
            for encoder in self.encoders:
                state = encoder(state)
                skips.append(state)
                state = F.max_pool2d(state, 2, ceil_mode=True)
            state = self.bottleneck(state)
            for decoder, skip in zip(self.decoders, reversed(skips), strict=True):
                state = F.interpolate(
                    state, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
                state = decoder(torch.cat((state, skip), dim=1))
            return self.head(state)


    @register_method("deeponet", aliases=("variable_sensor_deeponet",))
    class VariableSensorDeepONet(_PredictMixin, nn.Module):
        """DeepONet branch--trunk product with a mask-aware spatial branch.

        The upstream DeepONet branch receives an ordered, fixed sensor vector.
        A global mean/max set reduction is too lossy for structured 2-D PDE
        fields, so the source-faithful adapter can retain sensor ordering in
        a small convolutional branch before the canonical branch--trunk inner
        product.  The legacy set branch remains available for old checkpoints.
        """

        name = "deeponet"
        capabilities = _CAPABILITIES
        upstream_revision = "8d62345afd39e1df9c2c8c8d0e7c41882b06a9bf"

        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            hidden: int = 128,
            latent: int = 128,
            branch_layers: int = 2,
            trunk_layers: int = 2,
            geometry_channels: int = 1,
            branch_mode: str = "sensor_set",
            branch_grid: int = 4,
            branch_width: int | None = None,
        ) -> None:
            super().__init__()
            if min(hidden, latent, branch_layers, trunk_layers, branch_grid) < 1:
                raise ValueError("hidden, latent, and layer counts must be positive")
            branch_mode = str(branch_mode).strip().lower().replace("-", "_")
            if branch_mode not in {"sensor_set", "spatial_cnn"}:
                raise ValueError("branch_mode must be sensor_set or spatial_cnn")
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.latent = latent
            self.geometry_channels = geometry_channels
            self.branch_mode = branch_mode
            self.branch_grid = branch_grid
            sensor_width = in_channels + 1 + 2 + geometry_channels
            if branch_mode == "sensor_set":
                sensor_stack = [sensor_width, *([hidden] * branch_layers), latent]
                self.sensor_encoder: nn.Module | None = _mlp(sensor_stack, nn.ReLU)
                self.spatial_encoder: nn.Module | None = None
                branch_input = 2 * latent + 1
            else:
                resolved_branch_width = int(branch_width or min(hidden, 64))
                if resolved_branch_width < 1:
                    raise ValueError("branch_width must be positive")
                spatial_layers: list[nn.Module] = [
                    nn.Conv2d(sensor_width, resolved_branch_width, 3, padding=1),
                    nn.GELU(),
                ]
                for _ in range(branch_layers - 1):
                    spatial_layers.extend(
                        (
                            nn.Conv2d(
                                resolved_branch_width,
                                resolved_branch_width,
                                3,
                                padding=1,
                            ),
                            nn.GELU(),
                        )
                    )
                spatial_layers.append(nn.AdaptiveAvgPool2d((branch_grid, branch_grid)))
                self.sensor_encoder = None
                self.spatial_encoder = nn.Sequential(*spatial_layers)
                branch_input = resolved_branch_width * branch_grid * branch_grid
            self.branch_fusion = _mlp(
                [branch_input, hidden, out_channels * latent], nn.GELU
            )
            trunk_stack = [2 + geometry_channels, *([hidden] * trunk_layers), out_channels * latent]
            self.trunk = _mlp(trunk_stack, nn.ReLU)
            self.output_bias = nn.Parameter(torch.zeros(out_channels))
            self.apply(self._initialize)

        @staticmethod
        def _initialize(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(
            self,
            observations: Tensor,
            mask: Tensor | None = None,
            geometry: Tensor | None = None,
        ) -> Tensor:
            if observations.ndim != 4:
                raise ValueError("VariableSensorDeepONet expects observations with shape BCHW")
            visible = _mask_tensor(observations, mask)
            geometry_value = _geometry_tensor(
                observations, geometry, self.geometry_channels
            )
            coordinates = _coordinate_grid(observations)
            observed_values = observations * visible
            sensor_image = torch.cat(
                (observed_values, visible, coordinates, geometry_value), dim=1
            )
            if self.branch_mode == "spatial_cnn":
                assert self.spatial_encoder is not None
                branch_features = self.spatial_encoder(sensor_image).flatten(1)
            else:
                assert self.sensor_encoder is not None
                encoded = self.sensor_encoder(_flatten_channels(sensor_image))
                weights = _flatten_channels(visible)
                denominator = weights.sum(dim=1).clamp_min(1.0)
                mean = (encoded * weights).sum(dim=1) / denominator
                floor = torch.finfo(encoded.dtype).min
                maximum = encoded.masked_fill(weights == 0, floor).amax(dim=1)
                maximum = torch.where(
                    torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
                )
                density = weights.mean(dim=1)
                branch_features = torch.cat((mean, maximum, density), dim=-1)
            branch = self.branch_fusion(branch_features)
            branch = branch.reshape(observations.shape[0], self.out_channels, self.latent)

            query_features = _flatten_channels(torch.cat((coordinates, geometry_value), dim=1))
            trunk = self.trunk(query_features).reshape(
                observations.shape[0], -1, self.out_channels, self.latent
            )
            prediction = torch.einsum("bol,bnol->bon", branch, trunk)
            prediction = prediction + self.output_bias[None, :, None]
            return prediction.reshape(
                observations.shape[0], self.out_channels, *observations.shape[-2:]
            )


    class _PhysicsSliceAttention(nn.Module):
        def __init__(
            self,
            hidden: int,
            heads: int,
            slices: int,
            dropout: float,
        ) -> None:
            super().__init__()
            if hidden % heads:
                raise ValueError("hidden must be divisible by heads")
            self.heads = heads
            self.head_dim = hidden // heads
            self.slices = slices
            self.scale = self.head_dim**-0.5
            self.feature_projection = nn.Conv2d(hidden, hidden, 3, padding=1)
            self.weight_projection = nn.Conv2d(hidden, hidden, 3, padding=1)
            self.slice_projection = nn.Linear(self.head_dim, slices)
            self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
            self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
            self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)
            self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.Dropout(dropout))
            self.dropout = nn.Dropout(dropout)
            self.temperature = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
            nn.init.orthogonal_(self.slice_projection.weight)

        def forward(self, state: Tensor) -> Tensor:
            batch, hidden, height, width = state.shape
            points = height * width
            features = self.feature_projection(state).reshape(
                batch, self.heads, self.head_dim, points
            ).permute(0, 1, 3, 2)
            selectors = self.weight_projection(state).reshape(
                batch, self.heads, self.head_dim, points
            ).permute(0, 1, 3, 2)
            temperature = self.temperature.clamp(0.1, 5.0)
            weights = torch.softmax(self.slice_projection(selectors) / temperature, dim=-1)
            mass = weights.sum(dim=2).clamp_min(1.0e-5)
            tokens = torch.einsum("bhnd,bhns->bhsd", features, weights)
            tokens = tokens / mass[..., None]
            query = self.query(tokens)
            key = self.key(tokens)
            value = self.value(tokens)
            attention = torch.softmax(
                torch.matmul(query, key.transpose(-1, -2)) * self.scale, dim=-1
            )
            attended = torch.matmul(self.dropout(attention), value)
            restored = torch.einsum("bhsd,bhns->bhnd", attended, weights)
            restored = restored.permute(0, 2, 1, 3).reshape(batch, points, hidden)
            return self.output(restored).transpose(1, 2).reshape(batch, hidden, height, width)


    class _TransolverBlock(nn.Module):
        def __init__(
            self,
            hidden: int,
            heads: int,
            slices: int,
            mlp_ratio: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.norm_attention = nn.GroupNorm(1, hidden)
            self.attention = _PhysicsSliceAttention(hidden, heads, slices, dropout)
            self.norm_mlp = nn.GroupNorm(1, hidden)
            self.mlp = nn.Sequential(
                nn.Conv2d(hidden, hidden * mlp_ratio, 1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(hidden * mlp_ratio, hidden, 1),
            )

        def forward(self, state: Tensor) -> Tensor:
            state = state + self.attention(self.norm_attention(state))
            return state + self.mlp(self.norm_mlp(state))


    @register_method("transolver", aliases=("transolver_2d",))
    class MaskConditionedTransolver(_PredictMixin, nn.Module):
        """Structured-grid physics-slice attention adapted to masked recovery."""

        name = "transolver"
        capabilities = _CAPABILITIES
        upstream_revision = "75e0f67643806a81cd1d3f6adc88dd8c02416fe7"

        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            hidden: int = 256,
            layers: int = 5,
            heads: int = 8,
            slices: int = 32,
            mlp_ratio: int = 1,
            dropout: float = 0.0,
            geometry_channels: int = 1,
        ) -> None:
            super().__init__()
            if min(hidden, layers, heads, slices, mlp_ratio) < 1:
                raise ValueError("Transolver dimensions must be positive")
            if hidden % heads:
                raise ValueError("hidden must be divisible by heads")
            self.geometry_channels = geometry_channels
            feature_channels = in_channels + 1 + 2 + geometry_channels
            self.lift = nn.Sequential(
                nn.Conv2d(feature_channels, 2 * hidden, 1),
                nn.GELU(),
                nn.Conv2d(2 * hidden, hidden, 1),
            )
            self.blocks = nn.ModuleList(
                _TransolverBlock(hidden, heads, slices, mlp_ratio, dropout)
                for _ in range(layers)
            )
            self.norm = nn.GroupNorm(1, hidden)
            self.project = nn.Conv2d(hidden, out_channels, 1)
            self.apply(self._initialize)

        @staticmethod
        def _initialize(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(
            self,
            observations: Tensor,
            mask: Tensor | None = None,
            geometry: Tensor | None = None,
        ) -> Tensor:
            if observations.ndim != 4:
                raise ValueError("MaskConditionedTransolver expects observations with shape BCHW")
            visible = _mask_tensor(observations, mask)
            geometry_value = _geometry_tensor(
                observations, geometry, self.geometry_channels
            )
            state = self.lift(
                torch.cat(
                    (observations, visible, _coordinate_grid(observations), geometry_value),
                    dim=1,
                )
            )
            for block in self.blocks:
                state = block(state)
            return self.project(self.norm(state))


    class _NormalizedLinearAttention(nn.Module):
        def __init__(self, hidden: int, heads: int, dropout: float = 0.0) -> None:
            super().__init__()
            if hidden % heads:
                raise ValueError("hidden must be divisible by heads")
            self.hidden = hidden
            self.heads = heads
            self.head_dim = hidden // heads
            self.query = nn.Linear(hidden, hidden)
            self.key = nn.Linear(hidden, hidden)
            self.value = nn.Linear(hidden, hidden)
            self.output = nn.Linear(hidden, hidden)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            query_tokens: Tensor,
            source_tokens: Tensor | None = None,
            source_mask: Tensor | None = None,
        ) -> Tensor:
            source_tokens = query_tokens if source_tokens is None else source_tokens
            batch, queries, _ = query_tokens.shape
            sources = source_tokens.shape[1]
            query = self.query(query_tokens).reshape(
                batch, queries, self.heads, self.head_dim
            ).transpose(1, 2)
            key = self.key(source_tokens).reshape(
                batch, sources, self.heads, self.head_dim
            ).transpose(1, 2)
            value = self.value(source_tokens).reshape(
                batch, sources, self.heads, self.head_dim
            ).transpose(1, 2)
            query = torch.softmax(query, dim=-1)
            key = torch.softmax(key, dim=-1)
            if source_mask is not None:
                weights = source_mask[:, None, :, None].to(key)
                key = key * weights
                value = value * weights
            key_sum = key.sum(dim=-2, keepdim=True)
            denominator = (query * key_sum).sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
            context = torch.matmul(key.transpose(-2, -1), value)
            output = torch.matmul(query, context) / denominator
            output = output.transpose(1, 2).reshape(batch, queries, self.hidden)
            return self.output(self.dropout(output))


    class _GNOTBlock(nn.Module):
        def __init__(
            self,
            hidden: int,
            heads: int,
            experts: int,
            inner_ratio: int,
            gate_features: int,
            dropout: float,
        ) -> None:
            super().__init__()
            if experts < 1:
                raise ValueError("experts must be positive")
            self.cross_norm_query = nn.LayerNorm(hidden)
            self.cross_norm_source = nn.LayerNorm(hidden)
            self.cross = _NormalizedLinearAttention(hidden, heads, dropout)
            self.self_norm = nn.LayerNorm(hidden)
            self.self_attention = _NormalizedLinearAttention(hidden, heads, dropout)
            self.expert_norm_1 = nn.LayerNorm(hidden)
            self.expert_norm_2 = nn.LayerNorm(hidden)
            inner = inner_ratio * hidden
            self.experts_1 = nn.ModuleList(
                nn.Sequential(nn.Linear(hidden, inner), nn.GELU(), nn.Linear(inner, hidden))
                for _ in range(experts)
            )
            self.experts_2 = nn.ModuleList(
                nn.Sequential(nn.Linear(hidden, inner), nn.GELU(), nn.Linear(inner, hidden))
                for _ in range(experts)
            )
            self.gate = nn.Sequential(
                nn.Linear(gate_features, hidden),
                nn.GELU(),
                nn.Linear(hidden, experts),
            )

        def _mixture(self, state: Tensor, gate_features: Tensor, experts: nn.ModuleList) -> Tensor:
            weights = torch.softmax(self.gate(gate_features), dim=-1)
            candidates = torch.stack([expert(state) for expert in experts], dim=-1)
            return (candidates * weights.unsqueeze(-2)).sum(dim=-1)

        def forward(
            self,
            state: Tensor,
            source: Tensor,
            source_mask: Tensor,
            gate_features: Tensor,
        ) -> Tensor:
            state = state + self.cross(
                self.cross_norm_query(state),
                self.cross_norm_source(source),
                source_mask,
            )
            state = state + self.expert_norm_1(
                self._mixture(state, gate_features, self.experts_1)
            )
            state = state + self.self_attention(self.self_norm(state))
            return state + self.expert_norm_2(
                self._mixture(state, gate_features, self.experts_2)
            )


    @register_method("gnot", aliases=("mask_conditioned_gnot",))
    class MaskConditionedGNOT(_PredictMixin, nn.Module):
        """Geometry-gated normalized-attention operator for variable sensors."""

        name = "gnot"
        capabilities = _CAPABILITIES
        upstream_revision = "5ee2e6925a43f9a340a6016bad4da2c82a452cbe"

        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            hidden: int = 128,
            layers: int = 3,
            heads: int = 1,
            experts: int = 2,
            inner_ratio: int = 4,
            mlp_layers: int = 3,
            dropout: float = 0.0,
            geometry_channels: int = 1,
        ) -> None:
            super().__init__()
            if min(hidden, layers, heads, experts, inner_ratio, mlp_layers) < 1:
                raise ValueError("GNOT dimensions must be positive")
            if hidden % heads:
                raise ValueError("hidden must be divisible by heads")
            self.out_channels = out_channels
            self.geometry_channels = geometry_channels
            gate_features = 2 + geometry_channels
            source_features = in_channels + 1 + gate_features
            self.source_encoder = _mlp(
                [source_features, *([hidden] * mlp_layers), hidden], nn.GELU
            )
            self.query_encoder = _mlp(
                [gate_features, *([hidden] * mlp_layers), hidden], nn.GELU
            )
            self.blocks = nn.ModuleList(
                _GNOTBlock(
                    hidden,
                    heads,
                    experts,
                    inner_ratio,
                    gate_features,
                    dropout,
                )
                for _ in range(layers)
            )
            self.output = _mlp([hidden, hidden, out_channels], nn.GELU)

        def forward(
            self,
            observations: Tensor,
            mask: Tensor | None = None,
            geometry: Tensor | None = None,
        ) -> Tensor:
            if observations.ndim != 4:
                raise ValueError("MaskConditionedGNOT expects observations with shape BCHW")
            visible = _mask_tensor(observations, mask)
            geometry_value = _geometry_tensor(
                observations, geometry, self.geometry_channels
            )
            gate_features = _flatten_channels(
                torch.cat((_coordinate_grid(observations), geometry_value), dim=1)
            )
            source_features = _flatten_channels(
                torch.cat((observations, visible, _coordinate_grid(observations), geometry_value), dim=1)
            )
            source = self.source_encoder(source_features)
            state = self.query_encoder(gate_features)
            source_mask = _flatten_channels(visible).squeeze(-1)
            for block in self.blocks:
                state = block(state, source, source_mask, gate_features)
            prediction = self.output(state).transpose(1, 2)
            return prediction.reshape(
                observations.shape[0], self.out_channels, *observations.shape[-2:]
            )


else:

    class _TorchMissing:
        capabilities = _CAPABILITIES

        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


    @register_method("unet_paper_guided", aliases=("deep_mask_unet",))
    class PaperGuidedMaskUNet(_TorchMissing):
        name = "unet_paper_guided"


    @register_method("deeponet", aliases=("variable_sensor_deeponet",))
    class VariableSensorDeepONet(_TorchMissing):
        name = "deeponet"


    @register_method("transolver", aliases=("transolver_2d",))
    class MaskConditionedTransolver(_TorchMissing):
        name = "transolver"


    @register_method("gnot", aliases=("mask_conditioned_gnot",))
    class MaskConditionedGNOT(_TorchMissing):
        name = "gnot"


__all__ = [
    "PaperGuidedMaskUNet",
    "VariableSensorDeepONet",
    "MaskConditionedTransolver",
    "MaskConditionedGNOT",
]
