"""Compact PyTorch reference baselines.

The networks below are intentionally small, auditable architectural references
for benchmark smoke runs. They are **not exact reproductions** of corresponding
research papers or official repositories. Serious comparisons should record the
architecture and training configuration and cite the exact implementation used.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .base import MethodCapabilities, register_method

try:  # PyTorch is an optional dependency for interpolation-only installations.
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - exercised in minimal installations
    torch = None
    Tensor = Any
    nn = None
    F = None


_REFERENCE_NOTE = "Compact architectural reference; not an exact paper reproduction."
_RECOVERY_CAPS = MethodCapabilities(
    tasks=frozenset({"recovery", "forward", "inverse"}),
    trainable=True,
    requires_mask=True,
    reference_only=True,
    notes=_REFERENCE_NOTE,
)
_ROLLOUT_CAPS = MethodCapabilities(
    tasks=frozenset({"rollout"}),
    trainable=True,
    temporal=True,
    reference_only=True,
    notes=_REFERENCE_NOTE,
)
_ENCODER_CAPS = MethodCapabilities(
    tasks=frozenset({"retrieval", "classification", "supervised_multitask", "representation"}),
    trainable=True,
    requires_mask=True,
    reference_only=True,
    notes=(
        "Compact residual CNN encoder/classifier reference for continuous-latent retrieval "
        "and supervised multitask pretraining; not an exact paper reproduction."
    ),
)
_MAE_CAPS = MethodCapabilities(
    tasks=frozenset({"recovery", "forward", "inverse", "pretraining", "representation"}),
    trainable=True,
    requires_mask=True,
    reference_only=True,
    notes=(
        "Compact convolutional MAE-style reconstruction reference; not a ViT-MAE or exact "
        "paper reproduction."
    ),
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for neural baselines. Install pdeobs[train] or torch."
        )


if torch is not None:

    def _mask_channel(x: Tensor, mask: Tensor | None) -> Tensor:
        if mask is None:
            return torch.ones((x.shape[0], 1, *x.shape[-2:]), dtype=x.dtype, device=x.device)
        if mask.ndim == 3:
            mask = mask[:, None]
        if mask.ndim != 4:
            raise ValueError("mask must have shape BHW or BCHW")
        if mask.shape[1] != 1:
            mask = mask.amax(dim=1, keepdim=True)
        return mask.to(dtype=x.dtype, device=x.device)

    def _geometry_tensor(
        x: Tensor,
        geometry: Tensor | None,
        channels: int,
    ) -> Tensor | None:
        """Validate and align explicitly configured geometry channels."""

        if channels == 0:
            return None
        if geometry is None:
            raise ValueError(
                "geometry is required when geometry_channels is non-zero; "
                "obstacle geometry must never be silently discarded"
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
        return geometry.to(dtype=x.dtype, device=x.device)

    class _PredictMixin:
        """NumPy-friendly method protocol layered on top of ``nn.Module``."""

        @staticmethod
        def _to_numpy(output: Any) -> Any:
            if isinstance(output, Tensor):
                return output.detach().cpu().numpy()
            if isinstance(output, Mapping):
                return {name: _PredictMixin._to_numpy(value) for name, value in output.items()}
            if isinstance(output, tuple):
                return tuple(_PredictMixin._to_numpy(value) for value in output)
            if isinstance(output, list):
                return [_PredictMixin._to_numpy(value) for value in output]
            return np.asarray(output)

        def predict(self, observations: Any, mask: Any | None = None, **kwargs: Any) -> Any:
            parameter = next(self.parameters(), None)
            device = parameter.device if parameter is not None else torch.device("cpu")
            x = torch.as_tensor(observations, dtype=torch.float32, device=device)
            m = None if mask is None else torch.as_tensor(mask, dtype=torch.float32, device=device)
            self.eval()
            with torch.no_grad():
                output = self.forward(x, mask=m, **kwargs)
            return self._to_numpy(output)

    class _ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            groups = min(8, out_channels)
            while out_channels % groups:
                groups -= 1
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.GroupNorm(groups, out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(groups, out_channels),
                nn.GELU(),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.net(x)

    class _ResidualBlock(nn.Module):
        """Small pre-activation-free residual block used by the CNN anchor."""

        def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
            super().__init__()
            groups = min(8, out_channels)
            while out_channels % groups:
                groups -= 1
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(groups, out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, out_channels),
            )
            self.skip = (
                nn.Identity()
                if stride == 1 and in_channels == out_channels
                else nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            )

        def forward(self, x: Tensor) -> Tensor:
            return F.gelu(self.main(x) + self.skip(x))

    @register_method(
        "residual_cnn",
        aliases=("resnet", "resnet_encoder", "resnet_classifier", "supervised_multitask_small"),
    )
    class CompactResidualEncoder(_PredictMixin, nn.Module):
        """Mask-aware residual CNN for retrieval embeddings and semantic labels.

        ``forward`` returns a normalized embedding by default. Set ``output`` to
        ``"logits"`` to use the configured classification heads, or ``"both"``
        for a dictionary containing both representations. A mapping of class
        counts creates one head per semantic axis, which is the lightweight
        supervised-multitask protocol used by PDE-OBS.
        """

        name = "residual_cnn"
        capabilities = _ENCODER_CAPS

        def __init__(
            self,
            in_channels: int = 1,
            width: int = 32,
            embedding_dim: int = 128,
            class_counts: int | Mapping[str, int] | None = None,
            normalize_embeddings: bool = True,
            default_output: str = "embedding",
        ) -> None:
            super().__init__()
            if in_channels < 1 or width < 1 or embedding_dim < 1:
                raise ValueError("in_channels, width, and embedding_dim must be positive")
            if default_output not in {"embedding", "logits", "both"}:
                raise ValueError("default_output must be embedding, logits, or both")
            self.in_channels = in_channels
            self.embedding_dim = embedding_dim
            self.normalize_embeddings = normalize_embeddings
            self.default_output = default_output
            groups = min(8, width)
            while width % groups:
                groups -= 1
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels + 1, width, 5, padding=2, bias=False),
                nn.GroupNorm(groups, width),
                nn.GELU(),
            )
            self.encoder = nn.Sequential(
                _ResidualBlock(width, width),
                _ResidualBlock(width, width * 2, stride=2),
                _ResidualBlock(width * 2, width * 2),
                _ResidualBlock(width * 2, width * 4, stride=2),
                _ResidualBlock(width * 4, width * 4),
            )
            self.projection = nn.Linear(width * 4, embedding_dim)

            if isinstance(class_counts, int):
                resolved_counts = {"class": class_counts}
            else:
                resolved_counts = dict(class_counts or {})
            invalid = {name: count for name, count in resolved_counts.items() if int(count) < 2}
            if invalid:
                raise ValueError(f"Every classification head needs at least two classes: {invalid}")
            self.class_names = tuple(resolved_counts)
            self.class_heads = nn.ModuleDict(
                {
                    name: nn.Linear(embedding_dim, int(count))
                    for name, count in resolved_counts.items()
                }
            )

        def encode(self, observations: Tensor, mask: Tensor | None = None) -> Tensor:
            if observations.ndim != 4:
                raise ValueError("CompactResidualEncoder expects observations with shape BCHW")
            if observations.shape[1] != self.in_channels:
                raise ValueError(
                    f"Expected {self.in_channels} input channels, got {observations.shape[1]}"
                )
            visible = _mask_channel(observations, mask)
            features = self.encoder(self.stem(torch.cat((observations * visible, visible), dim=1)))
            embedding = self.projection(F.adaptive_avg_pool2d(features, 1).flatten(1))
            if self.normalize_embeddings:
                embedding = F.normalize(embedding, dim=-1)
            return embedding

        def classify(self, embedding: Tensor) -> Tensor | dict[str, Tensor]:
            if not self.class_heads:
                raise RuntimeError("No classification heads were configured; set class_counts")
            logits = {name: head(embedding) for name, head in self.class_heads.items()}
            return logits["class"] if self.class_names == ("class",) else logits

        def supervised_loss(
            self,
            logits: Tensor | Mapping[str, Tensor],
            targets: Tensor | Mapping[str, Tensor],
        ) -> Tensor:
            """Mean cross-entropy over one or more configured semantic axes."""

            if isinstance(logits, Tensor):
                if not isinstance(targets, Tensor):
                    raise TypeError("Single-head logits require a tensor target")
                return F.cross_entropy(logits, targets.long())
            if not isinstance(targets, Mapping):
                raise TypeError("Multitask logits require a target mapping")
            missing = set(logits) - set(targets)
            if missing:
                raise KeyError(f"Missing targets for classification heads: {sorted(missing)}")
            losses = [
                F.cross_entropy(value, targets[name].long()) for name, value in logits.items()
            ]
            return torch.stack(losses).mean()

        def forward(
            self,
            observations: Tensor,
            mask: Tensor | None = None,
            *,
            output: str | None = None,
        ) -> Any:
            embedding = self.encode(observations, mask)
            selected = output or self.default_output
            if selected == "embedding":
                return embedding
            logits = self.classify(embedding)
            if selected == "logits":
                return logits
            if selected == "both":
                return {"embedding": embedding, "logits": logits}
            raise ValueError("output must be embedding, logits, or both")

    @register_method(
        "mae_small",
        aliases=("masked_autoencoder", "masked_autoencoder_small", "mae_style_small"),
    )
    class MaskedAutoencoderSmall(_PredictMixin, nn.Module):
        """Compact convolutional MAE-style sparse reconstruction anchor.

        During training, the configured patch mask is combined with the
        benchmark observation mask. Visible values are copied to the output, so
        ordinary full-field MSE optimizes only hidden locations (up to a constant
        scale). Evaluation is deterministic and uses only the supplied mask.
        """

        name = "mae_small"
        capabilities = _MAE_CAPS

        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            width: int = 32,
            latent_channels: int = 128,
            patch_size: int = 8,
            mask_ratio: float = 0.75,
            preserve_visible: bool = True,
        ) -> None:
            super().__init__()
            if min(in_channels, out_channels, width, latent_channels, patch_size) < 1:
                raise ValueError("channel counts, width, and patch_size must be positive")
            if not 0.0 <= mask_ratio < 1.0:
                raise ValueError("mask_ratio must satisfy 0 <= mask_ratio < 1")
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.patch_size = patch_size
            self.mask_ratio = float(mask_ratio)
            self.preserve_visible = preserve_visible
            self.enc1 = _ConvBlock(in_channels + 1, width)
            self.enc2 = _ConvBlock(width, width * 2)
            self.latent = _ConvBlock(width * 2, latent_channels)
            self.dec2 = _ConvBlock(latent_channels, width * 2)
            self.dec1 = _ConvBlock(width * 2, width)
            self.head = nn.Conv2d(width, out_channels, 1)

        def _random_patch_mask(self, observations: Tensor) -> Tensor:
            height, width = observations.shape[-2:]
            grid_h = (height + self.patch_size - 1) // self.patch_size
            grid_w = (width + self.patch_size - 1) // self.patch_size
            keep = (
                torch.rand(
                    observations.shape[0],
                    1,
                    grid_h,
                    grid_w,
                    device=observations.device,
                )
                >= self.mask_ratio
            )
            flat = keep.flatten(1)
            empty = ~flat.any(dim=1)
            if empty.any():
                flat[empty, 0] = True
            keep = flat.reshape_as(keep).to(dtype=observations.dtype)
            return F.interpolate(keep, size=(height, width), mode="nearest")

        def mask_inputs(
            self,
            observations: Tensor,
            mask: Tensor | None = None,
            *,
            force_random: bool | None = None,
        ) -> tuple[Tensor, Tensor]:
            """Return masked values and the effective visible-location mask."""

            if observations.ndim != 4:
                raise ValueError("MaskedAutoencoderSmall expects observations with shape BCHW")
            if observations.shape[1] != self.in_channels:
                raise ValueError(
                    f"Expected {self.in_channels} input channels, got {observations.shape[1]}"
                )
            visible = _mask_channel(observations, mask)
            random_masking = self.training if force_random is None else force_random
            if random_masking and self.mask_ratio > 0:
                visible = visible * self._random_patch_mask(observations)
            return observations * visible, visible

        def _encode_masked(self, observations: Tensor, visible: Tensor) -> Tensor:
            first = self.enc1(torch.cat((observations, visible), dim=1))
            second = self.enc2(F.avg_pool2d(first, 2, ceil_mode=True))
            return self.latent(F.avg_pool2d(second, 2, ceil_mode=True))

        def encode(
            self,
            observations: Tensor,
            mask: Tensor | None = None,
            *,
            apply_random_mask: bool = False,
        ) -> Tensor:
            """Return the spatial latent map for transfer or retrieval studies."""

            masked, visible = self.mask_inputs(observations, mask, force_random=apply_random_mask)
            return self._encode_masked(masked, visible)

        def embedding(
            self,
            observations: Tensor,
            mask: Tensor | None = None,
            *,
            apply_random_mask: bool = False,
        ) -> Tensor:
            latent = self.encode(observations, mask, apply_random_mask=apply_random_mask)
            return F.adaptive_avg_pool2d(latent, 1).flatten(1)

        @staticmethod
        def reconstruction_loss(
            prediction: Tensor,
            target: Tensor,
            visible_mask: Tensor | None = None,
        ) -> Tensor:
            """MSE normalized over hidden values, the canonical MAE-style loss."""

            if prediction.shape != target.shape:
                raise ValueError("prediction and target must have identical shapes")
            if visible_mask is None:
                return F.mse_loss(prediction, target)
            visible = _mask_channel(target, visible_mask)
            hidden = 1.0 - visible
            squared = (prediction - target).square() * hidden
            denominator = hidden.sum() * target.shape[1]
            if float(denominator.detach()) == 0.0:
                return F.mse_loss(prediction, target)
            return squared.sum() / denominator

        def forward(self, observations: Tensor, mask: Tensor | None = None) -> Tensor:
            masked, visible = self.mask_inputs(observations, mask)
            latent = self._encode_masked(masked, visible)
            decoded = self.dec2(latent)
            decoded = F.interpolate(decoded, scale_factor=2, mode="bilinear", align_corners=False)
            decoded = self.dec1(decoded)
            decoded = F.interpolate(
                decoded,
                size=observations.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            reconstruction = self.head(decoded)
            if self.preserve_visible and self.in_channels == self.out_channels:
                reconstruction = torch.where(visible.bool(), observations, reconstruction)
            return reconstruction

    @register_method("unet", aliases=("mask_unet", "mask_channel_unet"))
    class MaskChannelUNet(_PredictMixin, nn.Module):
        """Small U-Net receiving the values and observation mask as channels."""

        name = "unet"
        capabilities = _RECOVERY_CAPS

        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            width: int = 32,
            geometry_channels: int = 0,
        ) -> None:
            super().__init__()
            self.in_channels = in_channels
            if geometry_channels < 0:
                raise ValueError("geometry_channels must be non-negative")
            self.geometry_channels = geometry_channels
            self.enc1 = _ConvBlock(in_channels + 1 + geometry_channels, width)
            self.enc2 = _ConvBlock(width, width * 2)
            self.bottleneck = _ConvBlock(width * 2, width * 4)
            self.dec2 = _ConvBlock(width * 4 + width * 2, width * 2)
            self.dec1 = _ConvBlock(width * 2 + width, width)
            self.head = nn.Conv2d(width, out_channels, 1)

        def forward(
            self,
            observations: Tensor,
            mask: Tensor | None = None,
            geometry: Tensor | None = None,
        ) -> Tensor:
            if observations.ndim != 4:
                raise ValueError("MaskChannelUNet expects observations with shape BCHW")
            inputs = [observations, _mask_channel(observations, mask)]
            geometry_tensor = _geometry_tensor(
                observations, geometry, self.geometry_channels
            )
            if geometry_tensor is not None:
                inputs.append(geometry_tensor)
            x = torch.cat(inputs, dim=1)
            e1 = self.enc1(x)
            e2 = self.enc2(F.avg_pool2d(e1, 2, ceil_mode=True))
            b = self.bottleneck(F.avg_pool2d(e2, 2, ceil_mode=True))
            up2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = self.dec2(torch.cat((up2, e2), dim=1))
            up1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            return self.head(self.dec1(torch.cat((up1, e1), dim=1)))

    class SpectralConv2d(nn.Module):
        """Truncated real-FFT convolution used by the compact FNO reference."""

        def __init__(self, in_channels: int, out_channels: int, modes_y: int, modes_x: int) -> None:
            super().__init__()
            self.in_channels, self.out_channels = in_channels, out_channels
            self.modes_y, self.modes_x = modes_y, modes_x
            scale = 1.0 / max(1, in_channels * out_channels)
            shape = (in_channels, out_channels, modes_y, modes_x)
            self.weight_pos = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
            self.weight_neg = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

        @staticmethod
        def _multiply(x: Tensor, weight: Tensor) -> Tensor:
            return torch.einsum("bixy,ioxy->boxy", x, weight)

        def forward(self, x: Tensor) -> Tensor:
            height, width = x.shape[-2:]
            result_dtype = x.dtype
            # CUDA FFT has no deterministic ComplexHalf implementation.  AMP
            # therefore cannot own this spectral block even though the local
            # convolutions around it can safely remain autocast.  Run both FFTs
            # and the complex multiply in float32/complex64, then restore the
            # activation dtype before merging with the local branch.
            with torch.autocast(device_type=x.device.type, enabled=False):
                transformed = torch.fft.rfft2(x.float(), norm="ortho")
                output = torch.zeros(
                    x.shape[0],
                    self.out_channels,
                    height,
                    width // 2 + 1,
                    device=x.device,
                    dtype=torch.cfloat,
                )
                my = min(self.modes_y, max(1, height // 2), self.weight_pos.shape[-2])
                mx = min(self.modes_x, width // 2 + 1, self.weight_pos.shape[-1])
                output[:, :, :my, :mx] = self._multiply(
                    transformed[:, :, :my, :mx], self.weight_pos[:, :, :my, :mx]
                )
                output[:, :, -my:, :mx] = self._multiply(
                    transformed[:, :, -my:, :mx], self.weight_neg[:, :, :my, :mx]
                )
                result = torch.fft.irfft2(output, s=(height, width), norm="ortho")
            return result.to(dtype=result_dtype)

    class _UFNOBranch2d(nn.Module):
        """Three-scale local U-shaped branch used inside a U-Fourier block.

        This is an independent 2-D adaptation of the architecture described by
        Wen et al.  It intentionally does not copy the authors' CC BY-NC-ND
        implementation.  All widths remain constant, matching the role of the
        U-Net path as a local residual correction to a Fourier layer.
        """

        def __init__(self, channels: int, dropout: float = 0.0) -> None:
            super().__init__()

            def down(stride: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Dropout(dropout),
                )

            def up(in_channels: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.ConvTranspose2d(in_channels, channels, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                )

            self.down_1 = down(2)
            self.down_2 = nn.Sequential(down(2), down(1))
            self.down_3 = nn.Sequential(down(2), down(1))
            self.up_2 = up(channels)
            self.up_1 = up(2 * channels)
            self.up_0 = up(2 * channels)
            self.output = nn.Conv2d(2 * channels, channels, 3, padding=1)

        def forward(self, state: Tensor) -> Tensor:
            level_1 = self.down_1(state)
            level_2 = self.down_2(level_1)
            level_3 = self.down_3(level_2)
            up_2 = self.up_2(level_3)
            if up_2.shape[-2:] != level_2.shape[-2:]:
                up_2 = F.interpolate(up_2, size=level_2.shape[-2:], mode="bilinear", align_corners=False)
            up_1 = self.up_1(torch.cat((level_2, up_2), dim=1))
            if up_1.shape[-2:] != level_1.shape[-2:]:
                up_1 = F.interpolate(up_1, size=level_1.shape[-2:], mode="bilinear", align_corners=False)
            up_0 = self.up_0(torch.cat((level_1, up_1), dim=1))
            if up_0.shape[-2:] != state.shape[-2:]:
                up_0 = F.interpolate(up_0, size=state.shape[-2:], mode="bilinear", align_corners=False)
            return self.output(torch.cat((state, up_0), dim=1))

    @register_method("ufno", aliases=("u_fno", "ufno2d", "u_fno_2d"))
    class UFNO2d(_PredictMixin, nn.Module):
        """Static-grid U-FNO-2D adaptation with 3 Fourier + 3 U-Fourier blocks.

        The official U-FNO release is a 3-D space--time model.  The benchmark
        uses the paper's six-block composition in two spatial dimensions and
        adds only the observation-mask, coordinates, and geometry channels
        required by PDE-OBS.  It must therefore be reported as ``U-FNO-2D
        (PDE-OBS adaptation)``, not as an exact upstream reproduction.
        """

        name = "ufno"
        capabilities = _RECOVERY_CAPS
        upstream_revision = "8315fd7b5bd75282b7efe42ee6b8de86543d13cc"

        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            width: int = 36,
            modes: int = 12,
            geometry_channels: int = 0,
            padding: int = 8,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            if min(width, modes) < 1 or padding < 0 or geometry_channels < 0:
                raise ValueError("U-FNO dimensions must be positive and padding non-negative")
            self.geometry_channels = geometry_channels
            self.padding = int(padding)
            feature_channels = in_channels + 1 + geometry_channels + 2
            self.lift = nn.Conv2d(feature_channels, width, 1)
            self.spectral = nn.ModuleList(
                SpectralConv2d(width, width, modes, modes) for _ in range(6)
            )
            self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(6))
            self.u_branches = nn.ModuleList(
                _UFNOBranch2d(width, dropout=dropout) for _ in range(3)
            )
            self.project = nn.Sequential(
                nn.Conv2d(width, 128, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, out_channels, 1),
            )

        def forward(
            self,
            observations: Tensor,
            mask: Tensor | None = None,
            geometry: Tensor | None = None,
        ) -> Tensor:
            if observations.ndim != 4:
                raise ValueError("UFNO2d expects observations with shape BCHW")
            height, width = observations.shape[-2:]
            y = torch.linspace(0, 1, height, dtype=observations.dtype, device=observations.device)
            x = torch.linspace(0, 1, width, dtype=observations.dtype, device=observations.device)
            gy, gx = torch.meshgrid(y, x, indexing="ij")
            grid = torch.stack((gy, gx))[None].expand(observations.shape[0], -1, -1, -1)
            inputs = [observations, _mask_channel(observations, mask), grid]
            geometry_tensor = _geometry_tensor(observations, geometry, self.geometry_channels)
            if geometry_tensor is not None:
                inputs.append(geometry_tensor)
            state = self.lift(torch.cat(inputs, dim=1))
            if self.padding:
                state = F.pad(state, (0, self.padding, 0, self.padding))
            for index, (spectral, local) in enumerate(
                zip(self.spectral, self.local, strict=True)
            ):
                update = spectral(state) + local(state)
                if index >= 3:
                    update = update + self.u_branches[index - 3](state)
                state = F.relu(update)
            if self.padding:
                state = state[..., :height, :width]
            return self.project(state)

    @register_method("fno", aliases=("fno2d", "compact_fno"))
    class CompactFNO2d(_PredictMixin, nn.Module):
        """Compact mask-channel FNO2d architectural reference.

        This is not an exact reproduction of the original FNO paper model.
        """

        name = "fno"
        capabilities = _RECOVERY_CAPS

        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            width: int = 32,
            modes: int = 12,
            layers: int = 4,
            geometry_channels: int = 0,
        ) -> None:
            super().__init__()
            if geometry_channels < 0:
                raise ValueError("geometry_channels must be non-negative")
            self.geometry_channels = geometry_channels
            self.lift = nn.Conv2d(in_channels + 1 + geometry_channels + 2, width, 1)
            self.spectral = nn.ModuleList(
                SpectralConv2d(width, width, modes, modes) for _ in range(layers)
            )
            self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
            self.norm = nn.ModuleList(nn.GroupNorm(min(8, width), width) for _ in range(layers))
            self.project = nn.Sequential(
                nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Conv2d(width * 2, out_channels, 1)
            )

        def forward(
            self,
            observations: Tensor,
            mask: Tensor | None = None,
            geometry: Tensor | None = None,
        ) -> Tensor:
            if observations.ndim != 4:
                raise ValueError("CompactFNO2d expects observations with shape BCHW")
            h, w = observations.shape[-2:]
            y = torch.linspace(0, 1, h, dtype=observations.dtype, device=observations.device)
            x = torch.linspace(0, 1, w, dtype=observations.dtype, device=observations.device)
            gy, gx = torch.meshgrid(y, x, indexing="ij")
            grid = torch.stack((gy, gx))[None].expand(observations.shape[0], -1, -1, -1)
            inputs = [observations, _mask_channel(observations, mask), grid]
            geometry_tensor = _geometry_tensor(
                observations, geometry, self.geometry_channels
            )
            if geometry_tensor is not None:
                inputs.append(geometry_tensor)
            state = self.lift(torch.cat(inputs, dim=1))
            for spectral, local, norm in zip(self.spectral, self.local, self.norm, strict=True):
                state = F.gelu(norm(spectral(state) + local(state)))
            return self.project(state)

    @register_method("pino", aliases=("pino_static", "physics_informed_fno"))
    class PhysicsInformedFNO2d(CompactFNO2d):
        """FNO recovery backbone that requires the reviewed PINO objective.

        PINO is distinguished from FNO by the trainer-side PDE residual, not by
        silently changing the spectral architecture. The marker below makes a
        missing physics objective a configuration error.
        """

        name = "pino"
        capabilities = MethodCapabilities(
            tasks=frozenset({"recovery", "forward", "rollout"}),
            trainable=True,
            temporal=True,
            requires_mask=True,
            reference_only=True,
            notes=(
                "PDE-OBS PINO adaptation with an FNO backbone and separately configured "
                "static-recovery or periodic free-rollout residual; not an exact paper "
                "reproduction."
            ),
        )

        def __init__(self, *args: Any, physics_contract: str = "pino_static_fd_v1", **kwargs: Any) -> None:
            if physics_contract not in {"pino_static_fd_v1", "pino_rollout_spectral_v1"}:
                raise ValueError(f"unsupported PINO physics contract {physics_contract!r}")
            super().__init__(*args, **kwargs)
            self.required_physics_loss = physics_contract

    class _AntiAliasedBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv = _ConvBlock(in_channels, out_channels)

        def forward(self, x: Tensor) -> Tensor:
            # A fixed binomial low-pass before learned convolution reduces aliasing
            # without claiming the complete continuous-discrete CNO construction.
            channels = x.shape[1]
            kernel_1d = torch.tensor([1.0, 2.0, 1.0], dtype=x.dtype, device=x.device) / 4.0
            kernel = torch.outer(kernel_1d, kernel_1d)[None, None].expand(channels, 1, -1, -1)
            filtered = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel, groups=channels)
            return self.conv(filtered)

    @register_method("cno", aliases=("cno2d", "compact_cno"))
    class CompactCNO2d(_PredictMixin, nn.Module):
        """Compact anti-aliased CNO-like reference, not an exact CNO reproduction."""

        name = "cno"
        capabilities = _RECOVERY_CAPS

        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            width: int = 32,
            geometry_channels: int = 0,
        ) -> None:
            super().__init__()
            if geometry_channels < 0:
                raise ValueError("geometry_channels must be non-negative")
            self.geometry_channels = geometry_channels
            self.enc = _AntiAliasedBlock(in_channels + 1 + geometry_channels + 2, width)
            self.mid = _AntiAliasedBlock(width, width * 2)
            self.dec = _AntiAliasedBlock(width * 2 + width, width)
            self.head = nn.Conv2d(width, out_channels, 1)

        def forward(
            self,
            observations: Tensor,
            mask: Tensor | None = None,
            geometry: Tensor | None = None,
        ) -> Tensor:
            if observations.ndim != 4:
                raise ValueError("CompactCNO2d expects observations with shape BCHW")
            h, w = observations.shape[-2:]
            y = torch.linspace(-1, 1, h, dtype=observations.dtype, device=observations.device)
            x = torch.linspace(-1, 1, w, dtype=observations.dtype, device=observations.device)
            gy, gx = torch.meshgrid(y, x, indexing="ij")
            grid = torch.stack((gy, gx))[None].expand(observations.shape[0], -1, -1, -1)
            inputs = [observations, _mask_channel(observations, mask), grid]
            geometry_tensor = _geometry_tensor(
                observations, geometry, self.geometry_channels
            )
            if geometry_tensor is not None:
                inputs.append(geometry_tensor)
            encoded = self.enc(torch.cat(inputs, dim=1))
            low = F.avg_pool2d(encoded, 2, ceil_mode=True)
            middle = self.mid(low)
            up = F.interpolate(
                middle,
                size=encoded.shape[-2:],
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            return self.head(self.dec(torch.cat((up, encoded), dim=1)))

    class ConvLSTMCell(nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
            super().__init__()
            self.hidden_channels = hidden_channels
            padding = kernel_size // 2
            self.gates = nn.Conv2d(
                in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding
            )

        def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
            hidden, cell = state
            i, f, g, o = self.gates(torch.cat((x, hidden), dim=1)).chunk(4, dim=1)
            cell = torch.sigmoid(f) * cell + torch.sigmoid(i) * torch.tanh(g)
            hidden = torch.sigmoid(o) * torch.tanh(cell)
            return hidden, cell

    @register_method("convlstm", aliases=("conv_lstm",))
    class ConvLSTM(_PredictMixin, nn.Module):
        """Small recurrent rollout baseline accepting BTCHW histories."""

        name = "convlstm"
        capabilities = _ROLLOUT_CAPS

        def __init__(
            self, in_channels: int = 1, out_channels: int = 1, hidden_channels: int = 32
        ) -> None:
            super().__init__()
            self.in_channels, self.out_channels = in_channels, out_channels
            self.encoder = nn.Conv2d(in_channels + 1, hidden_channels, 3, padding=1)
            self.cell = ConvLSTMCell(hidden_channels, hidden_channels)
            self.head = nn.Conv2d(hidden_channels, out_channels, 1)
            self.feedback = nn.Conv2d(out_channels + 1, hidden_channels, 3, padding=1)

        def forward(
            self,
            observations: Tensor,
            mask: Tensor | None = None,
            *,
            horizon: int = 1,
            teacher_forcing: Tensor | None = None,
        ) -> Tensor:
            if observations.ndim == 4:
                observations = observations[:, None]
            if observations.ndim != 5:
                raise ValueError("ConvLSTM expects observations with shape BTCHW or BCHW")
            if horizon < 1:
                raise ValueError("horizon must be at least one")
            batch, steps, _, height, width = observations.shape
            hidden = observations.new_zeros((batch, self.cell.hidden_channels, height, width))
            cell = torch.zeros_like(hidden)
            for step in range(steps):
                frame = observations[:, step]
                step_mask = mask[:, step] if mask is not None and mask.ndim == 5 else mask
                encoded = self.encoder(torch.cat((frame, _mask_channel(frame, step_mask)), dim=1))
                hidden, cell = self.cell(encoded, (hidden, cell))
            predictions = []
            full_mask = observations.new_ones((batch, 1, height, width))
            for step in range(horizon):
                prediction = self.head(hidden)
                predictions.append(prediction)
                feedback = (
                    teacher_forcing[:, step]
                    if teacher_forcing is not None and step < teacher_forcing.shape[1]
                    else prediction
                )
                encoded = self.feedback(torch.cat((feedback, full_mask), dim=1))
                hidden, cell = self.cell(encoded, (hidden, cell))
            return torch.stack(predictions, dim=1)

    @register_method("autoregressive", aliases=("autoregressive_unet", "ar_unet"))
    class AutoregressiveModel(_PredictMixin, nn.Module):
        """Turn any one-step BCHW model into a multi-step rollout model."""

        name = "autoregressive"
        capabilities = _ROLLOUT_CAPS

        def __init__(self, one_step_model: nn.Module | None = None, **unet_kwargs: Any) -> None:
            super().__init__()
            self.one_step_model = one_step_model or MaskChannelUNet(**unet_kwargs)
            self.required_physics_loss = getattr(self.one_step_model, "required_physics_loss", "none")

        def forward(
            self,
            observations: Tensor,
            mask: Tensor | None = None,
            *,
            horizon: int = 1,
            teacher_forcing: Tensor | None = None,
            geometry: Tensor | None = None,
            metadata: Any | None = None,
        ) -> Tensor:
            current = observations[:, -1] if observations.ndim == 5 else observations
            predictions = []
            for step in range(horizon):
                current_mask = mask[:, -1] if mask is not None and mask.ndim == 5 else mask
                callable_target = self.one_step_model.forward
                signature = inspect.signature(callable_target)
                accepts_extra = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
                context = {"mask": current_mask, "geometry": geometry, "metadata": metadata}
                kwargs = {
                    name: value
                    for name, value in context.items()
                    if value is not None and (accepts_extra or name in signature.parameters)
                }
                prediction = self.one_step_model(current, **kwargs)
                predictions.append(prediction)
                current = (
                    teacher_forcing[:, step]
                    if teacher_forcing is not None and step < teacher_forcing.shape[1]
                    else prediction
                )
                mask = None  # predicted states are fully specified after step one
            return torch.stack(predictions, dim=1)


else:

    class _TorchMissing:
        capabilities = _RECOVERY_CAPS

        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()

    @register_method("unet", aliases=("mask_unet", "mask_channel_unet"))
    class MaskChannelUNet(_TorchMissing):
        name = "unet"

    @register_method("fno", aliases=("fno2d", "compact_fno"))
    class CompactFNO2d(_TorchMissing):
        name = "fno"

    @register_method("ufno", aliases=("u_fno", "ufno2d", "u_fno_2d"))
    class UFNO2d(_TorchMissing):
        name = "ufno"

    @register_method("pino", aliases=("pino_static", "physics_informed_fno"))
    class PhysicsInformedFNO2d(_TorchMissing):
        name = "pino"
        required_physics_loss = "pino_static_fd_v1"

    @register_method("cno", aliases=("cno2d", "compact_cno"))
    class CompactCNO2d(_TorchMissing):
        name = "cno"

    @register_method("convlstm", aliases=("conv_lstm",))
    class ConvLSTM(_TorchMissing):
        name = "convlstm"
        capabilities = _ROLLOUT_CAPS

    @register_method("autoregressive", aliases=("autoregressive_unet", "ar_unet"))
    class AutoregressiveModel(_TorchMissing):
        name = "autoregressive"
        capabilities = _ROLLOUT_CAPS

    @register_method(
        "residual_cnn",
        aliases=("resnet", "resnet_encoder", "resnet_classifier", "supervised_multitask_small"),
    )
    class CompactResidualEncoder(_TorchMissing):
        name = "residual_cnn"
        capabilities = _ENCODER_CAPS

    @register_method(
        "mae_small",
        aliases=("masked_autoencoder", "masked_autoencoder_small", "mae_style_small"),
    )
    class MaskedAutoencoderSmall(_TorchMissing):
        name = "mae_small"
        capabilities = _MAE_CAPS


MODEL_FACTORIES: dict[str, Callable[..., Any]] = {
    "unet": MaskChannelUNet,
    "mask_channel_unet": MaskChannelUNet,
    "fno": CompactFNO2d,
    "fno2d": CompactFNO2d,
    "ufno": UFNO2d,
    "u_fno": UFNO2d,
    "ufno2d": UFNO2d,
    "u_fno_2d": UFNO2d,
    "pino": PhysicsInformedFNO2d,
    "pino_static": PhysicsInformedFNO2d,
    "physics_informed_fno": PhysicsInformedFNO2d,
    "cno": CompactCNO2d,
    "cno2d": CompactCNO2d,
    "convlstm": ConvLSTM,
    "autoregressive": AutoregressiveModel,
    "autoregressive_unet": AutoregressiveModel,
    "residual_cnn": CompactResidualEncoder,
    "resnet": CompactResidualEncoder,
    "resnet_encoder": CompactResidualEncoder,
    "resnet_classifier": CompactResidualEncoder,
    "supervised_multitask_small": CompactResidualEncoder,
    "mae_small": MaskedAutoencoderSmall,
    "masked_autoencoder": MaskedAutoencoderSmall,
    "masked_autoencoder_small": MaskedAutoencoderSmall,
    "mae_style_small": MaskedAutoencoderSmall,
}


@register_method("autoregressive_fno", aliases=("ar_fno",))
def create_autoregressive_fno(**kwargs: Any) -> Any:
    """Create the compact FNO one-step model behind the AR rollout wrapper."""

    one_step_model = CompactFNO2d(**kwargs)
    model = AutoregressiveModel(one_step_model=one_step_model)
    model.name = "autoregressive_fno"
    return model


MODEL_FACTORIES["autoregressive_fno"] = create_autoregressive_fno
MODEL_FACTORIES["ar_fno"] = create_autoregressive_fno


def create_model(name: str, /, **kwargs: Any) -> Any:
    """Create a neural baseline without importing the general method registry."""

    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        factory = MODEL_FACTORIES[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model {name!r}. Available models: {', '.join(sorted(MODEL_FACTORIES))}"
        ) from exc
    return factory(**kwargs)
