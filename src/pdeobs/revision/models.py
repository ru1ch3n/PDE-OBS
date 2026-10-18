"""Model construction for the October 2026 revision cohorts.

- FNO / CNO / the other benchmark networks come from the public method registry through the recipe's experiment
  configuration (`pdeobs.runner._method`).
- Voronoi-CNN is a PyTorch port of the authors' network: Fukami et al., "Global field reconstruction from sparse sensors
  with Voronoi tessellation-assisted deep learning", Nature Machine Intelligence 2021; code kfukami/Voronoi-CNN @99f043dc,
  Apache-2.0, whose README asks for academic research use only. The port follows the author code: 7 hidden 7x7
  convolutions with 48 filters + a linear 3x3 output; glorot-uniform weights, zero biases. The Voronoi image itself is the
  nearest-observation fill applied by the data pipeline (`pdeobs.revision.data.input_setup`).
- `NormWrap` is the fixed training-set scalar normaliser of the method-tuned (V4) recipes.
  - Network input: z~ = M * (z - mu) / sigma, or (z - mu) / sigma on every cell for nearest-filled inputs.
  - Output: mu + sigma * f(z~).
  - mu and sigma are persistent buffers, so a checkpoint carries them.

The upstream CNO (classic) and Senseiver code is not redistributed here; those cohorts need a local clone of the authors'
repositories and are outside the public replay.
"""
from __future__ import annotations

import inspect
from typing import Any, Mapping

import torch
from torch import nn


def _mask_channel(x, mask):
    if mask is None:
        return torch.ones((x.shape[0], 1, *x.shape[-2:]), dtype=x.dtype, device=x.device)
    if mask.ndim == 3:
        mask = mask[:, None]
    if mask.shape[1] != 1:
        mask = mask.amax(dim=1, keepdim=True)
    return mask.to(dtype=x.dtype, device=x.device)


class VoronoiCNN(nn.Module):
    """Voronoi-CNN (see module docstring). Inputs: (nearest-filled observations, mask, geometry) in BCHW."""

    def __init__(self, geometry_channels: int = 1, filters: int = 48, hidden_layers: int = 7, kernel: int = 7, out_kernel: int = 3):
        super().__init__()
        self.geometry_channels = int(geometry_channels)
        layers, c = [], 2 + self.geometry_channels
        for _ in range(hidden_layers):
            layers += [nn.Conv2d(c, filters, kernel, padding=kernel // 2), nn.ReLU()]
            c = filters
        layers.append(nn.Conv2d(filters, 1, out_kernel, padding=out_kernel // 2))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, observations, mask=None, geometry=None):
        parts = [observations, _mask_channel(observations, mask)]
        if self.geometry_channels:
            if geometry is None:
                raise ValueError("geometry is required when geometry_channels > 0")
            if geometry.ndim == 3:
                geometry = geometry[:, None]
            parts.append(geometry.to(dtype=observations.dtype, device=observations.device))
        return self.net(torch.cat(parts, dim=1))


def _call(module, args, kwargs):
    try:
        sig = inspect.signature(module.forward)
    except (TypeError, ValueError):
        return module(*args, **kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return module(*args, **kwargs)
    return module(*args, **{k: v for k, v in kwargs.items() if k in sig.parameters})


class NormWrap(nn.Module):
    """Fixed training-set scalar normaliser around any benchmark network (see module docstring)."""

    def __init__(self, base: nn.Module, mu: float = 0.0, sigma: float = 1.0, filled: bool = False):
        super().__init__()
        self.base = base
        self.register_buffer("v4_mu", torch.tensor(float(mu), dtype=torch.float32))
        self.register_buffer("v4_sigma", torch.tensor(float(sigma), dtype=torch.float32))
        self.filled = bool(filled)
        self.required_physics_loss = getattr(base, "required_physics_loss", "none")

    def forward(self, observations, mask=None, **kwargs):
        z = (observations - self.v4_mu) / self.v4_sigma
        if mask is not None and not self.filled:
            m = mask.to(observations.dtype)
            while m.dim() < observations.dim():
                m = m.unsqueeze(-3)
            z = z * m
        if kwargs.get("teacher_forcing") is not None:
            kwargs["teacher_forcing"] = (kwargs["teacher_forcing"] - self.v4_mu) / self.v4_sigma
        return self.v4_mu + self.v4_sigma * _call(self.base, (z,), dict(kwargs, mask=mask))


def build_model(recipe: Mapping[str, Any], cfg: Mapping[str, Any], norm: Mapping[str, float] | None = None) -> nn.Module:
    """The network of a recipe. For V4 recipes the normaliser statistics come with the checkpoint (buffers); pass `norm`
    ({"mu", "sigma"}, training-split statistics) only when training a new model."""
    from ..runner import _method
    method = recipe["method"]
    base_cfg = cfg["method"]["base"] if "base" in cfg["method"] else cfg["method"]
    geometry_channels = int(base_cfg.get("kwargs", {}).get("geometry_channels", 1))
    if method == "voronoi_cnn":
        model: nn.Module = VoronoiCNN(geometry_channels=geometry_channels)
        if cfg.get("task") == "rollout":
            from ..methods.neural import AutoregressiveModel
            model = AutoregressiveModel(one_step_model=model)
    elif method in ("cno_classic", "senseiver", "senseiver_v2"):
        raise NotImplementedError(f"{method} needs the authors' upstream code; it is not part of the public replay")
    else:
        model = _method(dict(cfg))
    if recipe.get("v4_norm"):
        filled = (recipe.get("input_representation") == "voronoi_mask_geometry" or recipe.get("fill_input") == "nearest"
                  or method == "voronoi_cnn")
        model = NormWrap(model, **(dict(norm) if norm else {}), filled=filled)
    return model


__all__ = ["NormWrap", "VoronoiCNN", "build_model"]
