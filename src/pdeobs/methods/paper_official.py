"""Fail-closed adapters for exact, commit-pinned author implementations.

The author source is not vendored into PDE-OBS.  Every construction verifies a
clean official checkout, its exact Git revision, and the selected source file
digest before importing it.  The adapters only translate PDE-OBS's BCHW sparse
recovery interface to the authors' tensor layout; they do not change the
authors' architecture.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from .base import MethodCapabilities, register_method

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - optional neural dependency
    torch = None
    Tensor = Any
    nn = None


OFFICIAL_CNO_REVISION = "bd9362c8c192d6f160129a7a85b1fe00d6c41523"
OFFICIAL_SOURCE_SHA256 = {
    "_OtherModels/BaselinesModules.py": (
        "b0899a7047a8240dc365550e7d2dae7214bfad1b1500def604bd38236f5ed8dd"
    ),
    "_OtherModels/FNOModules.py": (
        "a0872e258ff7b2d51e67f3c2b51cc8ce9db911116bd158087e5b9f1336af7608"
    ),
    "CNOModule.py": "ce87fc2f839a9aec802cf72a96777aa279b4e87b9edb3985632d5ccee279f8c7",
}

_RECOVERY_CAPS = MethodCapabilities(
    tasks=frozenset({"recovery"}),
    trainable=True,
    temporal=False,
    requires_mask=False,
    supports_multichannel=False,
    reference_only=False,
    notes=(
        "Exact author architecture loaded from a clean, commit-pinned checkout. "
        "PDE-OBS supplies a declared zero-filled sparse-recovery task adapter."
    ),
)


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("Paper-setting neural methods require PyTorch")


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Cannot attest official checkout {root}: {exc}") from exc
    return result.stdout.strip()


def _git_blob_sha256(root: Path, revision: str, source_file: str) -> str:
    """Hash committed blob bytes, independent of checkout line-ending filters."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{source_file}"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"Cannot read official source blob {revision}:{source_file}: {exc}"
        ) from exc
    return hashlib.sha256(result.stdout).hexdigest()


@lru_cache(maxsize=16)
def attest_official_checkout(
    upstream_root: str,
    upstream_revision: str,
    source_file: str,
    source_file_sha256: str,
) -> Path:
    """Return an attested source path or fail before importing author code."""

    root = Path(upstream_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Official model checkout does not exist: {root}")
    if upstream_revision != OFFICIAL_CNO_REVISION:
        raise ValueError("The adapter only supports the reviewed CNO paper-era revision")
    if source_file not in OFFICIAL_SOURCE_SHA256:
        raise ValueError(f"Unreviewed official source file: {source_file}")
    expected_source_sha = OFFICIAL_SOURCE_SHA256[source_file]
    if source_file_sha256 != expected_source_sha:
        raise ValueError("Configured official source SHA256 differs from the reviewed digest")
    actual_revision = _git(root, "rev-parse", "--verify", "HEAD")
    if actual_revision != upstream_revision:
        raise ValueError(
            f"Official checkout revision mismatch: {actual_revision} != {upstream_revision}"
        )
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Official checkout is not clean; refusing modified or untracked source")
    source = (root / source_file).resolve()
    if root not in source.parents or not source.is_file():
        raise ValueError("Official source path is missing or escaped its checkout")
    actual_source_sha = _git_blob_sha256(root, upstream_revision, source_file)
    if actual_source_sha != expected_source_sha:
        raise ValueError(
            f"Official source SHA256 mismatch: {actual_source_sha} != {expected_source_sha}"
        )
    return source


def _load_author_module(
    *,
    upstream_root: str,
    upstream_revision: str,
    source_file: str,
    source_file_sha256: str,
) -> Any:
    source = attest_official_checkout(
        upstream_root,
        upstream_revision,
        source_file,
        source_file_sha256,
    )
    root = Path(upstream_root).expanduser().resolve()
    module_name = "_pdeobs_official_" + hashlib.sha256(
        f"{root}|{source_file}|{upstream_revision}".encode()
    ).hexdigest()[:16]
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot construct import spec for {source}")
    module = importlib.util.module_from_spec(spec)
    search_paths = [str(root), str(root / "_OtherModels")]
    sys.path[:0] = search_paths
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        del sys.path[: len(search_paths)]
    sys.modules[module_name] = module
    return module


def _masked_observations(observations: Tensor, mask: Tensor | None) -> Tensor:
    if observations.ndim != 4 or observations.shape[1] != 1:
        raise ValueError("Paper-setting adapters require BCHW one-channel inputs")
    if tuple(observations.shape[-2:]) != (64, 64):
        raise ValueError("Reviewed CNO-paper settings require 64x64 inputs")
    if mask is None:
        return observations
    if mask.ndim == 3:
        mask = mask[:, None]
    if mask.shape[1] == 1 and observations.shape[1] != 1:
        mask = mask.expand(-1, observations.shape[1], -1, -1)
    if mask.shape != observations.shape:
        raise ValueError("Observation mask shape does not match the paper-setting input")
    return observations * mask.to(dtype=observations.dtype)


# The registry is also imported by NumPy-only baselines. Do not dereference an
# optional torch module during import; every concrete neural constructor still
# calls _require_torch() before initialization or external source access.
class _OfficialAdapter(nn.Module if nn is not None else object):
    capabilities = _RECOVERY_CAPS
    expected_by_pde: dict[str, dict[str, Any]] = {}

    def _verify_setting(self, paper_pde: str, values: dict[str, Any]) -> None:
        expected = self.expected_by_pde.get(paper_pde)
        if expected is None:
            raise ValueError(f"No reviewed paper setting exists for PDE {paper_pde!r}")
        if values != expected:
            raise ValueError(
                f"Official {paper_pde} setting mismatch: expected {expected}, received {values}"
            )


@register_method("paper_unet", hidden=True)
class OfficialPaperUNet(_OfficialAdapter):
    """Authors' U-Net baseline with Table 7's exact PDE-specific width."""

    name = "paper_unet"
    expected_by_pde = {
        "poisson": {"channels": 32},
        "navier_stokes": {"channels": 64},
    }

    def __init__(
        self,
        *,
        upstream_root: str,
        upstream_revision: str,
        source_file_sha256: str,
        paper_pde: str,
        channels: int,
        initialization_seed: int = 4,
    ) -> None:
        _require_torch()
        super().__init__()
        self._verify_setting(paper_pde, {"channels": int(channels)})
        if initialization_seed != 4:
            raise ValueError("Reviewed author-code initialization seed is 4")
        module = _load_author_module(
            upstream_root=upstream_root,
            upstream_revision=upstream_revision,
            source_file="_OtherModels/BaselinesModules.py",
            source_file_sha256=source_file_sha256,
        )
        torch.manual_seed(initialization_seed)
        # ``UNetOrg`` is the architecture whose width-32/64 parameter counts
        # match Table 7 (7.8M/31.0M).  The separate six-level ``UNet`` class is
        # not the selected benchmark architecture.
        self.model = module.UNetOrg(1, 1, int(channels), bilinear=False)

    def forward(self, observations: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.model(_masked_observations(observations, mask))


@register_method("paper_fno", hidden=True)
class OfficialPaperFNO2d(_OfficialAdapter):
    """Authors' FNO baseline with Table 11's exact PDE-specific architecture."""

    name = "paper_fno"
    expected_by_pde = {
        "poisson": {"width": 16, "modes": 16, "layers": 5, "padding": 0},
        "navier_stokes": {"width": 128, "modes": 16, "layers": 5, "padding": 0},
    }

    def __init__(
        self,
        *,
        upstream_root: str,
        upstream_revision: str,
        source_file_sha256: str,
        paper_pde: str,
        width: int,
        modes: int,
        layers: int,
        padding: int,
        initialization_seed: int = 4,
    ) -> None:
        _require_torch()
        super().__init__()
        setting = {
            "width": int(width),
            "modes": int(modes),
            "layers": int(layers),
            "padding": int(padding),
        }
        self._verify_setting(paper_pde, setting)
        if initialization_seed != 4:
            raise ValueError("Reviewed author-code initialization seed is 4")
        module = _load_author_module(
            upstream_root=upstream_root,
            upstream_revision=upstream_revision,
            source_file="_OtherModels/FNOModules.py",
            source_file_sha256=source_file_sha256,
        )
        architecture = {
            "width": setting["width"],
            "modes": setting["modes"],
            "n_layers": setting["layers"],
            "padding": setting["padding"],
            "include_grid": 1,
            "retrain": initialization_seed,
        }
        self.model = module.FNO2d(architecture, in_channels=1, out_channels=1, device="cpu")

    def forward(self, observations: Tensor, mask: Tensor | None = None) -> Tensor:
        values = _masked_observations(observations, mask)
        self.model.device = values.device
        result = self.model(values.permute(0, 2, 3, 1))
        return result.permute(0, 3, 1, 2).contiguous()


@register_method("paper_cno", hidden=True)
class OfficialPaperCNO2d(_OfficialAdapter):
    """Authors' classic CNO with Table 12's exact PDE-specific architecture."""

    name = "paper_cno"
    expected_by_pde = {
        "poisson": {"layers": 3, "channels": 16, "neck_residuals": 6, "level_residuals": 4},
        "navier_stokes": {
            "layers": 3,
            "channels": 32,
            "neck_residuals": 8,
            "level_residuals": 1,
        },
    }

    def __init__(
        self,
        *,
        upstream_root: str,
        upstream_revision: str,
        source_file_sha256: str,
        paper_pde: str,
        layers: int,
        channels: int,
        neck_residuals: int,
        level_residuals: int,
        initialization_seed: int = 4,
    ) -> None:
        _require_torch()
        super().__init__()
        setting = {
            "layers": int(layers),
            "channels": int(channels),
            "neck_residuals": int(neck_residuals),
            "level_residuals": int(level_residuals),
        }
        self._verify_setting(paper_pde, setting)
        if initialization_seed != 4:
            raise ValueError("Reviewed author-code initialization seed is 4")
        module = _load_author_module(
            upstream_root=upstream_root,
            upstream_revision=upstream_revision,
            source_file="CNOModule.py",
            source_file_sha256=source_file_sha256,
        )
        torch.manual_seed(initialization_seed)
        self.model = module.CNO(
            in_dim=1,
            in_size=64,
            N_layers=setting["layers"],
            N_res=setting["level_residuals"],
            N_res_neck=setting["neck_residuals"],
            channel_multiplier=setting["channels"],
            conv_kernel=3,
            cutoff_den=2.0001,
            filter_size=6,
            lrelu_upsampling=2,
            half_width_mult=0.8,
            radial=False,
            batch_norm=True,
            out_dim=1,
            out_size=1,
            expand_input=False,
            latent_lift_proj_dim=64,
            add_inv=True,
            activation="cno_lrelu",
        )

    def forward(self, observations: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.model(_masked_observations(observations, mask))


__all__ = [
    "OFFICIAL_CNO_REVISION",
    "OFFICIAL_SOURCE_SHA256",
    "OfficialPaperUNet",
    "OfficialPaperFNO2d",
    "OfficialPaperCNO2d",
    "attest_official_checkout",
]
