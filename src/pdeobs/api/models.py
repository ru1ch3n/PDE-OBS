"""One-line model construction with real, validated structure parameters."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

from . import specs as S


@dataclass
class ModelConfig:
    """Resolved architecture: the exact ``method`` mapping the runner/Trainer consume."""

    name: str
    task: str
    kwargs: dict[str, Any]
    preset: str | None
    temporal_wrapper: str | None
    physics_contract: str | None = None
    origin: str = "custom"  # paper | smoke | custom
    warnings: list[str] = field(default_factory=list)

    def method_config(self) -> dict[str, Any]:
        base = {"name": self.name, "kwargs": dict(self.kwargs)}
        if self.task == "rollout" and self.temporal_wrapper == "autoregressive":
            return {"name": "autoregressive", "base": base}
        return base

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "task": self.task, "kwargs": dict(self.kwargs), "preset": self.preset,
                "temporal_wrapper": self.temporal_wrapper if self.task == "rollout" else None,
                "physics_contract": self.physics_contract, "origin": self.origin, "method": self.method_config(),
                "warnings": list(self.warnings)}


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_model(model: str | Mapping[str, Any] | ModelConfig, *, task: str = "recovery", pde: str | None = None,
                  preset: str | None = None, params: Mapping[str, Any] | None = None,
                  in_channels: int | None = None) -> ModelConfig:
    """Validate a model request against the public parameter schema.

    ``preset`` may be ``smoke`` or ``paper``; explicit ``params`` override the preset
    (which then marks the configuration as ``custom``).  Unknown, mistyped or
    conflicting parameters raise before anything expensive happens.
    """
    if isinstance(model, ModelConfig):
        if params or preset:
            raise ValueError("resolve_model received an already resolved ModelConfig; build a new one to change it")
        return model
    if isinstance(model, Mapping):
        mapping = dict(model)
        name = mapping.pop("name", None)
        if name is None:
            raise ValueError("model mapping requires a 'name'")
        preset = mapping.pop("preset", preset)
        extra = dict(mapping.pop("kwargs", {}) or {})
        extra.update(mapping.pop("params", {}) or {})
        unknown_top = set(mapping)
        if unknown_top:
            raise ValueError(f"model mapping has unknown keys {sorted(unknown_top)}; use name/preset/params")
        merged = dict(extra)
        merged.update(params or {})
        return resolve_model(str(name), task=task, pde=pde, preset=preset, params=merged, in_channels=in_channels)

    task = str(task).lower().replace("-", "_")
    if task not in S.TASKS:
        raise ValueError(f"unknown task {task!r}; tasks: {S.TASKS}")
    name = S.normalize_model_name(model)
    spec = S.MODEL_SPECS[name]
    if task not in spec.tasks:
        raise ValueError(f"model {name!r} does not support task {task!r}; supported: {list(spec.tasks)}")
    if spec.upstream_wrapper:
        raise ValueError(f"{name!r} is an exact upstream wrapper requiring a pinned external checkout and explicit "
                         "attestation arguments; it is not served through the easy API")

    temporal = task == "rollout"
    if pde is not None and task == "rollout" and pde in S.STATIC_PDES:
        raise ValueError(f"{pde} is a static PDE family (T=1); rollout is not defined for it")
    kwargs: dict[str, Any] = {}
    origin = "custom"
    if preset is not None:
        key = str(preset).lower().replace("_", "-")
        if key in {"paper"}:
            key = "paper"
        table = spec.presets
        if key not in table:
            raise ValueError(f"model {name!r} has no preset {preset!r}; presets: {sorted(table)}")
        kwargs.update(copy.deepcopy(table[key]))
        if temporal and key in spec.temporal_presets:
            kwargs.update(copy.deepcopy(spec.temporal_presets[key]))
        origin = key
    given = dict(params or {})
    known = set(spec.param_names())
    unknown = sorted(set(given) - known)
    if unknown:
        raise ValueError(f"model {name!r} does not accept {unknown}; valid parameters: {sorted(known)}")
    # alias confusion guard: width/hidden and depth/layers are NOT interchangeable
    for wrong, right in (("hidden", "width"), ("width", "hidden"), ("depth", "layers"), ("layers", "depth"),
                         ("physics_slices", "slices"), ("blocks", "layers")):
        if wrong in given and wrong not in known and right in known:
            raise ValueError(f"model {name!r} uses {right!r}, not {wrong!r}")
    if given:
        if origin != "custom":
            origin = "custom"
    for p in spec.params:
        if p.name in given:
            kwargs[p.name] = p.validate(given[p.name], name)
        elif p.name not in kwargs and p.effect in {"io"} and p.name != "geometry_channels":
            kwargs[p.name] = p.default
    if in_channels is not None and "in_channels" in known:
        # io channels follow the served data (presets describe structure, not the field), unless given explicitly
        kwargs["in_channels"] = int(in_channels)
        if "out_channels" in known and "out_channels" not in given:
            kwargs["out_channels"] = int(in_channels)
    if spec.family == "neural" and "geometry_channels" in known:
        kwargs.setdefault("geometry_channels", 1)
    physics_contract = None
    if name == "pino":
        wanted = "pino_rollout_spectral_v1" if task == "rollout" else "pino_static_fd_v1"
        if task == "inverse":
            raise ValueError("PINO has no physics residual for the inverse task")
        contract = kwargs.get("physics_contract")
        if contract is not None and contract != wanted:
            raise ValueError(f"PINO physics_contract {contract!r} does not match task {task!r} (expected {wanted!r})")
        kwargs["physics_contract"] = wanted
        physics_contract = wanted
    _check_constraints(name, kwargs)
    warnings: list[str] = []
    if name == "ufno" and task != "rollout":
        warnings.append("U-FNO uses BatchNorm2d in its U-branches; use batch_size >= 2 for training")
    return ModelConfig(name=name, task=task, kwargs=kwargs, preset=preset if origin != "custom" else None,
                       temporal_wrapper=spec.temporal_wrapper, physics_contract=physics_contract, origin=origin,
                       warnings=warnings)


def _check_constraints(name: str, kwargs: Mapping[str, Any]) -> None:
    if name in {"fno", "pino"}:
        width = int(kwargs.get("width", 32))
        groups = min(8, width)
        if width % groups:
            raise ValueError(f"{name}: width={width} must be divisible by min(8, width)={groups} (GroupNorm); e.g. 8, 16, 24, 32, 40, 48, 64")
    if name in {"gnot", "transolver"}:
        hidden, heads = int(kwargs.get("hidden", 128)), int(kwargs.get("heads", 1))
        if hidden % heads:
            raise ValueError(f"{name}: hidden={hidden} must be divisible by heads={heads}")
    if name == "deeponet" and kwargs.get("branch_mode", "sensor_set") == "sensor_set" and kwargs.get("branch_width") is not None:
        raise ValueError("deeponet: branch_width only applies to branch_mode='spatial_cnn'")


def check_grid_compatibility(config: ModelConfig, resolution: tuple[int, int]) -> list[str]:
    """Cheap pre-flight checks of a resolved model against a grid size (before training)."""
    problems: list[str] = []
    h, w = resolution
    if config.name in {"fno", "pino", "ufno"}:
        modes = int(config.kwargs.get("modes", 12))
        if modes > max(1, h // 2) or modes > w // 2 + 1:
            problems.append(f"{config.name}: modes={modes} exceeds the grid {h}x{w} (max {max(1, h // 2)} x {w // 2 + 1}); "
                            "the spectral layer would silently clip them -- lower modes or raise the resolution")
    if config.name == "deeponet" and config.kwargs.get("branch_mode") == "spatial_cnn":
        grid = int(config.kwargs.get("branch_grid", 4))
        if grid > min(h, w):
            problems.append(f"deeponet: branch_grid={grid} exceeds the {h}x{w} grid")
    return problems


def build(config: ModelConfig) -> Any:
    """Construct the actual module through the existing runner factory (no duplicate logic)."""
    from ..runner import _method

    if S.MODEL_SPECS[config.name].requires_torch and not _torch_available():
        raise ImportError(f"model {config.name!r} requires PyTorch (install pdeobs[train]); the configuration itself is valid")
    return _method({"method": config.method_config()})


def count_parameters(model: Any) -> dict[str, int]:
    """Parameter count; complex weights count each real scalar (re + im) once."""
    try:
        import torch
    except ImportError:
        return {"total": 0, "trainable": 0}
    if not hasattr(model, "parameters"):
        return {"total": 0, "trainable": 0}
    total = trainable = 0
    for p in model.parameters():
        n = p.numel() * (2 if torch.is_complex(p) else 1)
        total += n
        if p.requires_grad:
            trainable += n
    return {"total": int(total), "trainable": int(trainable), "counting": "real scalars; complex weights count re+im"}


def summarize(config: ModelConfig, model: Any | None = None) -> dict[str, Any]:
    spec = S.MODEL_SPECS[config.name]
    out = {"model": config.name, "label": spec.label, "family": spec.family, "task": config.task,
           "origin": config.origin, "preset": config.preset, "resolved_architecture": dict(config.kwargs),
           "method_config": config.method_config(), "constraints": list(spec.constraints), "warnings": config.warnings}
    if model is not None:
        out["parameters"] = count_parameters(model)
        out["module"] = type(model).__name__
        inner = getattr(model, "one_step_model", None)
        if inner is not None:
            out["one_step_module"] = type(inner).__name__
    return out


def describe_model(name: str) -> dict[str, Any]:
    spec = S.model_spec(name)
    return {"name": spec.name, "label": spec.label, "family": spec.family, "aliases": list(spec.aliases),
            "tasks": list(spec.tasks), "main_paper_model": spec.main_paper_model, "requires_torch": spec.requires_torch,
            "parameters": [{"name": p.name, "type": p.type, "default": p.default, "doc": p.doc, "min": p.minimum,
                            "max": p.maximum, "choices": list(p.choices) if p.choices else None, "effect": p.effect}
                           for p in spec.params],
            "constraints": list(spec.constraints), "presets": {k: dict(v) for k, v in spec.presets.items()},
            "temporal_presets": {k: dict(v) for k, v in spec.temporal_presets.items()},
            "rollout_adapter": spec.temporal_wrapper, "notes": spec.notes, "upstream_wrapper": spec.upstream_wrapper,
            "example": f"pdeobs easy train --data <dataset> --task {spec.tasks[0]} --model {spec.name} "
                       + " ".join(f"--model-param {k}={v}" for k, v in list(spec.presets.get('smoke', {}).items())[:2])}


def list_models() -> list[dict[str, Any]]:
    return [{"name": s.name, "label": s.label, "family": s.family, "tasks": list(s.tasks),
             "main_paper_model": s.main_paper_model, "presets": sorted(s.presets), "aliases": list(s.aliases)}
            for s in S.MODEL_SPECS.values()]
