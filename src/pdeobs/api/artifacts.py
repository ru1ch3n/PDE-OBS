"""Reusable trained-model artifacts.

Layout of an artifact directory::

    model.json            manifest (schema, architecture, task, io schema, versions, hashes)
    weights.pt            plain state_dict, loadable with torch.load(weights_only=True)
    training_contract.json  data identity, observation, loss, budget, physics contract
    checkpoints/last.pt   the Trainer checkpoint (format_version 2, includes the training config)
    resolved_config.json  the full experiment configuration handed to Trainer/runner

``load_predictor`` rebuilds the exact structure from the manifest; an explicitly
supplied structure that disagrees is rejected, missing weights are an error, and
random initialisation is only reachable through ``allow_untrained=True`` which is
labelled as a development mode, never as inference.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .. import __solver_version__, __version__
from . import specs as S
from .models import ModelConfig, build, count_parameters, resolve_model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class ModelArtifact:
    path: Path
    manifest: dict[str, Any]

    @property
    def model_config(self) -> ModelConfig:
        arch = self.manifest["architecture"]
        return ModelConfig(name=arch["name"], task=self.manifest["task"], kwargs=dict(arch["kwargs"]), preset=arch.get("preset"),
                           temporal_wrapper=arch.get("temporal_wrapper"), physics_contract=arch.get("physics_contract"),
                           origin=arch.get("origin", "custom"))

    def _file_of(self, key: str) -> Path | None:
        # parameter-free methods store ``"weights": null`` (not a mapping); treat that as "no file"
        entry = self.manifest.get(key)
        name = entry.get("file") if isinstance(entry, Mapping) else None
        return self.path / name if name else None

    @property
    def weights(self) -> Path | None:
        return self._file_of("weights")

    @property
    def fitted_state(self) -> Path | None:
        return self._file_of("fitted_state")


def save_artifact(out: str | Path, *, model: Any, config: ModelConfig, training_contract: Mapping[str, Any],
                  resolved_config: Mapping[str, Any], io_schema: Mapping[str, Any], trainer_checkpoint: Path | None = None,
                  extra: Mapping[str, Any] | None = None) -> ModelArtifact:
    """Write the artifact directory for a trained model (torch module or fitted classical method)."""
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": S.MODEL_ARTIFACT_VERSION,
        "api_version": S.API_VERSION,
        "pdeobs_version": __version__,
        "numerical_kernel_version": __solver_version__,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": config.task,
        "architecture": {"name": config.name, "kwargs": dict(config.kwargs), "preset": config.preset, "origin": config.origin,
                         "temporal_wrapper": config.temporal_wrapper if config.task == "rollout" else None,
                         "physics_contract": config.physics_contract, "method": config.method_config(),
                         "parameters": count_parameters(model)},
        "io_schema": dict(io_schema),
        "training_contract_file": "training_contract.json",
        "resolved_config_file": "resolved_config.json",
        "kind": "example_checkpoint",  # smoke/example unless a run explicitly labels it otherwise
    }
    if extra:
        manifest.update(dict(extra))
    if hasattr(model, "state_dict"):
        import torch
        weights = directory / "weights.pt"
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(state, weights)
        manifest["weights"] = {"file": "weights.pt", "format": "torch_state_dict", "sha256": _sha256(weights),
                               "tensors": len(state)}
    elif hasattr(model, "save"):
        fitted = directory / "fitted_state.npz"
        model.save(fitted)
        manifest["fitted_state"] = {"file": fitted.name, "sha256": _sha256(fitted)}
    else:
        manifest["weights"] = None
        manifest["kind"] = "parameter_free_method"
    if trainer_checkpoint is not None and Path(trainer_checkpoint).is_file():
        manifest["trainer_checkpoint"] = {"file": str(Path(trainer_checkpoint).relative_to(directory)) if Path(trainer_checkpoint).is_relative_to(directory) else str(trainer_checkpoint),
                                          "sha256": _sha256(Path(trainer_checkpoint))}
    _json_dump(directory / "training_contract.json", training_contract)
    _json_dump(directory / "resolved_config.json", resolved_config)
    _json_dump(directory / "model.json", manifest)
    return ModelArtifact(directory, manifest)


def load_artifact(path: str | Path) -> ModelArtifact:
    source = Path(path)
    manifest_path = source / "model.json" if source.is_dir() else source
    if not manifest_path.is_file():
        raise FileNotFoundError(f"model artifact manifest not found: {manifest_path}")
    manifest = _json_load(manifest_path)
    if manifest.get("schema_version") != S.MODEL_ARTIFACT_VERSION:
        raise ValueError(f"unsupported model artifact schema {manifest.get('schema_version')!r}; expected {S.MODEL_ARTIFACT_VERSION!r}. "
                         "Legacy checkpoints (last.pt without model.json) need an explicit legacy adapter: see load_legacy_checkpoint")
    return ModelArtifact(manifest_path.parent, manifest)


@dataclass
class Predictor:
    """A model restored from an artifact, ready for target-free inference."""

    model: Any
    config: ModelConfig
    artifact: ModelArtifact | None
    io_schema: dict[str, Any]
    device: str = "cpu"
    trained: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def task(self) -> str:
        return self.config.task

    @property
    def is_torch(self) -> bool:
        return hasattr(self.model, "parameters") and hasattr(self.model, "state_dict")

    def to(self, device: str) -> Predictor:
        if self.is_torch:
            from ..training import resolve_device
            resolved = resolve_device(device)
            self.model.to(resolved)
            self.device = str(resolved)
        return self


def load_predictor(artifact: str | Path | ModelArtifact, *, device: str = "cpu", model: str | Mapping[str, Any] | None = None,
                   params: Mapping[str, Any] | None = None, allow_untrained: bool = False) -> Predictor:
    """Restore a model from its artifact; the manifest defines the structure.

    If ``model``/``params`` are given they must agree with the manifest exactly (a mismatch is an
    error, never a silent partial load).  ``allow_untrained=True`` is a development mode that builds
    the structure without weights; the resulting predictor is labelled untrained and cannot be used
    where a trained predictor is required.
    """
    art = artifact if isinstance(artifact, ModelArtifact) else load_artifact(artifact)
    config = art.model_config
    if model is not None or params is not None:
        requested = resolve_model(model or config.name, task=config.task, params=params or {})
        if requested.name != config.name or requested.kwargs != config.kwargs:
            raise ValueError(f"requested structure {requested.name} {requested.kwargs} differs from the artifact "
                             f"{config.name} {config.kwargs}; refusing to reconstruct a different model")
    module = build(config)
    weights = art.weights
    fitted = art.fitted_state
    trained = True
    if allow_untrained:
        # explicit development mode: build the structure only, never load weights;
        # the predictor is labelled untrained and predict() refuses learned inference
        trained = False
        weights = None
        fitted = None
    if weights is not None:
        import torch
        from ..training import load_checkpoint_payload
        if not weights.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights}")
        recorded = (art.manifest.get("weights") or {}).get("sha256")
        if recorded and _sha256(weights) != recorded:
            raise ValueError("artifact weights do not match the recorded sha256; refusing to load altered weights")
        state = load_checkpoint_payload(weights, map_location="cpu")
        state = state.get("model_state", state)
        missing, unexpected = module.load_state_dict(dict(state), strict=True), None
        del missing, unexpected
        from ..training import resolve_device
        module.to(resolve_device(device))
    elif fitted is not None:
        if not fitted.is_file():
            raise FileNotFoundError(f"fitted state missing: {fitted}")
        module.load(fitted)
    elif art.manifest.get("kind") == "parameter_free_method":
        trained = True  # nothing to learn
    elif not allow_untrained:
        raise ValueError("artifact has no trained weights; learned inference requires weights "
                         "(use allow_untrained=True only for development/initialisation tests)")
    if hasattr(module, "eval"):
        module.eval()
    return Predictor(model=module, config=config, artifact=art, io_schema=dict(art.manifest.get("io_schema", {})),
                     device=device, trained=trained,
                     provenance={"artifact": str(art.path), "weights_sha256": art.manifest.get("weights", {}).get("sha256") if art.manifest.get("weights") else None,
                                 "kind": art.manifest.get("kind")})


def load_legacy_checkpoint(checkpoint: str | Path, *, model: str | Mapping[str, Any], task: str,
                           params: Mapping[str, Any] | None = None, io_schema: Mapping[str, Any] | None = None,
                           device: str = "cpu") -> Predictor:
    """Explicit adapter for a v0.1.x Trainer checkpoint (``last.pt``/``best.pt``) without ``model.json``.

    The structure is *not* guessed from the file name: the caller states it, the checkpoint's own
    stored training config (format_version 2) is cross-checked for task/physics contract, and any
    field that is missing is recorded as unknown in the returned provenance.
    """
    from ..training import load_checkpoint_payload, resolve_device

    path = Path(checkpoint)
    payload = load_checkpoint_payload(path, map_location="cpu")
    config = resolve_model(model, task=task, params=params or {})
    module = build(config)
    state = payload.get("model_state", payload)
    module.load_state_dict(dict(state), strict=True)
    stored = payload.get("config") if isinstance(payload, Mapping) else None
    missing = []
    checked: dict[str, Any] = {}
    if isinstance(stored, Mapping):
        if stored.get("task") not in (None, task):
            raise ValueError(f"checkpoint was trained for task {stored.get('task')!r}, not {task!r}")
        if config.name == "pino" and stored.get("physics_loss") not in (None, config.physics_contract):
            raise ValueError("checkpoint physics contract differs from the requested PINO contract")
        checked = {k: stored.get(k) for k in ("task", "epochs", "loss", "physics_loss", "physics_loss_weight", "seed", "horizon", "history_steps")}
    else:
        missing.append("training config (legacy payload without format_version 2)")
    for key in ("observation", "data identity", "normalization", "training budget"):
        missing.append(key)
    module.to(resolve_device(device))
    module.eval()
    return Predictor(model=module, config=config, artifact=None, io_schema=dict(io_schema or {}), device=device, trained=True,
                     provenance={"legacy_checkpoint": str(path), "sha256": _sha256(path), "format_version": payload.get("format_version"),
                                 "epoch": payload.get("epoch"), "checked_training_config": checked, "unknown_fields": missing,
                                 "note": "structure supplied by the caller, not inferred from the file"})
