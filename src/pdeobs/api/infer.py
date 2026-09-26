"""Target-free inference: observations + mask (+ geometry) -> raw predictions.

No ground truth is required, no score is computed here.  The model logic is
the existing one (``training._forward`` for modules, ``predict`` for NumPy
methods); this module only packages tensors and provenance.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import specs as S
from .artifacts import Predictor
from .data import DatasetHandle, InferenceInput, inference_input_from_dataset


@dataclass
class PredictionBundle:
    predictions: np.ndarray  # channels-last (N,H,W,C) or (N,T,H,W,C)
    sample_ids: list[str]
    time_indices: list[int]  # source frame indices of the predicted frames (rollout) or [target frame]
    task: str
    provenance: dict[str, Any] = field(default_factory=dict)
    inputs: InferenceInput | None = None

    def __post_init__(self) -> None:
        # Validate identities and frame labels ONCE, at construction, with the strict scorer's own
        # rules. save() writes time_indices with dtype=int64; without this check a fractional or
        # boolean label would be truncated on disk and reload as a valid integer frame.
        from ..strict_score import StrictValidationError, _ids, _times

        prediction = np.asarray(self.predictions)
        if prediction.ndim not in (4, 5):
            raise ValueError("predictions must be channels-last (N,H,W,C) or (N,T,H,W,C)")
        try:
            ids = _ids(list(self.sample_ids), "sample_ids")
            times = _times(list(self.time_indices), "time_indices")
        except StrictValidationError as exc:
            raise ValueError(str(exc)) from exc
        if len(ids) != prediction.shape[0] or len(set(ids)) != len(ids):
            raise ValueError("sample_ids must be unique and match the prediction batch size")
        expected_frames = prediction.shape[1] if prediction.ndim == 5 else 1
        if len(times) != expected_frames:
            raise ValueError(f"time_indices has {len(times)} entries but the predictions carry {expected_frames} frame(s)")
        if self.inputs is not None and list(self.inputs.sample_ids) != ids:
            raise ValueError("inputs.sample_ids must equal the prediction sample_ids in the same order")
        self.sample_ids = ids
        self.time_indices = times

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.predictions.shape)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": S.PREDICTION_BUNDLE_VERSION, "task": self.task, "shape": list(self.shape),
                "sample_ids": list(self.sample_ids), "time_indices": list(self.time_indices), "provenance": self.provenance,
                "finite": bool(np.isfinite(self.predictions).all())}

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite predictions: {target}")
        meta = json.dumps(self.to_dict(), sort_keys=True, default=str)
        if target.suffix.lower() == ".npz":
            payload = {"prediction": self.predictions, "prediction_ids": np.asarray(self.sample_ids, dtype=str),
                       "prediction_time_indices": np.asarray(self.time_indices, dtype=np.int64), "schema": np.asarray(meta)}
            if self.inputs is not None:
                payload["observation"] = self.inputs.observations
                payload["mask"] = self.inputs.mask
                if self.inputs.geometry is not None:
                    payload["geometry"] = self.inputs.geometry
            np.savez_compressed(target, **payload)
        elif target.suffix.lower() in {".h5", ".hdf5"}:
            import h5py
            with h5py.File(target, "x") as handle:
                handle.attrs["schema_json"] = meta
                handle.attrs["artifact_version"] = S.PREDICTION_BUNDLE_VERSION
                handle.create_dataset("prediction", data=self.predictions, compression="gzip")
                handle.create_dataset("prediction_ids", data=np.asarray(self.sample_ids, dtype=object), dtype=h5py.string_dtype("utf-8"))
                handle.create_dataset("prediction_time_indices", data=np.asarray(self.time_indices, dtype=np.int64))
                if self.inputs is not None:
                    handle.create_dataset("observation", data=self.inputs.observations, compression="gzip")
                    handle.create_dataset("mask", data=self.inputs.mask, compression="gzip")
                    if self.inputs.geometry is not None:
                        handle.create_dataset("geometry", data=self.inputs.geometry, compression="gzip")
        else:
            raise ValueError("predictions are saved as .npz or .h5")
        return target

    @classmethod
    def load(cls, path: str | Path) -> PredictionBundle:
        target = Path(path)
        from ..strict_score import _times
        if target.suffix.lower() == ".npz":
            with np.load(target, allow_pickle=False) as data:
                meta = json.loads(str(data["schema"]))
                if meta.get("schema_version") != S.PREDICTION_BUNDLE_VERSION:
                    raise ValueError("unsupported prediction bundle schema")
                inputs = None
                if "observation" in data.files and "mask" in data.files:
                    inputs = InferenceInput(observations=data["observation"], mask=data["mask"],
                                            geometry=data["geometry"] if "geometry" in data.files else None,
                                            sample_ids=[str(s) for s in data["prediction_ids"]])
                return cls(predictions=data["prediction"], sample_ids=[str(s) for s in data["prediction_ids"]],
                           time_indices=_times(data["prediction_time_indices"], "prediction_time_indices"), task=meta["task"],
                           provenance=meta.get("provenance", {}), inputs=inputs)
        if target.suffix.lower() in {".h5", ".hdf5"}:
            import h5py
            with h5py.File(target, "r") as handle:
                meta = json.loads(str(handle.attrs["schema_json"]))
                if meta.get("schema_version") != S.PREDICTION_BUNDLE_VERSION:
                    raise ValueError("unsupported prediction bundle schema")
                if handle.attrs.get("artifact_version") != S.PREDICTION_BUNDLE_VERSION:
                    raise ValueError("unsupported prediction bundle artifact version")
                ids = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in handle["prediction_ids"][()]]
                inputs = None
                if "observation" in handle and "mask" in handle:
                    inputs = InferenceInput(observations=handle["observation"][()], mask=handle["mask"][()],
                                            geometry=handle["geometry"][()] if "geometry" in handle else None, sample_ids=ids)
                return cls(predictions=handle["prediction"][()], sample_ids=ids,
                           time_indices=_times(handle["prediction_time_indices"][()], "prediction_time_indices"), task=meta["task"],
                           provenance=meta.get("provenance", {}), inputs=inputs)
        raise ValueError("unsupported prediction file format")


def _numpy_predict(method: Any, observations: np.ndarray, mask: np.ndarray, geometry: np.ndarray | None,
                   task: str, horizon: int | None) -> np.ndarray:
    import inspect
    callable_target = method.predict if hasattr(method, "predict") else method
    kwargs: dict[str, Any] = {"horizon": int(horizon)} if task == "rollout" else {}
    try:
        signature = inspect.signature(callable_target)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        accepts_extra = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        if geometry is not None and (accepts_extra or "geometry" in signature.parameters):
            kwargs["geometry"] = geometry
        if not accepts_extra:
            kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return np.asarray(callable_target(observations, mask.astype(bool), **kwargs))


def predict(predictor: Predictor, inputs: InferenceInput | str | Path | tuple[Any, Any] | DatasetHandle, *,
            observation: Any = "stored", horizon: int | None = None, batch_size: int = 4, device: str | None = None,
            out: str | Path | None = None, free_geometry: bool = False, sample_ids: list[str] | None = None,
            max_samples: int | None = None) -> PredictionBundle:
    """Predict from observations only.

    ``inputs`` is an :class:`InferenceInput` (or a path to its ``.npz`` package), or a dataset handle/path
    of complete records together with a synthetic ``observation`` (benchmark mode; the targets stay
    with the caller for a later ``evaluate``).  ``observation="stored"`` uses the mask carried by the
    input package and never generates another one.
    """
    if not predictor.trained:
        raise ValueError("this predictor has no trained weights (development mode); learned inference is refused")
    task = predictor.task
    hidden_targets = None
    package_mode = isinstance(inputs, InferenceInput) or (isinstance(inputs, (str, Path)) and Path(inputs).suffix.lower() == ".npz")
    if package_mode and isinstance(inputs, (str, Path)):
        package = InferenceInput.load(inputs)
    elif package_mode:
        package = inputs
    else:
        spec = observation
        if isinstance(observation, str) and observation == "stored":
            raise ValueError("benchmark-mode inputs (complete records) need an explicit synthetic observation; "
                             "'stored' is only valid for observation packages that carry their own mask")
        history_steps = int(predictor.io_schema.get("history_steps", 1))
        package, hidden_targets = inference_input_from_dataset(inputs, spec, task=task, history_steps=history_steps,
                                                               sample_ids=sample_ids, max_samples=max_samples)
    if package_mode and observation != "stored":
        raise ValueError("an observation package already carries its mask; observation must be 'stored' "
                         "(generating a new mask would hide the user's actual sensor positions)")
    if package.is_temporal != (task == "rollout"):
        raise ValueError(f"predictor task {task!r} expects {'a history (N,T,H,W,C)' if task == 'rollout' else 'a static field (N,H,W,C)'}, "
                         f"got observations of shape {package.observations.shape}")
    in_ch = int(predictor.io_schema.get("in_channels", predictor.config.kwargs.get("in_channels", 1)))
    if package.observations.shape[-1] != in_ch:
        raise ValueError(f"predictor expects {in_ch} input channel(s); observations have {package.observations.shape[-1]}")
    trained_res = predictor.io_schema.get("resolution_trained")
    warnings: list[str] = []
    if trained_res and list(package.shape_hw) != list(trained_res):
        warnings.append(f"inputs are {list(package.shape_hw)} but the model was trained at {trained_res}; the model executes but "
                        "this resolution was not evaluated")
    needs_geometry = int(predictor.config.kwargs.get("geometry_channels", 0)) > 0
    geometry = package.geometry
    if needs_geometry and geometry is None:
        if not free_geometry:
            raise ValueError("this model consumes a geometry channel (1=solid); supply inputs.geometry or pass "
                             "free_geometry=True to declare an obstacle-free domain explicitly")
        geometry = np.zeros((package.batch_size, *package.shape_hw, 1), dtype=np.float32)
        warnings.append("geometry declared obstacle-free (all zeros) by free_geometry=True")
    if task == "rollout":
        trained_h = predictor.io_schema.get("horizon_trained")
        horizon = int(horizon or trained_h or 1)
        if trained_h and horizon > int(trained_h):
            warnings.append(f"horizon {horizon} exceeds the trained horizon {trained_h}: executable, but not an evaluated range")
    started = time.perf_counter()
    outputs: list[np.ndarray] = []
    if predictor.is_torch:
        import torch
        from ..training import TrainingConfig, _forward, _to_tensor, resolve_device
        dev = resolve_device(device or predictor.device)
        predictor.model.to(dev)
        predictor.model.eval()
        adapter = TrainingConfig(task=task, epochs=1, amp=False, data_layout="channels_last",
                                 history_steps=int(predictor.io_schema.get("history_steps", 1)),
                                 horizon=horizon or 1, rollout_target_offset=0, device=str(dev))
        with torch.no_grad():
            for start in range(0, package.batch_size, int(batch_size)):
                stop = min(package.batch_size, start + int(batch_size))
                x = _to_tensor(package.observations[start:stop], dev, "channels_last")
                m = _to_tensor(package.mask[start:stop], dev, "channels_last", is_mask=True)
                g = _to_tensor(geometry[start:stop], dev, "channels_last") if geometry is not None else None
                y = _forward(predictor.model, x, m, None, adapter, False, g, None, horizon=horizon)
                if not torch.isfinite(y).all():
                    raise FloatingPointError("model produced non-finite predictions")
                arr = y.detach().cpu().numpy()
                arr = np.moveaxis(arr, 2 if arr.ndim == 5 else 1, -1)
                outputs.append(arr)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
    else:
        for start in range(0, package.batch_size, int(batch_size)):
            stop = min(package.batch_size, start + int(batch_size))
            arr = _numpy_predict(predictor.model, package.observations[start:stop], package.mask[start:stop],
                                 geometry[start:stop] if geometry is not None else None, task, horizon)
            outputs.append(np.asarray(arr, dtype=np.float32))
    prediction = np.concatenate(outputs, axis=0)
    frames = package.metadata.get("stored_frame_indices") if package.metadata else None
    if task == "rollout":
        hist = int(predictor.io_schema.get("history_steps", 1))
        if frames is not None and len(frames) >= hist + horizon:
            time_indices = [int(v) for v in frames[hist:hist + horizon]]
        elif package.time_indices:
            last = int(package.time_indices[-1])
            stride = (package.time_indices[-1] - package.time_indices[-2]) if len(package.time_indices) > 1 else 1
            time_indices = [last + stride * (k + 1) for k in range(horizon)]
            warnings.append("predicted frame indices extrapolated from the input history stride")
        else:
            time_indices = list(range(1, horizon + 1))
            warnings.append("no source frame coordinates in the input; predicted frames are numbered 1..horizon")
    else:
        time_indices = [int(frames[0])] if frames else ([int(package.time_indices[0])] if package.time_indices else [0])
    provenance = {"predictor": predictor.provenance, "model": predictor.config.to_dict(), "observation": package.metadata.get("observation", {"namespace": "stored"}),
                  "inputs_sha256": hashlib.sha256(package.observations.tobytes() + package.mask.tobytes()).hexdigest(),
                  "horizon": horizon, "device": predictor.device if not predictor.is_torch else str(dev), "batch_size": int(batch_size),
                  "seconds": round(time.perf_counter() - started, 4), "warnings": warnings, "targets_used": False,
                  "executable_range": {"resolution": list(package.shape_hw), "horizon": horizon},
                  "evaluated_range": {"resolution": trained_res, "horizon": predictor.io_schema.get("horizon_trained")},
                  "api_version": S.API_VERSION}
    bundle = PredictionBundle(predictions=prediction.astype(np.float32), sample_ids=list(package.sample_ids), time_indices=time_indices,
                              task=task, provenance=provenance, inputs=package)
    if hidden_targets is not None:
        bundle.provenance["hidden_targets_available_to_caller"] = True
        bundle._hidden_targets = hidden_targets  # type: ignore[attr-defined]
    if out is not None:
        bundle.save(out)
    return bundle
