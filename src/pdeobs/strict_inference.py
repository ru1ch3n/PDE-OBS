"""Versioned inference-to-score bridge; legacy prediction writers are unchanged.

The manifest fixes source shards and physical identities. The exporter derives
frame indices from the actual task metadata, binds a caller-declared scoring
contract, and sends unmodified predictions to the float64 strict scorer. Hash
binding establishes consistency, not independent provenance authentication.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .dataset import BenchmarkDataset, collate_benchmark
from .evaluation import EvaluationConfig, _numpy_channels_last, predict_batch
from .one_setting import subset_dataset
from .storage import sha256_file
from .strict_score import (
    _canonical_hash, _ids, _times, invalid_report, score_prediction_file, write_report,
)
from .training import unpack_batch_context

ARTIFACT_VERSION = "pdeobs-strict-inference-v1"
MANIFEST_VERSION = "pdeobs-identity-manifest-v1"


def canonical_json_sha256(value: Any) -> str:
    """Portable hash used for manifest, unfitted-method and contract bindings."""
    return _canonical_hash(value)


def _contained(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("manifest shard paths must be nonempty relative paths")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("manifest shard path escapes the explicit data root")
    if not resolved.is_file():
        raise FileNotFoundError(f"manifest shard is missing: {relative}")
    return resolved


def create_identity_manifest(data_root: str | Path, shards: Sequence[str],
                             output: str | Path) -> dict[str, Any]:
    """Record explicit existing canonical shards; never discover or generate data."""
    from .storage import LazyHDF5Dataset

    root = Path(data_root).resolve()
    paths = [_contained(root, name) for name in shards]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("manifest needs nonempty, distinct shard paths")
    with LazyHDF5Dataset(paths) as source:
        rows = list(source.iter_metadata())
    ids = _ids([row.get("sample_id") for row in rows], "source sample_id")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate source sample_id")
    manifest = {"schema_version": MANIFEST_VERSION, "expected_ids": ids,
                "shards": [{"path": path.relative_to(root).as_posix(),
                            "sha256": sha256_file(path)} for path in paths]}
    write_report(output, manifest)
    return manifest


def load_identity_dataset(data_root: str | Path, manifest_path: str | Path,
                          **options: Any) -> tuple[BenchmarkDataset, dict[str, Any]]:
    """Hash/index only before selection; do not read any target array here."""
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"explicit data root does not exist: {root}")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_VERSION:
        raise ValueError(f"manifest must use {MANIFEST_VERSION}")
    ids = _ids(manifest.get("expected_ids"), "manifest expected_ids")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate manifest expected_ids")
    entries = manifest.get("shards")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest requires explicit shards")
    paths = []
    for entry in entries:
        path = _contained(root, entry["path"])
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError("every manifest shard requires a SHA-256")
        if sha256_file(path) != expected_hash.lower():
            raise ValueError(f"manifest shard SHA-256 differs: {entry['path']}")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate manifest shard paths")
    # No canonical split labels, fallback, or implicit max_samples are accepted.
    if any(key in options for key in ("split", "filters", "max_samples", "release_tier")):
        raise ValueError("identity manifest selection cannot be combined with split/filter shortcuts")
    dataset = BenchmarkDataset(paths, **options)
    actual = _ids([row.get("sample_id") for row in dataset.metadata], "source sample_id")
    if len(actual) != len(ids) or set(actual) != set(ids):
        dataset.close()
        raise ValueError("source identities differ from the explicit manifest")
    positions = {identity: i for i, identity in enumerate(actual)}
    dataset = subset_dataset(dataset, [positions[identity] for identity in ids])
    return dataset, manifest


def target_frame_metadata(row: Mapping[str, Any], config: EvaluationConfig) -> dict[str, Any]:
    """Select actual source frames; do not invent a time vector for missing metadata."""
    frames = _times(row.get("stored_frame_indices"), "stored_frame_indices")
    if row.get("T") != len(frames):
        raise ValueError("metadata T differs from stored_frame_indices")
    if config.task == "rollout":
        if config.rollout_target_offset != 0:
            raise ValueError("strict BenchmarkDataset rollout requires rollout_target_offset=0")
        positions = list(range(config.history_steps, config.history_steps + config.horizon))
    elif config.task in {"recovery", "forward"}:
        position = config.target_step if config.target_step >= 0 else len(frames) + config.target_step
        positions = [position]
    else:
        raise ValueError("strict inference currently supports recovery, forward and rollout; inverse needs an explicit condition-coordinate adapter")
    if any(position < 0 or position >= len(frames) for position in positions):
        raise ValueError("task target frames exceed stored trajectory metadata")
    physical = row.get("stored_time_values")
    if config.task == "rollout" or physical is not None:
        values = np.asarray(physical)
        if (values.ndim != 1 or len(values) != len(frames) or values.dtype.kind not in "fiu"
                or not np.isfinite(values).all() or np.any(np.diff(values) <= 0)):
            raise ValueError("dynamic stored_time_values must be finite, complete and strictly increasing")
        selected_times = [float(values[position]) for position in positions]
    else:
        selected_times = None
    return {"target_stored_positions": positions,
            "target_source_frame_indices": [frames[position] for position in positions],
            "target_physical_time_values": selected_times}


def metadata_target_shape(dataset: BenchmarkDataset, config: EvaluationConfig) -> list[int]:
    """Read HDF5 shape headers only, never held-out field values."""
    tails = set()
    for path in dataset.base.paths:
        with h5py.File(path, "r") as handle:
            tail = tuple(handle["trajectory"].shape[1:])
            if len(tail) != 4:
                raise ValueError("canonical trajectory must have NTHWC layout")
            tails.add(tail)
    if len(tails) != 1:
        raise ValueError("strict inference requires a consistent trajectory shape")
    time, height, width, channels = tails.pop()
    if any(row.get("T") != time for row in dataset.metadata):
        raise ValueError("trajectory time dimension differs from source metadata")
    if dataset.spatial_resolution is not None:
        raise ValueError("strict source export does not silently resample spatial fields")
    if dataset.state_representation != "native":
        # A source vorticity field is allowed; implicit velocity conversion is not.
        if any("vorticity" not in str(row.get("state_representation", ""))
               for row in dataset.metadata):
            raise ValueError("strict export requires the declared state representation in source shards")
    return ([config.horizon] if config.task == "rollout" else []) + [height, width, channels]


def batches(dataset: BenchmarkDataset, batch_size: int) -> Iterable[dict[str, Any]]:
    if isinstance(batch_size, bool) or int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(dataset), int(batch_size)):
        yield collate_benchmark([dataset[index] for index in range(start, min(len(dataset), start + int(batch_size)))])


class _StrictWriter:
    """Append-only versioned HDF5 export, isolated from the legacy writer."""
    def __init__(self, path: Path, contract: Mapping[str, Any]):
        self.path = path
        self.partial = path.with_suffix(path.suffix + ".partial")
        if path.suffix not in {".h5", ".hdf5"}:
            raise ValueError("strict inference output must be HDF5")
        if path.exists() or self.partial.exists():
            raise FileExistsError(f"refusing to overwrite inference artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.partial, "x")
        self.count = 0
        self.handle.attrs["artifact_version"] = ARTIFACT_VERSION
        self.handle.attrs["contract_sha256"] = _canonical_hash(contract)
        self.handle.attrs["contract_json"] = json.dumps(contract, sort_keys=True, allow_nan=False)
        self.handle.attrs["time_index_semantics"] = "source_solver_frame_indices"

    def append(self, name: str, values: Any, *, strings: bool = False) -> None:
        values = np.asarray(values, dtype=object if strings else None)
        if name not in self.handle:
            self.handle.create_dataset(name, shape=(0, *values.shape[1:]),
                                       maxshape=(None, *values.shape[1:]),
                                       dtype=h5py.string_dtype("utf-8") if strings else values.dtype,
                                       chunks=True, compression="gzip")
        result = self.handle[name]
        if result.shape[1:] != values.shape[1:]:
            raise ValueError(f"inconsistent {name} shapes across batches")
        start = result.shape[0]
        result.resize(start + len(values), axis=0)
        result[start:] = values

    def close(self, publish: bool) -> None:
        if self.handle.id.valid:
            self.handle.attrs["samples"] = self.count
            self.handle.flush()
            self.handle.close()
        if publish:
            # A hard-link publication is exclusive even if another process
            # creates the final name between the initial check and this point.
            import os
            os.link(self.partial, self.path)
            self.partial.unlink()


def evaluate_model_strict(method: Any, loader: Iterable[Any], *,
                          config: EvaluationConfig | Mapping[str, Any],
                          contract: Mapping[str, Any], predictions_path: str | Path,
                          report_path: str | Path | None = None) -> dict[str, Any]:
    """Normal model inference -> bound HDF5 -> strict scorer, no legacy reduction.

    Metadata must travel with each batch, as in ``collate_benchmark``. No manual
    post-hoc identity/time arrays or array repair are needed or performed.
    """
    writer = None
    try:
        if not isinstance(contract, Mapping):
            raise ValueError("scoring contract must be a JSON object")
        config = config if isinstance(config, EvaluationConfig) else EvaluationConfig(**dict(config))
        if config.observation_mode != "matched" or config.data_layout != "channels_last":
            raise ValueError("strict export requires matched observations and explicit channels_last layout")
        if contract.get("task") != config.task:
            raise ValueError("inference task differs from scoring contract")
        expected = _ids(contract.get("expected_ids"), "contract expected_ids")
        if len(set(expected)) != len(expected):
            raise ValueError("duplicate expected identities")
        expected_times = _times(contract.get("target_time_indices"), "contract target_time_indices")
        writer = _StrictWriter(Path(predictions_path), contract)
        seen: set[str] = set()
        for batch in loader:
            raw_input, raw_mask, _, raw_geometry, metadata = unpack_batch_context(batch)
            if not isinstance(metadata, (list, tuple)) or not metadata:
                raise ValueError("strict inference requires per-identity metadata")
            ids = _ids([row.get("sample_id") for row in metadata], "batch sample_id")
            if len(ids) != len(set(ids)) or seen.intersection(ids) or set(ids) - set(expected):
                raise ValueError("unexpected or duplicate batch identity")
            frame_rows = [target_frame_metadata(row, config) for row in metadata]
            if any(row["target_source_frame_indices"] != expected_times for row in frame_rows):
                raise ValueError("actual source target frames differ from scoring contract")
            prediction, target = predict_batch(method, batch, config)
            if list(target.shape[1:]) != contract.get("expected_shape"):
                raise ValueError("actual target shape differs from scoring contract")
            if len(ids) != len(prediction):
                raise ValueError("metadata and prediction batch lengths differ")
            for name, value in (("prediction", prediction), ("target", target),
                                ("observation", raw_input), ("mask", raw_mask),
                                ("geometry", raw_geometry)):
                if value is not None:
                    writer.append(name, _numpy_channels_last(value, "channels_last"))
            writer.append("prediction_ids", ids, strings=True)
            writer.append("target_ids", ids, strings=True)
            writer.append("metadata_json", [json.dumps({**dict(row), **frame}, sort_keys=True,
                          allow_nan=False) for row, frame in zip(metadata, frame_rows)], strings=True)
            seen.update(ids)
            writer.count += len(ids)
        if seen != set(expected):
            raise ValueError("inference did not cover every expected identity")
        for name in ("prediction_time_indices", "target_time_indices"):
            writer.handle.create_dataset(name, data=np.asarray(expected_times, dtype=np.int64))
        writer.close(True)
        writer = None
        report = score_prediction_file(predictions_path, contract)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        if writer is not None:
            writer.close(False)
        report = invalid_report([f"strict inference failed: {exc}"], contract)
    if report_path is not None:
        write_report(report_path, report)
    return report


def run_strict_infer(config_path: str | Path, contract_path: str | Path, *,
                     data_root: str | Path, manifest_path: str | Path,
                     output: str | Path, checkpoint: str | Path | None = None,
                     device: str = "cpu") -> dict[str, Any]:
    """CLI adapter for arbitrary registered methods and explicit test identities."""
    from .config import load_config
    from .runner import _evaluation_config, _load_checkpoint, _method

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    dataset = None
    contract = None
    try:
        config = load_config(config_path)
        contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
        if not isinstance(contract, dict):
            raise ValueError("scoring contract must be a JSON object")
        ec = replace(_evaluation_config(config), device=device, data_layout="channels_last")
        data = config.get("data", {})
        mask = dict(data.get("mask", {}))
        if contract.get("mask_config") != mask:
            raise ValueError("contract mask_config differs from actual inference mask")
        expected_checkpoint = sha256_file(checkpoint) if checkpoint is not None else "unfitted:" + _canonical_hash(config.get("method", {}))
        if contract.get("checkpoint_id") != expected_checkpoint:
            raise ValueError("actual checkpoint/method identity differs from contract")
        dataset, manifest = load_identity_dataset(data_root, manifest_path, task=ec.task,
                    mask=mask, target_step=ec.target_step, horizon=ec.horizon,
                    history_steps=ec.history_steps, seed=int(data.get("mask_seed", config.get("seed", 0))),
                    state_representation=data.get("state_representation", "native"))
        expected_manifest = _canonical_hash(manifest)
        if contract.get("dataset_manifest_sha256") != expected_manifest:
            raise ValueError("contract dataset_manifest_sha256 differs from actual manifest")
        positions = {row["sample_id"]: i for i, row in enumerate(dataset.metadata)}
        selected = _ids(contract.get("expected_ids"), "expected_ids")
        dataset = subset_dataset(dataset, [positions[identity] for identity in selected])
        if metadata_target_shape(dataset, ec) != contract.get("expected_shape"):
            raise ValueError("source shape header differs from scoring contract")
        method = _method(config)
        _load_checkpoint(method, checkpoint, device=device)
        write_report(destination / "contract.json", contract)
        report = evaluate_model_strict(method, batches(dataset, int(config.get("training", {}).get("batch_size", 4))),
                    config=ec, contract=contract, predictions_path=destination / "predictions.h5")
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as exc:
        report = invalid_report([f"strict inference setup failed: {exc}"], contract)
    finally:
        if dataset is not None:
            dataset.close()
    write_report(destination / "score.json", report)
    return report
