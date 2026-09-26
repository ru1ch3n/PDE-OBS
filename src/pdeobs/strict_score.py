"""Versioned, fail-closed scoring of explicit prediction/target identities.

This release path does not change legacy metrics or any training objective.
Raw arrays are validated before projection and before reduction. Contract
identity sets are supplied by the caller; a contract is not proof of provenance.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCORING_VERSION = "pdeobs-strict-v1"
CONTRACT_VERSION = "pdeobs-strict-contract-v1"
EPSILON = 1.0e-12


class StrictValidationError(ValueError):
    """A malformed artifact, rather than a poor finite prediction."""


def _ids(value: Any, label: str) -> list[str]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise StrictValidationError(f"{label} must be a one-dimensional string array")
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise StrictValidationError(f"{label} must be a non-empty string list")
    decoded = [item.decode("utf-8") if isinstance(item, bytes) else item for item in value]
    if any(not isinstance(item, str) or not item.strip() for item in decoded):
        raise StrictValidationError(f"{label} contains an empty or non-string identity")
    return decoded


def _times(value: Any, label: str) -> list[int]:
    array = np.asarray(value)
    if array.ndim != 1 or not len(array) or array.dtype.kind not in "iu":
        raise StrictValidationError(f"{label} must be a non-empty integer vector")
    result = [int(item) for item in array]
    if result[0] < 0 or any(a >= b for a, b in zip(result, result[1:])):
        raise StrictValidationError(f"{label} must be nonnegative and strictly increasing")
    return result


def _finite_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiub":
        raise StrictValidationError(f"{name} must be a real numeric array")
    if not array.size or not np.isfinite(array).all():
        bad = int(array.size - np.count_nonzero(np.isfinite(array)))
        raise StrictValidationError(f"{name} has {bad} non-finite entries or is empty")
    return array.astype(np.float64, copy=False)


def _norm(value: np.ndarray) -> float:
    # Scaled norm keeps finite large arrays from overflowing during squaring.
    scale = float(np.max(np.abs(value))) if value.size else 0.0
    if scale == 0:
        return 0.0
    result = scale * float(np.sqrt(np.sum(np.square(value / scale), dtype=np.float64)))
    if not np.isfinite(result):
        raise StrictValidationError("float64 norm overflow; no finite score was produced")
    return result


def _relative(prediction: np.ndarray, target: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        error = prediction - target
    if not np.isfinite(error).all():
        raise StrictValidationError("float64 subtraction overflow")
    value = _norm(error) / max(_norm(target), EPSILON)
    if not np.isfinite(value):
        raise StrictValidationError("float64 relative error overflow")
    return float(value)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def invalid_report(errors: Sequence[str], contract: Any = None) -> dict[str, Any]:
    """JSON-safe failure receipt, including malformed-contract cases."""
    contract = contract if isinstance(contract, Mapping) else {}
    expected = contract.get("expected_ids", [])
    return {
        "scoring_version": SCORING_VERSION, "status": "invalid", "errors": list(errors),
        "expected_identity_count": len(expected) if isinstance(expected, (list, tuple)) else None,
        "actual_prediction_identity_count": None, "actual_target_identity_count": None,
        "scored_identity_count": 0, "summary": None, "per_identity": [],
        "task": contract.get("task") if isinstance(contract.get("task"), str) else None,
        "observation_id": contract.get("observation_id") if isinstance(contract.get("observation_id"), str) else None,
        "checkpoint_id": contract.get("checkpoint_id") if isinstance(contract.get("checkpoint_id"), str) else None,
        "provenance_note": "Caller-declared contract; identity validation is not provenance authentication.",
    }


def score_arrays(
    prediction: Any, target: Any, *, prediction_ids: Any, target_ids: Any,
    prediction_time_indices: Any, target_time_indices: Any,
    contract: Mapping[str, Any], mask: Any = None, observation: Any = None,
) -> dict[str, Any]:
    """Validate all records, then score every expected identity or none.

    Layout is explicitly NHWC for static tasks and NTHWC for rollout. Identity
    order may differ in the two artifacts; each is aligned to contract order.
    Mask/observation belong to the target ordering. No record is dropped.
    """
    report = invalid_report([], contract)
    try:
        if not isinstance(contract, Mapping):
            raise StrictValidationError("contract must be a JSON object")
        json.dumps(contract, allow_nan=False)
        if contract.get("schema_version") != CONTRACT_VERSION:
            raise StrictValidationError(f"schema_version must be {CONTRACT_VERSION}")
        expected_ids = _ids(contract.get("expected_ids"), "expected_ids")
        pred_ids = _ids(prediction_ids, "prediction_ids")
        truth_ids = _ids(target_ids, "target_ids")
        report.update(actual_prediction_identity_count=len(pred_ids),
                      actual_target_identity_count=len(truth_ids))
        for label, values in (("expected_ids", expected_ids), ("prediction_ids", pred_ids),
                              ("target_ids", truth_ids)):
            duplicates = sorted(k for k, n in Counter(values).items() if n > 1)
            if duplicates:
                raise StrictValidationError(f"duplicate {label}: {duplicates}")
        for label, values in (("prediction_ids", pred_ids), ("target_ids", truth_ids)):
            missing, extra = sorted(set(expected_ids) - set(values)), sorted(set(values) - set(expected_ids))
            if missing or extra:
                raise StrictValidationError(f"{label}: missing={missing}; extra={extra}")
        split = contract.get("split")
        if split not in {"demo", "paper-test"}:
            raise StrictValidationError("split must be demo or paper-test")
        if split == "paper-test" and len(expected_ids) != 200:
            raise StrictValidationError("paper-test requires exactly 200 explicit expected identities")
        task = contract.get("task")
        if task not in {"recovery", "forward", "inverse", "rollout"}:
            raise StrictValidationError("unsupported task")
        for key in ("observation_id", "checkpoint_id"):
            if not isinstance(contract.get(key), str) or not contract[key]:
                raise StrictValidationError(f"contract requires a non-empty {key}")
        shape = contract.get("expected_shape")
        dimensions = 4 if task == "rollout" else 3
        if not isinstance(shape, list) or len(shape) != dimensions or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in shape
        ):
            raise StrictValidationError(f"expected_shape must contain {dimensions} positive integers")
        expected_times = _times(contract.get("target_time_indices"), "contract target_time_indices")
        pred_times = _times(prediction_time_indices, "prediction_time_indices")
        target_times = _times(target_time_indices, "target_time_indices")
        if pred_times != expected_times or target_times != expected_times:
            raise StrictValidationError("prediction/target time indices differ from the contract")
        if len(expected_times) != (shape[0] if task == "rollout" else 1):
            raise StrictValidationError("time index length does not match the task shape")
        # Raw validation precedes even identity reordering, projection or norms.
        pred, truth = _finite_array(prediction, "prediction"), _finite_array(target, "target")
        full_shape = (len(expected_ids), *shape)
        if pred.shape != full_shape or truth.shape != full_shape:
            raise StrictValidationError(f"shape mismatch: prediction={pred.shape}, target={truth.shape}, expected={full_shape}")
        pred_order = [pred_ids.index(identity) for identity in expected_ids]
        truth_order = [truth_ids.index(identity) for identity in expected_ids]
        pred, truth = pred[pred_order], truth[truth_order]
        projection = contract.get("projection", False)
        if not isinstance(projection, bool):
            raise StrictValidationError("projection must be a boolean")
        if projection and task != "recovery":
            raise StrictValidationError("data-consistency projection is only defined for static recovery")
        masks = observations = None
        if task == "recovery":
            if mask is None or observation is None:
                raise StrictValidationError("recovery requires mask and observation for static diagnostics")
            masks = _finite_array(mask, "mask")
            observations = _finite_array(observation, "observation")
            if masks.shape not in {full_shape, (*full_shape[:-1], 1)}:
                raise StrictValidationError("mask must be NHWC with one channel or target channel count")
            if observations.shape != full_shape:
                raise StrictValidationError("observation shape must equal the static target shape")
            if not np.isin(masks, (0, 1)).all():
                raise StrictValidationError("mask must contain only 0 and 1")
            masks = np.broadcast_to(masks, full_shape)[truth_order].astype(bool)
            observations = observations[truth_order]
            if not np.array_equal(observations[masks], truth[masks]):
                raise StrictValidationError("observed values disagree with the noiseless recovery target")
        elif mask is not None or observation is not None:
            # Non-recovery inputs are provenance only, but may not hide NaNs.
            if mask is not None:
                _finite_array(mask, "mask")
            if observation is not None:
                _finite_array(observation, "observation")
        rows = []
        for index, identity in enumerate(expected_ids):
            p, t = pred[index], truth[index]
            item: dict[str, Any] = {"identity": identity, "rel_l2_joint": _relative(p, t)}
            if task == "rollout":
                item["rel_l2_by_horizon"] = [
                    {"horizon": j + 1, "target_time_index": time_index, "rel_l2": _relative(p[j], t[j])}
                    for j, time_index in enumerate(expected_times)
                ]
            if task == "recovery":
                assert masks is not None and observations is not None
                m, error = masks[index], p - t
                denominator = max(_norm(t), EPSILON)
                hidden_count = int(np.count_nonzero(~m))
                item["static_diagnostics"] = {
                    "full_common_denominator": _norm(error) / denominator,
                    "observed_common_denominator": _norm(error[m]) / denominator,
                    "hidden_common_denominator": _norm(error[~m]) / denominator,
                    "observed_scalar_count": int(np.count_nonzero(m)),
                    "hidden_scalar_count": hidden_count,
                    "hidden_only_status": "valid" if hidden_count else "not_applicable",
                    "hidden_only_rel_l2": _norm(error[~m]) / max(_norm(t[~m]), EPSILON) if hidden_count else None,
                }
                if projection:
                    projected = np.where(m, observations[index], p)
                    item["projected_rel_l2_joint"] = _relative(projected, t)
            rows.append(item)
        # All denominators are the complete expected set. No nanmean/nansum.
        summary: dict[str, Any] = {"rel_l2_joint_mean": float(np.mean([r["rel_l2_joint"] for r in rows], dtype=np.float64)),
                                   "identity_count": len(expected_ids)}
        if task == "rollout":
            summary["rel_l2_by_horizon_mean"] = [
                {"horizon": j + 1, "target_time_index": time_index,
                 "rel_l2_mean": float(np.mean([r["rel_l2_by_horizon"][j]["rel_l2"] for r in rows], dtype=np.float64))}
                for j, time_index in enumerate(expected_times)
            ]
        if task == "recovery":
            keys = ("full_common_denominator", "observed_common_denominator", "hidden_common_denominator")
            summary["static_diagnostics"] = {key + "_mean": float(np.mean([r["static_diagnostics"][key] for r in rows], dtype=np.float64)) for key in keys}
            hidden_values = [r["static_diagnostics"]["hidden_only_rel_l2"] for r in rows]
            # Mixed applicability has no misleading mean with a reduced denominator.
            summary["static_diagnostics"].update(
                hidden_only_applicable_count=sum(v is not None for v in hidden_values),
                hidden_only_status="valid" if all(v is not None for v in hidden_values) else "not_applicable_for_all_identities",
                hidden_only_rel_l2_mean=float(np.mean(hidden_values, dtype=np.float64)) if all(v is not None for v in hidden_values) else None,
            )
            if projection:
                summary["projected_rel_l2_joint_mean"] = float(np.mean([r["projected_rel_l2_joint"] for r in rows], dtype=np.float64))
        report.update(status="valid", errors=[], scored_identity_count=len(rows), summary=summary,
                      per_identity=rows, config=dict(contract), config_sha256=_canonical_hash(contract),
                      identity_set_sha256=_canonical_hash(sorted(expected_ids)),
                      expected_identity_order=expected_ids, epsilon=EPSILON,
                      reduction="float64 per-identity relative L2, then arithmetic mean",
                      projection_applied=projection, raw_arrays_checked=True)
        # Last guard makes standard JSON a hard requirement, including extreme ratios.
        json.dumps(report, allow_nan=False)
    except (StrictValidationError, TypeError, ValueError, UnicodeDecodeError, OverflowError) as exc:
        report.update(status="invalid", errors=[str(exc)], scored_identity_count=0,
                      summary=None, per_identity=[])
    return report


def score_prediction_file(path: str | Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Load a portable NPZ/HDF5 artifact (never pickle), then apply strict v1."""
    source = Path(path)
    required = ("prediction", "target", "prediction_ids", "target_ids", "prediction_time_indices", "target_time_indices")
    optional = ("mask", "observation")
    try:
        binding = "unbound_legacy_or_external_artifact"
        if source.suffix.lower() == ".npz":
            with np.load(source, allow_pickle=False) as data:
                values = {key: data[key] for key in required}
                values.update({key: data[key] for key in optional if key in data})
        elif source.suffix.lower() in {".h5", ".hdf5"}:
            import h5py
            with h5py.File(source, "r") as data:
                version = data.attrs.get("artifact_version")
                if version is not None:
                    if version != "pdeobs-strict-inference-v1":
                        raise StrictValidationError("unsupported strict inference artifact version")
                    expected_hash = _canonical_hash(contract)
                    if data.attrs.get("contract_sha256") != expected_hash:
                        raise StrictValidationError("artifact is bound to a different scoring contract")
                    stored_contract = json.loads(data.attrs["contract_json"])
                    if _canonical_hash(stored_contract) != expected_hash:
                        raise StrictValidationError("embedded scoring contract differs from its binding")
                    binding = "pdeobs-strict-inference-v1_sha256_bound"
                values = {key: data[key][...] for key in required}
                values.update({key: data[key][...] for key in optional if key in data})
        else:
            raise StrictValidationError("predictions must be an NPZ or HDF5 artifact")
        report = score_arrays(contract=contract, **values)
        report["artifact_contract_binding"] = binding
        return report
    except (OSError, KeyError, ValueError, TypeError) as exc:
        return invalid_report([f"artifact loading failed: {exc}"], contract)


def write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    """Write standard JSON exclusively: old scores are never overwritten."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(text)
