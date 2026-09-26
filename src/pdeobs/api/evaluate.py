"""Strict evaluation: predictions + independent targets -> score, through the existing
``pdeobs-strict-v1`` scorer.  Nothing is scored without an explicit target."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import specs as S
from .artifacts import Predictor
from .data import DatasetHandle, load_dataset
from .infer import PredictionBundle
from .observation import ObservationSpec, make_observation


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def build_contract(*, task: str, sample_ids: Sequence[str], expected_shape: Sequence[int], target_time_indices: Sequence[int],
                   observation_id: str, checkpoint_id: str, mask_config: Mapping[str, Any] | None = None,
                   projection: bool = False, split: str | None = None, scope: str | None = None, **extra: Any) -> dict[str, Any]:
    from ..strict_score import CONTRACT_VERSION
    ids = [str(s) for s in sample_ids]
    if split is None:
        split = "paper-test" if len(ids) == 200 and extra.get("paper_split") else "demo"
    extra.pop("paper_split", None)
    contract: dict[str, Any] = {"schema_version": CONTRACT_VERSION, "task": task, "split": split, "expected_ids": ids,
                                "expected_shape": [int(v) for v in expected_shape], "target_time_indices": [int(v) for v in target_time_indices],
                                "observation_id": str(observation_id), "checkpoint_id": str(checkpoint_id),
                                "scope": scope or "user evaluation through the easy API; not a paper performance estimate"}
    if task == "recovery":
        contract["projection"] = bool(projection)
    if mask_config is not None:
        contract["mask_config"] = dict(mask_config)
    contract.update({k: v for k, v in extra.items() if v is not None})
    return contract


def evaluate(predictions: PredictionBundle | str | Path, targets: np.ndarray | Sequence[np.ndarray] | None = None, *,
             target_ids: Sequence[str] | None = None, target_time_indices: Sequence[int] | None = None,
             checkpoint_id: str | None = None, observation_id: str | None = None, out: str | Path | None = None,
             projection: bool = False, contract_extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Score a prediction bundle against explicit targets with the strict scorer.

    Identities, shapes and time indices must align exactly; missing, extra, repeated or non-finite
    entries invalidate the score instead of shrinking the denominator.
    """
    from ..strict_score import score_arrays, write_report

    bundle = predictions if isinstance(predictions, PredictionBundle) else PredictionBundle.load(predictions)
    if targets is None:
        hidden = getattr(bundle, "_hidden_targets", None)
        if hidden is None:
            raise ValueError("evaluate needs the independent target field; the prediction bundle carries none "
                             "(predictions made from observation-only inputs cannot be scored without targets)")
        targets = hidden
    target = np.asarray(targets, dtype=np.float64)
    ids = list(bundle.sample_ids)
    tids = list(ids if target_ids is None else target_ids)
    times = list(bundle.time_indices)
    ttimes = list(times if target_time_indices is None else target_time_indices)
    # Preserve invalid dtypes for strict rejection, instead of truncating 0.9 to 0.
    from ..strict_score import StrictValidationError, _times
    try:
        times = _times(times, "prediction_time_indices")
        ttimes = _times(ttimes, "target_time_indices")
    except (StrictValidationError, TypeError, ValueError) as exc:
        return _invalid([str(exc)], out)
    if target.shape != bundle.predictions.shape:
        return _invalid([f"target shape {target.shape} differs from prediction shape {bundle.predictions.shape}"], out)
    obs_meta = bundle.provenance.get("observation", {}) if bundle.provenance else {}
    contract = build_contract(task=bundle.task, sample_ids=ids, expected_shape=bundle.predictions.shape[1:],
                              target_time_indices=times, observation_id=observation_id or obs_meta.get("observation_id", "stored"),
                              checkpoint_id=checkpoint_id or (bundle.provenance.get("predictor", {}) or {}).get("weights_sha256") or "unknown-checkpoint",
                              mask_config=obs_meta.get("mask_config"), projection=projection, **(contract_extra or {}))
    mask = observation = None
    if bundle.task == "recovery":
        if bundle.inputs is None:
            return _invalid(["recovery scoring requires the observation and mask used for prediction"], out)
        mask = np.asarray(bundle.inputs.mask)
        observation = bundle.inputs.observations
        # score_arrays expects observations in TARGET order. The bundle carries
        # them in INPUT order, which can differ from the independently supplied targets.
        input_ids = list(bundle.inputs.sample_ids)
        if len(input_ids) != len(set(input_ids)) or set(input_ids) != set(ids):
            return _invalid(["input identities differ from prediction identities"], out)
        if all(isinstance(s, str) for s in tids) and len(tids) == len(ids) and set(tids) == set(ids):
            positions = {identity: index for index, identity in enumerate(input_ids)}
            order = [positions[identity] for identity in tids]
            mask = mask[order]
            observation = observation[order]
    report = score_arrays(np.asarray(bundle.predictions, dtype=np.float64), target, prediction_ids=ids, target_ids=tids,
                          prediction_time_indices=times, target_time_indices=ttimes, contract=contract, mask=mask, observation=observation)
    report["easy_api"] = {"api_version": S.API_VERSION, "prediction_provenance": bundle.provenance}
    if out is not None:
        out_dir = Path(out)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_report(out_dir / "contract.json", contract)
        write_report(out_dir / "score.json", report)
    return report


def _invalid(errors: list[str], out: str | Path | None) -> dict[str, Any]:
    from ..strict_score import invalid_report, write_report
    report = invalid_report(errors)
    if out is not None:
        out_dir = Path(out)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_report(out_dir / "score.json", report)
    return report


def evaluate_records(predictor: Predictor, data: DatasetHandle | str | Path, observation: Any, *, out: str | Path,
                     sample_ids: Sequence[str] | None = None, batch_size: int = 4, device: str | None = None,
                     projection: bool = False, split_label: str = "demo", seed: int = 0, max_samples: int | None = None,
                     contract_extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Strict model evaluation on complete records: the exact ``evaluate_model_strict`` path used by paper rows.

    Writes ``contract.json``, ``predictions.h5`` and ``score.json`` under ``out``.  The predictor is the
    same object used for target-free inference; here targets are read only to score.
    """
    from ..dataset import BenchmarkDataset
    from ..evaluation import EvaluationConfig
    from ..one_setting import subset_dataset
    from ..strict_inference import batches, evaluate_model_strict, metadata_target_shape, target_frame_metadata
    from ..strict_score import write_report

    handle = data if isinstance(data, DatasetHandle) else load_dataset(data)
    obs = observation if isinstance(observation, ObservationSpec) else make_observation(observation)
    if obs.namespace == "stored":
        raise ValueError("evaluate_records builds the observation from complete records; use a general or paper protocol")
    task = predictor.task
    history = int(predictor.io_schema.get("history_steps", 1))
    horizon = int(predictor.io_schema.get("horizon_trained") or max(1, handle.time_steps - history)) if task == "rollout" else 1
    rep = predictor.io_schema.get("state_representation") or ("vorticity" if handle.state_representation == "vorticity" else "native")
    dataset = BenchmarkDataset(handle.shards, task=task, mask=dict(obs.mask_config), history_steps=history,
                               horizon=horizon if task == "rollout" else 8, seed=seed, state_representation=rep, max_samples=max_samples)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=False)
    try:
        if sample_ids is not None:
            positions = {row["sample_id"]: i for i, row in enumerate(dataset.metadata)}
            dataset = subset_dataset(dataset, [positions[s] for s in sample_ids])
        ids = [str(row["sample_id"]) for row in dataset.metadata]
        ec = EvaluationConfig(task=task, device=device or predictor.device, data_layout="channels_last", history_steps=history,
                              horizon=horizon, horizons=tuple(range(1, horizon + 1)) if task == "rollout" else (1,),
                              rollout_target_offset=0, observation_mode="matched")
        frame = target_frame_metadata(dataset.metadata[0], ec)
        checkpoint_id = (predictor.provenance.get("weights_sha256") or predictor.provenance.get("sha256")
                         or f"unfitted:{_canonical_hash(predictor.config.method_config())}")
        contract = build_contract(task=task, sample_ids=ids, expected_shape=metadata_target_shape(dataset, ec),
                                  target_time_indices=frame["target_source_frame_indices"], observation_id=obs.observation_id,
                                  checkpoint_id=checkpoint_id, mask_config=obs.mask_config, projection=projection, split=split_label,
                                  method=predictor.config.name, test_view=obs.name, dataset_manifest_sha256=handle.sample_ids_sha256,
                                  time_index_semantics="source_solver_frame_indices", **(contract_extra or {}))
        write_report(out_dir / "contract.json", contract)
        report = evaluate_model_strict(predictor.model, batches(dataset, int(batch_size)), config=ec, contract=contract,
                                       predictions_path=out_dir / "predictions.h5", report_path=out_dir / "score.json")
    finally:
        dataset.close()
    return report


def evaluate_views(predictor: Predictor, data: DatasetHandle | str | Path, views: Sequence[Any] | None = None, *, out: str | Path,
                   sample_ids: Sequence[str] | None = None, batch_size: int = 4, device: str | None = None,
                   split_label: str = "demo", seed: int = 0, namespace: str = "paper") -> dict[str, Any]:
    """Evaluate one predictor under several observation views (default: the nine paper views)."""
    from .observation import paper_views
    specs = [v if isinstance(v, ObservationSpec) else make_observation(v, namespace=namespace) for v in views] if views else paper_views()
    out_dir = Path(out)
    ids = [spec.observation_id for spec in specs]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"evaluate_views received the same observation more than once: {dup}")
    # several general views may share a protocol name (e.g. random at two ratios); key those by observation id
    names = [spec.name for spec in specs]
    keys = [spec.name if names.count(spec.name) == 1 else spec.observation_id.replace(":", "_") for spec in specs]
    results: dict[str, Any] = {}
    for key, spec in zip(keys, specs):
        report = evaluate_records(predictor, data, spec, out=out_dir / key, sample_ids=sample_ids, batch_size=batch_size,
                                  device=device, split_label=split_label, seed=seed)
        results[key] = {"status": report.get("status"), "summary": report.get("summary"), "errors": report.get("errors", []),
                        "name": spec.name, "label": spec.label, "observation_id": spec.observation_id, "directory": key}
    summary = {"views": results, "all_valid": all(r["status"] == "valid" for r in results.values()), "count": len(results)}
    (out_dir / "views.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
