"""Inference-only strict rescoring of one mixed-pattern study checkpoint.

The fixed checkpoint is evaluated on every declared view over the frozen 200 held-out identities
of its macrodomain.  Every prediction, target, observation, mask and geometry array is retained in a
contract-bound HDF5 file per block (``archived_verification.export_bound_predictions``) and each
block is scored from that file with the unchanged ``pdeobs-strict-v1`` scorer.  Zero optimizer
updates; the study's own legacy evaluator receipts are not read here (the result builder compares
them afterwards).  Views absent from training (observation specs) are exported the same way and
labelled ``extra_view``; the nine frozen paper views are labelled ``frozen_paper_view``.

Inputs are bound, never inferred: the checkpoint digest must equal the one in the study's training
completion receipt, the dataset shards must equal the digests recorded by the main-grid
verification (``dataset_bindings.json``), and the reconstructed 1800/200 split must equal both the
study's split manifest and the main-grid binding.  Output directories are create-only.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pdeobs.archived_verification import VERSION as ADAPTER_VERSION
from pdeobs.archived_verification import assert_split, export_bound_predictions, require_hash
from pdeobs.dataset import BenchmarkDataset
from pdeobs.mask_specs import parse_mask_spec
from pdeobs.one_setting import VIEW_ORDER, stable_split, subset_dataset
from pdeobs.runner import _evaluation_config, _load_checkpoint, _method
from pdeobs.storage import sha256_file
from pdeobs.strict_inference import batches, metadata_target_shape, target_frame_metadata
from pdeobs.strict_score import CONTRACT_VERSION, _canonical_hash, score_arrays, write_report

DRIVER_VERSION = "pdeobs-mixed-strict-inference/20260926-v1"
WEIGHT_KEYS = ("model_state",)


def _load_weights(method: Any, checkpoint: Path, device: str, output: Path) -> dict[str, Any]:
    """Load the weights of a full-state study checkpoint.

    The repository refuses arbitrary-pickle checkpoints.  A study checkpoint written by the
    production trainer carries optimizer, scaler and RNG state next to ``model_state``; when the
    safe loader rejects those extra objects, the weights are extracted once into a create-only
    weights-only sibling inside the output directory and loaded from there.  Both digests are
    recorded so the mapping original -> stripped stays explicit, as for the public release.
    """
    import torch

    record: dict[str, Any] = {"checkpoint_original_sha256": sha256_file(checkpoint), "loaded_from": str(checkpoint),
                              "state_repair": None}
    try:
        _load_checkpoint(method, checkpoint, device=device)
        return record
    except ValueError as exc:
        if "weights-only" not in str(exc):
            raise
        safe_error = str(exc)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError("full-state checkpoint has no model_state mapping")
    state = dict(payload["model_state"])
    if "_metadata" in state and not isinstance(state["_metadata"], torch.Tensor):
        # Unprefixed constructor metadata written by some wrapped modules; holds no tensors.
        dropped = state.pop("_metadata")
        if any(isinstance(v, torch.Tensor) for v in (dropped.values() if isinstance(dropped, dict) else [dropped])):
            raise ValueError("_metadata entry unexpectedly holds tensors")
        record["state_repair"] = "dropped_unprefixed__metadata_key_without_tensors"
    stripped = {"model_state": state, "epoch": payload.get("epoch"), "optimizer_steps": payload.get("optimizer_steps")}
    for key in ("epoch", "optimizer_steps"):
        if stripped[key] is not None and not isinstance(stripped[key], (int, float)):
            stripped[key] = None
    target = output / "checkpoint-weights-only.pt"
    if target.exists():
        raise FileExistsError(target)
    torch.save(stripped, target)
    record.update(loaded_from=str(target), checkpoint_weights_only_sha256=sha256_file(target),
                  safe_loader_error=safe_error, dropped_keys=sorted(set(payload) - {"model_state"}))
    _load_checkpoint(method, target, device=device)
    return record


def _view_mask(spec_text: str, spatial: tuple[int, int]) -> tuple[dict[str, Any], str, str]:
    """Mask configuration for an extra view, exactly as the study's extra-view evaluator built it."""
    spec = parse_mask_spec(spec_text)
    if not spec.is_mixture and spec.components[0].protocol != "full":
        component = spec.components[0]
        return {"protocol": component.protocol, **component.resolved_kwargs(spatial)}, spec.view_id, spec.canonical_text()
    return {"protocol": spec.canonical_text()}, spec.view_id, spec.canonical_text()


def _projected_report(prediction_path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Diagnostic rescoring with data-consistency projection from the retained arrays (recovery only)."""
    import h5py

    with h5py.File(prediction_path, "r") as handle:
        values = {key: handle[key][...] for key in ("prediction", "target", "prediction_ids", "target_ids",
                                                    "prediction_time_indices", "target_time_indices", "mask", "observation")}
    report = score_arrays(contract={**contract, "projection": True}, **values)
    report["diagnostic_note"] = ("projection applied to the retained raw predictions; a separately labelled "
                                 "diagnostic, not a replacement of the raw score")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True, help="pde/method/train_view of the study row")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True, help="dataset_bindings.json of the main-grid verification")
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--views", default=",".join(VIEW_ORDER))
    parser.add_argument("--extra-view", action="append", default=[], help="observation spec absent from training")
    parser.add_argument("--source-manifest", type=Path, default=None, help="inference-code manifest to copy into provenance")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    begin = time.time()
    pde, method_name, train_view = args.identity.split("/")
    completion: dict[str, Any] = {"schema_version": DRIVER_VERSION, "adapter": ADAPTER_VERSION, "identity": args.identity,
                                  "status": "failed", "started_unix": begin, "blocks": [], "optimizer_updates": 0}
    dataset = None
    try:
        checkpoint = args.model_root / "checkpoints" / "last.pt"
        require_hash(checkpoint, args.expected_checkpoint_sha256)
        study = json.loads((args.model_root / "completion.json").read_text())
        if study.get("status") != "completed" or study.get("checkpoint_sha256") != args.expected_checkpoint_sha256:
            raise ValueError("study completion receipt does not bind this checkpoint")
        if (study.get("pde"), study.get("method"), study.get("training_view")) != (pde, method_name, train_view):
            raise ValueError("study completion receipt identity differs from the requested identity")
        split_manifest = json.loads((args.model_root / "split_manifest.json").read_text())
        config = yaml.safe_load((args.model_root / "resolved.yaml").read_text())
        campaign = yaml.safe_load(args.campaign.read_text())
        problem = campaign["problem_settings"][pde]
        if config["task"] != problem["task"]:
            raise ValueError("task differs between resolved configuration and campaign")
        if int(config.get("seed")) != int(split_manifest["seed"]):
            raise ValueError("mask seed differs from the study split seed")
        if config["data"].get("mask_seed") not in (None, int(config["seed"])):
            raise ValueError("resolved configuration carries a different mask seed")
        spec = parse_mask_spec(config["data"]["mask"]["protocol"])
        if not spec.is_mixture or spec.canonical_text() != study.get("mask_spec"):
            raise ValueError("training mask is not the study's deterministic mixture")
        bindings = json.loads(args.bindings.read_text())[pde]
        require_hash(args.data_root / "summary.json", study["dataset_summary_sha256"])
        shards = []
        for entry in bindings["shards"]:
            path = args.data_root / entry["path"]
            require_hash(path, entry["sha256"])
            stat = path.stat()
            if stat.st_size != entry["bytes"]:
                raise ValueError("dataset shard size differs from the main-grid binding")
            shards.append({"path": entry["path"], "sha256": entry["sha256"], "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        data_manifest = {"shards": shards, "original_summary_sha256": study["dataset_summary_sha256"]}
        ec = replace(_evaluation_config(config), device=args.device, data_layout="channels_last")
        options = dict(task=ec.task, target_step=ec.target_step, horizon=ec.horizon, history_steps=ec.history_steps,
                       seed=int(config["seed"]), mask=dict(campaign["views"][VIEW_ORDER[0]]["mask"]),
                       state_representation=config["data"].get("state_representation", "native"))
        paths = [args.data_root / s["path"] for s in shards]
        dataset = BenchmarkDataset(paths, **options)
        _, positions, split = stable_split(dataset.metadata, int(config["seed"]), split_manifest["macrodomain_identity"])
        assert_split(split, split_manifest)
        assert_split(split, bindings["split"])
        test = subset_dataset(dataset, positions)
        ids = [m["sample_id"] for m in test.metadata]
        if len(ids) != 200 or len(set(ids)) != 200:
            raise ValueError("not exactly 200 distinct held-out identities")
        frame_map = {m["sample_id"]: target_frame_metadata(m, ec) for m in test.metadata}
        for identity, frames in bindings["frame_mapping_by_identity"].items():
            if identity in frame_map and frame_map[identity] != frames:
                raise ValueError("target frame mapping differs from the main-grid binding")
        shape = metadata_target_shape(test, ec)
        target_positions = [1, 2, 3] if ec.task == "rollout" else [0]
        if any(f["target_stored_positions"] != target_positions for f in frame_map.values()):
            raise ValueError("not the original stored-frame task")
        spatial = (int(shape[-3]), int(shape[-2]))
        method = _method(config)
        weights = _load_weights(method, checkpoint, args.device, args.output)
        import torch

        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        batch_size = int(config["training"]["batch_size"])
        source_manifest = json.loads(args.source_manifest.read_text()) if args.source_manifest else None
        write_report(args.output / "provenance.json", {
            "schema_version": DRIVER_VERSION, "adapter": ADAPTER_VERSION, "identity": args.identity,
            "study_completion_receipt": study, "resolved_config_sha256": sha256_file(args.model_root / "resolved.yaml"),
            "resolved_config": config, "weights": weights, "dataset_manifest": data_manifest, "split": split,
            "mask_seed": int(config["seed"]), "inference_batch_size": batch_size,
            "training_mask_spec": spec.canonical_text(), "training_mask_view_id": spec.view_id,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
            "inference_code": source_manifest,
            "note": "Inference only; the study's training receipts and legacy evaluation are unchanged."})
        views: list[tuple[str, str, dict[str, Any], str | None]] = []
        for view in [v for v in args.views.split(",") if v]:
            views.append((view, "frozen_paper_view", dict(campaign["views"][view]["mask"]), None))
        for text in args.extra_view:
            mask, view_id, canonical = _view_mask(text, spatial)
            views.append((view_id, "extra_view", mask, canonical))
        if len({v[0] for v in views}) != len(views):
            raise ValueError("duplicate view identifiers")
        for view_id, kind, mask, canonical in views:
            block_begin = time.time()
            test.mask_config = dict(mask)
            dest = args.output / "blocks" / view_id
            dest.mkdir(parents=True, exist_ok=False)
            contract = {"schema_version": CONTRACT_VERSION, "task": ec.task, "split": "paper-test",
                        "expected_ids": ids, "expected_shape": shape, "target_time_indices": target_positions,
                        "time_index_semantics": "stored_frame_positions; per_identity_source_mapping_bound",
                        "frame_mapping_by_identity": frame_map, "projection": False,
                        "observation_id": view_id, "view_kind": kind, "mask_config": test.mask_config,
                        "extra_view_spec": canonical, "checkpoint_id": args.expected_checkpoint_sha256,
                        "dataset_manifest_sha256": _canonical_hash(data_manifest),
                        "historical_split_sha256": _canonical_hash(split),
                        "training_view": train_view, "training_mask_spec": spec.canonical_text(),
                        "training_cohort": "mixed_pattern_study", "training_protocol": study.get("training_protocol"),
                        "actual_epochs": study.get("actual_epochs"), "identity": args.identity,
                        "verification_adapter": ADAPTER_VERSION, "driver": DRIVER_VERSION}
            write_report(dest / "contract.json", contract)
            report = export_bound_predictions(method, batches(test, batch_size), config=ec, contract=contract,
                                              prediction_path=dest / "predictions.h5")
            write_report(dest / "score.json", report)
            item: dict[str, Any] = {"test_view": view_id, "view_kind": kind, "extra_view_spec": canonical,
                                    "mask_config": test.mask_config, "status": report["status"],
                                    "samples": report["scored_identity_count"], "seconds": time.time() - block_begin,
                                    "predictions_sha256": sha256_file(dest / "predictions.h5"),
                                    "score_sha256": sha256_file(dest / "score.json"),
                                    "rel_l2_joint_mean": (report["summary"] or {}).get("rel_l2_joint_mean") if report["status"] == "valid" else None}
            ratios = [float(m["observation_ratio"]) for m in test.metadata if m.get("observation_ratio") is not None]
            item["realized_observed_fraction"] = float(np.mean(ratios)) if len(ratios) == len(ids) else None
            if ec.task == "recovery" and report["status"] == "valid":
                projected = _projected_report(dest / "predictions.h5", contract)
                write_report(dest / "score.projected.json", projected)
                item["projected_status"] = projected["status"]
                item["projected_rel_l2_joint_mean"] = (projected["summary"] or {}).get("projected_rel_l2_joint_mean") if projected["status"] == "valid" else None
            write_report(dest / "completion.json", item)
            completion["blocks"].append(item)
            print(json.dumps({"identity": args.identity, **{k: item[k] for k in ("test_view", "status", "samples", "seconds", "rel_l2_joint_mean")}}), flush=True)
        for path, entry in zip(paths, shards):
            stat = path.stat()
            if stat.st_size != entry["bytes"] or stat.st_mtime_ns != entry["mtime_ns"]:
                raise ValueError("dataset changed while evaluating")
        completion["status"] = "complete" if all(b["status"] == "valid" for b in completion["blocks"]) else "invalid_predictions"
    except Exception as exc:  # noqa: BLE001 - the receipt must record any failure
        completion["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if dataset is not None:
            dataset.close()
        completion["finished_unix"] = time.time()
        completion["seconds"] = time.time() - begin
        write_report(args.output / "completion.json", completion)
    print(json.dumps({"identity": args.identity, "status": completion["status"], "blocks": len(completion["blocks"]),
                      "error": completion.get("error")}), flush=True)
    return 0 if completion["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
