"""Portable single-row execution for the original C500 protocol and small demos.

This is not a scheduler or a historical-result converter. It resolves the
retained method registry, selects identities independently of shard split labels,
trains without validation, reloads last.pt, and strictly evaluates fixed views.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import yaml

from .evaluation import EvaluationConfig
from .one_setting import (
    PUBLIC_LEARNED_METHODS, VIEW_ORDER, build_experiment_config, load_campaign,
    stable_split, subset_dataset,
)
from .storage import sha256_file
from .strict_inference import (
    batches, create_identity_manifest, evaluate_model_strict, load_identity_dataset,
    metadata_target_shape, target_frame_metadata,
)
from .strict_score import CONTRACT_VERSION, _canonical_hash, write_report

ROW_VERSION = "pdeobs-paper-row-v1"


def validate_c500_frames(metadata: Sequence[Mapping[str, Any]],
                         selected_frames: Sequence[Mapping[str, Any]], task: str) -> None:
    """Keep the initial paper-table adapter's time coordinate explicit.

    Generic strict inference supports thinned solver coordinates. The current
    C500-to-paper path instead requires the declared [1,2,3] source indices;
    remapping a thinned corpus requires a separately reviewed adapter.
    """
    prefix = [0, 1, 2, 3] if task == "rollout" else [0]
    targets = [1, 2, 3] if task == "rollout" else [0]
    if any(list(row["stored_frame_indices"])[:len(prefix)] != prefix for row in metadata):
        raise ValueError("C500 source time coordinates require unthinned first frames; a thinned corpus needs an explicit reviewed mapping adapter")
    if any(row["target_source_frame_indices"] != targets for row in selected_frames):
        raise ValueError("C500 target source indices differ from the supported paper contract")


def row_split(metadata: Sequence[Mapping[str, Any]], seed: int, identity: str,
              cohort: str) -> tuple[list[int], list[int], dict[str, Any]]:
    """C500 calls the existing exact stable_split, never shard train/test labels."""
    ids = [row.get("sample_id") for row in metadata]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("source identities must be nonempty and unique")
    if cohort == "original500":
        train, test, receipt = stable_split(metadata, seed, identity)
        if Counter(str(metadata[i]["regime"]) for i in train) != {"low": 600, "medium": 600, "high": 600}:
            raise ValueError("C500 training requires exactly 600 identities per regime")
    elif cohort == "demo":
        # This explicitly reduced split is not labeled the 1800/200 algorithm.
        # Model construction, Trainer, last reload and strict inference below
        # are the same control path as original500.
        if not 6 <= len(metadata) <= 30:
            raise ValueError("demo mode requires 6 to 30 actual records")
        train, test, counts = [], [], {}
        if {row.get("regime") for row in metadata} != {"low", "medium", "high"}:
            raise ValueError("demo needs all three regimes")
        for regime in ("low", "medium", "high"):
            positions = [i for i, row in enumerate(metadata) if row["regime"] == regime]
            if len(positions) < 2:
                raise ValueError("demo needs at least two actual records per regime")
            positions.sort(key=lambda i: hashlib.sha256(f"{seed}|{identity}|{ids[i]}".encode()).hexdigest())
            test.append(positions[0])
            train.extend(positions[1:])
            counts[regime] = {"records": len(positions), "train": len(positions) - 1, "test": 1}
        receipt = {"schema_version": "pdeobs-paper-row-demo-split-v1",
                   "algorithm": "sha256_rank_one_test_per_regime_demo_only",
                   "seed": seed, "macrodomain_identity": identity,
                   "records": len(metadata), "train_records": len(train),
                   "test_records": len(test), "validation_records": 0, "regime_counts": counts}
    else:
        raise ValueError("only original500 and demo are executable; fixed200/min120/recovered cohorts require their separately versioned attempt adapter")
    receipt = {**receipt, "train_ids": [ids[i] for i in train], "test_ids": [ids[i] for i in test]}
    return train, test, receipt


def resolve_row(config_path: str | Path, data_root: str | Path,
                device: str = "cpu") -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = Path(config_path).resolve()
    row = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(row, dict) or row.get("schema_version") != ROW_VERSION:
        raise ValueError(f"row config must use {ROW_VERSION}")
    allowed = {"schema_version", "campaign", "registry", "cohort", "pde", "method",
               "train_view", "training_seed", "run_id", "attempt_id", "demo_epochs", "demo_test_views"}
    if set(row) - allowed:
        raise ValueError(f"unknown row config keys: {sorted(set(row) - allowed)}")
    cohort = row.get("cohort")
    if cohort not in {"original500", "demo"}:
        raise ValueError("unsupported cohort; only original500 and demo are executable, not fixed200/min120/recovered")
    for key in ("run_id", "attempt_id"):
        if not isinstance(row.get(key), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", row[key]):
            raise ValueError(f"row requires an explicit portable {key} (1-128 letters, digits, dot, underscore or hyphen)")
    if row.get("method") not in PUBLIC_LEARNED_METHODS:
        raise ValueError("row must select one of the seven public learned methods")
    if cohort == "original500" and any(key in row for key in ("demo_epochs", "demo_test_views")):
        raise ValueError("demo overrides cannot be applied to original500")
    campaign_path = (path.parent / row["campaign"]).resolve()
    registry_path = (path.parent / row["registry"]).resolve()
    campaign = load_campaign(campaign_path)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != "pdeobs.source-faithful-method-registry/v1":
        raise ValueError("row requires the current method registry")
    epochs = 500 if cohort == "original500" else int(row.get("demo_epochs", 1))
    if cohort == "demo" and not 1 <= epochs <= 3:
        raise ValueError("demo_epochs must be between 1 and 3")
    resolved = build_experiment_config(campaign, registry, dataset_root=data_root,
                pde=row["pde"], method=row["method"], view=row["train_view"], epochs=500)
    seed = row.get("training_seed", campaign["seed"])
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("training_seed must be an integer in [0, 2**32)")
    resolved["seed"] = seed
    resolved["data"]["mask_seed"] = int(campaign["seed"])
    resolved["training"].update(epochs=epochs, device=device, data_layout="channels_last",
                num_workers=0, pin_memory=device.startswith("cuda"), persistent_workers=False)
    # Operational CPU loader adaptation only. All scientific optimizer/model
    # values remain supplied by the selected current registry.
    views = list(VIEW_ORDER) if cohort == "original500" else list(row.get("demo_test_views", [row["train_view"]]))
    if not views or len(set(views)) != len(views) or set(views) - set(VIEW_ORDER):
        raise ValueError("test views must be distinct registered paper views")
    row = {**row, "training_seed": seed, "epochs": epochs, "test_views": views,
           "split_seed": int(campaign["seed"]), "campaign_sha256": sha256_file(campaign_path),
           "registry_sha256": sha256_file(registry_path)}
    return row, resolved, campaign


def run_paper_row(config_path: str | Path, *, data_root: str | Path,
                   manifest_path: str | Path, output: str | Path,
                   device: str = "cpu") -> dict[str, Any]:
    """Execute exactly one new attempt, or leave an explicit failure receipt."""
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    dataset = None
    stage = "configuration"
    receipt: dict[str, Any] = {"schema_version": ROW_VERSION, "status": "failed",
                "historical_results_replayed": False, "validation_records": 0,
                "test_arrays_accessed_before_final_checkpoint": 0}
    try:
        row, config, campaign = resolve_row(config_path, data_root, device)
        receipt.update(row)
        problem = campaign["problem_settings"][row["pde"]]
        ec = EvaluationConfig(task=problem["task"], data_layout="channels_last", device=device,
                              target_step=-1, history_steps=1, horizon=3, rollout_target_offset=0)
        stage = "manifest_and_metadata"
        dataset, manifest = load_identity_dataset(data_root, manifest_path, task=ec.task,
                     mask=config["data"]["mask"], target_step=-1, horizon=3, history_steps=1,
                     state_representation=problem["state_representation"], seed=int(campaign["seed"]))
        if any(any(meta.get(key) != value for key, value in
                   (("pde", row["pde"]), ("boundary", problem["boundary"]), ("setting", problem["setting"])))
                   for meta in dataset.metadata):
            raise ValueError("identity manifest mixes or differs from the selected physical macrodomain")
        shape = metadata_target_shape(dataset, ec)
        if shape[-1] != 1 or (row["cohort"] == "original500" and shape[-3:-1] != [128, 128]):
            raise ValueError("paper rows require scalar source fields and C500 requires 128x128")
        frames = [target_frame_metadata(meta, ec) for meta in dataset.metadata]
        if any(item["target_source_frame_indices"] != frames[0]["target_source_frame_indices"] for item in frames):
            raise ValueError("source solver target frame indices differ between identities")
        if row["cohort"] == "original500":
            validate_c500_frames(dataset.metadata, frames, ec.task)
        # The exact delimiter is part of the original SHA256 ranking input.
        identity = "|".join((row["pde"], problem["boundary"], problem["setting"]))
        train_indices, test_indices, split = row_split(dataset.metadata, row["split_seed"], identity, row["cohort"])
        train = subset_dataset(dataset, train_indices)
        test = subset_dataset(dataset, test_indices)
        config["training"]["scheduler_steps_per_epoch"] = math.ceil(len(train) / config["training"]["batch_size"])
        training_hash = _canonical_hash(config)
        recipe = copy.deepcopy(config)
        recipe.pop("seed", None)
        recipe["data"].pop("root", None)
        for key in ("device", "num_workers", "pin_memory", "persistent_workers"):
            recipe["training"].pop(key, None)
        recipe_hash = _canonical_hash(recipe)
        dataset_hash = _canonical_hash(manifest)
        receipt.update(training_records=len(train), test_records=len(test),
                       training_config_sha256=training_hash, training_recipe_sha256=recipe_hash,
                       dataset_manifest_sha256=dataset_hash,
                       task=ec.task, checkpoint_selection="final_last_only", stop_reason="not_completed")
        write_report(destination / "row_resolved.json", {"row": row, "experiment": config})
        write_report(destination / "split.json", split)
        write_report(destination / "identity_manifest.json", manifest)
        from .runner import _loader, _load_checkpoint, _method, _training_config
        from .training import Trainer, load_checkpoint_payload, seed_everything

        stage = "training"
        seed_everything(row["training_seed"])
        method = _method(config)
        trainer = Trainer(method, _training_config(config, destination, None))
        history = trainer.fit(_loader(train, config, shuffle=True), val_loader=None)
        checkpoint = destination / "checkpoints" / "last.pt"
        saved = load_checkpoint_payload(checkpoint, map_location="cpu")
        if saved.get("epoch") != row["epochs"] or len(saved.get("history", [])) != row["epochs"]:
            raise ValueError("last checkpoint did not complete the explicitly declared budget")
        checkpoint_hash = sha256_file(checkpoint)
        write_report(destination / "training_completion.json", {
            "schema_version": "pdeobs-paper-row-training-completion-v1",
            "cohort": row["cohort"], "actual_epochs": saved["epoch"],
            "checkpoint_id": checkpoint_hash, "training_config_sha256": training_hash,
            "training_recipe_sha256": recipe_hash,
            "dataset_manifest_sha256": dataset_hash, "split_sha256": _canonical_hash(split),
            "training_records": len(train), "validation_records": 0,
            "test_arrays_accessed": 0, "checkpoint_selection": "final_last_only",
            "history": history, "health_events": saved.get("health_events", []),
            "health_warnings": saved.get("health_warnings", []),
            "scientific_acceptance": "not_inferred_from_budget_completion",
        })
        # Discard the trained model and reload a new instance from final last.pt.
        # This is not a new-process test (the separate E5 workflow provides one).
        del saved, trainer, method
        method = _method(config)
        _load_checkpoint(method, checkpoint, device=device)
        receipt.update(actual_epochs=row["epochs"], checkpoint_id=checkpoint_hash,
                       stop_reason="declared_budget_complete", final_checkpoint_reloaded=True)
        stage = "strict_evaluation"
        block_results = []
        for view in row["test_views"]:
            test.mask_config = dict(campaign["views"][view]["mask"])
            block = destination / "evaluation" / view
            block.mkdir(parents=True, exist_ok=False)
            contract = {"schema_version": CONTRACT_VERSION, "task": ec.task,
                "split": "paper-test" if row["cohort"] == "original500" else "demo",
                "expected_ids": split["test_ids"], "expected_shape": shape,
                "target_time_indices": frames[0]["target_source_frame_indices"],
                "time_index_semantics": "source_solver_frame_indices", "projection": False,
                "observation_id": view, "mask_config": test.mask_config,
                "checkpoint_id": checkpoint_hash, "dataset_manifest_sha256": dataset_hash,
                "training_config_sha256": training_hash, "registry_sha256": row["registry_sha256"],
                "training_recipe_sha256": recipe_hash,
                "campaign_sha256": row["campaign_sha256"], "run_id": row["run_id"],
                "training_seed": row["training_seed"], "attempt_id": row["attempt_id"],
                "training_cohort": row["cohort"], "actual_epochs": row["epochs"],
                "pde": row["pde"], "method": row["method"], "train_view": row["train_view"],
                "test_view": view, "stop_reason": "declared_budget_complete"}
            write_report(block / "contract.json", contract)
            report = evaluate_model_strict(method, batches(test, config["training"]["batch_size"]),
                       config=ec, contract=contract, predictions_path=block / "predictions.h5",
                       report_path=block / "score.json")
            block_results.append({"test_view": view, "status": report["status"],
                                  "scored_identity_count": report["scored_identity_count"],
                                  "score_sha256": sha256_file(block / "score.json")})
        receipt.update(status="complete" if all(item["status"] == "valid" for item in block_results) else "invalid_evaluation",
                       evaluation_blocks=block_results, stage="finished")
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as exc:
        receipt.update(status="failed", stage=stage, errors=[str(exc)])
    finally:
        if dataset is not None:
            dataset.close()
    write_report(destination / "receipt.json", receipt)
    return receipt


def create_paper_row_demo_data(output: str | Path, pde: str) -> dict[str, Any]:
    """Explicit small data generation, separate from the no-fallback row runner."""
    from .generation import GenerationJob, generate_job

    if pde not in {"poisson", "heat"}:
        raise ValueError("paper-row demo data supports poisson or heat")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    shards, results = [], []
    for index, regime in enumerate(("low", "medium", "high")):
        name = f"{regime}.h5"
        job = GenerationJob(pde=pde, boundary="periodic" if pde == "heat" else "dirichlet",
                  setting="smooth_grf", regime=regime, sample_start=0, sample_count=3,
                  shard_index=index, output_path=str(root / name), resolution=16,
                  seed=20260804, time_steps=4 if pde == "heat" else 1,
                  macro_size=9, tier="custom", quality={"profile": "report"})
        result = generate_job(job, resume=False, overwrite=False)
        shards.append(name)
        results.append({"regime": regime, "job": job.to_dict(), "samples": result.sample_count})
    manifest = create_identity_manifest(root, shards, root / "identities.json")
    receipt = {"schema_version": "pdeobs-paper-row-demo-data-v1", "pde": pde,
               "actual_records": len(manifest["expected_ids"]), "generation": results}
    write_report(root / "generation.json", receipt)
    return receipt
