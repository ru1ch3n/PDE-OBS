"""Training inventory of the 441 credited models and the mixture-assignment tables.  No training, no scores.

Sources (all declared, all metadata):
    results/prediction_verification_20260924/index.json     credited identities: cohort, epochs, stop reason, training
                                                            configuration, checkpoint digests
    results/archived_campaign/index.json                    training receipts of the same attempts (health.json optimizer
                                                            steps, repository commit, attempt identity); its legacy scores
                                                            are never read
    results/revision_v2/inputs/resolved_configs_441.json    resolved.yaml of every released model (batch size, schedule),
                                                            digest-bound to the verification index
    results/mixed_pattern_study/receipts/*/                 mask specification and staged assignment counts of the mixed rows
    results/revision_v2/inputs/contract_shapes.json         spatial shape per PDE

Outputs (in --out): training_inventory.csv (441 rows), trial_inventory.json, mixture_assignments.csv,
mixture_counts.csv and audit_summary.json.  Unknown evidence is written as ``unknown``; a planned budget is never
turned into an observed update count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from common import (PDES, REPO, TEST_IDENTITIES_PER_BLOCK, TRAIN_IDENTITIES, VIEWS, git_head, load_json, load_yaml, require, sha256_file,
                    utc_now, write_csv, write_json)

sys.path.insert(0, str(REPO / "src"))

AUDIT_VERSION = "pdeobs-revision-audit-training/20260926-v2"
LOADER_EVIDENCE = ("code: src/pdeobs/runner.py::_loader batches the 1800 training records without drop_last in one process; "
                   "src/pdeobs/training.py takes one optimizer update per batch (no gradient accumulation)")
SCHEDULE_KEYS = {"step": ("scheduler_step_size", "scheduler_gamma"),
                 "one_cycle": ("scheduler_pct_start", "scheduler_div_factor", "scheduler_final_div_factor", "scheduler_anneal_strategy", "epochs")}
INVENTORY_FIELDS = [
    "run_id", "training_cohort", "parent_attempt_id", "pde", "method", "task", "training_observation", "training_seed",
    "training_identities_sha256", "checkpoint_sha256", "checkpoint_released_sha256", "optimizer", "learning_rate", "weight_decay", "scheduler",
    "schedule_parameters", "grad_clip", "batch_size", "world_size", "gradient_accumulation", "drop_last", "planned_epochs", "completed_epochs",
    "actual_optimizer_updates", "derived_updates_from_epochs", "updates_consistency", "actual_examples_processed", "stopping_rule", "stop_reason",
    "training_status", "evaluation_status_strict", "training_code_commit", "last_verified_timestamp", "evidence_source"]


def steps_per_epoch(batch_size: int) -> int:
    return math.ceil(TRAIN_IDENTITIES / int(batch_size))


def unknown(value: Any) -> Any:
    return "unknown" if value is None else value


def schedule_parameters(training: Mapping[str, Any]) -> dict[str, Any]:
    return {k: training.get(k) for k in SCHEDULE_KEYS.get(str(training.get("scheduler")), ())}


def grid_inventory(vrecords: Mapping[str, Any], arecords: Mapping[str, Any], resolved: Mapping[str, Any], bindings: Mapping[str, Any],
                   campaign: Mapping[str, Any], verified_stamp: str) -> list[dict[str, Any]]:
    rows = []
    for identity in sorted(vrecords):
        v, a, rc = vrecords[identity], arecords[identity], resolved[identity]
        require(rc["digest_matches_index"], f"{identity}: resolved.yaml digest not bound to the verification index")
        require(a["checkpoint_id"] == v["checkpoint_original_sha256"], f"{identity}: archived and verified checkpoint identities differ")
        pde, method, view = identity.split("/")
        training = rc["training"]
        ce = v.get("cohort_evidence") or {}
        health = a.get("health") or {}
        planned = ce.get("planned_epochs") if ce.get("planned_epochs") is not None else training.get("epochs")
        completed = v["actual_epochs"]
        steps = health.get("optimizer_steps")
        bs = int(training["batch_size"])
        derived = completed * steps_per_epoch(bs)
        consistency = "unknown" if steps is None else ("equal" if steps == derived else ("more_than_completed_epochs" if steps > derived else "less_than_completed_epochs"))
        continuation = ce.get("continuation_records") or []
        early = bool(ce.get("patience")) or training.get("early_stopping_patience") is not None
        if ce.get("training_protocol"):
            rule = ce["training_protocol"]
        elif planned == 500 and not early:
            rule = "fixed_500_epochs_last_checkpoint (original protocol; derived from planned epochs and no early stopping)"
        else:
            rule = "unknown"
        if v.get("stop_reason"):
            stop, stop_src = v["stop_reason"], "index.stop_reason"
        elif completed == planned == 500 and not early:
            stop, stop_src = "max_epochs_500", "derived: completed == planned == 500 and no early stopping configured"
        else:
            stop, stop_src = "unknown", "no stop reason recorded"
        evidence = {
            "planned_epochs": "index.cohort_evidence.planned_epochs" if ce.get("planned_epochs") is not None else "resolved.yaml training.epochs",
            "completed_epochs": "prediction_verification index actual_epochs (history entries of the credited checkpoint)",
            "actual_optimizer_updates": "archived_campaign index health.optimizer_steps (health.json of the credited attempt)" if steps is not None else "unknown: no optimizer step counter in the credited receipts",
            "actual_examples_processed": "derived: completed epochs x 1800 training records; " + LOADER_EVIDENCE,
            "batch_size/optimizer/schedule": "resolved.yaml of the released model (digest bound to the verification index)",
            "stop_reason": stop_src, "training_status": "index.cohort_evidence.training_status" if ce.get("training_status") else "archived health.status",
            "segment_note": ("optimizer step counter and history may cover the last continuation segment only: " + ";".join(continuation)) if continuation else None}
        rows.append({
            "run_id": a.get("training_attempt_identity") or "unknown", "training_cohort": v["training_cohort"],
            "parent_attempt_id": ("continuation_of:" + ";".join(continuation)) if continuation else "", "pde": pde, "method": method,
            "task": campaign["problem_settings"][pde]["task"], "training_observation": view, "training_seed": rc.get("seed"),
            "training_identities_sha256": bindings[pde]["split"]["train_sample_ids_sha256"], "checkpoint_sha256": v["checkpoint_original_sha256"],
            "checkpoint_released_sha256": v.get("checkpoint_released_sha256"), "optimizer": training.get("optimizer"), "learning_rate": training.get("learning_rate"),
            "weight_decay": training.get("weight_decay"), "scheduler": training.get("scheduler"), "schedule_parameters": json.dumps(schedule_parameters(training), sort_keys=True),
            "grad_clip": training.get("grad_clip"), "batch_size": bs, "world_size": 1, "gradient_accumulation": 1, "drop_last": False,
            "planned_epochs": planned, "completed_epochs": completed, "actual_optimizer_updates": unknown(steps), "derived_updates_from_epochs": derived,
            "updates_consistency": consistency, "actual_examples_processed": completed * TRAIN_IDENTITIES, "stopping_rule": rule, "stop_reason": stop,
            "training_status": ce.get("training_status") or health.get("status") or "unknown",
            "evaluation_status_strict": "complete_valid (nine strict blocks, 200 identities each)" if v["evaluation_status"] == "complete_prediction_valid" else v["evaluation_status"],
            "training_code_commit": a.get("repository_commit") or "unknown", "last_verified_timestamp": verified_stamp,
            "evidence_source": json.dumps(evidence, sort_keys=True)})
    return rows


def reconstruct_identities(pde: str, boundary: str, setting: str, regime_counts: Mapping[str, Mapping[str, int]], seed: int) -> list[dict[str, Any]]:
    rows = []
    for regime in ("low", "medium", "high"):
        for index in range(int(regime_counts[regime]["records"])):
            rows.append({"sample_id": f"seed-{seed}/{pde}/{boundary}/{setting}/{regime}/{index:06d}", "pde": pde, "boundary": boundary,
                         "setting": setting, "regime": regime, "regime_sample_index": index})
    return rows


def assign_mixture(spec_text: str, mask_seed: int, identities: list[dict[str, Any]], spatial: tuple[int, int], split_of: Mapping[str, str]) -> list[dict[str, Any]]:
    """The release assignment logic (``pdeobs.mask_specs`` / ``dataset._spec_mask``) run over frozen identity lists."""
    import numpy as np

    from pdeobs.mask_specs import parse_mask_spec
    from pdeobs.schema import derive_seed

    spec = parse_mask_spec(spec_text)
    require(spec.is_mixture, "training mask is not a mixture")
    out = []
    for row in identities:
        stratum = (row["pde"], row["boundary"], row["setting"], row["regime"])
        index, selection_seed, cycle_position = spec.select(mask_seed=mask_seed, stratum=stratum, position=int(row["regime_sample_index"]))
        component = spec.components[index]
        seed = derive_seed(mask_seed, "mask", row["sample_id"], component.protocol)
        mask = component.generate(spatial, seed)
        packed = np.packbits(np.asarray(mask, dtype=bool).ravel()).tobytes()
        out.append({"physical_identity": row["sample_id"], "pde": row["pde"], "regime": row["regime"], "regime_sample_index": int(row["regime_sample_index"]),
                    "split": split_of[row["sample_id"]], "assigned_component": component.base_id, "component_canonical": component.canonical_text(),
                    "component_index": int(index), "cycle_position": int(cycle_position), "selection_seed": int(selection_seed), "mask_seed": int(seed),
                    "component_mask_hash": hashlib.sha256(packed + json.dumps(list(spatial)).encode()).hexdigest(),
                    "observed_count": int(np.count_nonzero(mask)), "observed_fraction": float(np.count_nonzero(mask)) / float(mask.size)})
    return out


def mixture_tables(spec_text: str, pdes: list[str], campaign: Mapping[str, Any], bindings: Mapping[str, Any], shapes: Mapping[str, Any],
                   stage_audits: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from pdeobs.one_setting import stable_split

    mask_seed = int(bindings[pdes[0]]["split"]["seed"])
    assignments: list[dict[str, Any]] = []
    counts: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    for pde in pdes:
        problem = campaign["problem_settings"][pde]
        split_info = bindings[pde]["split"]
        identities = reconstruct_identities(pde, problem["boundary"], problem["setting"], split_info["regime_counts"], int(split_info["seed"]))
        train_idx, test_idx, split = stable_split(identities, int(split_info["seed"]), split_info["macrodomain_identity"])
        require(split["train_sample_ids_sha256"] == split_info["train_sample_ids_sha256"] and split["test_sample_ids_sha256"] == split_info["test_sample_ids_sha256"],
                f"{pde}: reconstructed identities do not reproduce the frozen split digests")
        split_of = {identities[i]["sample_id"]: "train" for i in train_idx}
        split_of.update({identities[i]["sample_id"]: "test" for i in test_idx})
        shape = shapes[pde]["expected_shape"]
        spatial = (int(shape[-3]), int(shape[-2]))
        rows = assign_mixture(spec_text, mask_seed, identities, spatial, split_of)
        again = {r["physical_identity"]: (r["assigned_component"], r["component_mask_hash"]) for r in assign_mixture(spec_text, mask_seed, list(reversed(identities)), spatial, split_of)}
        require(all(again[r["physical_identity"]] == (r["assigned_component"], r["component_mask_hash"]) for r in rows), f"{pde}: assignment depends on identity order")
        assignments.extend(rows)
        tally: Counter = Counter((r["split"], r["regime"], r["assigned_component"]) for r in rows)
        components = sorted({r["assigned_component"] for r in rows})
        for split_name in ("train", "test"):
            for regime in ("low", "medium", "high"):
                for component in components:
                    counts.append({"pde": pde, "split": split_name, "regime": regime, "component": component, "count": tally[(split_name, regime, component)]})
        train_total = sum(v for (s, _, _), v in tally.items() if s == "train")
        test_total = sum(v for (s, _, _), v in tally.items() if s == "test")
        require(train_total == TRAIN_IDENTITIES and test_total == TEST_IDENTITIES_PER_BLOCK, f"{pde}: counts do not sum to the split sizes")
        recorded = {}
        for row_name, audit in stage_audits.items():
            if row_name.startswith(pde + "__"):
                reproduced = {regime: {c: tally[("train", regime, c)] for c in components} for regime in ("low", "medium", "high")}
                recorded[row_name] = {"matches_reproduction": audit["mask_assignment"]["per_regime"] == reproduced}
        require(all(v["matches_reproduction"] for v in recorded.values()), f"{pde}: reproduced counts differ from the staged mask_assignment record")
        checks[pde] = {"split_digests_reproduced": True, "order_independent": True, "train_total": train_total, "test_total": test_total, "spatial": list(spatial),
                       "train_counts_per_component": {c: sum(tally[("train", g, c)] for g in ("low", "medium", "high")) for c in components},
                       "stage_audit_agreement": recorded}
    return assignments, counts, checks


def run(out: Path) -> dict[str, Any]:
    vindex = load_json(REPO / "results/prediction_verification_20260924/index.json")
    vrecords = vindex["records"]
    require(len(vrecords) == 441, "verification index is not the 441 grid")
    arecords = load_json(REPO / "results/archived_campaign/index.json")["records"]
    resolved_doc = load_json(REPO / "results/revision_v2/inputs/resolved_configs_441.json")
    require(resolved_doc["n"] == 441 and not resolved_doc["digest_mismatches"], "resolved configurations are not digest-bound to the verification index")
    shapes = load_json(REPO / "results/revision_v2/inputs/contract_shapes.json")["shapes"]
    bindings = load_json(REPO / "results/prediction_verification_20260924/dataset_bindings.json")
    campaign = load_yaml(REPO / "configs/campaign/all_pde_one_setting_10method_9x9.yaml")
    receipts = sorted(p for p in (REPO / "results/mixed_pattern_study/receipts").glob("*") if (p / "completion.json").exists())
    stage_audits = {p.name: load_json(p / "stage-audit.json") for p in receipts if (p / "stage-audit.json").exists()}
    specs = {load_json(p / "completion.json")["mask_spec"] for p in receipts}
    require(len(specs) == 1, "the mixed rows do not share one mask specification")
    spec_text = specs.pop()
    mixed_pdes = sorted({p.name.split("__")[0] for p in receipts})
    verified_stamp = "2026-09-25 (pdeobs-prediction-verification/20260925-v1)"
    grid_rows = grid_inventory(vrecords, arecords, resolved_doc["records"], bindings, campaign, verified_stamp)
    assignments, counts, mixture_checks = mixture_tables(spec_text, mixed_pdes, campaign, bindings, shapes, stage_audits)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "training_inventory.csv", grid_rows, INVENTORY_FIELDS)
    write_csv(out / "mixture_assignments.csv", assignments, ["physical_identity", "pde", "regime", "regime_sample_index", "split", "assigned_component", "component_canonical",
                                                             "component_index", "cycle_position", "selection_seed", "mask_seed", "component_mask_hash", "observed_count", "observed_fraction"])
    write_csv(out / "mixture_counts.csv", counts, ["pde", "split", "regime", "component", "count"])
    trials = {"schema": AUDIT_VERSION, "generated_utc": utc_now(), "n": len(grid_rows), "by_cohort": dict(Counter(r["training_cohort"] for r in grid_rows)),
              "by_training_status": dict(Counter(r["training_status"] for r in grid_rows)), "by_updates_consistency": dict(Counter(r["updates_consistency"] for r in grid_rows)),
              "by_stop_reason": dict(Counter(r["stop_reason"] for r in grid_rows)), "by_batch_size": {f"{k[0]}:{k[1]}": v for k, v in sorted(Counter((r["method"], r["batch_size"]) for r in grid_rows).items())},
              "unknown_optimizer_updates": sorted(f"{r['pde']}/{r['method']}/{r['training_observation']}" for r in grid_rows if r["actual_optimizer_updates"] == "unknown"),
              "evaluation_status_strict": dict(Counter(r["evaluation_status_strict"] for r in grid_rows)), "last_verified_timestamp": verified_stamp,
              "note": "one credited attempt per identity; all 441 have nine valid strict blocks of 200 identities in the released verification"}
    write_json(out / "trial_inventory.json", trials)
    summary = {"schema": AUDIT_VERSION, "generated_utc": utc_now(), "analysis_code": git_head(),
               "inputs": {"verification_index_sha256": sha256_file(REPO / "results/prediction_verification_20260924/index.json"),
                          "archived_index_sha256": sha256_file(REPO / "results/archived_campaign/index.json"),
                          "resolved_configs_sha256": sha256_file(REPO / "results/revision_v2/inputs/resolved_configs_441.json"),
                          "campaign_sha256": sha256_file(REPO / "configs/campaign/all_pde_one_setting_10method_9x9.yaml"),
                          "stage_audits": {k: sha256_file(REPO / "results/mixed_pattern_study/receipts" / k / "stage-audit.json") for k in stage_audits}},
               "checks": {"resolved_config_digests_bound": True, "archived_and_verified_checkpoints_agree": True, "mixture": mixture_checks, "mask_spec": spec_text},
               "counts": {"inventory_rows": len(grid_rows), "assignments": len(assignments), "count_rows": len(counts)},
               "notes": ["main-grid optimizer updates come from health.json step counters; for continued runs the counter may cover the last segment only (parent_attempt_id column)",
                         "examples processed are derived from completed epochs and the loader semantics; they are not an independent observation", LOADER_EVIDENCE]}
    write_json(out / "audit_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "results/revision_v2")
    args = parser.parse_args()
    summary = run(args.out)
    print(json.dumps({"counts": summary["counts"], "mixture": {k: v["train_counts_per_component"] for k, v in summary["checks"]["mixture"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
