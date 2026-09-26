"""Rebuild every mixed-pattern comparison from one declared set of result sources.

Numerators are the strict (``pdeobs-strict-v1``) scores of the mixed models produced by the
inference-only rescoring (``tools/revision/mix_strict_inference.py``); denominators are the strict
scores of the main grid in the declared prediction-verification release.  Both are aligned to the
same 200 held-out identities, the same frozen view masks, the same target frames and the same
scoring version, and the builder refuses anything else.  A missing strict score is an error; there
is no archive fallback.  The study's legacy evaluator scalars are read for an audit column only.

Outputs (in --out):
    mixed_comparisons.csv          one row per (pair, destination view)
    mixed_extra_views.csv          views absent from training, separately labelled (no specialist)
    mixed_summary.json             explicit populations and aggregation definitions
    mixed_comparison_inputs.json   every input file digest and every raw value used
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

from common import (PDES, REPO, SCORING_VERSION, TEST_IDENTITIES_PER_BLOCK, VIEWS, SourceError, canonical_hash, geometric_mean,
                    git_head, ids_hash, load_json, load_per_identity, load_sources, load_yaml, require, resolve, sha256_file,
                    utc_now, verified_main_grid, write_csv, write_json)

BUILDER_VERSION = "pdeobs-revision-build-results/20260926-v1"
TARGET_FRAMES = {"recovery": [0], "rollout": [1, 2, 3]}
LEGACY_TOLERANCE = {"atol": 1e-7, "rtol": 1e-4}
DEFINITIONS = {
    "E_mix_w": "strict rel-L2 (joint, float64 per-identity then arithmetic mean over the 200 held-out identities) of the mixed model tested on view w",
    "E_ww": "same statistic for the specialist trained on view w (main-grid strict score)",
    "ratio_w": "E_mix_w / E_ww",
    "mean_mix_W": "arithmetic mean of E_mix_w over the declared destination set W",
    "mean_specialist_W": "arithmetic mean of E_ww over the same W",
    "geomean_ratio_W": "exp(mean over w in W of log(E_mix_w / E_ww)); all three statistics of a population use the same W",
    "transfer_stats_w": "descriptive: median/min/max over the eight specialists trained on v != w and tested on w (E_vw); not budget matched",
    "extra_views": "views absent from every training set; no specialist exists, so no ratio is formed",
}


def assert_no_forbidden_sources(sources: Mapping[str, Any]) -> None:
    forbidden = [str(p) for p in sources.get("forbidden_sources", [])]

    def walk(value: Any) -> None:
        if isinstance(value, str):
            for prefix in forbidden:
                if value.startswith(prefix):
                    raise SourceError(f"forbidden score source declared: {value}")
        elif isinstance(value, Mapping):
            for k, v in value.items():
                if k != "forbidden_sources":
                    walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)

    walk({k: v for k, v in sources.items() if k != "forbidden_sources"})


def _population_views(spec: Mapping[str, Any]) -> list[str]:
    views = spec["views"]
    if views == "all_nine":
        return list(VIEWS)
    require(isinstance(views, list) and views and all(v in VIEWS for v in views) and len(set(views)) == len(views),
            f"population views must be distinct frozen views: {views}")
    return list(views)


def load_strict_block(folder: Path, view: str) -> dict[str, Any]:
    score_path = folder / "blocks" / view / "score.json"
    contract_path = folder / "blocks" / view / "contract.json"
    require(score_path.exists() and contract_path.exists(), f"missing strict score for {folder.name}/{view}: {score_path}")
    score = load_json(score_path)
    contract = load_json(contract_path)
    block_completion = load_json(folder / "blocks" / view / "completion.json") if (folder / "blocks" / view / "completion.json").exists() else {}
    require(score.get("status") == "valid", f"{folder.name}/{view}: strict score is not valid: {score.get('errors')}")
    require(score.get("scoring_version") == SCORING_VERSION, f"{folder.name}/{view}: scoring version {score.get('scoring_version')}")
    require(score.get("scored_identity_count") == TEST_IDENTITIES_PER_BLOCK, f"{folder.name}/{view}: not 200 scored identities")
    require(canonical_hash(score["config"]) == score["config_sha256"], f"{folder.name}/{view}: score is not bound to its contract")
    require(canonical_hash(contract) == score["config_sha256"], f"{folder.name}/{view}: contract.json differs from the scored contract")
    require(score.get("artifact_contract_binding") == "pdeobs-strict-inference-v1_sha256_bound", f"{folder.name}/{view}: predictions not hash-bound")
    per_identity = {row["identity"]: float(row["rel_l2_joint"]) for row in score["per_identity"]}
    require(len(per_identity) == TEST_IDENTITIES_PER_BLOCK, f"{folder.name}/{view}: duplicate identities in per-identity records")
    expected = contract["expected_ids"]
    require(len(expected) == len(set(expected)) == TEST_IDENTITIES_PER_BLOCK and set(expected) == set(per_identity), f"{folder.name}/{view}: identity list mismatch")
    require(score["identity_set_sha256"] == canonical_hash(sorted(expected)), f"{folder.name}/{view}: identity digest mismatch")
    mean = score["summary"]["rel_l2_joint_mean"]
    recomputed = statistics.fmean(per_identity.values())
    require(math.isclose(mean, recomputed, rel_tol=1e-12, abs_tol=1e-15), f"{folder.name}/{view}: summary mean differs from per-identity mean")
    return {"score": score, "contract": contract, "per_identity": per_identity, "mean": float(mean),
            "score_sha256": sha256_file(score_path), "contract_sha256": sha256_file(contract_path),
            "predictions_sha256": block_completion.get("predictions_sha256"), "realized_observed_fraction": block_completion.get("realized_observed_fraction"),
            "horizons": score["summary"].get("rel_l2_by_horizon_mean")}


def check_frozen_view_contract(block: Mapping[str, Any], *, pde: str, view: str, task: str, campaign: Mapping[str, Any],
                               checkpoint_sha256: str, test_ids_digest: str) -> None:
    c = block["contract"]
    require(c["task"] == task, f"{view}: task {c['task']} differs from {task}")
    require(c["observation_id"] == view and c.get("view_kind") == "frozen_paper_view", f"{view}: contract is not the frozen paper view")
    require(c["mask_config"] == campaign["views"][view]["mask"], f"{view}: mask differs from the frozen campaign view")
    require(c["target_time_indices"] == TARGET_FRAMES[task], f"{view}: target frames differ from the frozen task")
    require(c["split"] == "paper-test" and c["projection"] is False, f"{view}: contract is not the raw paper-test contract")
    require(c["checkpoint_id"] == checkpoint_sha256, f"{view}: checkpoint differs from the study completion receipt")
    require(ids_hash(c["expected_ids"]) == test_ids_digest, f"{view}: test identities differ from the frozen split")


def compare_pair(*, pde: str, method: str, task: str, mix_blocks: Mapping[str, Mapping[str, Any]], records: Mapping[str, Any],
                 grid_per_identity: Mapping[tuple[str, str], Mapping[str, float]], legacy_blocks: Mapping[str, float]) -> list[dict[str, Any]]:
    """One comparison row per destination view; every alignment check is fail-closed."""
    rows = []
    for view in VIEWS:
        mix = mix_blocks[view]
        specialist_id = f"{pde}/{method}/{view}"
        record = records[specialist_id]
        joint = record["blocks"][view]["joint"]
        spec_ids = grid_per_identity[(specialist_id, view)]
        require(set(spec_ids) == set(mix["per_identity"]), f"{specialist_id}: test identities differ between mixed and specialist blocks")
        require(math.isclose(statistics.fmean(spec_ids.values()), joint["mean"], rel_tol=1e-9, abs_tol=1e-12), f"{specialist_id}: index mean differs from per-identity mean")
        require(joint["mean"] > 0, f"{specialist_id}: zero denominator")
        transfers = {v: records[f"{pde}/{method}/{v}"]["blocks"][view]["joint"]["mean"] for v in VIEWS if v != view}
        ratio = mix["mean"] / joint["mean"]
        legacy = legacy_blocks.get(view)
        legacy_diff = abs(mix["mean"] - legacy) if legacy is not None else None
        rows.append({
            "pde": pde, "method": method, "train_view": "mixed_nine_views", "test_view": view, "task": task,
            "mix_rel_l2": mix["mean"], "specialist_rel_l2": joint["mean"], "specialist_rel_l2_sd200": joint.get("std"),
            "ratio_mix_over_specialist": ratio, "log10_ratio": math.log10(ratio),
            "specialist_identity": specialist_id, "specialist_cohort": record["training_cohort"], "specialist_actual_epochs": record["actual_epochs"],
            "specialist_stop_reason": record.get("stop_reason"), "specialist_training_protocol": (record.get("cohort_evidence") or {}).get("training_protocol"),
            "transfer_n": len(transfers), "transfer_median": statistics.median(transfers.values()), "transfer_min": min(transfers.values()),
            "transfer_max": max(transfers.values()), "transfer_argmin_train_view": min(transfers, key=transfers.get),
            "mix_below_transfer_median": mix["mean"] < statistics.median(transfers.values()), "mix_below_transfer_min": mix["mean"] < min(transfers.values()),
            "identity_count": TEST_IDENTITIES_PER_BLOCK, "test_identities_sha256": ids_hash(mix["per_identity"]),
            "mix_checkpoint_sha256": mix["contract"]["checkpoint_id"], "mix_score_sha256": mix["score_sha256"], "mix_predictions_sha256": mix["predictions_sha256"],
            "specialist_checkpoint_original_sha256": record["checkpoint_original_sha256"], "specialist_block_contract_sha256": record["blocks"][view].get("contract_sha256"),
            "legacy_evaluator_rel_l2": legacy, "legacy_abs_diff": legacy_diff,
            "legacy_within_tolerance": (legacy_diff <= LEGACY_TOLERANCE["atol"] + LEGACY_TOLERANCE["rtol"] * abs(legacy)) if legacy is not None else None,
            "mix_horizons": json.dumps(mix["horizons"]) if mix["horizons"] else None,
            "specialist_horizons": json.dumps(record["blocks"][view].get("horizons")) if record["blocks"][view].get("horizons") else None,
        })
    return rows


def population_summary(rows: list[dict[str, Any]], views: list[str], name: str) -> dict[str, Any]:
    by_view = {r["test_view"]: r for r in rows}
    require(all(v in by_view for v in views), f"population {name}: a declared destination has no comparison row")
    selected = [by_view[v] for v in views]
    ratios = {r["test_view"]: r["ratio_mix_over_specialist"] for r in selected}
    return {
        "population": name, "views": list(views), "n": len(views),
        "mean_mix": statistics.fmean(r["mix_rel_l2"] for r in selected),
        "mean_specialist": statistics.fmean(r["specialist_rel_l2"] for r in selected),
        "geomean_ratio": geometric_mean(list(ratios.values())),
        "ratio_min": min(ratios.values()), "ratio_min_view": min(ratios, key=ratios.get),
        "ratio_max": max(ratios.values()), "ratio_max_view": max(ratios, key=ratios.get),
        "n_ratio_below_one": sum(v < 1 for v in ratios.values()),
        "n_mix_below_transfer_median": sum(bool(r["mix_below_transfer_median"]) for r in selected),
        "n_mix_below_transfer_min": sum(bool(r["mix_below_transfer_min"]) for r in selected),
        "specialist_cohorts": sorted({r["specialist_cohort"] for r in selected}),
        "specialist_actual_epochs": sorted({int(r["specialist_actual_epochs"]) for r in selected}),
        "descriptive_transfer_median_over_views": statistics.median(r["transfer_median"] for r in selected),
    }


def build(sources_path: Path, out: Path) -> dict[str, Any]:
    sources = load_sources(sources_path)
    assert_no_forbidden_sources(sources)
    grid = verified_main_grid(sources)
    records = grid["records"]
    campaign_path = resolve(sources["frozen_contracts"]["campaign"])
    require(sha256_file(campaign_path) == sources["frozen_contracts"]["campaign_sha256"], "campaign digest differs from the declaration")
    campaign = load_yaml(campaign_path)
    bindings = load_json(resolve(sources["main_grid"]["dataset_bindings"]))
    inputs: dict[str, Any] = {"schema": BUILDER_VERSION, "generated_utc": utc_now(), "sources_file": str(sources_path),
                              "sources_sha256": sha256_file(resolve(sources_path)), "main_grid_digests": grid["digests"],
                              "campaign_sha256": sha256_file(campaign_path), "rows": {}}
    summary: dict[str, Any] = {"schema": BUILDER_VERSION, "generated_utc": utc_now(), "release_input_version": sources["release_input_version"],
                               "scoring_version": SCORING_VERSION, "definitions": DEFINITIONS, "pairs": {}, "pending": {},
                               "provenance": {"training_code": sources["provenance"]["training_code"], "inference_code": dict(sources["provenance"]["inference_code"]),
                                              "analysis_code": {"tool": "tools/revision/build_results.py", "version": BUILDER_VERSION, **git_head()}}}
    comparison_rows: list[dict[str, Any]] = []
    extra_rows: list[dict[str, Any]] = []
    per_identity_cache: dict[str, Any] = {}
    for entry in sources["mixed_models"]:
        identity = entry["identity"]
        pde, method, train_view = identity.split("/")
        require(train_view == "mixed_nine_views" and pde in PDES, f"unexpected mixed identity {identity}")
        if entry.get("status") != "complete":
            summary["pending"][identity] = {"status": entry.get("status"), "reason": entry.get("reason")}
            continue
        receipts = resolve(entry["receipts"])
        completion = load_json(receipts / "completion.json")
        split_manifest = load_json(receipts / "split_manifest.json")
        health = load_json(receipts / "health.json")
        require(completion["status"] == "completed" and completion["training_view"] == "mixed_nine_views", f"{identity}: training not completed")
        checkpoint = completion["checkpoint_sha256"]
        task = campaign["problem_settings"][pde]["task"]
        test_digest = split_manifest["test_sample_ids_sha256"]
        require(test_digest == bindings[pde]["split"]["test_sample_ids_sha256"], f"{identity}: study split differs from the main-grid binding")
        strict_dir = resolve(entry["strict_scores"])
        provenance = load_json(strict_dir / "provenance.json")
        require(provenance["study_completion_receipt"]["checkpoint_sha256"] == checkpoint, f"{identity}: inference provenance binds another checkpoint")
        row_completion = load_json(strict_dir / "completion.json")
        require(row_completion["status"] == "complete" and row_completion.get("optimizer_updates", 0) == 0, f"{identity}: inference run incomplete or not inference-only")
        legacy_blocks = {}
        legacy_dir = resolve(entry["legacy_evaluator_blocks"]) if entry.get("legacy_evaluator_blocks") else None
        if legacy_dir is not None and legacy_dir.exists():
            for view in VIEWS:
                p = legacy_dir / f"{view}.json"
                if p.exists():
                    legacy_blocks[view] = float(load_json(p)["metrics"]["relative_l2"])
        mix_blocks = {}
        for view in VIEWS:
            block = load_strict_block(strict_dir, view)
            check_frozen_view_contract(block, pde=pde, view=view, task=task, campaign=campaign, checkpoint_sha256=checkpoint, test_ids_digest=test_digest)
            mix_blocks[view] = block
        if pde not in per_identity_cache:
            per_identity_cache[pde] = load_per_identity(sources, pde)
        rows = compare_pair(pde=pde, method=method, task=task, mix_blocks=mix_blocks, records=records,
                            grid_per_identity=per_identity_cache[pde], legacy_blocks=legacy_blocks)
        populations = {}
        for name, spec in entry["populations"].items():
            views = _population_views(spec)
            populations[name] = {**population_summary(rows, views, name), "reason": spec.get("reason"), "subject_to": spec.get("subject_to")}
        for r in rows:
            for name, spec in entry["populations"].items():
                r[f"population_{name}"] = r["test_view"] in _population_views(spec)
        comparison_rows.extend(rows)
        extras = {}
        legacy_extra = load_json(receipts / "extra-views-completion.json")["blocks"] if (receipts / "extra-views-completion.json").exists() else {}
        for view_id in entry.get("extra_views", []):
            block = load_strict_block(strict_dir, view_id)
            c = block["contract"]
            require(c.get("view_kind") == "extra_view" and c["checkpoint_id"] == checkpoint and ids_hash(c["expected_ids"]) == test_digest, f"{identity}/{view_id}: extra view contract mismatch")
            legacy = legacy_extra.get(view_id, {}).get("relative_l2")
            extras[view_id] = {"spec": c.get("extra_view_spec"), "mask_config": c["mask_config"], "mix_rel_l2": block["mean"],
                               "realized_observed_fraction": block["realized_observed_fraction"], "legacy_evaluator_rel_l2": legacy,
                               "legacy_abs_diff": abs(block["mean"] - legacy) if legacy is not None else None,
                               "score_sha256": block["score_sha256"], "predictions_sha256": block["predictions_sha256"], "horizons": block["horizons"]}
            extra_rows.append({"pde": pde, "method": method, "train_view": "mixed_nine_views", "view_id": view_id, **{k: v for k, v in extras[view_id].items() if k not in ("mask_config", "horizons")},
                               "note": "no specialist trained on this view; descriptive only"})
        legacy_diffs = [r["legacy_abs_diff"] for r in rows if r["legacy_abs_diff"] is not None]
        summary["pairs"][identity] = {
            "status": "complete", "pde": pde, "method": method, "task": task,
            "training": {"protocol": completion.get("training_protocol"), "planned_epochs": completion.get("planned_epochs"), "actual_epochs": completion.get("actual_epochs"),
                         "stop_reason": completion.get("stop_reason"), "optimizer_steps": health.get("optimizer_steps"), "mask_spec": completion.get("mask_spec"),
                         "mask_seed": completion.get("mask_seed"), "checkpoint_sha256": checkpoint, "training_code": {"repository_commit": completion.get("repository_commit"),
                         "overlay_manifest_sha256": completion.get("overlay_manifest_sha256"), "study_adapter_sha256": completion.get("study_adapter_sha256"),
                         "protocol_adapter_sha256": completion.get("protocol_adapter_sha256")}},
            "inference": {"driver": provenance.get("schema_version"), "adapter": provenance.get("adapter"), "code": provenance.get("inference_code"),
                          "device": provenance.get("device_name"), "torch": provenance.get("torch_version"), "weights": provenance.get("weights"),
                          "row_completion_sha256": sha256_file(strict_dir / "completion.json")},
            "per_view": {r["test_view"]: {k: r[k] for k in ("mix_rel_l2", "specialist_rel_l2", "ratio_mix_over_specialist", "specialist_cohort", "specialist_actual_epochs",
                                                            "transfer_median", "transfer_min", "transfer_max", "legacy_evaluator_rel_l2", "legacy_within_tolerance")} for r in rows},
            "populations": populations, "extra_views": extras,
            "legacy_evaluator_audit": {"n": len(legacy_diffs), "max_abs_diff": max(legacy_diffs) if legacy_diffs else None,
                                       "all_within_tolerance": all(r["legacy_within_tolerance"] for r in rows if r["legacy_within_tolerance"] is not None) if legacy_diffs else None,
                                       "note": "legacy evaluator scalars are an audit column, never a numerator"},
        }
        inputs["rows"][identity] = {
            "receipts": {name: sha256_file(receipts / name) for name in ("completion.json", "health.json", "split_manifest.json", "eligibility.json", "stage-audit.json") if (receipts / name).exists()},
            "legacy_evaluator_blocks": {view: sha256_file(legacy_dir / f"{view}.json") for view in legacy_blocks} if legacy_dir else {},
            "strict": {view: {"score_sha256": b["score_sha256"], "contract_sha256": b["contract_sha256"], "predictions_sha256": b["predictions_sha256"],
                              "mean": b["mean"], "identity_set_sha256": b["score"]["identity_set_sha256"]} for view, b in mix_blocks.items()},
            "strict_extra": {view_id: {"score_sha256": e["score_sha256"], "predictions_sha256": e["predictions_sha256"], "mean": e["mix_rel_l2"]} for view_id, e in extras.items()},
            "specialists": {view: {"identity": f"{pde}/{method}/{view}", "joint": records[f"{pde}/{method}/{view}"]["blocks"][view]["joint"],
                                   "checkpoint_original_sha256": records[f"{pde}/{method}/{view}"]["checkpoint_original_sha256"],
                                   "contract_sha256": records[f"{pde}/{method}/{view}"]["blocks"][view].get("contract_sha256")} for view in VIEWS},
            "transfer_matrix_column": {view: {v: records[f"{pde}/{method}/{v}"]["blocks"][view]["joint"]["mean"] for v in VIEWS} for view in VIEWS},
            "test_identities_sha256": test_digest, "provenance_sha256": sha256_file(strict_dir / "provenance.json"),
        }
    fieldnames = ["pde", "method", "train_view", "test_view", "task"] + [k for k in comparison_rows[0] if k.startswith("population_")] + [
        "mix_rel_l2", "specialist_rel_l2", "ratio_mix_over_specialist", "log10_ratio", "specialist_rel_l2_sd200", "specialist_identity", "specialist_cohort",
        "specialist_actual_epochs", "specialist_stop_reason", "specialist_training_protocol", "transfer_n", "transfer_median", "transfer_min", "transfer_max",
        "transfer_argmin_train_view", "mix_below_transfer_median", "mix_below_transfer_min", "identity_count", "test_identities_sha256", "mix_checkpoint_sha256",
        "mix_score_sha256", "mix_predictions_sha256", "specialist_checkpoint_original_sha256", "specialist_block_contract_sha256", "legacy_evaluator_rel_l2",
        "legacy_abs_diff", "legacy_within_tolerance", "mix_horizons", "specialist_horizons"] if comparison_rows else ["pde"]
    for r in comparison_rows:
        for k in fieldnames:
            r.setdefault(k, None)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "mixed_comparisons.csv", comparison_rows, fieldnames)
    write_csv(out / "mixed_extra_views.csv", extra_rows, ["pde", "method", "train_view", "view_id", "spec", "mix_rel_l2", "realized_observed_fraction",
                                                          "legacy_evaluator_rel_l2", "legacy_abs_diff", "score_sha256", "predictions_sha256", "note"])
    summary["counts"] = {"pairs_complete": len(summary["pairs"]), "pairs_pending": len(summary["pending"]), "comparison_rows": len(comparison_rows),
                         "extra_view_rows": len(extra_rows),
                         "rows_mix_below_transfer_median": sum(bool(r["mix_below_transfer_median"]) for r in comparison_rows),
                         "rows_mix_below_transfer_min": sum(bool(r["mix_below_transfer_min"]) for r in comparison_rows),
                         "max_ratio_all_rows": max((r["ratio_mix_over_specialist"] for r in comparison_rows), default=None),
                         "min_ratio_all_rows": min((r["ratio_mix_over_specialist"] for r in comparison_rows), default=None)}
    write_json(out / "mixed_summary.json", summary)
    write_json(out / "mixed_comparison_inputs.json", inputs)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=REPO / "configs/revision/result_sources.yaml")
    parser.add_argument("--out", type=Path, default=REPO / "results/revision_v2")
    args = parser.parse_args()
    summary = build(args.sources, args.out)
    for identity, pair in summary["pairs"].items():
        for name, pop in pair["populations"].items():
            print(json.dumps({"pair": identity, "population": name, "n": pop["n"], "mean_mix": round(pop["mean_mix"], 6),
                              "mean_specialist": round(pop["mean_specialist"], 6), "geomean_ratio": round(pop["geomean_ratio"], 6)}))
    for identity, item in summary["pending"].items():
        print(json.dumps({"pair": identity, **item}))
    print(json.dumps(summary["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
