"""Observed/hidden error decomposition and data-consistency projection for every stationary block.

Two modes, no model inference in either:

    score      runs where the retained prediction files live.  For every stationary block
               (darcy, poisson, helmholtz: 7 methods x 9 training views x 9 test views = 1,701 blocks)
               it reopens ``predictions.h5`` and rescores the retained arrays with the unchanged strict
               scorer (``pdeobs.strict_score.score_arrays``) once with ``projection: true``.  The raw
               per-identity errors must equal the sealed ``score.json``; the decomposition identity and the
               projection bounds are checked per identity.  Writes one ``per_identity.csv.gz`` per PDE plus
               a ``blocks.jsonl`` receipt.

    aggregate  runs anywhere.  Reads the per-PDE outputs and writes the release tables:
                   stationary_diagnostics_per_identity.csv.gz
                   stationary_diagnostics_summary.csv
                   stationary_raw_vs_projected.csv
                   stationary_diagnostics_checks.json

Definitions (per identity i, d_i = max(||u_i||, 1e-12)):
    e_i      = ||u_hat_i - u_i|| / d_i                   raw (the released score)
    e_obs,i  = ||M_i (u_hat_i - u_i)|| / d_i             observed contribution, common denominator
    e_hid,i  = ||(1 - M_i)(u_hat_i - u_i)|| / d_i        hidden contribution, common denominator
    e_proj,i = ||where(M_i, y_i, u_hat_i) - u_i|| / d_i  raw prediction with the supplied observations pasted in
Raw scores are never replaced: projection is a separately labelled diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

STATIONARY = ("darcy", "poisson", "helmholtz")
COLUMNS = ["pde", "method", "train_view", "test_view", "identity", "regime", "raw_rel_l2", "observed_common_denominator",
           "hidden_common_denominator", "hidden_only_rel_l2", "projected_rel_l2", "observed_scalar_count", "hidden_scalar_count"]
SCORE_VERSION = "pdeobs-revision-static-diagnostics/20260926-v1"


def _score_block(block_dir: Path, score_arrays) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import h5py
    import numpy as np

    contract = json.loads((block_dir / "contract.json").read_text())
    sealed = json.loads((block_dir / "score.json").read_text())
    if sealed.get("status") != "valid" or contract.get("task") != "recovery":
        raise ValueError(f"{block_dir}: not a valid stationary block")
    with h5py.File(block_dir / "predictions.h5", "r") as handle:
        values = {key: handle[key][...] for key in ("prediction", "target", "prediction_ids", "target_ids",
                                                    "prediction_time_indices", "target_time_indices", "mask", "observation")}
    report = score_arrays(contract={**contract, "projection": True}, **values)
    if report["status"] != "valid":
        raise ValueError(f"{block_dir}: projected rescoring invalid: {report['errors']}")
    sealed_raw = {r["identity"]: r["rel_l2_joint"] for r in sealed["per_identity"]}
    rows = []
    worst = {"raw_vs_sealed": 0.0, "decomposition": 0.0, "projected_vs_hidden": 0.0, "projected_minus_raw": -math.inf}
    for r in report["per_identity"]:
        d = r["static_diagnostics"]
        raw, obs, hid, proj = r["rel_l2_joint"], d["observed_common_denominator"], d["hidden_common_denominator"], r["projected_rel_l2_joint"]
        worst["raw_vs_sealed"] = max(worst["raw_vs_sealed"], abs(raw - sealed_raw[r["identity"]]))
        worst["decomposition"] = max(worst["decomposition"], abs(raw * raw - (obs * obs + hid * hid)) / max(raw * raw, 1e-300))
        worst["projected_vs_hidden"] = max(worst["projected_vs_hidden"], abs(proj - hid) / max(raw, 1e-300))
        worst["projected_minus_raw"] = max(worst["projected_minus_raw"], proj - raw)
        rows.append({"identity": r["identity"], "regime": r["identity"].split("/")[-2], "raw_rel_l2": raw, "observed_common_denominator": obs,
                     "hidden_common_denominator": hid, "hidden_only_rel_l2": d["hidden_only_rel_l2"], "projected_rel_l2": proj,
                     "observed_scalar_count": d["observed_scalar_count"], "hidden_scalar_count": d["hidden_scalar_count"]})
    if worst["raw_vs_sealed"] > 1e-12:
        raise ValueError(f"{block_dir}: retained arrays do not reproduce the sealed score ({worst['raw_vs_sealed']})")
    if worst["decomposition"] > 1e-9:
        raise ValueError(f"{block_dir}: decomposition identity violated ({worst['decomposition']})")
    if worst["projected_vs_hidden"] > 1e-9 or worst["projected_minus_raw"] > 1e-9:
        raise ValueError(f"{block_dir}: projection bounds violated ({worst})")
    receipt = {"block": str(block_dir), "test_view": contract["observation_id"], "identities": len(rows), "worst": worst,
               "sealed_summary": sealed["summary"]["rel_l2_joint_mean"], "projected_summary": report["summary"]["projected_rel_l2_joint_mean"],
               "mask_config": contract["mask_config"], "contract_sha256": report["config_sha256"]}
    return rows, receipt


def cmd_score(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(args.source))
    from pdeobs.strict_score import score_arrays  # noqa: E402 - frozen verification source

    out = args.out / args.pde
    out.mkdir(parents=True, exist_ok=False)
    began = time.time()
    identities = sorted(p for p in (args.controller / "outputs" / args.pde).glob("*/*") if (p / "completion.json").exists())
    receipts = []
    with gzip.open(out / "per_identity.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for ident in identities:
            method, train_view = ident.parts[-2], ident.parts[-1]
            for block_dir in sorted((ident / "blocks").glob("*")):
                rows, receipt = _score_block(block_dir, score_arrays)
                for r in rows:
                    writer.writerow({"pde": args.pde, "method": method, "train_view": train_view, "test_view": receipt["test_view"], **r})
                receipt.update(pde=args.pde, method=method, train_view=train_view)
                receipts.append(receipt)
                print(json.dumps({"pde": args.pde, "method": method, "train_view": train_view, "test_view": receipt["test_view"],
                                  "projected": receipt["projected_summary"], "raw": receipt["sealed_summary"]}), flush=True)
    with (out / "blocks.jsonl").open("w", encoding="utf-8") as handle:
        for r in receipts:
            handle.write(json.dumps(r, sort_keys=True) + "\n")
    (out / "completion.json").write_text(json.dumps({"schema": SCORE_VERSION, "pde": args.pde, "identities": len(identities), "blocks": len(receipts),
                                                     "seconds": time.time() - began, "status": "complete"}, indent=1))
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    from common import REPO, VIEWS, load_json, write_csv, write_json  # noqa: E402

    index = load_json(REPO / "results/prediction_verification_20260924/index.json")["records"]
    per_identity: list[dict[str, Any]] = []
    blocks: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    completion = {}
    for pde in STATIONARY:
        folder = args.scored / pde
        completion[pde] = load_json(folder / "completion.json")
        if completion[pde].get("status") != "complete":
            raise ValueError(f"{pde}: scoring incomplete")
        with gzip.open(folder / "per_identity.csv.gz", "rt", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for k in COLUMNS[6:11]:
                    row[k] = float(row[k])
                row["observed_scalar_count"] = int(row["observed_scalar_count"]); row["hidden_scalar_count"] = int(row["hidden_scalar_count"])
                per_identity.append(row)
                blocks.setdefault((row["pde"], row["method"], row["train_view"], row["test_view"]), []).append(row)
    expected_blocks = {(p, m, v, w) for p in STATIONARY for m in ("ufno_2d", "fno", "cno", "deeponet", "gnot", "transolver", "pino") for v in VIEWS for w in VIEWS}
    if set(blocks) != expected_blocks:
        raise ValueError(f"stationary blocks incomplete: {len(blocks)} of {len(expected_blocks)}")
    summary_rows, raw_vs_projected = [], []
    worst_index_diff = 0.0
    for key in sorted(blocks):
        rows = blocks[key]
        if len(rows) != 200 or len({r["identity"] for r in rows}) != 200:
            raise ValueError(f"{key}: block is not 200 distinct identities")
        pde, method, train_view, test_view = key
        n = len(rows)
        raw_mean = sum(r["raw_rel_l2"] for r in rows) / n
        indexed = index[f"{pde}/{method}/{train_view}"]["blocks"][test_view]["joint"]["mean"]
        worst_index_diff = max(worst_index_diff, abs(raw_mean - indexed))
        sum_full_sq = sum(r["raw_rel_l2"] ** 2 for r in rows)
        sum_obs_sq = sum(r["observed_common_denominator"] ** 2 for r in rows)
        item = {"pde": pde, "method": method, "train_view": train_view, "test_view": test_view, "matched": train_view == test_view, "n": n,
                "raw_mean": raw_mean, "released_index_mean": indexed, "observed_common_mean": sum(r["observed_common_denominator"] for r in rows) / n,
                "hidden_common_mean": sum(r["hidden_common_denominator"] for r in rows) / n, "hidden_only_mean": sum(r["hidden_only_rel_l2"] for r in rows) / n,
                "projected_mean": sum(r["projected_rel_l2"] for r in rows) / n, "observed_squared_error_share": sum_obs_sq / sum_full_sq,
                "observed_fraction_mean": sum(r["observed_scalar_count"] / (r["observed_scalar_count"] + r["hidden_scalar_count"]) for r in rows) / n}
        summary_rows.append(item)
        raw_vs_projected.append({k: item[k] for k in ("pde", "method", "train_view", "test_view", "matched", "n", "raw_mean", "projected_mean")}
                                | {"projected_over_raw": item["projected_mean"] / item["raw_mean"], "hidden_common_mean": item["hidden_common_mean"]})
    if worst_index_diff > 1e-9:
        raise ValueError(f"per-identity raw means differ from the released index ({worst_index_diff})")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "stationary_diagnostics_per_identity.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in per_identity:
            writer.writerow({k: row[k] for k in COLUMNS})
    write_csv(out / "stationary_diagnostics_summary.csv", summary_rows, list(summary_rows[0].keys()))
    write_csv(out / "stationary_raw_vs_projected.csv", raw_vs_projected, list(raw_vs_projected[0].keys()))
    matched = [r for r in summary_rows if r["matched"]]
    cross = [r for r in summary_rows if not r["matched"]]
    checks = {"schema": SCORE_VERSION, "blocks": len(summary_rows), "identities_per_block": 200, "per_identity_rows": len(per_identity),
              "scoring_receipts": completion, "worst_raw_mean_vs_released_index": worst_index_diff,
              "per_identity_checks": "raw equals the sealed score (1e-12), raw^2 = observed^2 + hidden^2 (rel 1e-9), projected = hidden contribution and projected <= raw (1e-9); enforced by the score step, a violation aborts",
              "definitions": {"observed_squared_error_share": "sum_i e_obs,i^2 / sum_i e_i^2 over the 200 identities of a block; a share of squared error, not a fraction of wrong cells",
                              "projected": "where(mask, supplied observation, raw prediction); observations are noiseless so the observed entries become exact"},
              "aggregate": {"matched_blocks": len(matched), "cross_blocks": len(cross),
                            "matched_median_observed_share": _median([r["observed_squared_error_share"] for r in matched]),
                            "cross_median_observed_share": _median([r["observed_squared_error_share"] for r in cross]),
                            "matched_median_projected_over_raw": _median([r["projected_mean"] / r["raw_mean"] for r in matched]),
                            "cross_median_projected_over_raw": _median([r["projected_mean"] / r["raw_mean"] for r in cross])}}
    write_json(out / "stationary_diagnostics_checks.json", checks)
    print(json.dumps(checks["aggregate"], indent=1))
    return 0


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="rescore retained stationary prediction files with projection (host with the files)")
    s.add_argument("--controller", type=Path, required=True, help="verification controller root (outputs/<pde>/<method>/<view>/blocks/<view>/)")
    s.add_argument("--source", type=Path, required=True, help="directory containing the frozen pdeobs package (added to sys.path)")
    s.add_argument("--pde", choices=STATIONARY, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(func=cmd_score)
    a = sub.add_parser("aggregate", help="build the release tables from the per-PDE score outputs")
    a.add_argument("--scored", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.set_defaults(func=cmd_aggregate)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
