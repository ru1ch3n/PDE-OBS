"""Paired test-identity bootstrap intervals for the 441-model grid (CPU only, no inference).

For each PDE the released per-record errors are arranged as E[method, train_view, test_view, identity]
(7 x 9 x 9 x 200; rollout PDEs also carry the three horizon errors).  Each replicate draws one identity
list with replacement *within each regime group* (67 / 67 / 66) and uses that same list for every
method, view and horizon of the PDE, so every contrast is paired.  Within a replicate the actual
statistic is recomputed from the resampled cell means:

    D = mean of the nine matched cells E[v,v];  C = mean of the 72 cross cells E[v,w], v != w
    C - D,  C / D,  E[v,w] - E[w,w],  E[v,w] / E[w,w]  (destination reference),
    per-horizon C / D (rollout),  density ratios E[v,w] / E[v,R50] for v, w in the random views,
    median of C / D over the pairs of a PDE, over all 49 pairs, and over the fixed 13-pair subset.

Intervals are percentile intervals of the replicate distribution.  They are *conditional test-identity
intervals*: they describe sampling variation over the 200 held-out identities of each PDE for the fixed
trained models, not training-seed uncertainty.  Ratios of means are never replaced by means of ratios.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from common import (METHODS, PDES, REPO, VIEWS, SourceError, load_json, load_per_identity, load_sources, require, resolve,
                    sha256_file, utc_now, write_csv, write_json)

BOOTSTRAP_VERSION = "pdeobs-revision-bootstrap/20260926-v1"
ROLLOUT = ("heat", "reaction_diffusion", "burgers", "navier_stokes")
RANDOM_VIEWS = ("random_50pct", "random_65pct", "random_80pct")
FIXED13 = ["burgers/cno", "darcy/cno", "darcy/fno", "darcy/pino", "darcy/ufno_2d", "heat/cno", "helmholtz/cno", "helmholtz/deeponet",
           "helmholtz/fno", "helmholtz/pino", "helmholtz/transolver", "navier_stokes/cno", "reaction_diffusion/cno"]
FIELDS = ["statistic", "scope", "pde", "method", "train_view", "test_view", "horizon", "estimate", "ci_low", "ci_high", "replicates",
          "seed", "n_identities", "regime_groups", "denominator_flag", "fraction_replicates_above_one"]


def load_grid(sources: dict[str, Any], pde: str) -> tuple[np.ndarray, np.ndarray | None, list[str], dict[str, list[int]]]:
    """E[method, train, test, identity] (+ horizons) from the released per-identity records, with ordered identities."""
    blocks = load_per_identity(sources, pde)
    identities = None
    for key, block in blocks.items():
        ids = sorted(block)
        if identities is None:
            identities = ids
        elif ids != identities:
            raise SourceError(f"{pde}: block {key} has a different identity set")
    require(identities is not None and len(identities) == 200, f"{pde}: no complete identity set")
    E = np.zeros((len(METHODS), len(VIEWS), len(VIEWS), 200), dtype=np.float64)
    for mi, method in enumerate(METHODS):
        for vi, train in enumerate(VIEWS):
            for wi, test in enumerate(VIEWS):
                block = blocks[(f"{pde}/{method}/{train}", test)]
                E[mi, vi, wi] = [block[i] for i in identities]
    H = None
    if pde in ROLLOUT:
        import gzip

        H = np.zeros((len(METHODS), len(VIEWS), len(VIEWS), 200, 3), dtype=np.float64)
        position = {i: k for k, i in enumerate(identities)}
        with gzip.open(resolve(sources["main_grid"]["per_identity"].format(pde=pde)), "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                _, method, train = row["model"].split("/")
                H[METHODS.index(method), VIEWS.index(train), VIEWS.index(row["test_view"]), position[row["identity"]]] = row["rel_l2_by_horizon"]
    groups: dict[str, list[int]] = {}
    for k, identity in enumerate(identities):
        groups.setdefault(identity.split("/")[-2], []).append(k)
    require(set(groups) == {"low", "medium", "high"}, f"{pde}: regime groups are not the three paper regimes")
    return E, H, identities, groups


def resample_weights(groups: dict[str, list[int]], n: int, replicates: int, seed: int) -> np.ndarray:
    """Counts matrix (replicates x n): one regime-stratified identity draw per replicate, shared by everything in the PDE."""
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, n), dtype=np.float64)
    for regime in sorted(groups):
        members = np.asarray(groups[regime])
        draws = rng.choice(members, size=(replicates, len(members)), replace=True)
        for b in range(replicates):
            counts = np.bincount(draws[b], minlength=n)
            weights[b] += counts
    require(np.all(weights.sum(axis=1) == n), "each replicate must draw exactly n identities")
    return weights


def cell_means(E: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Resampled cell means: E[..., identity] x counts -> [..., replicate]."""
    flat = E.reshape(-1, E.shape[-1])
    return (flat @ weights.T / weights.sum(axis=1)).reshape(*E.shape[:-1], weights.shape[0])


def pair_statistics(M: np.ndarray) -> dict[str, np.ndarray]:
    """M[method, train, test, replicate] -> D, C, transfer differences/ratios per replicate."""
    idx = np.arange(len(VIEWS))
    diag = M[:, idx, idx, :]                                   # [method, view, replicate]
    off = ~np.eye(len(VIEWS), dtype=bool)
    D = diag.mean(axis=1)                                      # [method, replicate]
    C = M[:, off, :].mean(axis=1)                              # [method, replicate]
    reference = M[:, idx, idx, :][:, None, :, :]               # E[w,w] broadcast over train axis -> [method, 1, test, replicate]
    diff = M - reference
    ratio = M / reference
    return {"D": D, "C": C, "C_minus_D": C - D, "C_over_D": C / D, "transfer_diff": diff, "transfer_ratio": ratio}


def percentile_row(statistic: str, scope: str, estimate: float, samples: np.ndarray, *, replicates: int, seed: int, n: int, groups: str,
                   pde: str = "", method: str = "", train_view: str = "", test_view: str = "", horizon: str = "") -> dict[str, Any]:
    finite = np.isfinite(samples)
    flag = "" if finite.all() else f"nonfinite in {int((~finite).sum())} replicates (zero denominator)"
    values = samples[finite] if finite.any() else samples
    lo, hi = (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if finite.any() else (math.nan, math.nan)
    return {"statistic": statistic, "scope": scope, "pde": pde, "method": method, "train_view": train_view, "test_view": test_view, "horizon": horizon,
            "estimate": estimate, "ci_low": lo, "ci_high": hi, "replicates": replicates, "seed": seed, "n_identities": n, "regime_groups": groups,
            "denominator_flag": flag, "fraction_replicates_above_one": float(np.mean(values > 1)) if statistic.endswith("ratio") or "over" in statistic else ""}


def run(sources_path: Path, out: Path, replicates: int, seed: int) -> dict[str, Any]:
    sources = load_sources(sources_path)
    index = load_json(resolve(sources["main_grid"]["index"]))["records"]
    statistics = load_json(resolve(sources["main_grid"]["statistics"]))
    rows: list[dict[str, Any]] = []
    c_over_d_replicates: dict[str, np.ndarray] = {}
    c_over_d_point: dict[str, float] = {}
    checks: dict[str, Any] = {}
    for pde in PDES:
        E, H, identities, groups = load_grid(sources, pde)
        n = len(identities)
        group_text = "/".join(f"{k}:{len(v)}" for k, v in sorted(groups.items()))
        # point estimates from the full identity set must reproduce the released index means
        point = E.mean(axis=-1)
        worst = 0.0
        for mi, method in enumerate(METHODS):
            for vi, train in enumerate(VIEWS):
                for wi, test in enumerate(VIEWS):
                    worst = max(worst, abs(point[mi, vi, wi] - index[f"{pde}/{method}/{train}"]["blocks"][test]["joint"]["mean"]))
        require(worst < 1e-9, f"{pde}: per-identity means do not reproduce the released index ({worst})")
        weights = resample_weights(groups, n, replicates, seed)
        M = cell_means(E, weights)
        stats = pair_statistics(M)
        point_stats = pair_statistics(point[..., None])
        for mi, method in enumerate(METHODS):
            pair = f"{pde}/{method}"
            c_over_d_replicates[pair] = stats["C_over_D"][mi]
            c_over_d_point[pair] = float(point_stats["C_over_D"][mi, 0])
            for name in ("C_minus_D", "C_over_D"):
                rows.append(percentile_row(name, "pair", float(point_stats[name][mi, 0]), stats[name][mi], replicates=replicates, seed=seed, n=n, groups=group_text, pde=pde, method=method))
            for vi, train in enumerate(VIEWS):
                for wi, test in enumerate(VIEWS):
                    if vi == wi:
                        continue
                    rows.append(percentile_row("transfer_diff", "cell", float(point_stats["transfer_diff"][mi, vi, wi, 0]), stats["transfer_diff"][mi, vi, wi],
                                               replicates=replicates, seed=seed, n=n, groups=group_text, pde=pde, method=method, train_view=train, test_view=test))
                    rows.append(percentile_row("transfer_ratio", "cell", float(point_stats["transfer_ratio"][mi, vi, wi, 0]), stats["transfer_ratio"][mi, vi, wi],
                                               replicates=replicates, seed=seed, n=n, groups=group_text, pde=pde, method=method, train_view=train, test_view=test))
            for train in RANDOM_VIEWS:
                for test in ("random_65pct", "random_80pct"):
                    vi, wi, r50 = VIEWS.index(train), VIEWS.index(test), VIEWS.index("random_50pct")
                    rows.append(percentile_row("density_ratio", "cell", float(point[mi, vi, wi] / point[mi, vi, r50]), M[mi, vi, wi] / M[mi, vi, r50],
                                               replicates=replicates, seed=seed, n=n, groups=group_text, pde=pde, method=method, train_view=train, test_view=test))
        if H is not None:
            for h in range(3):
                Mh = cell_means(H[..., h], weights)
                sh = pair_statistics(Mh)
                ph = pair_statistics(H[..., h].mean(axis=-1)[..., None])
                for mi, method in enumerate(METHODS):
                    rows.append(percentile_row("horizon_C_over_D", "pair", float(ph["C_over_D"][mi, 0]), sh["C_over_D"][mi], replicates=replicates, seed=seed, n=n,
                                               groups=group_text, pde=pde, method=method, horizon=str(h + 1)))
        pde_median = np.median(np.stack([c_over_d_replicates[f"{pde}/{m}"] for m in METHODS]), axis=0)
        rows.append(percentile_row("median_C_over_D", "pde", float(np.median([c_over_d_point[f"{pde}/{m}"] for m in METHODS])), pde_median,
                                   replicates=replicates, seed=seed, n=n, groups=group_text, pde=pde))
        checks[pde] = {"identities": n, "regime_groups": dict(sorted((k, len(v)) for k, v in groups.items())), "worst_point_vs_index": worst,
                       "identity_set_sha256": sha256_of(identities)}
    all_pairs = sorted(c_over_d_replicates)
    global_median = np.median(np.stack([c_over_d_replicates[p] for p in all_pairs]), axis=0)
    global_point = float(np.median([c_over_d_point[p] for p in all_pairs]))
    require(abs(global_point - statistics["stats"]["median_C_over_D"]) < 1e-9, "global median C/D does not reproduce the released statistics")
    rows.append(percentile_row("median_C_over_D", "global_49_pairs", global_point, global_median, replicates=replicates, seed=seed, n=200, groups="per PDE"))
    fixed_median = np.median(np.stack([c_over_d_replicates[p] for p in FIXED13]), axis=0)
    fixed_point = float(np.median([c_over_d_point[p] for p in FIXED13]))
    require(abs(fixed_point - statistics["fixed_subset"]["median_C_over_D"]) < 1e-9, "fixed-subset median C/D does not reproduce the released statistics")
    rows.append(percentile_row("median_C_over_D", "fixed_13_pairs", fixed_point, fixed_median, replicates=replicates, seed=seed, n=200, groups="per PDE"))
    pairs_c_gt_d = np.stack([c_over_d_replicates[p] > 1 for p in all_pairs]).sum(axis=0)
    rows.append(percentile_row("pairs_with_C_greater_than_D", "global_49_pairs", float(sum(c_over_d_point[p] > 1 for p in all_pairs)), pairs_c_gt_d.astype(float),
                               replicates=replicates, seed=seed, n=200, groups="per PDE"))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "bootstrap_intervals.csv", rows, FIELDS)
    config = {"schema": BOOTSTRAP_VERSION, "generated_utc": utc_now(), "replicates": replicates, "seed": seed, "interval": "95 % percentile",
              "resampling": "identities drawn with replacement within each regime group; one draw per replicate shared by all methods, views and horizons of a PDE (paired)",
              "interpretation": "conditional test-identity intervals for the fixed trained models; not training-seed uncertainty",
              "statistics": ["C_minus_D", "C_over_D", "transfer_diff (E[v,w]-E[w,w])", "transfer_ratio (E[v,w]/E[w,w])", "density_ratio (E[v,w]/E[v,R50])",
                             "horizon_C_over_D (rollout)", "median_C_over_D (pde, global 49 pairs, fixed 13 pairs)", "pairs_with_C_greater_than_D"],
              "sources": {"index_sha256": sha256_file(resolve(sources["main_grid"]["index"])), "statistics_sha256": sha256_file(resolve(sources["main_grid"]["statistics"])),
                          "per_identity_sha256": {pde: sha256_file(resolve(sources["main_grid"]["per_identity"].format(pde=pde))) for pde in PDES}},
              "checks": checks, "point_estimates_reproduce_released_statistics": True, "rows": len(rows),
              "global": {"median_C_over_D": {"estimate": global_point, "ci": [rows[-3]["ci_low"], rows[-3]["ci_high"]]},
                         "fixed13_median_C_over_D": {"estimate": fixed_point, "ci": [rows[-2]["ci_low"], rows[-2]["ci_high"]]},
                         "pairs_C_gt_D": {"estimate": rows[-1]["estimate"], "ci": [rows[-1]["ci_low"], rows[-1]["ci_high"]]}}}
    write_json(out / "bootstrap_config.json", config)
    return config


def sha256_of(identities: list[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(sorted(identities)).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=Path, default=REPO / "configs/revision/result_sources.yaml")
    parser.add_argument("--out", type=Path, default=REPO / "results/revision_v2")
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    config = run(args.sources, args.out, args.replicates, args.seed)
    print(json.dumps(config["global"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
