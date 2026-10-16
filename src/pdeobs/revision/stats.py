"""Paired parent statistics used for the October 2026 confirmation analyses.

Method differences are computed per parent (paired: same parent, same mask) and summarised with a percentile bootstrap of
the parents, drawn within each physical regime (stratified) and SHARED by every method, view, mask and training seed of a
population, so that all comparisons of one population use the same resamples. Training seeds are kept separate: a parent
interval does not measure seed-to-seed variation.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


class ParentResamples:
    """Stratified bootstrap resamples (B x N indices) of a fixed, sorted parent population."""

    def __init__(self, ids: Sequence[str], regime: Mapping[str, str] | None = None, boot: int = 2000, seed: int = 0):
        self.ids = sorted(str(i) for i in ids)
        self.pos = {i: k for k, i in enumerate(self.ids)}
        g = np.asarray([str((regime or {}).get(i, "all")) for i in self.ids])
        rng = np.random.default_rng(seed)
        groups = [np.flatnonzero(g == r) for r in sorted(set(g))]
        self.regimes = {str(r): int(len(x)) for r, x in zip(sorted(set(g)), groups)}
        self.R = np.concatenate([rng.choice(x, size=(boot, len(x)), replace=True) for x in groups], axis=1)

    def vector(self, per_record: Mapping[str, float]) -> np.ndarray:
        """Values on the population order; parents without a value are NaN (reported as missing, never imputed)."""
        v = np.full(len(self.ids), np.nan)
        for i, e in per_record.items():
            k = self.pos.get(str(i))
            if k is not None:
                v[k] = e
        return v

    def boot_mean(self, v: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return np.nanmean(v[self.R], axis=1)


def paired_difference(res: ParentResamples, a: np.ndarray, b: np.ndarray) -> dict:
    """d = a - b per parent (positive: b better): mean, 95 % percentile interval, median difference, share b better."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    ok = np.isfinite(d)
    if not ok.any():
        return {"n": 0}
    lo, hi = np.nanquantile(res.boot_mean(d), [0.025, 0.975])
    return {"n": int(ok.sum()), "mean_diff": float(np.nanmean(d)), "ci95": [float(lo), float(hi)],
            "median_diff": float(np.nanmedian(d)), "share_b_better": float((d[ok] > 0).mean())}


def matched_offpattern(errors: Mapping[tuple[str, str], np.ndarray], layouts: Sequence[str], res: ParentResamples) -> dict:
    """D = mean matched error of the layout specialists, C = mean of their off-pattern errors; errors[(train, test)] are
    per-parent vectors of ONE seed. Returns D, C, C/D and C - D with intervals from the shared resamples."""
    diag = [errors[(p, p)] for p in layouts]
    off = [errors[(p, q)] for p in layouts for q in layouts if q != p]
    D, C = np.mean([np.nanmean(v) for v in diag]), np.mean([np.nanmean(v) for v in off])
    bD = np.mean([res.boot_mean(v) for v in diag], axis=0)
    bC = np.mean([res.boot_mean(v) for v in off], axis=0)
    return {"D": float(D), "C": float(C), "C_over_D": float(C / D), "C_over_D_ci95": [float(x) for x in np.quantile(bC / bD, [0.025, 0.975])],
            "C_minus_D": float(C - D), "C_minus_D_ci95": [float(x) for x in np.quantile(bC - bD, [0.025, 0.975])]}


__all__ = ["ParentResamples", "matched_offpattern", "paired_difference"]
