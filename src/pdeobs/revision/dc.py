"""Data-consistency split of static, noise-free recovery errors.

With M the observed cells, t the target and p the prediction (per record, flattened), and the full-field denominator ||t||:

    raw     = ||p - t|| / ||t||
    obs     = ||M (p - t)|| / ||t||              error on the cells the method was given
    hid     = ||(1 - M)(p - t)|| / ||t||         error of the data-consistent prediction M z + (1 - M) p (z = t on M)
    raw^2   = obs^2 + hid^2
    hid_rel = ||(1 - M)(p - t)|| / ||(1 - M) t||  hidden-region relative error (its own denominator; reported separately)

Only valid for static recovery with noise-free observations; it must not be applied to forecasts or noisy measurements.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def dc_split(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Per-record raw / obs / hid / hid_rel for arrays of shape (N, ...) with matching flattened sizes."""
    n = len(target)
    p = np.asarray(pred, dtype=np.float64).reshape(n, -1)
    t = np.asarray(target, dtype=np.float64).reshape(n, -1)
    m = np.asarray(mask).reshape(n, -1) > 0.5
    if not (p.shape == t.shape == m.shape):
        raise ValueError(f"shape mismatch: pred {p.shape}, target {t.shape}, mask {m.shape}")
    d2 = (p - t) ** 2
    den = np.maximum(np.sqrt((t ** 2).sum(1)), EPS)
    return {"raw": np.sqrt(d2.sum(1)) / den,
            "obs": np.sqrt((d2 * m).sum(1)) / den,
            "hid": np.sqrt((d2 * ~m).sum(1)) / den,
            "hid_rel": np.sqrt((d2 * ~m).sum(1)) / np.maximum(np.sqrt(((t ** 2) * ~m).sum(1)), EPS)}


__all__ = ["dc_split"]
