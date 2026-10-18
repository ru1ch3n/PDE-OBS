"""Blind re-check of the H1 gappy-POD baseline (V2 plan 3.1). Contract ``pdeobs.rev8k.h1audit/1``.

This is an independent implementation of the H1 static method (rank-r POD of the training targets; ridge gappy
coefficients). Where H1 solved least squares on the augmented system [A; sqrt(lam) I], this audit uses the SVD of the
masked basis A = U_r[M]:

    c = V diag(s / (s^2 + lam)) W^T b,   b = z - mu[M],   lam = reg * ||A||_F^2 / r

with the same pseudo-inverse cutoff as numpy.lstsq (rcond = eps * max(A.shape)) when lam = 0. It also returns
sigma_min, the condition number and lam.

The predictor reads only the observation package (values at the observed cells, the mask). :func:`hidden_independence`
proves this: the prediction from a field whose hidden cells hold NaN or 1e6 is compared bit for bit with the clean
call.

Reported per record and view: raw, data-consistent (observed cells pasted back) and hidden-only relative L2;
the full-field projection error of the target onto the selected basis (mask independent); lam; sigma_min; condition
number. Per view: mean, median, 95th, max and failure count. Basis statistics: r95 / r99 / r99.9 / r99.99 of the
centred POD energy (cumulative energy only; never the selected rank).
"""
from __future__ import annotations

import numpy as np

H1_AUDIT_CONTRACT = "pdeobs.rev8k.h1audit/1"


def pod_basis(X: np.ndarray, r: int):
    """Centred POD of the rows of X: (mean, first r right singular vectors as columns, singular values)."""
    mu = X.mean(0)
    _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, vt[:r].T.copy(), s


def energy_ranks(s: np.ndarray) -> dict:
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    return {name: int(np.searchsorted(e, q) + 1) for name, q in (("r95", 0.95), ("r99", 0.99), ("r99.9", 0.999),
                                                                  ("r99.99", 0.9999))}


def gappy_svd(U: np.ndarray, mu: np.ndarray, z: np.ndarray, m: np.ndarray, reg: float):
    """Ridge gappy coefficients by the SVD of the masked basis; returns (c, info)."""
    A = U[m]
    b = z - mu[m]
    r = A.shape[1]
    W, s, Vt = np.linalg.svd(A, full_matrices=False)
    lam = reg * float(np.sum(s ** 2)) / r if reg > 0 else 0.0
    if lam > 0:
        f = s / (s ** 2 + lam)
    else:
        cut = np.finfo(np.float64).eps * max(A.shape) * (s[0] if s.size else 0.0)
        f = np.where(s > cut, 1.0 / np.where(s > cut, s, 1.0), 0.0)
    c = Vt.T @ (f * (W.T @ b))
    if not np.all(np.isfinite(c)):
        raise FloatingPointError("non-finite gappy coefficients")
    smin = float(s[-1]) if s.size else 0.0
    return c, {"lam": lam, "sigma_min": smin, "sigma_max": float(s[0]) if s.size else 0.0,
               "cond": float(s[0] / smin) if smin > 0 else float("inf")}


def predict(field: np.ndarray, mask: np.ndarray, U: np.ndarray, mu: np.ndarray, reg: float):
    """Prediction from a full field array of which ONLY the observed cells are read."""
    m = np.asarray(mask, dtype=bool).reshape(-1)
    z = np.asarray(field, dtype=np.float64).reshape(-1)[m]
    c, info = gappy_svd(U, mu, z, m, reg)
    return mu + U @ c, info


def hidden_independence(field, mask, U, mu, reg) -> dict:
    """Prediction with hidden cells replaced by NaN and by 1e6 must equal the clean prediction bit for bit."""
    m = np.asarray(mask, dtype=bool).reshape(-1)
    base, _ = predict(field, m, U, mu, reg)
    out = {}
    for name, val in (("nan", np.nan), ("big", 1.0e6)):
        f = np.asarray(field, dtype=np.float64).reshape(-1).copy()
        f[~m] = val
        p, _ = predict(f, m, U, mu, reg)
        out[name] = bool(np.array_equal(p, base))
    return out


def record_errors(p: np.ndarray, t: np.ndarray, m: np.ndarray, U: np.ndarray, mu: np.ndarray) -> dict:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    m = np.asarray(m, dtype=bool).reshape(-1)
    nt = max(float(np.linalg.norm(t)), 1e-12)
    q = p.copy()
    q[m] = t[m]
    proj = mu + U @ (U.T @ (t - mu))
    return {"raw": float(np.linalg.norm(p - t) / nt), "dc": float(np.linalg.norm(q - t) / nt),
            "hidden": float(np.linalg.norm((p - t)[~m]) / nt), "projection": float(np.linalg.norm(proj - t) / nt)}


def summarize(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.quantile(v, 0.95)),
            "max": float(v.max())}


__all__ = ["H1_AUDIT_CONTRACT", "energy_ranks", "gappy_svd", "hidden_independence", "pod_basis", "predict",
           "record_errors", "summarize"]
