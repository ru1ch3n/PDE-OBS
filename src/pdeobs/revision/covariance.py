"""Covariance-weighted linear reconstruction (the strongest static reference of the October 2026 revision).

Given training fields X (N x d), the estimator uses the empirical mean ``mu`` and the leading POD modes ``U`` (d x r) with
eigenvalues ``lam`` (the training covariance restricted to the retained modes, C = U diag(lam) U^T). For one record with
observed cells S (a boolean mask) and observed values z, the reconstruction is the Gaussian conditional mean

    u_hat = mu + C S^T (S C S^T + lam_reg I)^-1 (z - S mu),   lam_reg = reg * trace(S C S^T) / K,

with K = number of observed cells and ``reg`` a dimensionless regularisation chosen on a validation split (never on test
data). Two numerically independent solves are provided: the latent-space form through the smaller of the r x r / K x K
systems (``latent``) and the direct K-space solve (``kspace``); they agree to floating-point precision on well-posed
problems. ``reg = 0`` gives the minimum-norm solution (pseudo-inverse on the numerically non-zero eigenvalues).

The reconstruction is linear in z and affine-equivariant; it uses training information only (mu, U, lam) and the observed
values of the record. It needs noise-aware regularisation under sensor noise, and it extrapolates poorly outside the span
of the training fields (see docs/october_revision.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


def pod_basis(X: np.ndarray, r_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centred POD of the rows of X (float64): (mean (d,), modes (d, r), all singular values). Uses the smaller Gram
    matrix; modes with zero singular value are dropped."""
    import scipy.linalg as sl
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    mu = X.mean(0)
    Xc = X - mu
    if N <= d:
        w, V = sl.eigh(Xc @ Xc.T, check_finite=False)
        w, V = w[::-1].clip(min=0), V[:, ::-1]
        s = np.sqrt(w)
        keep = s > 1e-12 * max(float(s[0]), 1e-300)
        r = int(min(r_max, keep.sum()))
        U = (Xc.T @ V[:, :r]) / s[:r]
    else:
        w, V = sl.eigh(Xc.T @ Xc, check_finite=False)
        w, V = w[::-1].clip(min=0), V[:, ::-1]
        s = np.sqrt(w)
        keep = s > 1e-12 * max(float(s[0]), 1e-300)
        r = int(min(r_max, keep.sum()))
        U = np.ascontiguousarray(V[:, :r])
    return mu, U, s


def _apply(e, Q, qb, lam):
    if lam > 0:
        return Q @ (qb / (e + lam))
    keep = e > 1e-12 * max(float(e.max()), 1e-300)
    return Q @ np.where(keep, qb / np.where(keep, e, 1.0), 0.0)


@dataclass
class CovarianceReconstructor:
    """Covariance-weighted reconstruction fitted on training fields; ``reg`` is chosen with :meth:`select_reg`."""

    rank: int = 2048
    reg: float | None = None
    mu: np.ndarray | None = field(default=None, repr=False)
    U: np.ndarray | None = field(default=None, repr=False)
    lam: np.ndarray | None = field(default=None, repr=False)

    def fit(self, X_train: np.ndarray) -> "CovarianceReconstructor":
        X_train = np.asarray(X_train, dtype=np.float64).reshape(len(X_train), -1)
        self.mu, U, s = pod_basis(X_train, self.rank)
        self.U = U
        self.lam = (s[:U.shape[1]] ** 2) / max(X_train.shape[0] - 1, 1)
        return self

    def coefficients(self, z: np.ndarray, mask: np.ndarray, regs: Sequence[float]) -> list[np.ndarray]:
        """Latent coefficients c (u_hat = mu + U c) for every reg, through the smaller symmetric system."""
        m = np.asarray(mask, dtype=bool).reshape(-1)
        SU = self.U[m]
        b = np.asarray(z, dtype=np.float64).reshape(-1) - self.mu[m]
        sw = np.sqrt(self.lam)
        B = SU * sw
        t = float(np.einsum("ij,ij->", B, B)) / SU.shape[0]
        K, r = B.shape
        out = []
        if r <= K:
            import scipy.linalg as sl
            e, Q = sl.eigh(B.T @ B, check_finite=False)
            e = e.clip(min=0)
            qb = Q.T @ (B.T @ b)
            out = [sw * _apply(e, Q, qb, reg * t) for reg in regs]
        else:
            import scipy.linalg as sl
            e, Q = sl.eigh(B @ B.T, check_finite=False)
            e = e.clip(min=0)
            qb = Q.T @ b
            out = [sw * (B.T @ _apply(e, Q, qb, reg * t)) for reg in regs]
        return out

    def predict(self, z: np.ndarray, mask: np.ndarray, reg: float | None = None, solve: str = "latent") -> np.ndarray:
        """Full-field reconstruction from the observed values z at the cells of ``mask`` (flat order)."""
        reg = self.reg if reg is None else reg
        if reg is None:
            raise ValueError("reg is not set: call select_reg on validation data or pass reg")
        if solve == "latent":
            return self.mu + self.U @ self.coefficients(z, mask, [reg])[0]
        if solve == "kspace":
            import scipy.linalg as sl
            m = np.asarray(mask, dtype=bool).reshape(-1)
            SU = self.U[m]
            G = (SU * self.lam) @ SU.T
            K = G.shape[0]
            G = G + (reg * float(np.trace(G)) / K) * np.eye(K)
            b = np.asarray(z, dtype=np.float64).reshape(-1) - self.mu[m]
            return self.mu + self.U @ (self.lam * (SU.T @ sl.solve(G, b, assume_a="pos", check_finite=False)))
        raise ValueError(f"unknown solve {solve!r}")

    def select_reg(self, records: Iterable[tuple[np.ndarray, np.ndarray]], regs: Sequence[float]) -> dict:
        """Choose reg by the mean relative L2 error over validation records (target field, mask); returns the scores."""
        regs = list(regs)
        sc = np.zeros(len(regs))
        n = 0
        for target, mask in records:
            t = np.asarray(target, dtype=np.float64).reshape(-1)
            m = np.asarray(mask, dtype=bool).reshape(-1)
            for k, c in enumerate(self.coefficients(t[m], m, regs)):
                sc[k] += np.linalg.norm(self.mu + self.U @ c - t) / max(np.linalg.norm(t), 1e-300)
            n += 1
        if n == 0:
            raise ValueError("no validation records")
        self.reg = float(regs[int(np.argmin(sc))])
        return {"reg": self.reg, "scores": dict(zip(map(float, regs), (sc / n).tolist())), "records": n}


__all__ = ["CovarianceReconstructor", "pod_basis"]
