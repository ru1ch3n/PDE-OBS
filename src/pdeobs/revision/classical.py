"""H1 classical baselines of the rev8k v1 campaign as pure numpy functions on arrays. Contract
``pdeobs.rev8k.v1.classical/1``.

The numerics are those of the executed campaign module (``rev8k.classical8k``); only its data loader and file output
are left out. Arrays are float64 and flattened per state: a static target is (HW,), a dynamic record (4, HW) = u0 and
frames 1-3; ``m`` is a flat boolean mask over HW and ``z`` the values at the observed cells (z = field[m]).

Static (Darcy, Poisson, Helmholtz), recovery of the full target from the observed cells only:
  train_mean, observed_mean, linear_interp_nearest_fallback (fallback fraction recorded), gappy_pod_ridge, zero_output;
  raw and data-consistent (observed cells pasted back) errors for every method.
Dynamic (Heat, Reaction-diffusion, Burgers, Navier-Stokes), frames 1-3 from the partial u0:
  mean_future_trajectory, interp_persistence, pod_persistence, pod_affine_dynamics, zero_output. The POD basis and the
  affine map use the TRAINING TARGET frames 1-3 only (never the full training u0 or frames 4-14).
Gappy POD: c = argmin ||S_M U_r c - (z - S_M mu)||^2 + lam ||c||^2, solved by least squares on the augmented system
  [A; sqrt(lam) I] c = [b; 0] (no explicit inverse); lam = reg * trace(A^T A) / r; rank in RANKS, reg in REGS.
  One (rank, reg) per PDE, chosen on the development split (:func:`select_rank_reg`: basis from the dev-train records,
  score = mean error over the views on the dev-val records; dynamic: the three-step error); the basis is then refit on
  all training records.
Affine dynamics c' = c W + b (b unregularised), fitted on the 1->2 and 2->3 latent transitions with the same ridge scale.
Data complexity (T02): centred POD energy r95 / r99 and the radially averaged power spectrum per regime.
"""
from __future__ import annotations

import numpy as np

CLASSICAL_CONTRACT = "pdeobs.rev8k.v1.classical/1"
RANKS = (16, 32, 64, 128)
REGS = (0.0, 1e-8, 1e-6, 1e-4, 1e-2)
STATIC_METHODS = ("train_mean", "observed_mean", "linear_interp_nearest_fallback", "gappy_pod_ridge", "zero_output")
DYNAMIC_METHODS = ("mean_future_trajectory", "interp_persistence", "pod_persistence", "pod_affine_dynamics", "zero_output")


def rel(p, t):
    return float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-12))


def pod(X, r):
    """Centred POD of the rows of X: (mean, first r right singular vectors as columns, singular values)."""
    mu = X.mean(0)
    _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, vt[:r].T.copy(), s


def gappy_coeffs(U, mu, z, m, reg):
    """Ridge gappy POD by least squares on [A; sqrt(lam) I] c = [b; 0]; lam = reg * trace(A^T A) / r."""
    A = U[m]
    b = z - mu[m]
    r = A.shape[1]
    if reg > 0:
        lam = reg * float(np.einsum("ij,ij->", A, A)) / r
        A = np.vstack([A, np.sqrt(lam) * np.eye(r)])
        b = np.concatenate([b, np.zeros(r)])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    if not np.all(np.isfinite(c)):
        raise FloatingPointError("non-finite gappy POD coefficients")
    return c


def interp(z, m, shape=(128, 128)):
    """Piecewise-linear interpolation of the observed values (scipy griddata), nearest-neighbour fallback outside the
    convex hull; returns (flat field, fallback fraction)."""
    from scipy.interpolate import griddata
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    pts = np.stack([yy.ravel()[m], xx.ravel()[m]], 1)
    lin = griddata(pts, z, (yy, xx), method="linear").reshape(-1)
    bad = ~np.isfinite(lin)
    if bad.any():
        lin[bad] = griddata(pts, z, (yy, xx), method="nearest").reshape(-1)[bad]
    return lin, float(bad.mean())


def _side(hw: int) -> int:
    n = int(round(hw ** 0.5))
    if n * n != hw:
        raise ValueError(f"the radial spectrum needs square fields; {hw} cells is not a square")
    return n


def complexity(X, regimes, static):
    """T02: centred POD energy (r95 / r99 of the training targets) and the radially averaged power spectrum per regime.
    X: static (N, HW); dynamic (N, 4, HW) (frames 1-3 enter the POD energy, frame 1 the spectrum). Square n x n fields;
    the spectrum keeps the radial bins 0..n/2 (65 bins at n = 128, the executed setting)."""
    out = {}
    states = X if static else X[:, 1:4].reshape(-1, X.shape[-1])
    _, s, _ = np.linalg.svd(states - states.mean(0), full_matrices=False)
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    out["pod_r95"] = int(np.searchsorted(e, 0.95) + 1)
    out["pod_r99"] = int(np.searchsorted(e, 0.99) + 1)
    out["pod_energy_at"] = {str(k): float(e[k - 1]) for k in (1, 4, 16, 64, 128, 256) if k <= len(e)}
    n = _side(X.shape[-1])
    k = np.fft.fftfreq(n) * n
    kr = np.rint(np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)).astype(int)
    for g in sorted(set(regimes)):
        sel = [i for i, r in enumerate(regimes) if r == g]
        f = (X[sel] if static else X[sel, 1]).reshape(len(sel), n, n)
        p = (np.abs(np.fft.fft2(f)) ** 2).mean(0)
        spec = np.bincount(kr.ravel(), p.ravel()) / np.bincount(kr.ravel())
        out[f"spectrum_{g}"] = [float(x) for x in spec[:n // 2 + 1]]
    return out


def pod_basis(X, r, static):
    """POD of the training targets: static rows of X, or the frames 1-3 of dynamic records."""
    return pod(X if static else X[:, 1:4].reshape(-1, X.shape[-1]), r)


def affine_dynamics(X, mu, U, reg):
    """Latent affine map c' = c W + b fitted on the 1->2 and 2->3 transitions of the training frames 1-3 (b not
    regularised; lam = reg * ||P - mean P||_F^2 / r)."""
    C = (X[:, 1:4] - mu) @ U                                          # (N, 3, r) latent of frames 1-3
    P = np.concatenate([C[:, 0], C[:, 1]])                           # 1->2 and 2->3 transitions
    Q = np.concatenate([C[:, 1], C[:, 2]])
    r = U.shape[1]
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)                             # intercept b is not regularised
    lam = reg * float(np.einsum("ij,ij->", Pc, Pc)) / r if reg > 0 else 0.0
    Aug = np.vstack([Pc, np.sqrt(lam) * np.eye(r)]) if lam > 0 else Pc
    Rhs = np.vstack([Qc, np.zeros((r, r))]) if lam > 0 else Qc
    W, *_ = np.linalg.lstsq(Aug, Rhs, rcond=None)                     # c' = c W + b
    b = Q.mean(0) - P.mean(0) @ W
    return W, b


def predict_dynamic(kind, mu, U, W, b, z, m, reg):
    """Frames 1-3 (3, HW) from the observed u0 values: gappy coefficients of u0, then persistence (pod_persistence) or
    three applications of the affine map (pod_affine_dynamics)."""
    c = gappy_coeffs(U, mu, z, m, reg)
    if kind == "pod_persistence":
        return np.stack([mu + U @ c] * 3)
    out = []
    for _ in range(3):
        c = c @ W + b
        out.append(mu + U @ c)
    return np.stack(out)


def data_consistent(p, z, m):
    """Copy of a flat prediction with the observed values pasted back at the observed cells."""
    q = p.copy()
    q[m] = z
    return q


def static_predictions(z, m, mu, U, reg, mean_field, shape=(128, 128)):
    """The five static predictions (flat, HW) from the observed values only; returns (dict, interp fallback fraction).
    mean_field is the training-target mean; (mu, U) the selected basis."""
    z = np.asarray(z, dtype=np.float64)
    om = np.full(m.shape, z.mean(), dtype=np.float64)
    om[m] = z
    ip, fb = interp(z, m, shape)
    gp = mu + U @ gappy_coeffs(U, mu, z, m, reg)
    preds = {"train_mean": mean_field, "observed_mean": om, "linear_interp_nearest_fallback": ip,
             "gappy_pod_ridge": gp, "zero_output": np.zeros(m.shape, dtype=np.float64)}
    return preds, fb


def dynamic_predictions(z, m, mu, U, W, b, reg, mean_future, shape=(128, 128)):
    """The five dynamic predictions (3, HW) of frames 1-3 from the observed u0 values; returns (dict, fallback fraction).
    mean_future is the training mean of frames 1-3 (3, HW); (W, b) the affine map of :func:`affine_dynamics`."""
    z = np.asarray(z, dtype=np.float64)
    ip, fb = interp(z, m, shape)
    preds = {"mean_future_trajectory": mean_future, "interp_persistence": np.stack([ip] * 3),
             "pod_persistence": predict_dynamic("pod_persistence", mu, U, None, None, z, m, reg),
             "pod_affine_dynamics": predict_dynamic("pod_affine_dynamics", mu, U, W, b, z, m, reg),
             "zero_output": np.zeros((3,) + m.shape, dtype=np.float64)}
    return preds, fb


def select_rank_reg(X_dev_train, X_dev_val, view_masks, static, ranks=RANKS, regs=REGS) -> dict:
    """(rank, reg) selection on the development split, as executed: basis of rank max(ranks) from the dev-train records,
    every (r, reg) scored by the mean relative error over all views and dev-val records (static: gappy_pod_ridge;
    dynamic: pod_affine_dynamics, joint over frames 1-3); a non-finite solve counts as inf. view_masks: sequence over
    views of flat boolean masks (N_dev_val, HW). Returns {"scores": {"r|reg": mean}, "rank", "reg"} (first minimum)."""
    mu_d, U_d, _ = pod_basis(X_dev_train, max(ranks), static)
    scores = {}
    for r in ranks:
        for reg in regs:
            errs = []
            if not static:
                W, b = affine_dynamics(X_dev_train, mu_d, U_d[:, :r], reg)
            for masks in view_masks:
                for j in range(len(X_dev_val)):
                    m = np.asarray(masks[j], dtype=bool).reshape(-1)
                    try:
                        if static:
                            p = mu_d + U_d[:, :r] @ gappy_coeffs(U_d[:, :r], mu_d, X_dev_val[j][m], m, reg)
                            errs.append(rel(p, X_dev_val[j]))
                        else:
                            p = predict_dynamic("pod_affine_dynamics", mu_d, U_d[:, :r], W, b, X_dev_val[j][0][m], m, reg)
                            errs.append(rel(p, X_dev_val[j][1:4]))
                    except FloatingPointError:
                        errs.append(float("inf"))
            scores[f"{r}|{reg}"] = float(np.mean(errs))
    best = min(scores, key=scores.get)
    return {"scores": scores, "rank": int(best.split("|")[0]), "reg": float(best.split("|")[1]),
            "rule": "mean dev-val error over the views", "dev_val_records": len(X_dev_val)}


def gappy_pod_recover(X_train, X_test, masks, rank, reg, basis=None):
    """Static gappy POD: basis of the given rank from X_train (N, HW) (or ``basis`` = (mu, U) already computed by
    :func:`pod`); for every test record j the prediction from X_test[j][masks[j]] only. Returns (predictions
    (N_test, HW), per-record raw errors, per-record data-consistent errors)."""
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    mu, U = basis if basis is not None else pod(X_train, int(rank))[:2]
    preds, raw, dc = [], [], []
    for j in range(len(X_test)):
        m = np.asarray(masks[j], dtype=bool).reshape(-1)
        t = X_test[j]
        z = t[m]
        p = mu + U @ gappy_coeffs(U, mu, z, m, reg)
        preds.append(p)
        raw.append(rel(p, t))
        dc.append(rel(data_consistent(p, z, m), t))
    return np.stack(preds) if preds else np.zeros((0, X_train.shape[1])), raw, dc


__all__ = ["CLASSICAL_CONTRACT", "DYNAMIC_METHODS", "RANKS", "REGS", "STATIC_METHODS", "affine_dynamics", "complexity",
           "data_consistent", "dynamic_predictions", "gappy_coeffs", "gappy_pod_recover", "interp", "pod", "pod_basis",
           "predict_dynamic", "rel", "select_rank_reg", "static_predictions"]
