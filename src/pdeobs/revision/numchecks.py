"""Per-record numerical checks on the reference route (rev8k H8 / E7.1 keys). Contract ``pdeobs.rev8k.numchecks/1``.

The checks re-solve one stored record with the benchmark's own solvers through :mod:`pdeobs.revision.reference`. They
are self-consistency checks (replay = self_replay, refinement = self_refinement: the same implementation, NOT an
independent solver); :func:`manufactured` is the analytic check of a static family.

Static Dirichlet families (Poisson, Darcy), :func:`static_refinement_record`:
  levels r = 1, 2, 4 solve the stored problem on r(N-1)+1 points of the stored boundary-inclusive interval
  (:func:`pdeobs.revision.reference.static_solve`, source / log-coefficient by trigonometric continuation of the stored
  field) and are compared at the N stored points:
  replay_rel = |c1 - stored| / |stored|, e_h_h2 = |c1 - c2| / |c2|, e_h2_h4 = |c2 - c4| / |c4|,
  observed_order = log2(e_h_h2 / e_h2_h4), stored_vs_finest = |stored - c4| / |c4|.
Periodic families (Burgers, Navier-Stokes), :func:`periodic_refinement_record`, at the stored frames 1-3 (dense index
k (T - 1) / 14):
  replay_rel (native route), time refinement with the internal step divided by 2 and 4 (e_t_dt_dt2, e_t_dt2_dt4,
  time_order, stored_vs_finest_time), space refinement on 2N^2 and 4N^2 from the trigonometric continuation of the
  stored initial state, evaluated back at the stored cell centres (e_s_h_h2, e_s_h2_h4, space_order,
  stored_vs_finest_space). ``last_frame=True`` adds the T_* checks at the last stored frame (H8: first two cases).
Every record also gets the spectral tail of its input (energy fraction above |k| = N/3) and ``alert`` (any replay /
stored-vs-finest difference > 1e-3; an engineering alert, not an exclusion threshold).

Relation to the executed v1 code (``supp.numerics``): the periodic records are bit-identical to its periodic_record /
last_frame_checks. Static records follow the V2 reference route: the r = 1 level also uses the trigonometric
continuation, and the Darcy forcing has its grid mean removed as in the generator (round-off-level differences).
Helmholtz, Heat and Reaction-diffusion have no reference route here (only :func:`manufactured` covers Helmholtz).
"""
from __future__ import annotations

import numpy as np

from . import reference as R
from .v2spec import STORED_FRAMES

NUMCHECK_CONTRACT = "pdeobs.rev8k.numchecks/1"
ALERT = 1e-3
STATIC_PDES = ("poisson", "darcy")
PERIODIC_PDES = ("burgers", "navier_stokes")
LABELS = {"replay_rel": "self_replay", "refinement": "self_refinement"}


def spectral_tail(field) -> float:
    """Fraction of the power spectrum with |k_x| > N/3 or |k_y| > N/3."""
    field = np.asarray(field, dtype=np.float64)
    n = field.shape[0]
    p = np.abs(np.fft.fft2(field)) ** 2
    k = np.abs(np.fft.fftfreq(n) * n)
    high = (k[:, None] > n / 3) | (k[None, :] > n / 3)
    return float(p[high].sum() / max(p.sum(), 1e-300))


def _alert(row: dict) -> bool:
    refs = [v for key, v in row.items() if key.startswith(("stored_vs_finest", "replay_rel", "T_replay")) and isinstance(v, float)]
    return bool(refs and max(refs) > ALERT)


def _head(pde, record_id, regime, n):
    row = {"contract": NUMCHECK_CONTRACT, "pde": pde, "n": int(n)}
    if record_id is not None:
        row["id"] = str(record_id)
    if regime is not None:
        row["regime"] = str(regime)
    return row


def static_refinement_record(pde: str, condition, native_solution, *, record_id=None, regime=None) -> dict:
    """Replay and r = 1, 2, 4 refinement of one stored static record (condition and solution N x N, float)."""
    if pde not in STATIC_PDES:
        raise ValueError(f"static reference route defined for {STATIC_PDES}, not {pde!r}")
    cond = np.asarray(condition, dtype=np.float64)
    stored = np.asarray(native_solution, dtype=np.float64)
    n = cond.shape[0]
    if cond.shape != (n, n) or stored.shape != (n, n):
        raise ValueError(f"condition and solution must be one N x N grid; got {cond.shape} / {stored.shape}")
    sols, solver = {}, {}
    for r in (1, 2, 4):
        sols[r], info = R.static_solve(pde, cond, n, r)
        solver[str(r)] = {k: info[k] for k in ("points", "iterations", "relative_residual")}
    c1, c2, c4 = sols[1], sols[2], sols[4]
    e12, e24 = R.rel(c1, c2), R.rel(c2, c4)
    row = _head(pde, record_id, regime, n)
    row.update(replay_rel=R.rel(c1, stored), e_h_h2=e12, e_h2_h4=e24, observed_order=R.order(e12, e24),
               stored_vs_finest=R.rel(stored, c4), condition_tail_energy_above_n_over_3=spectral_tail(cond),
               labels=dict(LABELS), solver=solver)
    row["alert"] = _alert(row)
    return row


def periodic_refinement_record(pde: str, regime: str, trajectory, *, record_id=None, last_frame: bool = False) -> dict:
    """Replay, time refinement (r_t = 2, 4) and space refinement (2N, 4N) of one stored periodic record.
    trajectory: the stored frames (T >= 4, N x N); frame 0 is the initial state, frames 1-3 are compared (and frame 14
    with ``last_frame``; the runs then continue to frame 14, which leaves frames 1-3 unchanged)."""
    if pde not in PERIODIC_PDES:
        raise ValueError(f"periodic reference route defined for {PERIODIC_PDES}, not {pde!r}")
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 3 or traj.shape[0] < 4 or traj.shape[1] != traj.shape[2]:
        raise ValueError(f"trajectory must be (T >= 4, N, N); got {traj.shape}")
    last = STORED_FRAMES - 1
    if last_frame and traj.shape[0] <= last:
        raise ValueError(f"last-frame checks need all {STORED_FRAMES} stored frames; got {traj.shape[0]}")
    n = traj.shape[-1]
    init, stored = traj[0], traj[1:4]
    horizon = last if last_frame else 3

    def run(n_ref, r_t, upto=horizon):
        return R.periodic_run(pde, regime, init, n_ref, r_t, last_frame=upto)[0]

    runs = {"base": run(n, 1), "t2": run(n, 2), "t4": run(n, 4), "s2": run(2 * n, 1), "s4": run(4 * n, 1, 3)}
    base, t2, t4, f2, f4 = (R.sample_periodic(runs[k][1:4], n) for k in ("base", "t2", "t4", "s2", "s4"))
    row = _head(pde, record_id, regime, n)
    row.update(replay_rel=R.rel(base, stored), e_t_dt_dt2=R.rel(base, t2), e_t_dt2_dt4=R.rel(t2, t4),
               time_order=R.order(R.rel(base, t2), R.rel(t2, t4)), stored_vs_finest_time=R.rel(stored, t4),
               e_s_h_h2=R.rel(base, f2), e_s_h2_h4=R.rel(f2, f4), space_order=R.order(R.rel(base, f2), R.rel(f2, f4)),
               stored_vs_finest_space=R.rel(stored, f4), condition_tail_energy_above_n_over_3=spectral_tail(init),
               labels=dict(LABELS))
    if last_frame:
        b, l2, l4, lf = (R.sample_periodic([runs[k][last]], n)[0] for k in ("base", "t2", "t4", "s2"))
        row.update(T_replay_rel=R.rel(b, traj[last]), T_e_t_dt_dt2=R.rel(b, l2), T_e_t_dt2_dt4=R.rel(l2, l4),
                   T_time_order=R.order(R.rel(b, l2), R.rel(l2, l4)), T_e_s_h_h2=R.rel(b, lf))
    row["alert"] = _alert(row)
    return row


def manufactured(pde: str, n: int = 128) -> dict:
    """Analytic check u* = sin(pi xi) sin(pi eta) on the boundary-inclusive interval of the n-point grid, analytic
    right-hand side, relative error vs u* at r = 1, 2, 4 (Helmholtz at the medium wavenumber; Darcy with the coefficient
    exp(0.5 sin(2 pi x) cos(2 pi y)))."""
    from ..pdes.numerics import solve_elliptic
    if pde not in ("poisson", "helmholtz", "darcy"):
        raise ValueError(f"manufactured solutions exist for poisson, helmholtz and darcy, not {pde!r}")
    k = None
    if pde == "helmholtz":
        from ..pdes import helmholtz as H
        k = H.WAVENUMBER_BY_REGIME["medium"]
    out = {}
    for r in (1, 2, 4):
        x, h = R.static_coords(n, r, n)
        L = x[-1] - x[0]
        xx, yy = np.meshgrid(x, x)
        xi, eta = (xx - x[0]) / L, (yy - x[0]) / L
        u = np.sin(np.pi * xi) * np.sin(np.pi * eta)
        lap = -2.0 * (np.pi / L) ** 2 * u
        if pde == "poisson":
            u_num = solve_elliptic(-lap, "dirichlet", h, h, rtol=R.STATIC_RTOL, maxiter=R.STATIC_MAXITER)[0]
        elif pde == "helmholtz":
            u_num = solve_elliptic(-lap - k ** 2 * u, "dirichlet", h, h, reaction=-(k ** 2), rtol=R.STATIC_RTOL,
                                   maxiter=R.STATIC_MAXITER)[0]
        else:
            a = np.exp(0.5 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy))
            ax = a * 0.5 * 2 * np.pi * np.cos(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
            ay = -a * 0.5 * 2 * np.pi * np.sin(2 * np.pi * xx) * np.sin(2 * np.pi * yy)
            ux = (np.pi / L) * np.cos(np.pi * xi) * np.sin(np.pi * eta)
            uy = (np.pi / L) * np.sin(np.pi * xi) * np.cos(np.pi * eta)
            u_num = solve_elliptic(-(ax * ux + ay * uy + a * lap), "dirichlet", h, h, coefficient=a, rtol=R.STATIC_RTOL,
                                   maxiter=R.STATIC_MAXITER)[0]
        out[r] = R.rel(u_num, u)
    return {"err": out, "order_1_2": R.order(out[1], out[2]), "order_2_4": R.order(out[2], out[4])}


__all__ = ["ALERT", "NUMCHECK_CONTRACT", "PERIODIC_PDES", "STATIC_PDES", "manufactured", "periodic_refinement_record",
           "spectral_tail", "static_refinement_record"]
