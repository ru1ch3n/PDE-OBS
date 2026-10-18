"""Numerical reference routes for the V2 labels (plan 5.3; resolution addon 3.2-3.4).

The same continuous problem is solved on finer grids and sampled at the coordinates of the stored grid:

* Static Dirichlet families (Poisson, Darcy). The N stored points per axis include the zero boundary at
  a = 1/256 and b = 255/256 (the outer cell centres of the original 128 grid). x_j = a + j (b - a)/(N - 1).
  Refinement level m solves on m(N - 1) + 1 points of the same interval, so every stored point is a fine point and the
  boundary does not move (plan 5.3: 128 / 255 / 509 points for m = 1 / 2 / 4).
  * Poisson source: exact trigonometric interpolation of the stored 128^2 source.
  * Darcy coefficient: exp of the trigonometric interpolant of the log of the stored 128^2 coefficient.
  * Darcy forcing: the generator's analytic forcing minus the same 128-grid mean.
  The solver is the benchmark's own ``solve_elliptic`` (FD2 flux operator, CG), at rtol 1e-11.
* Periodic families (Burgers, Navier-Stokes). Cell centres x_j = (j + 0.5)/N on [0, 1).
  * The stored 128^2 initial state is extended by exact trigonometric interpolation to n_ref^2.
  * It is integrated with the benchmark's own integrators: the NS internal step is divided by r_t; the Burgers CFL
    factor 0.35 is divided by r_t.
  * Every state is evaluated at the target grid by the phase-correct trigonometric interpolant (no stride sampling
    of cell-centred grids).
  * All 15 stored frames are produced at the paper's physical times t_k = k (T - 1)/14 * dt_dense.

Labels: ``native`` is the paper route (m = 1 or n_ref = 128, r_t = 1); ``reference`` is a finer route fixed by the
pilot gate. The input fields (source / coefficient / initial state) are the stored float32 128^2 fields, interpreted
as samples of one band-limited continuous field, so every grid sees the same physical problem and normalisation.
"""
from __future__ import annotations

import inspect
import math
import textwrap
import time

import numpy as np

from . import V2_REFERENCE_CONTRACT
from .v2spec import DENSE_T, FINAL_T, STORED_FRAMES

STATIC_A = 1.0 / 256.0
STATIC_B = 255.0 / 256.0
STATIC_RTOL = 1.0e-11
STATIC_MAXITER = 400000


# --------------------------------------------------------------------------------------------------------- helpers
def trig_eval(values: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Exact trigonometric interpolant of n x n samples at cell centres (i + 0.5)/n, evaluated on ys x xs.
    (Identical to supp.numerics.trig_eval of the executed v1 H8 route.)"""
    n = values.shape[0]
    c = np.fft.fft2(np.asarray(values, dtype=np.float64)) / (n * n)
    k = np.fft.fftfreq(n) * n
    x0 = 0.5 / n
    ex = np.exp(2j * np.pi * np.outer(np.asarray(xs) - x0, k))
    ey = np.exp(2j * np.pi * np.outer(np.asarray(ys) - x0, k))
    return np.real(ey @ c @ ex.T)


def rel(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def order(e1: float, e2: float):
    if e1 <= 1e-12 and e2 <= 1e-12:
        return None
    return float(math.log2(e1 / e2)) if e1 > 0 and e2 > 0 else None


def static_coords(n_points: int, m: int = 1, n_base: int = 128):
    """m-fold refinement of the N-point boundary-inclusive static grid on [0.5/n_base, 1 - 0.5/n_base] (the outer cell
    centres of the n_base grid; [1/256, 255/256] for the paper's 128 grid): coordinates and spacing."""
    a, b = 0.5 / n_base, 1.0 - 0.5 / n_base
    intervals = m * (int(n_points) - 1)
    h = (b - a) / intervals
    return a + np.arange(intervals + 1) * h, h


def periodic_centres(n: int) -> np.ndarray:
    return (np.arange(int(n)) + 0.5) / int(n)


# ------------------------------------------------------------------------------------------------------- static
def _darcy_forcing(xx, yy):
    f = np.sin(2.0 * np.pi * xx) * np.sin(2.0 * np.pi * yy)
    return f + 0.2 * np.sin(4.0 * np.pi * xx + 0.3) * np.sin(2.0 * np.pi * yy)


def _darcy_forcing_mean(n_base: int) -> float:
    x = periodic_centres(n_base)
    xx, yy = np.meshgrid(x, x)
    return float(np.mean(_darcy_forcing(xx, yy)))


def static_solve(pde: str, condition128: np.ndarray, n_points: int, m: int) -> tuple[np.ndarray, dict]:
    """Solve on the m-fold refinement of the N-point grid; return the values at the N stored points.
    ``condition128`` is the stored base-grid input (128^2 in production)."""
    from ..pdes.numerics import solve_elliptic
    cond = np.asarray(condition128, dtype=np.float64)
    n_base = cond.shape[0]
    x, h = static_coords(n_points, m, n_base)
    t0 = time.process_time()
    if pde == "poisson":
        src = trig_eval(cond, x, x)
        sol, info = solve_elliptic(src, "dirichlet", h, h, rtol=STATIC_RTOL, maxiter=STATIC_MAXITER)
    elif pde == "darcy":
        coef = np.exp(trig_eval(np.log(cond), x, x))
        xx, yy = np.meshgrid(x, x)
        forcing = _darcy_forcing(xx, yy) - _darcy_forcing_mean(n_base)
        sol, info = solve_elliptic(forcing, "dirichlet", h, h, coefficient=coef, rtol=STATIC_RTOL, maxiter=STATIC_MAXITER)
    else:
        raise ValueError(f"static reference route not defined for {pde!r}")
    out = sol[::m, ::m]
    assert out.shape == (n_points, n_points)
    return out, {"m": m, "points": len(x), "h": h, "iterations": int(info.iterations),
                 "relative_residual": float(info.relative_residual), "cpu_s": round(time.process_time() - t0, 3)}


# ------------------------------------------------------------------------------------------------------ periodic
def _burgers_step_with_cfl(cfl: float):
    """advance_burgers with its CFL factor 0.35 replaced by ``cfl`` (same algorithm, smaller internal step).
    (The executed v1 H8 route, supp.numerics.burgers_step_with_cfl.)"""
    from ..pdes import numerics as N
    src = textwrap.dedent(inspect.getsource(N.advance_burgers))
    assert src.count("stable_dt = 0.35 /") == 1, "advance_burgers changed; refusing to patch its CFL factor"
    src = src.replace("stable_dt = 0.35 /", "stable_dt = %r /" % float(cfl)).replace("def advance_burgers(", "def _advance_burgers_cfl(")
    ns = dict(N.__dict__)
    exec(compile(src, "<advance_burgers_cfl>", "exec"), ns)  # noqa: S102 - source of this package's own kernel
    return ns["_advance_burgers_cfl"]


def frame_schedule(pde: str):
    """(dense steps between stored frames m, dense frame dt) of the paper corpus."""
    T = DENSE_T[pde]
    m = (T - 1) // (STORED_FRAMES - 1)
    assert m * (STORED_FRAMES - 1) == T - 1
    return m, FINAL_T[pde] / (T - 1)


def periodic_run(pde: str, regime: str, initial128: np.ndarray, n_ref: int, r_t: int, last_frame: int = STORED_FRAMES - 1):
    """Integrate the stored initial state on an n_ref^2 grid; return the states at stored frames 0..last_frame and
    diagnostics. The initial state is the trigonometric continuation of the 128^2 stored field."""
    from ..pdes import numerics as N
    from ..pdes import burgers as B
    from ..pdes import navier_stokes as NS
    init = np.asarray(initial128, dtype=np.float64)
    n = int(n_ref)
    c = periodic_centres(n)
    state = init.copy() if n == init.shape[0] else trig_eval(init, c, c)
    dx = 1.0 / n
    m, frame_dt = frame_schedule(pde)
    t0 = time.process_time()
    frames = [state]
    substeps = 0
    if pde == "navier_stokes":
        xx, yy = np.meshgrid(c, c)
        forcing = 0.1 * (np.sin(2.0 * np.pi * (xx + yy)) + np.cos(2.0 * np.pi * (xx + yy)))
        nu = NS.VISCOSITY_BY_REGIME[regime]
        for k in range(1, last_frame + 1):
            for _ in range(m):
                state, s = N.advance_periodic_vorticity(state, forcing, nu, frame_dt, dx, dx,
                                                        internal_dt=NS.PERIODIC_INTERNAL_TIME_STEP / r_t)
                substeps += int(s)
            frames.append(state)
    elif pde == "burgers":
        step = N.advance_burgers if r_t == 1 else _burgers_step_with_cfl(0.35 / r_t)
        nu = B.VISCOSITY_BY_REGIME[regime]
        for k in range(1, last_frame + 1):
            for _ in range(m):
                state, d = step(state, nu, frame_dt, dx, dx, "periodic", advection_scheme="spectral")
                substeps += int(d["substeps"])
            frames.append(state)
    else:
        raise ValueError(f"periodic reference route not defined for {pde!r}")
    for f in frames:
        if not np.all(np.isfinite(f)):
            raise RuntimeError(f"non-finite state in the {pde} reference run (n={n}, r_t={r_t})")
    return frames, {"n_ref": n, "r_t": int(r_t), "substeps": substeps, "cpu_s": round(time.process_time() - t0, 3)}


def sample_periodic(frames, n_out: int, initial128: np.ndarray | None = None) -> np.ndarray:
    """States evaluated at the n_out cell centres (phase-correct trigonometric interpolation). Frame 0 at 128 is the
    stored initial state itself when it is supplied."""
    c = periodic_centres(n_out)
    out = []
    for k, f in enumerate(frames):
        f = np.asarray(f, dtype=np.float64)
        if k == 0 and initial128 is not None and n_out == initial128.shape[0]:
            out.append(np.asarray(initial128, dtype=np.float64))
        elif f.shape[0] == n_out:
            out.append(f)
        else:
            out.append(trig_eval(f, c, c))
    return np.stack(out)


def contract() -> dict:
    return {"contract": V2_REFERENCE_CONTRACT, "static_interval": [STATIC_A, STATIC_B], "static_rtol": STATIC_RTOL,
            "static_solver": "pdeobs.pdes.numerics.solve_elliptic (fd2 flux operator, Jacobi-preconditioned CG)",
            "periodic_solvers": {"navier_stokes": "advance_periodic_vorticity, internal_dt = 1e-4 / r_t",
                                 "burgers": "advance_burgers (spectral), CFL factor 0.35 / r_t"},
            "sampling": {"static": "every m-th point of the m(N-1)+1 grid", "periodic": "trigonometric interpolation at (j+0.5)/N"},
            "inputs": "stored float32 128^2 fields as samples of one band-limited continuous field"}


__all__ = ["STATIC_A", "STATIC_B", "contract", "frame_schedule", "order", "periodic_centres", "periodic_run", "rel",
           "sample_periodic", "static_coords", "static_solve", "trig_eval"]
