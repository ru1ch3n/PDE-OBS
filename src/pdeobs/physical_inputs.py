"""Small public adapters for complete custom physical problems.

These functions do not reconstruct unknown coefficients or initial conditions.
They validate user arrays and delegate to the existing numerical kernels.
Grid spacings and boundary conventions are explicit, never inferred from masks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .pdes.numerics import crank_nicolson_diffusion, solve_elliptic


def _field(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 3 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 2D field with both dimensions at least 3")
    return array


def _spacing(dx: float, dy: float) -> None:
    if not np.isfinite([dx, dy]).all() or min(dx, dy) <= 0:
        raise ValueError("dx and dy must be finite and positive")


def solve_custom_elliptic(source: Any, *, boundary: str, dx: float, dy: float,
                          coefficient: Any = None, reaction: float = 0.0):
    """Solve -div(a grad u) + reaction*u = source using the existing kernel."""
    rhs = _field(source, "source")
    coeff = None if coefficient is None else _field(coefficient, "coefficient")
    _spacing(dx, dy)
    if not np.isfinite(reaction):
        raise ValueError("reaction must be finite")
    return solve_elliptic(rhs, boundary, dx, dy, coefficient=coeff, reaction=reaction)


def advance_custom_heat(initial_state: Any, *, diffusivity: float, dt: float,
                        boundary: str, dx: float, dy: float):
    """Advance a fully specified initial field; no partial-input imputation."""
    state = _field(initial_state, "initial_state")
    _spacing(dx, dy)
    if not np.isfinite([diffusivity, dt]).all() or min(diffusivity, dt) < 0:
        raise ValueError("diffusivity and dt must be finite and nonnegative")
    return crank_nicolson_diffusion(state, diffusivity, dt, dx, dy, boundary)
