"""Nearest-observation (Voronoi) fill of a partially observed field: the executed v1 H7 input rule (pre-locked).
Contract ``pdeobs.rev8k.v1.fill/1``.

Every hidden cell takes the value of its nearest observed cell in grid-index distance; ties go to the observed cell with
the smallest flat index (row-major). Stationary (Dirichlet) data use the plain Euclidean distance, periodic data the
torus distance. In the v1 campaign only the zero-filled observation channel of the first input state was replaced; the
mask and geometry channels stay unchanged.

Exact and fast: an exact Euclidean distance transform gives each hidden cell's nearest squared distance d2; the winner is
then the smallest-flat-index observed cell among the integer offsets with dy^2 + dx^2 == d2. :func:`brute_force_index`
is the O(N_hidden x N_observed) reference used by the tests. Outputs are bit-identical to the campaign module
(``rev8k.fill``).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

FILL_CONTRACT = "pdeobs.rev8k.v1.fill/1"


def nearest_index(mask: np.ndarray, periodic: bool) -> np.ndarray:
    """mask (H, W) bool -> flat index of the chosen observed cell for every cell (observed cells map to themselves)."""
    from scipy.ndimage import distance_transform_edt
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    if not mask.any():
        raise ValueError("no observed cell")
    if periodic:
        tiled = np.tile(mask, (3, 3))
        dist = distance_transform_edt(~tiled)[h:2 * h, w:2 * w]
    else:
        dist = distance_transform_edt(~mask)
    d2 = np.rint(dist ** 2).astype(np.int64)
    flat = np.arange(h * w, dtype=np.int64).reshape(h, w)
    out = np.where(mask, flat, -1)
    ys, xs = np.nonzero(~mask)
    if ys.size == 0:
        return out
    cd2 = d2[ys, xs]
    table_d2, table_dy, table_dx = _offset_table(int(cd2.max()))
    start = np.searchsorted(table_d2, cd2, side="left")
    count = np.searchsorted(table_d2, cd2, side="right") - start
    cell = np.repeat(np.arange(ys.size), count)                          # every (hidden cell, offset at its distance)
    first = np.repeat(np.cumsum(count) - count, count)
    k = start[cell] + np.arange(cell.size) - first
    yy, xx = ys[cell] + table_dy[k], xs[cell] + table_dx[k]
    if periodic:
        yy, xx = yy % h, xx % w
        ok = np.ones(cell.size, dtype=bool)
    else:
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        yy, xx = np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)
    ok &= mask[yy, xx]
    cand = np.where(ok, yy * w + xx, h * w)
    best = np.minimum.reduceat(cand, np.cumsum(count) - count)
    if (best >= h * w).any():
        raise AssertionError("distance transform and offset search disagree")
    out[ys, xs] = best
    return out


@lru_cache(maxsize=8)
def _offset_table(max_d2: int):
    """Integer offsets with dy^2 + dx^2 <= max_d2, sorted by squared distance."""
    r = math.isqrt(max_d2)
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    d2 = (dy * dy + dx * dx).ravel()
    keep = d2 <= max_d2
    order = np.argsort(d2[keep], kind="stable")
    return d2[keep][order], dy.ravel()[keep][order], dx.ravel()[keep][order]


def nearest_fill(values: np.ndarray, mask: np.ndarray, periodic: bool) -> np.ndarray:
    """values (H, W[, C]) with any content at hidden cells -> filled copy using only observed values."""
    idx = nearest_index(mask, periodic)
    v = np.asarray(values)
    flat = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
    return flat[idx.ravel()].reshape(v.shape)


def nearest_fill_records(values: np.ndarray, masks: np.ndarray, periodic: bool) -> np.ndarray:
    """Per-record fill: values (N, H, W[, C]) with masks (N, H, W); record i is :func:`nearest_fill` of its own mask."""
    v = np.asarray(values)
    m = np.asarray(masks, dtype=bool)
    if m.ndim != 3 or v.shape[:3] != m.shape:
        raise ValueError(f"per-record fill needs masks (N, H, W) aligned with values (N, H, W[, C]); got {m.shape} / {v.shape}")
    return np.stack([nearest_fill(v[i], m[i], periodic) for i in range(m.shape[0])]) if m.shape[0] else v.copy()


def brute_force_index(mask: np.ndarray, periodic: bool) -> np.ndarray:
    """Reference implementation for the tests (O(N_hidden x N_observed))."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    oy, ox = np.nonzero(mask)
    oflat = oy * w + ox
    out = np.where(mask, np.arange(h * w).reshape(h, w), -1)
    for y, x in zip(*np.nonzero(~mask)):
        dy, dx = np.abs(oy - y), np.abs(ox - x)
        if periodic:
            dy, dx = np.minimum(dy, h - dy), np.minimum(dx, w - dx)
        d2 = dy * dy + dx * dx
        m = d2.min()
        out[y, x] = oflat[d2 == m].min()
    return out


__all__ = ["FILL_CONTRACT", "brute_force_index", "nearest_fill", "nearest_fill_records", "nearest_index"]
