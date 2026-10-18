"""V2 observation protocols (plan section 6; resolution addon 4.1). Contract ``pdeobs.rev8k.v2.masks/1``.

Eligible cells: the unknown grid points. Static Dirichlet families exclude the known boundary ring ((N-2)^2 cells);
periodic families use the whole grid (N^2). Every protocol records K, K/N^2 and K/N_eligible.

Seeds: ``v2seed("mask_order", family, pde_key, parent_id, rep, N)``. One ordering belongs to a parent. It does not
depend on the count K, the correlation length, the model or the training seed (the handoff reference rule). So the three
ell records of a parent see the same layout, and K-nested protocols are prefixes of one ordering.

* U<K>   uniform permutation of the eligible cells; the first K.
* C<K>   four Gaussian clusters. Centres are uniform over the eligible index box, sigma = 0.08 N index units per axis
         (0.08 of the domain length on every grid), non-wrapped Euclidean distance, equal weights. Sampling without
         replacement in proportion to the mixture density via Gumbel top-k (ties by flat index).
         Algorithm of the handoff reference ``ordering(..., 'C')``.
* L<K>   K/64 distinct rows, 64 distinct columns in each row (rows from the eligible rows, columns drawn per row from
         the eligible columns). One row permutation and per-row column permutations, so L256 is contained in L1024
         (an explicitly defined nesting, plan 6).
* G256   16 x 16 centre-of-bin grid over the eligible box (evaluation only).
* MXR3_K256        U256 / C256 / L256, one component per record and epoch, (j + epoch) mod 3.
* MXR6_K256_K1024  U/C/L x 256/1024, (j + epoch) mod 6.
* MXR2_K256        U256 / C256, (j + epoch) mod 2: the RPB 64^2 two-layout mixture (L256 does not exist there; not MXR3).
  j = rank of the record in its regime by ``v2seed("mixture_order", pde, parent)`` + ``v2seed("mixture_offset", pde,
  regime) mod q``. Epoch 0 uses the record's rep-0 layout. Epoch e >= 1 redraws the positions with
  rep = ``v2seed("mixture_positions", pde, parent, component, e)`` (positions refreshed every epoch; never a union).
Specialists (U256, C256, L256, U1024, ...) train on the record's fixed rep-0 layout. Tests use rep 0 for every model;
mechanism studies add rep 1 and rep 2.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from .contracts import v2seed
from .v2spec import PDE_KEY, STATIC, canonical_pde

MASK_CONTRACT = "pdeobs.rev8k.v2.masks/1"
CLUSTERS = 4
CLUSTER_SIGMA = 0.08
LINE_COLUMNS = 64
MIXTURES = {"MXR3_K256": ("U256", "C256", "L256"),
            "MXR2_K256": ("U256", "C256"),          # release 31 (D-0076): two-layout mixture for 64^2 Dirichlet grids (no L256)
            "MXR6_K256_K1024": ("U256", "C256", "L256", "U1024", "C1024", "L1024")}
NESTED_K = (64, 128, 256, 512, 1024, 4096)


def eligible(pde: str, n: int = 128) -> np.ndarray:
    pde = canonical_pde(pde)
    a = np.ones((n, n), dtype=bool)
    if pde in STATIC:
        a[[0, -1], :] = False
        a[:, [0, -1]] = False
    return a


def _rng(*parts) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(v2seed(*parts)))


def ordering(family: str, pde: str, parent: str, rep: int = 0, n: int = 128) -> np.ndarray:
    """Full ordering of the eligible flat indices for family U or C."""
    pde = canonical_pde(pde)
    el = eligible(pde, n)
    idx = np.flatnonzero(el)
    rng = _rng("mask_order", family, PDE_KEY[pde], str(parent), int(rep), int(n))
    if family == "U":
        return rng.permutation(idx)
    if family != "C":
        raise ValueError(f"ordering family must be U or C, not {family!r}")
    coords = np.stack(np.unravel_index(idx, (n, n)), axis=1).astype(np.float64)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    centres = rng.uniform(lo, hi, size=(CLUSTERS, 2))
    scale = CLUSTER_SIGMA * np.array([n, n], dtype=np.float64)
    ex = -0.5 * np.sum(((coords[:, None, :] - centres[None, :, :]) / scale) ** 2, axis=-1)
    m = ex.max(axis=1)
    logw = m + np.log(np.exp(ex - m[:, None]).mean(axis=1))
    u = np.clip(rng.random(len(idx)), np.finfo(float).tiny, 1 - np.finfo(float).eps)
    key = logw - np.log(-np.log(u))
    return idx[np.lexsort((idx, -key))]


def lines_mask(k: int, pde: str, parent: str, rep: int = 0, n: int = 128) -> np.ndarray:
    pde = canonical_pde(pde)
    if k % LINE_COLUMNS:
        raise ValueError("line protocols need K divisible by 64")
    rows_needed = k // LINE_COLUMNS
    el = eligible(pde, n)
    rows = np.flatnonzero(el.any(axis=1))
    cols = np.flatnonzero(el.any(axis=0))
    if rows_needed > len(rows) or LINE_COLUMNS > len(cols):
        raise ValueError(f"L{k} does not fit a {n}x{n} grid with {len(rows)} eligible rows / {len(cols)} columns")
    rng = _rng("mask_order", "L", PDE_KEY[pde], str(parent), int(rep), int(n))
    row_order = rng.permutation(rows)
    mask = np.zeros((n, n), dtype=bool)
    for r in row_order[:rows_needed].tolist():
        rr = _rng("mask_order", "L-row", PDE_KEY[pde], str(parent), int(rep), int(n), int(r))
        mask[r, rr.permutation(cols)[:LINE_COLUMNS]] = True
    return mask


def grid_mask(k: int, pde: str, n: int = 128) -> np.ndarray:
    side = int(round(k ** 0.5))
    if side * side != k:
        raise ValueError("G protocols need a square K")
    el = eligible(pde, n)
    rows = np.flatnonzero(el.any(axis=1))
    lo, length = int(rows[0]), int(len(rows))
    ind = lo + np.floor((np.arange(side) + 0.5) * length / side).astype(int)
    mask = np.zeros((n, n), dtype=bool)
    mask[np.ix_(ind, ind)] = True
    return mask


def grid_bins(k: int, pde: str, n: int = 128):
    """G<K> geometry: (row/column indices of the grid points, [(lo, hi)] inclusive bin of each index). The bins split the
    eligible rows (= columns) into sqrt(K) equal parts, as the point placement of :func:`grid_mask` does."""
    side = int(round(k ** 0.5))
    if side * side != k:
        raise ValueError("G protocols need a square K")
    el = eligible(pde, n)
    rows = np.flatnonzero(el.any(axis=1))
    lo, length = int(rows[0]), int(len(rows))
    ind = lo + np.floor((np.arange(side) + 0.5) * length / side).astype(int)
    edges = lo + np.floor(np.arange(side + 1) * length / side).astype(int)
    return ind, [(int(edges[i]), int(edges[i + 1]) - 1) for i in range(side)]


# N3 (D-0062): fixed-count perturbations of the G256 grid. GT<d><a>_K<K>: every point shifted by d cells along axis a
# (x = column, y = row); refused (never clipped or wrapped) when a shifted point leaves the eligible cells.
# GJ<r>_K<K>_r<q>: every point jittered inside its own grid bin, offsets uniform on {-r..r}^2 intersected with the bin and
# the eligible cells, one point per bin; realization q is one fixed layout shared by every parent (evaluation).
# GJ<r>_K<K> (no realization): training augmentation, a fresh jitter per (parent, epoch) with the same support.
N3_PROTO = re.compile(r"^G(T)(\d)([xy])_K(\d+)$|^G(J)(\d)_K(\d+)(?:_r(\d+))?$")


def n3_mask(protocol: str, pde: str, parent: str = "", epoch: int = 0, n: int = 128):
    """(mask, info) of an N3 protocol; info records the displacement of every point (cells) for the report."""
    m = N3_PROTO.match(protocol)
    if not m:
        raise ValueError(f"not an N3 grid-perturbation protocol: {protocol!r}")
    el = eligible(pde, n)
    if m.group(1) == "T":
        d, axis, k = int(m.group(2)), m.group(3), int(m.group(4))
        ind, _ = grid_bins(k, pde, n)
        rr, cc = np.meshgrid(ind, ind, indexing="ij")
        if axis == "y":
            rr = rr + d
        else:
            cc = cc + d
        if rr.min() < 0 or cc.min() < 0 or rr.max() >= n or cc.max() >= n or not el[rr, cc].all():
            raise ValueError(f"{protocol}: a shifted grid point leaves the eligible cells (infeasible, not clipped)")
        mask = np.zeros((n, n), dtype=bool)
        mask[rr, cc] = True
        return mask, {"kind": "shift", "d": d, "axis": axis, "max_disp": float(d), "mean_disp": float(d)}
    r, k, q = int(m.group(6)), int(m.group(7)), m.group(8)
    ind, bins = grid_bins(k, pde, n)
    if q is not None:
        rng = _rng("n3_grid_jitter_eval", PDE_KEY[canonical_pde(pde)], int(k), int(r), int(q), int(n))
    else:
        rng = _rng("n3_grid_jitter_train", PDE_KEY[canonical_pde(pde)], str(parent), int(k), int(r), int(epoch), int(n))
    mask = np.zeros((n, n), dtype=bool)
    disp = []
    for i, ci in enumerate(ind):
        for j, cj in enumerate(ind):
            (rlo, rhi), (clo, chi) = bins[i], bins[j]
            cand = [(a, b) for a in range(max(rlo, ci - r), min(rhi, ci + r) + 1)
                    for b in range(max(clo, cj - r), min(chi, cj + r) + 1) if el[a, b]]
            a, b = cand[int(rng.integers(len(cand)))]
            mask[a, b] = True
            disp.append(((a - ci) ** 2 + (b - cj) ** 2) ** 0.5)
    disp = np.asarray(disp)
    return mask, {"kind": "jitter", "r": r, "realization": None if q is None else int(q),
                  "max_disp": float(disp.max()), "mean_disp": float(disp.mean()),
                  "disp_hist": np.bincount(np.rint(disp).astype(int)).tolist()}


_PROTO = re.compile(r"^([UCLG])(\d+)$")


def protocol_mask(protocol: str, pde: str, parent: str, *, rep: int = 0, n: int = 128) -> np.ndarray:
    """Fixed layout of a single-component protocol (U<K>, C<K>, L<K>, G<K>) for one parent and replicate."""
    m = _PROTO.match(protocol)
    if not m:
        raise ValueError(f"not a single-layout protocol: {protocol!r}")
    fam, k = m.group(1), int(m.group(2))
    if fam in "UC":
        order = ordering(fam, pde, parent, rep, n)
        if k > len(order):
            raise ValueError(f"{protocol} exceeds the {len(order)} eligible cells")
        mask = np.zeros(n * n, dtype=bool)
        mask[order[:k]] = True
        return mask.reshape(n, n)
    if fam == "L":
        return lines_mask(k, pde, parent, rep, n)
    return grid_mask(k, pde, n)


def mixture_positions(parents_by_regime: dict, pde: str, q: int) -> dict:
    """{regime: [parent ids]} -> {parent id: j}."""
    key = PDE_KEY[canonical_pde(pde)]
    out = {}
    for regime, parents in sorted(parents_by_regime.items()):
        order = sorted((str(p) for p in parents), key=lambda p: (v2seed("mixture_order", key, p), p))
        offset = v2seed("mixture_offset", key, str(regime)) % q
        for rank, p in enumerate(order):
            out[p] = rank + offset
    return out


def mixture_mask(protocol: str, pde: str, parent: str, j: int, epoch: int, *, n: int = 128):
    comps = MIXTURES[protocol]
    comp = comps[(int(j) + int(epoch)) % len(comps)]
    if int(epoch) == 0:
        rep = 0
    else:
        rep = v2seed("mixture_positions", PDE_KEY[canonical_pde(pde)], str(parent), comp, int(epoch)) % (2 ** 31)
    return protocol_mask(comp, pde, parent, rep=rep, n=n), {"component": comp, "rep": int(rep), "epoch": int(epoch), "j": int(j)}


def describe(protocol: str, pde: str, n: int = 128) -> dict:
    el = int(eligible(pde, n).sum())
    m = _PROTO.match(protocol)
    k = int(m.group(2)) if m else None
    return {"protocol": protocol, "K": k, "N": n, "eligible": el, "K_over_N2": (k / (n * n)) if k else None,
            "K_over_eligible": (k / el) if k else None, "contract": MASK_CONTRACT}


def mask_digest(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(mask, dtype=bool).ravel()).tobytes()).hexdigest()


__all__ = ["CLUSTERS", "CLUSTER_SIGMA", "LINE_COLUMNS", "MASK_CONTRACT", "MIXTURES", "N3_PROTO", "NESTED_K", "describe", "eligible", "grid_bins", "n3_mask",
           "grid_mask", "lines_mask", "mask_digest", "mixture_mask", "mixture_positions", "ordering", "protocol_mask"]
