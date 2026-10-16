"""seed64 and the executed rev8k v1 contracts (decision D-0001), reproduced bit for bit.

seed64(domain, *parts) = first 8 bytes (big-endian) of SHA-256 over the canonical JSON array [domain, *parts] with
separators "," and ":" and ASCII escapes (ints stay ints, so 9101 and "9101" differ).

This is NOT the function of the v1 handoff reference file (which prepends "rev8k_reference_v1" to the array); the
difference table is in the campaign audit (v1_contract_difference_table.md). Executed results keep this function.
V2 contracts call it with a versioned domain tag through :func:`v2seed`.

The executed v1 rules built on seed64 (contract ``rev8k-contracts-D0001-v1``) follow below, unchanged in behaviour:
the development split (:func:`dev_partition`), the mixture rank and component (:func:`mixture_positions`,
:func:`mixture_component`), the MXR3 epoch seed (:func:`mxr3_seed`) and the mixture receipt (:func:`mixture_receipt`).
PDE arguments are family names (``navier_stokes`` ...), mapped to the plan's logical keys by :data:`LOGICAL`; record
arguments are the benchmark's canonical sample ids.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict

from . import V1_CONTRACT

SEED64_DEFINITION = "sha256(json.dumps([domain,*parts],separators=(',',':'),ensure_ascii=True))[:8] big-endian"
V2_TAG = "rev8k-v2"


def seed64(domain: str, *parts) -> int:
    text = json.dumps([domain, *parts], separators=(",", ":"), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(text.encode("ascii")).digest()[:8], "big")


def v2seed(name: str, *parts) -> int:
    """seed64 under the V2 domain tag: seed64("rev8k-v2/<name>", *parts)."""
    return seed64(f"{V2_TAG}/{name}", *parts)


def ids_sha256(ids) -> str:
    """Digest of an id set (order-free): SHA-256 of the sorted ids joined by newlines."""
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids)).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ executed v1 rules (rev8k-contracts-D0001-v1)
CONTRACT_VERSION = V1_CONTRACT                      # "rev8k-contracts-D0001-v1"
LOGICAL = {"darcy": "darcy", "poisson": "poisson", "helmholtz": "helmholtz", "heat": "heat",
           "reaction_diffusion": "reaction_diffusion", "burgers": "burgers", "navier_stokes": "ns"}
FAMILY = {v: k for k, v in LOGICAL.items()}

DEV_SEED = 9101
TRAIN_SEEDS = (1101, 1102, 1103)
LR_CANDIDATES = (3e-4, 1e-3)
DEV_VAL_PER_REGIME = 100
DEV_TRAIN_PER_REGIME = 500
DEV_VIEWS = ("R50", "H", "CL")
MIX_COMPONENTS = {"MXF3": ("R50", "H", "CL"), "MXC3": ("R50", "H", "CL"), "MXR3": ("R50", "H", "CL"), "MXC2": ("U256", "C256")}


def dev_partition(records, family: str, *, any_count: bool | None = None):
    """records: (dataset index, sample id, regime) of the 1,800 original training records of one PDE.
    Per regime, order by seed64("dev_split", 9101, logical pde, sample id) (ties by id); the first 100 are the
    development validation records, the remaining 500 the development training records.
    Returns (dev_train indices, dev_val indices, receipt), each index list in the input order.

    ``any_count`` (default: the executed switch, environment REV8K_TEST_ANY_COUNT == "1") accepts partial data: then
    the first max(1, min(100, n // 6)) records of a regime are dev-val and the receipt says ``partial_test_mode``."""
    pde = LOGICAL[family]
    by_regime = defaultdict(list)
    for idx, sid, regime in records:
        by_regime[str(regime)].append((seed64("dev_split", DEV_SEED, pde, str(sid)), str(sid), int(idx)))
    if any_count is None:
        any_count = os.environ.get("REV8K_TEST_ANY_COUNT") == "1"      # local tests on partial data only
    any_count = bool(any_count)
    val, counts = set(), {}
    for regime, rows in sorted(by_regime.items()):
        rows.sort()
        if not any_count:
            assert len(rows) == DEV_VAL_PER_REGIME + DEV_TRAIN_PER_REGIME, (regime, len(rows))
        k = min(DEV_VAL_PER_REGIME, len(rows) // 6 if any_count else DEV_VAL_PER_REGIME) or 1
        val.update(i for _, _, i in rows[:k])
        counts[regime] = {"dev_val": k, "dev_train": len(rows) - k}
    order = [int(idx) for idx, _, _ in records]
    dev_train = [i for i in order if i not in val]
    dev_val = [i for i in order if i in val]
    sid = {int(idx): str(s) for idx, s, _ in records}
    receipt = {"contract": CONTRACT_VERSION, "seed64": SEED64_DEFINITION, "domain": "dev_split", "dev_seed": DEV_SEED,
               "pde_key": pde, "per_regime": counts, "dev_train_records": len(dev_train), "dev_val_records": len(dev_val),
               "dev_train_ids_sha256": ids_sha256(sid[i] for i in dev_train),
               "dev_val_ids_sha256": ids_sha256(sid[i] for i in dev_val),
               "partial_test_mode": any_count}
    return dev_train, dev_val, receipt


def mixture_positions(records, family: str, q: int) -> dict:
    """Plan 4.3: records = (record id, regime) of the CURRENT training list (parent id for paired new data, the original
    record id for the old data).  Within a regime, rank by seed64("mixture_order", pde, id) (ties by id) from 0, then add
    seed64("mixture_offset", pde, regime) % q.  Independent of model, training seed and ell.
    (The V2 mixtures use :func:`pdeobs.revision.v2masks.mixture_positions` under the rev8k-v2 seed tag.)"""
    pde = LOGICAL[family]
    by = defaultdict(list)
    for rid, regime in records:
        by[str(regime)].append(str(rid))
    out = {}
    for regime, ids in sorted(by.items()):
        ids.sort(key=lambda r: (seed64("mixture_order", pde, r), r))
        offset = seed64("mixture_offset", pde, regime) % q
        for rank, rid in enumerate(ids):
            out[rid] = rank + offset
    return out


def mixture_component(protocol: str, j: int, epoch: int) -> str:
    """MXF3: j mod 3 (fixed); MXC3 / MXR3 / MXC2: (j + epoch) mod q (one component per record and epoch, never a union)."""
    comps = MIX_COMPONENTS[protocol]
    return comps[int(j) % len(comps)] if protocol == "MXF3" else comps[(int(j) + int(epoch)) % len(comps)]


def mxr3_seed(family: str, record_id: str, view: str, epoch: int) -> int:
    """MXR3, epoch >= 1: positions re-drawn with the same factory and parameters from an independent epoch seed."""
    return seed64("mxr3_positions", LOGICAL[family], str(record_id), str(view), int(epoch))


def mixture_receipt(positions: dict, protocol: str) -> dict:
    """Receipt of a mixture assignment {record id: j}: epoch-0 component counts and a positions digest."""
    comps, contract = MIX_COMPONENTS.get(protocol), CONTRACT_VERSION
    if comps is None:                                     # V2 mixtures (pdeobs.revision.v2masks, rev8k-v2 seed tag)
        from .v2masks import MASK_CONTRACT, MIXTURES
        comps, contract = MIXTURES[protocol], MASK_CONTRACT
    q = len(comps)
    counts = defaultdict(int)
    for j in positions.values():
        counts[comps[int(j) % q]] += 1
    return {"protocol": protocol, "components": list(comps), "records": len(positions),
            "epoch0_component_counts": dict(counts),
            "positions_sha256": ids_sha256(f"{k}:{v}" for k, v in positions.items()), "contract": contract}


__all__ = ["CONTRACT_VERSION", "DEV_SEED", "DEV_TRAIN_PER_REGIME", "DEV_VAL_PER_REGIME", "DEV_VIEWS", "FAMILY", "LOGICAL",
           "LR_CANDIDATES", "MIX_COMPONENTS", "SEED64_DEFINITION", "TRAIN_SEEDS", "V2_TAG", "dev_partition", "ids_sha256",
           "mixture_component", "mixture_positions", "mixture_receipt", "mxr3_seed", "seed64", "v2seed"]
