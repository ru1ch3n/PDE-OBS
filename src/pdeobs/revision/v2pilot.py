"""V2 reference pilot (plan 5.3; resolution addon 3.4): space, time and joint refinement of fixed pilot problems.

Pilot problems: per slice and regime, the first k parents of the train + validation hash order
(:func:`pdeobs.revision.v2spec.pilot_parents`), never test. That is 12 slices x 3 regimes x 10 = 360 problems.

Per problem, all comparisons use the same physical problem at the coordinates of the stored grid:

* static (m = refinement of the 128-point boundary-inclusive grid):
  native vs m1 replay; e(m1, m2); e(m2, m4); e(m4, m8); observed orders. With --addon (rough Poisson), the 64- and
  256-point views at m = 1, 2, 4.
* periodic (n_ref, r_t), target frames 1-3 jointly:
  * native vs (128, 1) replay;
  * time: e((128, 1), (128, 2)) and e((128, 2), (128, 4));
  * space: e((128, 1), (256, 1)) and e((256, 1), (512, 1));
  * joint candidates: e((256, 1), (512, 2)), e((256, 2), (512, 4)), and on request e((512, 2), (1024, 4));
  * last stored frame: for the first ``last_frame_cases`` parents of a regime, the candidate pair again at frame 14.

The gate is registered before the pilot runs (plan 5.3): for the label route R and its finer comparator F, over a
PDE's pilot problems, 95th percentile <= 3e-4 and max <= 1e-3 of the discrete reference difference e(R, F). One route
is chosen per PDE for all three ell (no ell-dependent numerical method). A failing PDE is reported. It is never fixed
by dropping cases or changing seeds.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import REVISION_VERSION, V2_REFERENCE_CONTRACT
from . import reference as R
from .v2data import TARGET_FRAMES, _stored_native, native_output
from .v2spec import ADDON_ELL, ADDON_GRIDS, ELL_OF, STATIC, parent_id, pilot_parents

GATE = {"p95": 3.0e-4, "max": 1.0e-3}
STATIC_CANDIDATES = ({"kind": "static", "m": 2, "finer_m": 4}, {"kind": "static", "m": 4, "finer_m": 8})
PERIODIC_CANDIDATES = ({"kind": "periodic", "n_ref": 256, "r_t": 1, "finer": [512, 2]},
                       {"kind": "periodic", "n_ref": 256, "r_t": 2, "finer": [512, 4]},
                       {"kind": "periodic", "n_ref": 512, "r_t": 2, "finer": [1024, 4]})


def _t(frames):
    return np.stack([frames[k] for k in TARGET_FRAMES])


def static_problem(pde, regime, ordinal, ellcode, *, m_max=8, addon=False, grid=128) -> dict:
    setting = ELL_OF[ellcode][1]
    t0 = time.process_time()
    out = native_output(pde, regime, ordinal, setting, grid)
    native = _stored_native(pde, out)[-1].astype(np.float64)
    cond = np.asarray(out.condition, dtype=np.float64)[..., 0]
    row = {"parent_id": parent_id(pde, regime, ordinal), "pde": pde, "regime": regime, "ell": ellcode,
           "cpu_s": {"native": round(time.process_time() - t0, 3)}}
    sol = {}
    m = 1
    while m <= m_max:
        sol[m], info = R.static_solve(pde, cond, grid, m)
        row["cpu_s"][f"m{m}"] = info["cpu_s"]
        row.setdefault("cg_iterations", {})[f"m{m}"] = info["iterations"]
        m *= 2
    row["replay_native_m1"] = R.rel(sol[1], native)
    ms = sorted(sol)
    for a, b in zip(ms, ms[1:]):
        row[f"e_m{a}_m{b}"] = R.rel(sol[a], sol[b])
        row[f"native_vs_m{b}"] = R.rel(native, sol[b])
    for a, b, c in zip(ms, ms[1:], ms[2:]):
        row[f"order_m{a}_m{b}_m{c}"] = R.order(row[f"e_m{a}_m{b}"], row[f"e_m{b}_m{c}"])
    if addon and pde in ADDON_GRIDS and ellcode == ADDON_ELL and grid == 128:
        for n in ADDON_GRIDS[pde]:
            v = {mm: R.static_solve(pde, cond, n, mm) for mm in (1, 2, 4)}
            row[f"grid{n}"] = {"e_m1_m2": R.rel(v[1][0], v[2][0]), "e_m2_m4": R.rel(v[2][0], v[4][0]),
                               "cpu_s": {f"m{mm}": v[mm][1]["cpu_s"] for mm in v}}
    return row


def _pair_tag(a, b) -> str:
    return f"e_{int(a[0])}r{int(a[1])}_{int(b[0])}r{int(b[1])}"


def addon_problem(pde, regime, ordinal, ellcode, *, grid_ms=None, pairs=None, grid=128) -> dict:
    """Addon-view qualification of a route on the addon grids themselves (resolution addon 3.4).

    * static: for every addon grid n in ``grid_ms`` ({n: (m_a, m_b, ...)}), the n-point views of successive refinements
      and their differences e(m_a, m_b). Used where the pilot's addon views (m = 1, 2, 4) do not qualify m = 2 on a grid:
      then e(m4, m8) decides m = 4.
    * periodic: for every addon grid n (the keys of ``grid_ms``; default: all addon grids of the PDE) and every candidate
      pair ((n_ref, r_t), (n_ref', r_t')) in ``pairs`` (128-grid equivalents; default the first registered candidate),
      the joint difference over target frames 1-3 of the two runs sampled at the n-grid cell centres. The pilot measured
      the addon grids only for the (256, 2) and (512, 2) candidates."""
    if pde not in STATIC:
        return _periodic_addon_problem(pde, regime, ordinal, ellcode, grids=grid_ms, pairs=pairs, grid=grid)
    setting = ELL_OF[ellcode][1]
    out = native_output(pde, regime, ordinal, setting, grid)
    cond = np.asarray(out.condition, dtype=np.float64)[..., 0]
    row = {"parent_id": parent_id(pde, regime, ordinal), "pde": pde, "regime": regime, "ell": ellcode, "addon_check": True}
    for n, ms in sorted((grid_ms or {64: (4, 8)}).items(), key=lambda kv: int(kv[0])):
        v = {int(mm): R.static_solve(pde, cond, int(n), int(mm)) for mm in ms}
        mm = sorted(v)
        entry = {f"e_m{a}_m{b}": R.rel(v[a][0], v[b][0]) for a, b in zip(mm, mm[1:])}
        entry["cpu_s"] = {f"m{k}": v[k][1]["cpu_s"] for k in mm}
        row[f"grid{int(n)}"] = entry
    return row


def _periodic_addon_problem(pde, regime, ordinal, ellcode, *, grids=None, pairs=None, grid=128) -> dict:
    setting = ELL_OF[ellcode][1]
    t0 = time.process_time()
    out = native_output(pde, regime, ordinal, setting, grid)
    init = _stored_native(pde, out).astype(np.float64)[0]
    row = {"parent_id": parent_id(pde, regime, ordinal), "pde": pde, "regime": regime, "ell": ellcode, "addon_check": True,
           "cpu_s": {"native": round(time.process_time() - t0, 3)}}
    raw = {}

    def frames(n, rt):
        if (n, rt) not in raw:
            raw[(n, rt)], info = R.periodic_run(pde, regime, init, int(n) * grid // 128, int(rt), last_frame=3)
            row["cpu_s"][f"n{int(n)}_r{int(rt)}_f3"] = info["cpu_s"]
        return raw[(n, rt)]

    views = sorted(int(n) for n in (grids or ADDON_GRIDS[pde]))
    pairs = [tuple(map(tuple, pq)) for pq in (pairs or [((256, 1), (512, 2))])]
    for n in views:
        row[f"grid{n}"] = {_pair_tag(a, b): R.rel(_t(R.sample_periodic(frames(*a), n, None)),
                                                    _t(R.sample_periodic(frames(*b), n, None))) for a, b in pairs}
    return row


def run_addon_problem(p: dict, *, grid_ms=None, pairs=None, grid=128) -> dict:
    t0 = time.time()
    try:
        row = addon_problem(p["pde"], p["regime"], p["ordinal"], p["ell"], grid_ms=grid_ms, pairs=pairs, grid=grid)
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - a failed check is reported, never dropped
        row = {"parent_id": parent_id(p["pde"], p["regime"], p["ordinal"]), "pde": p["pde"], "regime": p["regime"],
               "ell": p["ell"], "addon_check": True, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    row["wall_s"] = round(time.time() - t0, 2)
    row["revision_version"] = REVISION_VERSION
    return row


def addon_routes(rows, pde) -> dict:
    """Per addon grid of ``pde``: the first registered candidate (static: m2 vs m4, then m4 vs m8; periodic: the order of
    PERIODIC_CANDIDATES) whose difference on that grid passes the gate on every rough pilot parent. ``rows`` = the pilot
    rows plus the addon_check rows (static: grid<N>{e_m..}; periodic: pilot keys grid<N>_e_.. or check rows
    grid<N>{e_..}; a parent counts once per metric). Every rough parent must have the metric.
    Candidates are taken strictly in order: a candidate without measurements ("UNMEASURED") or with missing / failed rows
    ("INCOMPLETE") stops the search, so a later candidate is never chosen in place of an earlier one not yet decided."""
    rough = [r for r in rows if r.get("pde") == pde and r.get("ell") == ADDON_ELL]
    parents = {r["parent_id"] for r in rough}
    failed = sorted({r["parent_id"] for r in rough if r.get("status") != "ok"})
    out = {"gate": GATE, "rough_parents": len(parents), "failed_rows": failed}
    static = pde in STATIC
    for n in ADDON_GRIDS[pde]:
        res = {"candidates": []}
        for c in (STATIC_CANDIDATES if static else PERIODIC_CANDIDATES):
            if static:
                key = f"e_m{c['m']}_m{c['finer_m']}"
                route = {"kind": "static", "m": c["m"]}
            else:
                key = _pair_tag((c["n_ref"], c["r_t"]), c["finer"])
                route = {"kind": "periodic", "n_ref": c["n_ref"], "r_t": c["r_t"]}
            vals = {}
            for r in rough:
                if r.get("status") != "ok":
                    continue
                g = r.get(f"grid{n}")
                v = g.get(key) if isinstance(g, dict) else None
                if v is None and not static:
                    v = r.get(f"grid{n}_{key}")
                if v is not None:
                    vals[r["parent_id"]] = float(v)
            entry = {"route": route, "metric": key, "n": len(vals)}
            res["candidates"].append(entry)
            if not vals:
                entry["passes"] = None
                res["status"] = "UNMEASURED"
                break
            a = np.array(list(vals.values()), dtype=np.float64)
            entry.update(p95=float(np.quantile(a, 0.95)), max=float(a.max()), median=float(np.median(a)))
            complete = len(vals) == len(parents) and not failed
            entry["passes"] = bool(entry["p95"] <= GATE["p95"] and entry["max"] <= GATE["max"] and complete)
            if entry["passes"]:
                res["status"] = "PASS"
                break
            if not complete:
                res["status"] = "INCOMPLETE"
                break
        else:
            res["status"] = "NOT_PASSED"
        res["chosen_route"] = res["candidates"][-1]["route"] if res["status"] == "PASS" else None
        out[str(n)] = res
    return out


def periodic_problem(pde, regime, ordinal, ellcode, *, finest=False, last_frame=False, addon=False, grid=128) -> dict:
    """Runs are named by their 128-grid equivalents (n128 = base, n256 = 2x, n512 = 4x, n1024 = 8x the base grid)."""
    setting = ELL_OF[ellcode][1]
    t0 = time.process_time()
    out = native_output(pde, regime, ordinal, setting, grid)
    native = _stored_native(pde, out).astype(np.float64)
    init = native[0]
    row = {"parent_id": parent_id(pde, regime, ordinal), "pde": pde, "regime": regime, "ell": ellcode,
           "cpu_s": {"native": round(time.process_time() - t0, 3)}}
    raw, runs = {}, {}

    def raw_run(n, rt, last=3):
        key = (n, rt, last)
        if key not in raw:
            raw[key], info = R.periodic_run(pde, regime, init, n * grid // 128, rt, last_frame=last)
            row["cpu_s"][f"n{n}_r{rt}_f{last}"] = info["cpu_s"]
        return raw[key]

    def run(n, rt, last=3, view=None):
        view = view or grid
        key = (n, rt, last, view)
        if key not in runs:
            runs[key] = R.sample_periodic(raw_run(n, rt, last), view, init if view == grid else None)
        return runs[key]

    row["replay_native_128_r1"] = R.rel(_t(run(128, 1)), _t(native))
    row["time_e_r1_r2"] = R.rel(_t(run(128, 1)), _t(run(128, 2)))
    row["time_e_r2_r4"] = R.rel(_t(run(128, 2)), _t(run(128, 4)))
    row["time_order"] = R.order(row["time_e_r1_r2"], row["time_e_r2_r4"])
    row["space_e_128_256"] = R.rel(_t(run(128, 1)), _t(run(256, 1)))
    row["space_e_256_512"] = R.rel(_t(run(256, 1)), _t(run(512, 1)))
    row["space_order"] = R.order(row["space_e_128_256"], row["space_e_256_512"])
    row["joint_e_256r1_512r2"] = R.rel(_t(run(256, 1)), _t(run(512, 2)))
    row["joint_e_256r2_512r4"] = R.rel(_t(run(256, 2)), _t(run(512, 4)))
    row["native_vs_512r4"] = R.rel(_t(native), _t(run(512, 4)))
    if finest or (addon and pde in ADDON_GRIDS and ellcode == ADDON_ELL and grid == 128):
        row["joint_e_512r2_1024r4"] = R.rel(_t(run(512, 2)), _t(run(1024, 4)))
    if addon and pde in ADDON_GRIDS and ellcode == ADDON_ELL and grid == 128:
        # the addon views must be qualified on their own grids: sample the same runs at 64 and 256 centres
        for n in ADDON_GRIDS[pde]:
            row[f"grid{n}_e_256r2_512r4"] = R.rel(_t(run(256, 2, view=n)), _t(run(512, 4, view=n)))
            row[f"grid{n}_e_512r2_1024r4"] = R.rel(_t(run(512, 2, view=n)), _t(run(1024, 4, view=n)))
    if last_frame:
        n_last = len(native) - 1
        a = run(256, 2, n_last)
        b = run(512, 4, n_last)
        row["last_e_256r2_512r4"] = R.rel(a[-1], b[-1])
        row["last_native_vs_512r4"] = R.rel(native[-1], b[-1])
    return row


def pilot_problems(pdes, k=10, regimes=("low", "medium", "high"), ells=("ell022", "ell009", "ell0025")):
    out = []
    for pde in pdes:
        for regime in regimes:
            for j, ordinal in enumerate(pilot_parents(pde, regime, k)):
                for ellcode in ells:
                    out.append({"pde": pde, "regime": regime, "ordinal": int(ordinal), "ell": ellcode, "rank": j})
    return out


def run_problem(p: dict, *, last_frame_cases=2, finest=False, addon=True, grid=128) -> dict:
    t0 = time.time()
    try:
        if p["pde"] in STATIC:
            row = static_problem(p["pde"], p["regime"], p["ordinal"], p["ell"], addon=addon, grid=grid)
        else:
            row = periodic_problem(p["pde"], p["regime"], p["ordinal"], p["ell"], finest=finest,
                                   last_frame=p.get("rank", 99) < last_frame_cases, addon=addon, grid=grid)
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - a failed pilot case is reported, never dropped
        row = {"parent_id": parent_id(p["pde"], p["regime"], p["ordinal"]), "pde": p["pde"], "regime": p["regime"],
               "ell": p["ell"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    row["wall_s"] = round(time.time() - t0, 2)
    row["revision_version"] = REVISION_VERSION
    return row


def gate_summary(rows) -> dict:
    """Per PDE: the cheapest registered candidate route whose e(R, F) passes the gate on all its pilot problems."""
    by = {}
    for r in rows:
        by.setdefault(r["pde"], []).append(r)
    out = {"gate": GATE, "contract": V2_REFERENCE_CONTRACT}
    for pde, rs in sorted(by.items()):
        ok = [r for r in rs if r.get("status") == "ok"]
        res = {"problems": len(rs), "failed_problems": len(rs) - len(ok), "candidates": []}
        cands = STATIC_CANDIDATES if pde in STATIC else PERIODIC_CANDIDATES
        for c in cands:
            if c["kind"] == "static":
                key = f"e_m{c['m']}_m{c['finer_m']}"
            else:
                key = {(256, 1): "joint_e_256r1_512r2", (256, 2): "joint_e_256r2_512r4", (512, 2): "joint_e_512r2_1024r4"}[(c["n_ref"], c["r_t"])]
            vals = np.array([r[key] for r in ok if r.get(key) is not None], dtype=np.float64)
            entry = {"route": c, "metric": key, "n": int(vals.size)}
            if vals.size:
                entry.update(p95=float(np.quantile(vals, 0.95)), max=float(vals.max()), median=float(np.median(vals)),
                             per_ell={e: float(np.max([r[key] for r in ok if r["ell"] == e and r.get(key) is not None] or [np.nan]))
                                      for e in ("ell022", "ell009", "ell0025")},
                             per_regime={g: float(np.max([r[key] for r in ok if r["regime"] == g and r.get(key) is not None] or [np.nan]))
                                         for g in ("low", "medium", "high")})
                entry["passes"] = bool(entry["p95"] <= GATE["p95"] and entry["max"] <= GATE["max"]
                                       and vals.size == len(rs) and res["failed_problems"] == 0)
            else:
                entry["passes"] = False
            res["candidates"].append(entry)
        chosen = next((e for e in res["candidates"] if e["passes"]), None)
        res["chosen_route"] = chosen["route"] if chosen else None
        res["status"] = "PASS" if chosen else "NOT_PASSED"
        out[pde] = res
    return out


def write_rows(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


__all__ = ["GATE", "PERIODIC_CANDIDATES", "STATIC_CANDIDATES", "addon_problem", "addon_routes", "gate_summary",
           "periodic_problem", "pilot_problems", "run_addon_problem", "run_problem", "static_problem", "write_rows"]
