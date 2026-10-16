"""V2 data generation (plan 5.1-5.4 and the resolution addon 3.1-3.3).

One unit of work is a *part*: (PDE, regime, split, ordinal range). For every parent in the part the paper generator
is called once per correlation length with the parent seed (so the three ell records share the white noise). That
gives the stored inputs and the native 128^2 solution. The reference route of :mod:`pdeobs.revision.reference`
then gives the label.

Output (HDF5 shards in the corpus schema, :class:`pdeobs.storage.AtomicHDF5ShardWriter`):

    <out>/<dataset_id>/<regime>/<split>/part_<start:05d>.h5          condition, reference trajectory, geometry
    <out>/<dataset_id>__grid<N>/<regime>/<split>/part_<start:05d>.h5   addon views (rough Poisson / NS, N = 64, 256)
    <out>/receipts/<pde>/<regime>/<split>/part_<start:05d>.json       per-record diagnostics, timing, versions

* The stored trajectory is the reference label: static T = 1; temporal 15 frames at the paper times, with frame 0 the
  stored initial state.
* Every record carries the native-vs-reference difference: joint relative L2 over frames 1-3, or the static target.
* A failed record is written to the receipt with its error and is never replaced by another seed. The part is then
  incomplete, and the shards are not published.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from . import REVISION_VERSION, V2_DATA_CONTRACT
from . import reference as R
from .contracts import V2_TAG
from .v2spec import (ADDON_ELL, ADDON_GRIDS, BOUNDARY, DENSE_T, ELLS, GRID, STATIC, STORED_FRAMES, dataset_id, parent_id,
                     parent_seed, record_id, split_of, split_range)

TARGET_FRAMES = (1, 2, 3)


def default_route(pde: str) -> dict:
    """Candidate production route before the pilot decides (recorded in every spec; the pilot may replace it)."""
    if pde in STATIC:
        return {"kind": "static", "m": 4}
    return {"kind": "periodic", "n_ref": 256, "r_t": 2}


def native_output(pde: str, regime: str, ordinal: int, setting: str, grid: int = GRID):
    """The paper generator, unchanged, at the parent seed (float32 storage dtype)."""
    from ..pdes import generate_sample
    kwargs = {}
    if pde not in STATIC:
        kwargs["time_steps"] = DENSE_T[pde]
    return generate_sample(family=pde, boundary=BOUNDARY[pde], setting=setting, regime=regime,
                           seed=parent_seed(pde, regime, ordinal), resolution=grid, dtype=np.float32, **kwargs)


def _stored_native(pde: str, out) -> np.ndarray:
    traj = np.asarray(out.trajectory, dtype=np.float32)[..., 0]
    if pde in STATIC:
        return traj[-1:]
    stride = (DENSE_T[pde] - 1) // (STORED_FRAMES - 1)
    return traj[::stride][:STORED_FRAMES]


def reference_label(pde: str, regime: str, condition: np.ndarray, initial: np.ndarray | None, route: dict,
                    grids=(GRID,)) -> tuple[dict, dict]:
    """Reference label on each requested grid (dict N -> array [T, N, N]) and diagnostics."""
    labels, diag = {}, {"route": dict(route)}
    if route["kind"] == "static":
        for n in grids:
            sol, info = R.static_solve(pde, condition, n, int(route["m"]))
            labels[n] = sol[None]
            diag[f"grid{n}"] = info
    else:
        frames, info = R.periodic_run(pde, regime, initial, int(route["n_ref"]), int(route["r_t"]))
        diag["run"] = info
        for n in grids:
            labels[n] = R.sample_periodic(frames, n, initial if n == initial.shape[0] else None)
    return labels, diag


def _target_rel(pde: str, a: np.ndarray, b: np.ndarray) -> float:
    if pde in STATIC:
        return R.rel(a[-1], b[-1])
    idx = list(TARGET_FRAMES)
    return R.rel(a[idx], b[idx])


def _sample(pde, regime, ordinal, ellcode, setting, out, label, n, route, extra):
    from ..schema import Sample
    from .. import __version__ as pdeobs_version
    cond = np.asarray(out.condition, dtype=np.float32)
    geom = np.asarray(out.geometry, dtype=np.float32)
    if n != cond.shape[0]:
        # addon view: inputs are the stored 128^2 fields evaluated at the view grid (same continuous field)
        if pde in STATIC:
            x, _ = R.static_coords(n, 1, cond.shape[0])
            c64 = np.asarray(cond[..., 0], dtype=np.float64)
            field = R.trig_eval(c64, x, x) if pde == "poisson" else np.exp(R.trig_eval(np.log(c64), x, x))
            gview = np.zeros((n, n), dtype=np.float32)
            gview[[0, -1], :] = 1.0
            gview[:, [0, -1]] = 1.0
        else:
            field = label[0]
            gview = np.zeros((n, n), dtype=np.float32)
        cond = np.asarray(field, dtype=np.float32)[..., None]
        geom = gview[..., None]
    params = dict(getattr(out, "parameters", {}) or {})
    md = {"sample_id": record_id(pde, regime, ordinal, ellcode) + ("" if n == GRID else f"/grid{n}"),
          "parent_id": parent_id(pde, regime, ordinal), "dataset_id": dataset_id(pde, ellcode) + ("" if n == GRID else f"__grid{n}"),
          "schema_version": "1.0", "pde": pde, "boundary": BOUNDARY[pde], "setting": setting, "regime": regime,
          "ell_code": ellcode, "state_representation": "vorticity" if pde == "navier_stokes" else "scalar",
          "solver_fidelity": "reference_qualified", "solver_version": V2_DATA_CONTRACT,
          "solver_implementation": "pdeobs.revision.reference", "pdeobs_version": pdeobs_version,
          "revision_version": REVISION_VERSION, "resolution": [n, n], "T": int(label.shape[0]),
          "regime_sample_index": int(ordinal), "split": split_of(ordinal), "tier": "rev8k-v2",
          "seed": parent_seed(pde, regime, ordinal), "generation_seed": f"{V2_TAG}/parent",
          "label_route": route, "parameters": params, **extra}
    if pde not in STATIC:
        m, frame_dt = R.frame_schedule(pde)
        md["stored_frame_indices"] = [k * m for k in range(STORED_FRAMES)]
        md["stored_time_values"] = [k * m * frame_dt for k in range(STORED_FRAMES)]
    traj = np.asarray(label, dtype=np.float32)[..., None]
    return Sample(cond, traj, geom, md)


def addon_route_for(addon_route: dict | None, n: int) -> dict | None:
    """The route of addon view ``n``: ``addon_route`` is either one route for every addon grid (a dict with "kind") or
    one route per grid ({"64": route, "256": route}; each qualified on its own grid by the pilot)."""
    if addon_route is None:
        return None
    if "kind" in addon_route:
        return addon_route
    r = addon_route.get(str(int(n)), addon_route.get(int(n)))
    if r is None:
        raise ValueError(f"no addon route for grid {n}: {sorted(addon_route)}")
    return r


def generate_parent(pde: str, regime: str, ordinal: int, route: dict, *, grid: int = GRID, addon: bool = True,
                    addon_route: dict | None = None) -> dict:
    """All records of one parent: {(ellcode, N): Sample} plus per-record diagnostics. Raises on failure.
    The 128^2 label of every ell always comes from ``route`` (one route per PDE for all ell). The addon views (64^2, 256^2
    of the rough slice) come from ``addon_route`` when given: a finer route qualified on those grids by the pilot
    (resolution addon 3.4), either one route for both grids or one per grid (:func:`addon_route_for`); otherwise from
    ``route``."""
    samples, diags = {}, {}
    for ellcode, ell, setting in ELLS:
        t0 = time.process_time()
        out = native_output(pde, regime, ordinal, setting, grid)
        native = _stored_native(pde, out)
        cond = np.asarray(out.condition, dtype=np.float64)[..., 0]
        init = None if pde in STATIC else np.asarray(out.trajectory, dtype=np.float32)[0, ..., 0].astype(np.float64)
        t_native = time.process_time() - t0
        grids = (grid,)
        extra_grids = ()
        if addon and ellcode == ADDON_ELL and pde in ADDON_GRIDS and grid == GRID:
            extra_grids = tuple(ADDON_GRIDS[pde])
            if addon_route is None:
                grids = (grid,) + extra_grids
            elif "kind" not in addon_route:                 # per grid: views qualified on the main route share its run
                grids = (grid,) + tuple(n for n in extra_grids if addon_route_for(addon_route, n) == route)
        labels, diag = reference_label(pde, regime, cond, init, route, grids)
        if extra_grids and addon_route is not None and "kind" in addon_route:
            add_labels, add_diag = reference_label(pde, regime, cond, init, addon_route, extra_grids)
            labels.update(add_labels)
            diag["addon"] = add_diag
            grids = (grid,) + extra_grids
        elif extra_grids and addon_route is not None:                  # one qualified route per addon grid
            groups = {}
            for n in extra_grids:
                if n in grids:
                    continue                                            # already labelled with the main route
                r_n = addon_route_for(addon_route, n)
                groups.setdefault(json.dumps(r_n, sort_keys=True), (r_n, []))[1].append(n)
            on_main = [n for n in extra_grids if n in grids]
            diag["addon"] = {"grids_" + "_".join(str(n) for n in on_main): "main_route"} if on_main else {}
            for r_n, ns in groups.values():
                add_labels, add_diag = reference_label(pde, regime, cond, init, r_n, tuple(ns))
                labels.update(add_labels)
                diag["addon"]["grids_" + "_".join(str(n) for n in ns)] = add_diag
            grids = (grid,) + extra_grids
        if pde not in STATIC:
            assert np.array_equal(labels[grid][0].astype(np.float32), native[0]), "frame 0 must be the stored initial state"
        d = {"native_vs_reference_target": _target_rel(pde, native.astype(np.float64), labels[grid]),
             "cpu_s_native": round(t_native, 3), "reference": diag, "cpu_s_total": round(time.process_time() - t0, 3)}
        if pde not in STATIC:
            d["native_vs_reference_last"] = R.rel(native[-1], labels[grid][-1])
        diags[ellcode] = d
        for n in grids:
            extra = {"native_vs_reference_target": d["native_vs_reference_target"]} if n == grid else {"addon_view": True}
            r_n = route if (n == grid or addon_route is None) else addon_route_for(addon_route, n)
            samples[(ellcode, n)] = _sample(pde, regime, ordinal, ellcode, setting, out, labels[n], n, r_n, extra)
    return {"samples": samples, "diagnostics": diags}


TRAIN_VIEW_FRAMES = 4      # "t4" view of temporal records: stored frames 0..3 (input u0 + rollout targets 1-3) only


def _views(pde: str, sample):
    """The full record, plus for temporal families the compact training view "t4" (same record id, frames 0..3; the
    full 15-frame trajectory stays in the full view). Training stages only the compact view."""
    yield "full", sample
    if pde not in STATIC:
        from ..schema import Sample
        md = dict(sample.metadata)
        md["T"] = TRAIN_VIEW_FRAMES
        md["view"] = "t4"
        for k in ("stored_frame_indices", "stored_time_values"):
            if k in md:
                md[k] = list(md[k])[:TRAIN_VIEW_FRAMES]
        yield "t4", Sample(sample.condition, np.asarray(sample.trajectory)[:TRAIN_VIEW_FRAMES], sample.geometry, md)


def part_paths(out_root: Path, pde: str, regime: str, split: str, start: int):
    return {"receipt": out_root / "receipts" / pde / regime / split / f"part_{start:05d}.json"}


def generate_part(pde: str, regime: str, split: str, start: int, count: int, out_root, route: dict | None = None, *,
                  grid: int = GRID, addon: bool = True, addon_route: dict | None = None, compression: str | None = "gzip",
                  level: int = 4) -> dict:
    """Generate ordinals [start, start + count) of one (PDE, regime, split); write the shards and a receipt."""
    from ..storage import AtomicHDF5ShardWriter
    out_root = Path(out_root)
    route = dict(route or default_route(pde))
    lo, hi = split_range(split)
    if not (lo <= start and start + count <= hi):
        raise ValueError(f"ordinals {start}..{start + count - 1} are not all in the {split} range {lo}..{hi - 1}")
    receipt_path = part_paths(out_root, pde, regime, split, start)["receipt"]
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old.get("status") == "completed":
            return old
    spec = {"contract": V2_DATA_CONTRACT, "revision_version": REVISION_VERSION, "pde": pde, "regime": regime, "split": split,
            "ordinals": [start, start + count], "grid": grid, "route": route, "addon": bool(addon), "addon_route": addon_route}
    writers = {}
    records, failures = [], []
    t0 = time.time()
    c0 = time.process_time()
    try:
        for k, ordinal in enumerate(range(start, start + count)):
            try:
                res = generate_parent(pde, regime, ordinal, route, grid=grid, addon=addon, addon_route=addon_route)
            except Exception as exc:  # noqa: BLE001 - recorded, never replaced
                failures.append({"parent_id": parent_id(pde, regime, ordinal), "error": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc()[-2000:]})
                continue
            for (ellcode, n), sample in sorted(res["samples"].items()):
                for view, smp in _views(pde, sample):
                    key = (ellcode, n, view)
                    if key not in writers:
                        ds = dataset_id(pde, ellcode) + ("" if n == grid else f"__grid{n}") + ("" if view == "full" else f"__{view}")
                        path = out_root / ds / regime / split / f"part_{start:05d}.h5"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        writers[key] = AtomicHDF5ShardWriter(path, expected_count=count,
                                                             spec=dict(spec, dataset_id=ds, view_grid=n, view=view),
                                                             resume=False, overwrite=True, compression=compression,
                                                             compression_opts=level)
                    writers[key].append(smp)
            records.append({"parent_id": parent_id(pde, regime, ordinal), **{e: res["diagnostics"][e] for e in res["diagnostics"]}})
        status = "completed" if not failures else "failed_records"
        files = {}
        for (ellcode, n, view), w in sorted(writers.items()):
            if failures:
                w.close()
            else:
                man = w.finalize()
                files[f"{ellcode}/grid{n}/{view}"] = {"path": str(Path(w.path).relative_to(out_root)), "sha256": man["sha256"],
                                                      "count": int(man["sample_count"])}
    except BaseException:
        for w in writers.values():
            w.close()
        raise
    receipt = {"status": status, "spec": spec, "records": records, "failures": failures, "files": files,
               "wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - c0, 1),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}       # no host / user / job identity in data receipts
    tmp = receipt_path.with_suffix(".json.part")
    tmp.write_text(json.dumps(receipt, indent=1))
    os.replace(tmp, receipt_path)
    return receipt


__all__ = ["TARGET_FRAMES", "TRAIN_VIEW_FRAMES", "addon_route_for", "default_route", "generate_parent", "generate_part", "native_output", "reference_label"]
