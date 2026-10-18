"""One-line commands of the PDE-OBS revision tools: ``python -m pdeobs.revision <command> ...``.

  version                       versions and contracts (cite these with any result)
  plan                          V2 data-plan arithmetic and the 12 dataset ids
  v2-parts                      list the generation parts of one PDE (for a CPU job's work list)
  v2-generate                   generate one part (one PDE, regime, split, ordinal range) into an output root
  v2-run                        run a list of parts or pilot problems on a local process pool (one CPU job)
  v2-pilot                      list the pilot problems or evaluate the gate from pilot rows
  v2-addon-check                qualify an addon-view route on its own grids (m=4 vs m=8 at 64^2; 256r1 vs 512r2)
  fill                          nearest-observation fill of a field from a mask (executed v1 H7 rule)
  score                         strict block score (executed v1 scorer) [+ stationary diagnostics with --mask]
  classical                     gappy POD (H1 numerics) fit on training targets, recovery of test targets from masks
  numcheck                      replay / refinement checks of one stored record, or the manufactured-solution check
  contracts                     executed v1 contracts: dev-split, mixture positions, seed64

Examples (each line is complete and citable together with ``version``):
  python -m pdeobs.revision v2-generate --pde poisson --regime high --split train --start 0 --count 50 --out /data/v2
  python -m pdeobs.revision v2-run --pilot --pdes poisson,navier_stokes --workers 32 --out /data/v2-pilot
  python -m pdeobs.revision v2-pilot --gate /data/v2-pilot/pilot_rows.jsonl
  python -m pdeobs.revision v2-addon-check --pde poisson --grid 64 --ms 4,8 --out /data/v2-pilot/addon_rows.jsonl
  python -m pdeobs.revision v2-addon-check --pde navier_stokes --grid 64,256 --pairs 256r1:512r2 --out /data/v2-pilot/addon_rows.jsonl
  python -m pdeobs.revision fill --mask mask.npy --field field.npy --out filled.npy --periodic
  python -m pdeobs.revision score --pred pred.npy --target target.npy --mask masks.npy
  python -m pdeobs.revision classical --train train.npz --test test.npz --masks masks.npz --rank 64 --reg 1e-6
  python -m pdeobs.revision numcheck --pde poisson --shard part_00000.h5 --index 0
  python -m pdeobs.revision contracts dev-split --pde poisson --ids train_records.tsv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from . import REVISION_VERSION, V1_CONTRACT, V2_DATA_CONTRACT, V2_REFERENCE_CONTRACT


def _versions() -> dict:
    from .. import __version__ as pdeobs_version
    from .classical import CLASSICAL_CONTRACT
    from .contracts import SEED64_DEFINITION, V2_TAG
    from .fill import FILL_CONTRACT
    from .numchecks import NUMCHECK_CONTRACT
    from .scoring import SCORER_VERSION, STATIONARY_CONTRACT
    return {"pdeobs": pdeobs_version, "revision": REVISION_VERSION, "v1_contract": V1_CONTRACT, "v2_data": V2_DATA_CONTRACT,
            "v2_reference": V2_REFERENCE_CONTRACT, "seed64": SEED64_DEFINITION, "v2_seed_tag": V2_TAG,
            "v1_fill": FILL_CONTRACT, "v1_scorer": SCORER_VERSION, "v1_classical": CLASSICAL_CONTRACT,
            "stationary": STATIONARY_CONTRACT, "numchecks": NUMCHECK_CONTRACT}


def _route(text: str | None, pde: str) -> dict:
    from .v2data import default_route
    if not text or text == "default":
        return default_route(pde)
    if text.startswith("{"):
        return json.loads(text)
    return json.loads(Path(text).read_text())


def cmd_plan(_a) -> int:
    from .v2spec import ELLS, PDES, dataset_id, plan_counts
    print(json.dumps({"versions": _versions(), "counts": plan_counts(),
                      "datasets": [dataset_id(p, e) for p in PDES for e, _, _ in ELLS]}, indent=1))
    return 0


def parts(pde: str, part_size: int):
    from .v2spec import REGIMES, SPLITS
    out = []
    for regime in REGIMES:
        for split, lo, hi in SPLITS:
            for start in range(lo, hi, part_size):
                out.append({"pde": pde, "regime": regime, "split": split, "start": start, "count": min(part_size, hi - start)})
    return out


def cmd_parts(a) -> int:
    from .v2spec import canonical_pde
    for p in parts(canonical_pde(a.pde), a.part_size):
        print(json.dumps(p))
    return 0


def cmd_generate(a) -> int:
    from .v2data import generate_part
    from .v2spec import canonical_pde
    pde = canonical_pde(a.pde)
    rec = generate_part(pde, a.regime, a.split, a.start, a.count, a.out, _route(a.route, pde), addon=not a.no_addon)
    print(json.dumps({"status": rec["status"], "files": rec.get("files"), "failures": len(rec.get("failures", [])),
                      "wall_s": rec.get("wall_s"), "versions": _versions()}))
    return 0 if rec["status"] == "completed" else 1


def _worker_init():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")


def _do_item(item: dict) -> dict:
    import resource
    t0, c0 = time.time(), time.process_time()
    if item["kind"] == "pilot":
        from .v2pilot import run_problem
        row = run_problem(item["problem"], last_frame_cases=item.get("last_frame_cases", 2), finest=item.get("finest", False))
    else:
        from .v2data import generate_part
        rec = generate_part(item["pde"], item["regime"], item["split"], item["start"], item["count"], item["out"], item["route"],
                            addon=item.get("addon", True))
        row = {"part": {k: item[k] for k in ("pde", "regime", "split", "start", "count")}, "status": rec["status"],
               "failures": len(rec.get("failures", [])), "files": rec.get("files")}
    row["worker_wall_s"] = round(time.time() - t0, 2)
    row["worker_cpu_s"] = round(time.process_time() - c0, 2)
    row["worker_maxrss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    return row


def cmd_run(a) -> int:
    """One CPU job: a process pool over pilot problems or generation parts; every result line is appended at once."""
    import multiprocessing as mp
    from .v2spec import canonical_pde
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    if a.pilot:
        from .v2pilot import pilot_problems
        probs = pilot_problems([canonical_pde(p) for p in a.pdes.split(",")], k=a.k,
                               ells=tuple(a.ells.split(",")) if a.ells else ("ell022", "ell009", "ell0025"))
        if a.limit:
            probs = probs[: a.limit]
        items = [{"kind": "pilot", "problem": p, "last_frame_cases": a.last_frame_cases, "finest": a.finest} for p in probs]
        rows_path = out / (a.rows_name or "pilot_rows.jsonl")
    else:
        for line in Path(a.parts_file).read_text().splitlines():
            if line.strip():
                p = json.loads(line)
                pde = canonical_pde(p["pde"])
                items.append({"kind": "part", "pde": pde, "regime": p["regime"], "split": p["split"], "start": int(p["start"]),
                              "count": int(p["count"]), "out": str(out), "route": _route(a.route, pde), "addon": not a.no_addon})
        rows_path = out / (a.rows_name or "parts_log.jsonl")
    done = set()
    if rows_path.exists() and a.resume:
        for line in rows_path.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") in ("ok", "completed"):
                done.add(r.get("item_key"))
    for it in items:
        it["item_key"] = json.dumps(it.get("problem") or {k: it[k] for k in ("pde", "regime", "split", "start", "count")}, sort_keys=True)
    todo = [it for it in items if it["item_key"] not in done]
    if a.pilot:
        # longest-first (LPT) so the expensive periodic problems do not form the tail of the pool
        weight = {"navier_stokes": 4.0, "burgers": 3.0, "darcy": 1.0, "poisson": 0.5}
        todo.sort(key=lambda it: -(weight.get(it["problem"]["pde"], 1.0) * (2.0 if it["problem"]["ell"] == "ell0025" else 1.0)
                                   * (1.5 if it["problem"].get("rank", 99) < a.last_frame_cases else 1.0)))
    t0 = time.time()
    meta = {"event": "start", "items": len(items), "todo": len(todo), "workers": a.workers, "versions": _versions(),
            "host": os.uname().nodename if hasattr(os, "uname") else "", "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with (out / "run_events.jsonl").open("a") as ev:
        ev.write(json.dumps(meta) + "\n")
    deadline = time.time() + a.max_seconds if a.max_seconds else None
    n_ok = 0
    cpu = 0.0
    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    with ctx.Pool(a.workers, initializer=_worker_init, maxtasksperchild=a.maxtasks or None) as pool, rows_path.open("a") as f:
        pending = [pool.apply_async(_do_item, (it,)) for it in todo]
        for it, res in zip(todo, pending):
            while True:
                try:
                    row = res.get(timeout=30)
                    break
                except mp.TimeoutError:
                    if deadline and time.time() > deadline:
                        pool.terminate()
                        with (out / "run_events.jsonl").open("a") as ev:
                            ev.write(json.dumps({"event": "deadline", "done": n_ok, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
                        return 75
            row["item_key"] = it["item_key"]
            f.write(json.dumps(row) + "\n")
            f.flush()
            n_ok += row.get("status") in ("ok", "completed")
            cpu += float(row.get("worker_cpu_s", 0.0))
    wall = time.time() - t0
    end = {"event": "end", "done_ok": n_ok, "items": len(todo), "wall_s": round(wall, 1), "worker_cpu_s": round(cpu, 1),
           "useful_cpu_fraction": round(cpu / max(wall * a.workers, 1e-9), 4),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with (out / "run_events.jsonl").open("a") as ev:
        ev.write(json.dumps(end) + "\n")
    print(json.dumps(end))
    return 0 if n_ok == len(todo) else 1


def cmd_pilot(a) -> int:
    from .v2pilot import gate_summary, pilot_problems
    from .v2spec import PDES, canonical_pde
    if a.gate:
        rows = [json.loads(l) for l in Path(a.gate).read_text().splitlines() if l.strip()]
        res = gate_summary(rows)
        from .v2pilot import addon_routes
        from .v2spec import ADDON_GRIDS
        for pde in sorted({r.get("pde") for r in rows} & set(ADDON_GRIDS)):
            res.setdefault("addon_routes", {})[pde] = addon_routes(rows, pde)     # per addon grid (resolution addon 3.4)
        res["versions"] = _versions()
        res["rows_file"] = str(a.gate)
        print(json.dumps(res, indent=1))
        if a.write:
            Path(a.write).write_text(json.dumps(res, indent=1))
        return 0
    pdes = [canonical_pde(p) for p in a.pdes.split(",")] if a.pdes else list(PDES)
    for p in pilot_problems(pdes, k=a.k):
        print(json.dumps(p))
    return 0


def cmd_addon_check(a) -> int:
    """Rough-slice pilot parents of one static PDE: the addon view(s) at the listed refinements, one JSON row each."""
    from .v2pilot import pilot_problems, run_addon_problem
    from .v2spec import ADDON_ELL, ADDON_GRIDS, STATIC, canonical_pde
    pde = canonical_pde(a.pde)
    if pde not in ADDON_GRIDS:
        raise _InputError(f"v2-addon-check covers the families with addon views: {sorted(ADDON_GRIDS)}")
    grids = [int(g) for g in a.grid.split(",")]
    if not set(grids) <= set(ADDON_GRIDS[pde]):
        raise _InputError(f"addon grids of {pde}: {list(ADDON_GRIDS[pde])}")
    ms = tuple(int(m) for m in a.ms.split(","))
    pairs = None
    if pde not in STATIC:                                     # e.g. --pairs 256r1:512r2
        try:
            pairs = [tuple(tuple(int(x) for x in side.split("r")) for side in pq.split(":")) for pq in a.pairs.split(",")]
        except ValueError as exc:
            raise _InputError(f"--pairs must look like 256r1:512r2[,256r2:512r4]: {exc}") from exc
    problems = [p for p in pilot_problems([pde], k=a.k) if p["ell"] == ADDON_ELL]
    out = Path(a.out).open("a", encoding="utf-8") if a.out else None
    n_ok = 0
    for p in problems:
        row = run_addon_problem(p, grid_ms={g: ms for g in grids}, pairs=pairs)
        row["versions"] = _versions()
        n_ok += row["status"] == "ok"
        line = json.dumps(row)
        print(line)
        if out:
            out.write(line + "\n")
            out.flush()
    if out:
        out.close()
    return 0 if n_ok == len(problems) else 1


# ------------------------------------------------------------------------------- v1 campaign tools (array commands)
V1_TOOL_COMMANDS = ("fill", "score", "classical", "numcheck", "contracts")


class _InputError(ValueError):
    """A user input that violates a command's documented contract (reported as JSON, exit code 2)."""


def _emit(payload: dict) -> int:
    payload["versions"] = _versions()
    print(json.dumps(payload, indent=1))
    return 0


def _load_array(path: str, key: str | None = None):
    import numpy as np
    obj = np.load(path, allow_pickle=False)
    if hasattr(obj, "files"):                                          # .npz: the named array, else the only one
        with obj:
            if key and key in obj.files:
                return obj[key]
            if len(obj.files) == 1:
                return obj[obj.files[0]]
            raise _InputError(f"{path}: expected an array named {key!r} (found {obj.files})")
    return obj


def _load_npz(path: str) -> dict:
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _as_mask(arr, name: str):
    import numpy as np
    arr = np.asarray(arr)
    if arr.dtype == bool:
        return arr
    if arr.size and not np.isin(arr, (0, 1)).all():
        raise _InputError(f"{name} must be boolean or 0/1 (True / 1 = observed)")
    return arr.astype(bool)


def _file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _id_list(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _family(name: str) -> str:
    name = str(name).strip().lower()
    return {"ns": "navier_stokes", "rd": "reaction_diffusion"}.get(name, name)


def _records(path: str) -> list[tuple[int, str, str]]:
    """(index, record id, regime) per line: 'id<TAB>regime', 'id,regime', 'id regime' or a JSON object with
    sample_id / id / parent_id, regime and an optional index (default: the line's position)."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            d = json.loads(line)
            rid = next((d[k] for k in ("sample_id", "id", "parent_id") if k in d), None)
            if rid is None or "regime" not in d:
                raise _InputError(f"{path}: a JSON line needs sample_id (or id / parent_id) and regime: {line[:80]}")
            rows.append((int(d.get("index", len(rows))), str(rid), str(d["regime"])))
            continue
        parts = line.split("\t") if "\t" in line else (line.rsplit(",", 1) if "," in line else line.split())
        if len(parts) != 2:
            raise _InputError(f"{path}: expected '<record id> <regime>' per line: {line[:80]}")
        rows.append((len(rows), parts[0].strip(), parts[1].strip()))
    return rows


def cmd_fill(a) -> int:
    """Nearest-observation fill: mask (H, W) with field (H, W[, C]), or masks (N, H, W) with fields (N, H, W[, C])."""
    import numpy as np
    from .fill import FILL_CONTRACT, nearest_fill, nearest_fill_records
    mask = _as_mask(_load_array(a.mask, "mask"), "--mask")
    field = _load_array(a.field, "field")
    if mask.ndim == 2 and field.shape[:2] == mask.shape:
        out = nearest_fill(field, mask, a.periodic)
    elif mask.ndim == 3 and field.shape[:3] == mask.shape:
        out = nearest_fill_records(field, mask, a.periodic)
    else:
        raise _InputError(f"mask {mask.shape} must be (H, W) or (N, H, W) and lead the field shape {field.shape}")
    out_path = a.out if str(a.out).endswith(".npy") else str(a.out) + ".npy"
    np.save(out_path, out)
    return _emit({"command": "fill", "contract": FILL_CONTRACT, "metric": "torus" if a.periodic else "euclidean",
                  "tie_rule": "smallest flat (row-major) index", "mask_shape": list(mask.shape),
                  "field_shape": list(field.shape), "dtype": str(out.dtype), "observed": int(mask.sum()),
                  "hidden": int((~mask).sum()), "out": out_path, "out_sha256": _file_sha256(out_path)})


def cmd_score(a) -> int:
    """Strict block score; with --mask also the stationary raw / observed / hidden / data-consistent split."""
    import numpy as np
    from .scoring import stationary_diagnostics, strict_score, summarize
    pred = _load_array(a.pred, "prediction")
    target = _load_array(a.target, "target")
    target_ids = _id_list(a.ids) if a.ids else [str(i) for i in range(len(target))]
    if len(target_ids) != len(target):
        raise _InputError(f"--ids lists {len(target_ids)} ids for {len(target)} target records")
    if a.pred_ids:
        pred_ids = _id_list(a.pred_ids)
        if len(pred_ids) != len(pred):
            raise _InputError(f"--pred-ids lists {len(pred_ids)} ids for {len(pred)} predicted records")
    else:
        pred_ids = target_ids if len(pred) == len(target) else [str(i) for i in range(len(pred))]
    res = {"command": "score", "pred": str(a.pred), "target": str(a.target),
           "score": strict_score(pred_ids, pred, target_ids, target, target_ids)}
    if a.mask and res["score"].get("valid"):
        p_index = {i: k for k, i in enumerate(pred_ids)}
        p = np.asarray(pred)[[p_index[i] for i in target_ids]]
        t = np.asarray(target)
        squeezed = p.ndim == 4 and p.shape[-1] == 1
        if squeezed:
            p, t = p[..., 0], t[..., 0]
        mask = _as_mask(_load_array(a.mask, "mask"), "--mask")
        if mask.ndim == 2:
            mask = np.broadcast_to(mask, t.shape)
        diag = stationary_diagnostics(p, t, mask)
        keys = ("raw", "observed", "hidden", "data_consistent")
        res["stationary"] = {"contract": diag["contract"], "n": diag["n"], "squeezed_channel": squeezed,
                             "summary": {k: summarize(diag[k]) for k in keys},
                             "per_record": [[i, *(float(diag[k][j]) for k in keys)] for j, i in enumerate(target_ids)],
                             "per_record_columns": ["id", *keys]}
    _emit(res)
    return 0 if res["score"].get("valid") else 1


def cmd_classical(a) -> int:
    """Static gappy POD (H1 numerics): rank-r basis of the training targets, recovery of every test target from its
    observed cells only; raw and data-consistent errors. --all-methods adds the other four static baselines."""
    import numpy as np
    from .classical import CLASSICAL_CONTRACT, complexity, data_consistent, gappy_pod_recover, pod, rel, static_predictions
    from .scoring import summarize
    tr, te = _load_npz(a.train), _load_npz(a.test)
    if "X" not in tr or "X" not in te:
        raise _InputError("--train and --test must hold an array X (N, HW) of static targets")
    Xtr, Xte = np.asarray(tr["X"], dtype=np.float64), np.asarray(te["X"], dtype=np.float64)
    if Xtr.ndim != 2 or Xte.ndim != 2 or Xtr.shape[1] != Xte.shape[1]:
        raise _InputError(f"X must be (N, HW) with one HW for train and test; got {Xtr.shape} / {Xte.shape}")
    masks = _as_mask(_load_array(a.masks, "masks"), "--masks")
    hw = Xte.shape[1]
    masks = masks.reshape(-1, hw) if masks.size % hw == 0 else None
    if masks is None or len(masks) not in (1, len(Xte)):
        raise _InputError(f"--masks must hold one flat mask of {hw} cells, or one per test record")
    if len(masks) == 1:
        masks = np.repeat(masks, len(Xte), axis=0)
    ids = [str(i) for i in te["ids"]] if "ids" in te else [str(i) for i in range(len(Xte))]
    if len(ids) != len(Xte):
        raise _InputError(f"--test ids has {len(ids)} entries for {len(Xte)} records")
    mu, U, _ = pod(Xtr, int(a.rank))
    preds, raw, dc = gappy_pod_recover(Xtr, Xte, masks, a.rank, a.reg, basis=(mu, U))
    res = {"command": "classical", "contract": CLASSICAL_CONTRACT, "method": "gappy_pod_ridge",
           "rank": int(a.rank), "reg": float(a.reg), "train_records": int(len(Xtr)), "test_records": int(len(Xte)),
           "basis_columns": int(U.shape[1]),
           "lam_rule": "reg * trace(A^T A) / r on the masked basis A; least squares on [A; sqrt(lam) I]",
           "summary": {"raw": summarize(raw), "data_consistent": summarize(dc)},
           "per_record": [[i, e, d] for i, e, d in zip(ids, raw, dc, strict=True)],
           "per_record_columns": ["id", "raw", "data_consistent"]}
    if a.all_methods:
        side = int(round(hw ** 0.5))
        if side * side != hw:
            raise _InputError("--all-methods needs square fields (linear interpolation on the n x n grid)")
        mean_field = Xtr.mean(0)
        rows, fallback = {}, []
        for j in range(len(Xte)):
            m, t = masks[j], Xte[j]
            p_all, fb = static_predictions(t[m], m, mu, U, a.reg, mean_field, (side, side))
            fallback.append(fb)
            for k, p in p_all.items():
                rows.setdefault(k, ([], []))
                rows[k][0].append(rel(p, t))
                rows[k][1].append(rel(data_consistent(p, t[m], m), t))
        res["methods"] = {k: {"raw": summarize(r), "data_consistent": summarize(d)} for k, (r, d) in rows.items()}
        res["interp_fallback_fraction_mean"] = float(np.mean(fallback))
    if a.complexity:
        regimes = [str(r) for r in tr["regimes"]] if "regimes" in tr else ["all"] * len(Xtr)
        res["complexity"] = complexity(Xtr, regimes, True)
    if a.out:
        out_path = a.out if str(a.out).endswith(".npy") else str(a.out) + ".npy"
        np.save(out_path, preds)
        res["out"], res["out_sha256"] = out_path, _file_sha256(out_path)
    return _emit(res)


def cmd_numcheck(a) -> int:
    """Replay / refinement checks of record --index of a shard, or the manufactured-solution check."""
    import numpy as np
    from .numchecks import PERIODIC_PDES, STATIC_PDES, manufactured, periodic_refinement_record, static_refinement_record
    pde = _family(a.pde)
    if a.manufactured:
        return _emit({"command": "numcheck", "check": "manufactured_or_analytic", "pde": pde, "n": int(a.n),
                      "result": manufactured(pde, int(a.n))})
    if not a.shard:
        raise _InputError("numcheck needs --shard and --index, or --manufactured")
    if pde not in STATIC_PDES + PERIODIC_PDES:
        raise _InputError(f"no reference route for {pde!r}; supported: {STATIC_PDES + PERIODIC_PDES}")
    from ..storage import LazyHDF5Dataset
    with LazyHDF5Dataset(a.shard) as ds:
        s = ds[int(a.index)]
    md = dict(s.metadata)
    stored_pde = md.get("pde", md.get("family"))
    if stored_pde is not None and _family(stored_pde) != pde:
        raise _InputError(f"record {a.index} of {a.shard} is {stored_pde!r}, not {pde!r}")
    traj = np.asarray(s.trajectory, dtype=np.float64)[..., 0]
    if pde in STATIC_PDES:
        row = static_refinement_record(pde, np.asarray(s.condition, dtype=np.float64)[..., 0], traj[-1],
                                       record_id=md.get("sample_id"), regime=md.get("regime"))
    else:
        if md.get("regime") is None:
            raise _InputError(f"record {a.index} of {a.shard} has no regime in its metadata (needed for the viscosity)")
        row = periodic_refinement_record(pde, str(md["regime"]), traj, record_id=md.get("sample_id"), last_frame=a.last_frame)
    return _emit({"command": "numcheck", "shard": str(a.shard), "index": int(a.index), "record": row})


def cmd_contracts(a) -> int:
    """Executed v1 contracts (rev8k-contracts-D0001-v1) on id lists."""
    from . import contracts as C
    if a.what == "seed64":
        parts = []
        for p in a.parts:
            try:
                parts.append(json.loads(p))
            except ValueError:
                parts.append(p)
        return _emit({"command": "contracts seed64", "domain": a.domain, "parts": parts, "seed64": C.seed64(a.domain, *parts),
                      "definition": C.SEED64_DEFINITION})
    family = _family(a.pde)
    if family not in C.LOGICAL:
        raise _InputError(f"unknown PDE {a.pde!r}; expected one of {sorted(C.LOGICAL)}")
    recs = _records(a.ids)
    if a.what == "dev-split":
        per_regime = {}
        for _, _, regime in recs:
            per_regime[regime] = per_regime.get(regime, 0) + 1
        full = C.DEV_VAL_PER_REGIME + C.DEV_TRAIN_PER_REGIME
        if not a.any_count and any(v != full for v in per_regime.values()):
            raise _InputError(f"dev-split takes the {full} original training records of every regime (got {per_regime}); "
                              "--any-count is for partial software tests only")
        dev_train, dev_val, receipt = C.dev_partition(recs, family, any_count=bool(a.any_count))
        sid = {i: s for i, s, _ in recs}
        return _emit({"command": "contracts dev-split", "ids_file": str(a.ids), "receipt": receipt, "dev_train": dev_train,
                      "dev_val": dev_val, "dev_val_ids": [sid[i] for i in dev_val]})
    from .v2masks import MIXTURES
    if a.protocol in C.MIX_COMPONENTS:
        comps = C.MIX_COMPONENTS[a.protocol]
        positions = C.mixture_positions([(rid, regime) for _, rid, regime in recs], family, len(comps))
        component = lambda j, e: C.mixture_component(a.protocol, j, e)       # noqa: E731
    elif a.protocol in MIXTURES:
        from .v2masks import mixture_positions as v2_positions
        from .v2spec import PDES as V2_PDES
        if family not in V2_PDES:
            raise _InputError(f"V2 mixtures are defined for {V2_PDES}, not {family!r}")
        comps = MIXTURES[a.protocol]
        by = {}
        for _, rid, regime in recs:
            by.setdefault(regime, []).append(rid)
        positions = v2_positions(by, family, len(comps))
        component = lambda j, e: comps[(int(j) + int(e)) % len(comps)]       # noqa: E731
    else:
        raise _InputError(f"unknown mixture {a.protocol!r}; expected one of {sorted(C.MIX_COMPONENTS) + sorted(MIXTURES)}")
    res = {"command": "contracts mixture", "ids_file": str(a.ids), "receipt": C.mixture_receipt(positions, a.protocol),
           "positions": positions}
    if a.epoch is not None:
        res["epoch"] = int(a.epoch)
        res["components"] = {rid: component(j, a.epoch) for rid, j in positions.items()}
    return _emit(res)


def _add_v1_tool_parsers(sub) -> None:
    p = sub.add_parser("fill", help="nearest-observation fill (executed v1 H7 rule)")
    p.add_argument("--mask", required=True, help=".npy/.npz boolean mask (H, W) or (N, H, W); True = observed")
    p.add_argument("--field", required=True, help=".npy/.npz field (H, W[, C]) or (N, H, W[, C]); hidden values unused")
    p.add_argument("--out", required=True, help="output .npy")
    p.add_argument("--periodic", action="store_true", help="torus distance (periodic data); default Euclidean")
    p.set_defaults(fn=cmd_fill)
    p = sub.add_parser("score", help="strict block score (executed v1 scorer) [+ stationary diagnostics]")
    p.add_argument("--pred", required=True, help=".npy/.npz predictions (N, ...)")
    p.add_argument("--target", required=True, help=".npy/.npz targets (N, ...)")
    p.add_argument("--mask", default=None, help="boolean (N, H, W) or (H, W) mask -> raw / observed / hidden / data-consistent")
    p.add_argument("--ids", default=None, help="text file, one target id per line (default: row numbers)")
    p.add_argument("--pred-ids", default=None, help="text file, one prediction id per line (default: the target ids)")
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("classical", help="gappy POD fit + recovery (H1 numerics, static targets)")
    p.add_argument("--train", required=True, help=".npz with X (N, HW) training targets [regimes (N,)]")
    p.add_argument("--test", required=True, help=".npz with X (N_test, HW) test targets [ids (N_test,)]")
    p.add_argument("--masks", required=True, help=".npz/.npy masks (N_test, HW | H, W) or one shared mask")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--reg", type=float, default=0.0, help="lam = reg * trace(A^T A) / r (0 = plain least squares)")
    p.add_argument("--out", default=None, help="write the gappy POD predictions (N_test, HW) to this .npy")
    p.add_argument("--all-methods", action="store_true", help="also train_mean, observed_mean, interpolation, zero_output")
    p.add_argument("--complexity", action="store_true", help="POD energy r95 / r99 and radial spectra of the training targets")
    p.set_defaults(fn=cmd_classical)
    p = sub.add_parser("numcheck", help="replay / refinement of one stored record, or a manufactured solution")
    p.add_argument("--pde", required=True, help="poisson, darcy, burgers, navier_stokes (manufactured: poisson, helmholtz, darcy)")
    p.add_argument("--shard", default=None, help="HDF5 shard in the corpus schema")
    p.add_argument("--index", type=int, default=0, help="record index within the shard")
    p.add_argument("--last-frame", action="store_true", help="periodic: also the T_* checks at stored frame 14")
    p.add_argument("--manufactured", action="store_true", help="the analytic check instead of a stored record")
    p.add_argument("--n", type=int, default=128, help="grid of the manufactured check")
    p.set_defaults(fn=cmd_numcheck)
    p = sub.add_parser("contracts", help="executed v1 contracts: dev-split, mixture, seed64")
    csub = p.add_subparsers(dest="what", required=True)
    q = csub.add_parser("dev-split", help="development split (100 dev-val + 500 dev-train per regime)")
    q.add_argument("--pde", required=True)
    q.add_argument("--ids", required=True, help="records file: '<sample id> <regime>' per line (or JSON lines)")
    q.add_argument("--any-count", action="store_true", help="partial data (software tests only): no 600-per-regime check")
    q.set_defaults(fn=cmd_contracts)
    q = csub.add_parser("mixture", help="mixture positions j and receipt (v1 MXF3/MXC3/MXR3/MXC2, V2 MXR3_K256/MXR6_K256_K1024)")
    q.add_argument("--pde", required=True)
    q.add_argument("--protocol", required=True)
    q.add_argument("--ids", required=True, help="records file: '<record id> <regime>' per line (V2: parent ids)")
    q.add_argument("--epoch", type=int, default=None, help="also list each record's component at this epoch")
    q.set_defaults(fn=cmd_contracts)
    q = csub.add_parser("seed64", help="seed64(domain, *parts); parts are JSON values, other words are strings")
    q.add_argument("domain")
    q.add_argument("parts", nargs="*")
    q.set_defaults(fn=cmd_contracts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pdeobs.revision", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version").set_defaults(fn=lambda a: print(json.dumps(_versions(), indent=1)) or 0)
    sub.add_parser("plan").set_defaults(fn=cmd_plan)
    p = sub.add_parser("v2-parts")
    p.add_argument("--pde", required=True)
    p.add_argument("--part-size", type=int, default=50)
    p.set_defaults(fn=cmd_parts)
    p = sub.add_parser("v2-generate")
    p.add_argument("--pde", required=True)
    p.add_argument("--regime", required=True, choices=("low", "medium", "high"))
    p.add_argument("--split", required=True, choices=("train", "validation", "test"))
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--route", default="default", help="'default', a JSON object, or a JSON file (e.g. the pilot gate decision)")
    p.add_argument("--no-addon", action="store_true", help="skip the 64/256 views of rough Poisson / NS")
    p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("v2-run")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--pdes", default="darcy,poisson,burgers,navier_stokes")
    p.add_argument("--ells", default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--last-frame-cases", type=int, default=2)
    p.add_argument("--finest", action="store_true", help="also e((512,2),(1024,4)) for every periodic pilot problem")
    p.add_argument("--parts-file", default=None)
    p.add_argument("--route", default="default")
    p.add_argument("--no-addon", action="store_true")
    p.add_argument("--rows-name", default=None)
    p.add_argument("--resume", action="store_true", help="skip items already recorded as ok/completed in the rows file")
    p.add_argument("--max-seconds", type=float, default=0)
    p.add_argument("--maxtasks", type=int, default=0)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("v2-pilot")
    p.add_argument("--pdes", default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--gate", default=None, help="pilot rows jsonl -> gate summary")
    p.add_argument("--write", default=None)
    p.set_defaults(fn=cmd_pilot)
    p = sub.add_parser("v2-addon-check")
    p.add_argument("--pde", required=True)
    p.add_argument("--grid", required=True, help="addon grid(s), comma separated")
    p.add_argument("--ms", default="4,8", help="static: refinements compared in order, e.g. 4,8")
    p.add_argument("--pairs", default="256r1:512r2", help="periodic: candidate pairs (128-grid equivalents), e.g. 256r1:512r2")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", default=None, help="append the rows to this jsonl file")
    p.set_defaults(fn=cmd_addon_check)
    _add_v1_tool_parsers(sub)
    a = ap.parse_args(argv)
    try:
        return int(a.fn(a) or 0)
    except ValueError as exc:                     # input contract violations of the array commands: JSON, exit code 2
        if not (isinstance(exc, _InputError) or a.cmd in V1_TOOL_COMMANDS):
            raise
        print(json.dumps({"command": a.cmd, "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
