"""Weights-inference replay of an October 2026 result (reproduction path C).

The script:
1. rebuilds the model of a run from its frozen recipe (configs/paper/october/recipes.yaml);
2. loads its checkpoint;
3. re-predicts the run's test views on the same parents and per-parent sensor layouts;
4. scores every record with the full-field relative L2 (float64);
5. compares every view of the suite with the stored result in results/revision_october (runs_long): the replayed view
   mean against the stored mean and the replayed record count against the stored one; with --per-record, every record's
   error against the published per-record export of the suite, aligned by sample identity.

    python reproduce/october/replay_inference.py --run V4C__poisson__ell009__fno__U256__s1101 \
        --checkpoint <dir>/checkpoints/last.pt --data-root <V2 store> [--suite v2eval7|n5nested] [--device cuda] \
        [--per-record <download>/october2026/results/part-A/update_20261021/per_record/v2eval7.csv.gz]

The V2 store is the directory that holds the V2 views (e.g. v2_poisson_ell009_ref128/). The verdict is `status`:
`agrees` (exit 0) when every view of the suite has a stored score, every replayed mean is finite and within --rtol
(default 1e-4) of it with the stored record count, and, with --per-record, every record of every view is found once and
within --per-record-rtol (default 1e-3) of its stored error; `disagrees` (exit 1) when a difference exceeds its
tolerance or a replayed value is not finite; `incomplete` (exit 1) when a stored score, a record or a per-record row is
missing or repeated. Bit-identical agreement is not expected across devices.

FNO checkpoints on a CPU: the October FNOs were trained and scored with cuFFT, whose large-batch inverse real FFT keeps
the imaginary parts of the kx = 0 / Nyquist bins that CPU FFT libraries drop (see pdeobs.revision.fftcompat). With
--fft auto (default) a CPU replay uses the cuFFT-compatible transform; --fft native shows the plain CPU result.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import gzip
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

V2EVAL7 = ("U256", "C256", "L256", "U1024", "C1024", "L1024", "G256")
NESTED_K = (64, 128, 256, 512, 1024, 4096)


def parse_run(run: str) -> dict:
    p = run.split("__")
    proto = "__".join(p[4:-1]) or {"N3GJ2": "GJ2_K256", "N5MXR6": "MXR6_K256_K1024"}.get(p[0], "")
    grid = 128
    if "__g" in proto:
        proto, g = proto.rsplit("__g", 1)
        grid = int(g)
    pde = {"ns": "navier_stokes"}.get(p[1], p[1])
    return {"cohort": p[0], "key": p[1], "pde": pde, "ell": p[2], "method": p[3], "protocol": proto, "seed": int(p[-1][1:]),
            "grid": grid, "dataset_id": f"v2_{p[1]}_{p[2]}_ref128"}


def recipe_of(run: str, recipes_path: str) -> dict:
    info = parse_run(run)
    groups = yaml.safe_load(open(recipes_path, encoding="utf-8"))["groups"]
    for g in groups:
        if g["cohort"] == info["cohort"] and g["dataset_id"] == info["dataset_id"] and g["method"] == info["method"] and "recipe" in g:
            rec = dict(g["recipe"])
            rec.update(id=run, pde=info["pde"], dataset_id=info["dataset_id"], protocol=info["protocol"], seed=info["seed"],
                       grid=info["grid"] if info["grid"] != 128 else rec.get("grid"))
            return rec
    raise SystemExit(f"no frozen recipe for {run} in {recipes_path}")


def parse_overrides(items) -> dict:
    """--set KEY=VALUE items -> recipe overrides (VALUE read as JSON when it parses, else as a string)."""
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set needs KEY=VALUE, got {item!r}")
        try:
            out[key] = json.loads(value)
        except ValueError:
            out[key] = value
    return out


def stored_scores(run: str, suite: str, results: str) -> dict:
    """{view label: {"mean", "n", "run_id"}} of the run's stored result rows (runs_long.part-*.csv); the row of a view
    must be unique and valid."""
    out = {}
    for f in sorted(glob.glob(os.path.join(results, "runs_long.part-*.csv"))):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            own = suite == "v2eval7" and r["run_id"] == run and r["suite"] == "v2eval7"
            nested = suite == "n5nested" and r["suite"] == "n5nested" and r["parent_run_id"] == run
            if not (own or nested):
                continue
            if r["view"] in out:
                raise SystemExit(f"{run}: two stored rows for suite {suite}, view {r['view']}")
            if r["valid"] != "True":
                raise SystemExit(f"{run}: the stored row of {suite} / {r['view']} is not a valid block")
            out[r["view"]] = {"mean": float(r["mean"]), "n": int(r["n"]) if r["n"] else None, "run_id": r["run_id"]}
    return out


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def load_per_record(path: str, run_ids) -> tuple[dict, Counter]:
    """{(run_id, view, sample_id): error} of the wanted evaluation ids from a per-record export (csv or csv.gz; columns
    run_id, view, sample_id, error); the second value counts keys that occur more than once."""
    wanted = set(run_ids)
    out: dict = {}
    dups: Counter = Counter()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        for need in ("run_id", "view", "sample_id", "error"):
            if need not in col:
                raise SystemExit(f"{path}: column {need} missing (columns {header})")
        i_run, i_view, i_id, i_err = col["run_id"], col["view"], col["sample_id"], col["error"]
        for row in reader:
            if row[i_run] not in wanted:
                continue
            key = (row[i_run], row[i_view], row[i_id])
            if key in out:
                dups[key[:2]] += 1
                continue
            out[key] = float(row[i_err]) if row[i_err] != "" else float("nan")
    return out, dups


def compare_per_record(table: dict, dups: Counter, run_id: str, view: str, ids, errors) -> dict:
    """Replayed per-record errors against the stored ones of (run_id, view), aligned by sample identity."""
    rel, missing = [], 0
    for sid, e in zip(ids, errors):
        s = table.get((run_id, view, str(sid)))
        if s is None:
            missing += 1
            continue
        rel.append(abs(float(e) - s) / max(abs(s), 1e-12) if finite(s) and finite(float(e)) else float("nan"))
    finite_rel = [x for x in rel if math.isfinite(x)]
    return {"compared": len(rel), "missing": missing, "repeated_rows": int(dups.get((run_id, view), 0)),
            "non_finite": len(rel) - len(finite_rel), "max_rel_diff": max(finite_rel) if finite_rel else None,
            "mean_rel_diff": (sum(finite_rel) / len(finite_rel)) if finite_rel else None}


def decide(views: dict, rtol: float, per_record_rtol: float | None = None) -> dict:
    """The verdict over every requested view: agrees / disagrees / incomplete, with the reasons.

    views: {label: {"replay_mean", "stored_mean" (None when no stored row), "records", "stored_records",
    "rel_diff", "per_record" (None or the dict of compare_per_record)}}."""
    disagree, incomplete, diffs, pr_diffs = [], [], [], []
    for label, v in views.items():
        if v.get("stored_mean") is None:
            incomplete.append(f"{label}: no stored score")
            continue
        if not finite(v.get("replay_mean")) or not finite(v.get("rel_diff")):
            disagree.append(f"{label}: non-finite replayed mean")
        else:
            diffs.append(float(v["rel_diff"]))
            if v["rel_diff"] > rtol:
                disagree.append(f"{label}: relative difference of the view mean {v['rel_diff']:.3g} exceeds {rtol:g}")
        if v.get("stored_records") is not None and v.get("records") != v["stored_records"]:
            incomplete.append(f"{label}: {v.get('records')} records replayed, {v['stored_records']} stored")
        pr = v.get("per_record")
        if per_record_rtol is not None:
            if not pr:
                incomplete.append(f"{label}: no per-record comparison")
                continue
            if pr.get("missing") or pr.get("repeated_rows"):
                incomplete.append(f"{label}: per-record rows missing {pr.get('missing', 0)}, repeated {pr.get('repeated_rows', 0)}")
            if pr.get("non_finite") or not finite(pr.get("max_rel_diff")):
                disagree.append(f"{label}: non-finite per-record difference")
            else:
                pr_diffs.append(float(pr["max_rel_diff"]))
                if pr["max_rel_diff"] > per_record_rtol:
                    disagree.append(f"{label}: per-record relative difference {pr['max_rel_diff']:.3g} exceeds {per_record_rtol:g}")
    status = "disagrees" if disagree else ("incomplete" if incomplete else "agrees")
    return {"status": status, "agrees": status == "agrees", "problems": disagree + incomplete, "compared_views": len(diffs),
            "missing_views": [l for l, v in views.items() if v.get("stored_mean") is None],
            "max_rel_diff": max(diffs) if diffs else None, "per_record_max_rel_diff": max(pr_diffs) if pr_diffs else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--suite", default="v2eval7", choices=("v2eval7", "n5nested"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--results", default="results/revision_october")
    ap.add_argument("--rtol", type=float, default=1e-4, help="tolerance of the relative difference of a view mean")
    ap.add_argument("--per-record", default=None, help="per-record export of the suite (csv / csv.gz): compare every record by identity")
    ap.add_argument("--per-record-rtol", type=float, default=1e-3, help="tolerance of the relative difference of a per-record error")
    ap.add_argument("--per-record-out", default=None, help="write the replayed per-record errors here (csv: run_id, view, sample_id, error)")
    ap.add_argument("--out", default=None, help="write the report JSON here")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="recipe override of a controlled variant trained with train_run.py --set (e.g. v4_norm=train_scalar)")
    ap.add_argument("--fft", default="auto", choices=("auto", "native", "cufft"),
                    help="inverse real 2-D FFT of FNO layers on CPU: cuFFT-compatible (auto on CPU, cufft) or the CPU library's")
    a = ap.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from pdeobs.dataset import collate_benchmark
    from pdeobs.runner import _training_config
    from pdeobs.training import _forward, prepare_batch_with_context
    from pdeobs.revision.data import apply_layout, input_setup, v2_config, v2_split
    from pdeobs.revision.fftcompat import cufft_c2r_on_cpu
    from pdeobs.revision.models import build_model
    from pdeobs.revision.v2masks import protocol_mask

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    recipe = recipe_of(a.run, os.path.join(a.repo_root, "configs/paper/october/recipes.yaml"))
    overrides = parse_overrides(a.set)
    recipe.update(overrides)
    cfg = v2_config(recipe, a.data_root, a.repo_root)
    test, receipt = v2_split(cfg, recipe, "test")
    input_setup(recipe, test)
    model = build_model(recipe, cfg)
    payload = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state)
    device = torch.device(a.device)
    model.to(device).eval()
    tconf = _training_config(cfg, Path(tempfile.gettempdir()) / "pdeobs-replay", None)   # inference only: nothing is written
    grid = int(recipe.get("grid") or 128)
    if a.suite == "v2eval7":
        views = []
        for v in V2EVAL7:
            try:
                protocol_mask(v, recipe["pde"], "feasibility", rep=0, n=grid)
            except ValueError:
                continue
            views.append((v, 0, f"{v}__rep0"))
    else:
        views = [(f"U{k}", q, f"U{k}__q{q}") for k in NESTED_K for q in range(3)]
    stored = stored_scores(a.run, a.suite, a.results)
    per_record = dups = None
    if a.per_record:
        per_record, dups = load_per_record(a.per_record, {s["run_id"] for s in stored.values()})
    fft = "cufft" if a.fft == "cufft" or (a.fft == "auto" and device.type == "cpu") else "native"
    report = {"run": a.run, "suite": a.suite, "recipe": recipe, "overrides": overrides, "test_receipt": receipt, "device": str(device),
              "torch": torch.__version__, "fft": fft if device.type == "cpu" else "native (cuda)", "rtol": a.rtol,
              "per_record_file": a.per_record, "per_record_rtol": a.per_record_rtol if a.per_record else None, "views": {}}
    rows_out = []
    with torch.no_grad(), (cufft_c2r_on_cpu() if fft == "cufft" else contextlib.nullcontext()):
        for proto, rep, label in views:
            apply_layout(test, proto, pde=recipe["pde"], rep=rep)
            errs, ids = [], []
            for b in DataLoader(test, batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate_benchmark):
                inputs, mask, target, geometry, metadata = prepare_batch_with_context(b, tconf, device)
                pred = _forward(model, inputs, mask, target, tconf, False, geometry, metadata)
                p = pred.double().cpu().numpy().reshape(len(metadata), -1)
                t = target.double().cpu().numpy().reshape(len(metadata), -1)
                errs.append(np.sqrt(((p - t) ** 2).sum(1)) / np.maximum(np.sqrt((t ** 2).sum(1)), 1e-12))
                ids.extend(str(m.get("sample_id")) for m in metadata)
            e = np.concatenate(errs)
            s = stored.get(label)
            mean = float(e.mean())
            entry = {"replay_mean": mean, "stored_mean": s["mean"] if s else None, "records": int(len(e)),
                     "stored_records": s["n"] if s else None, "stored_run_id": s["run_id"] if s else None,
                     "rel_diff": (abs(mean - s["mean"]) / max(abs(s["mean"]), 1e-300)) if s else None, "per_record": None}
            if per_record is not None and s:
                entry["per_record"] = compare_per_record(per_record, dups, s["run_id"], label, ids, e)
            if a.per_record_out:
                rows_out.extend((s["run_id"] if s else a.run, label, sid, float(x)) for sid, x in zip(ids, e))
            report["views"][label] = entry
    verdict = decide(report["views"], a.rtol, a.per_record_rtol if per_record is not None else None)
    report.update(verdict)
    report["seconds"] = round(time.time() - t0, 1)
    if a.per_record_out:
        with open(a.per_record_out, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run_id", "view", "sample_id", "error"])
            w.writerows(rows_out)
    txt = json.dumps(report, indent=1, default=str)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt)
    print(txt)
    return 0 if report["agrees"] else 1


if __name__ == "__main__":
    sys.exit(main())
