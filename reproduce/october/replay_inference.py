"""Weights-inference replay of an October 2026 result (reproduction path C).

The script:
1. rebuilds the model of a run from its frozen recipe (configs/paper/october/recipes.yaml);
2. loads its checkpoint;
3. re-predicts the run's test views on the same parents and per-parent sensor layouts;
4. scores every record with the full-field relative L2 (float64);
5. compares the view means with the stored scores in results/revision_october (runs_long).

    python reproduce/october/replay_inference.py --run V4C__poisson__ell009__fno__U256__s1101 \
        --checkpoint <dir>/checkpoints/last.pt --data-root <V2 store> [--suite v2eval7|n5nested] [--device cuda]

The V2 store is the directory that holds the V2 views (e.g. v2_poisson_ell009_ref128/). Scores are compared with a
relative tolerance (default 1e-4); bit-identical agreement is not expected across devices.

FNO checkpoints on a CPU: the October FNOs were trained and scored with cuFFT, whose large-batch inverse real FFT keeps
the imaginary parts of the kx = 0 / Nyquist bins that CPU FFT libraries drop (see pdeobs.revision.fftcompat). With
--fft auto (default) a CPU replay uses the cuFFT-compatible transform; --fft native shows the plain CPU result.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import json
import os
import sys
import tempfile
import time
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
    out = {}
    for f in sorted(glob.glob(os.path.join(results, "runs_long.part-*.csv"))):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if suite == "v2eval7" and r["run_id"] == run and r["suite"] == "v2eval7":
                out[r["view"]] = float(r["mean"])
            if suite == "n5nested" and r["suite"] == "n5nested" and r["parent_run_id"] == run:
                out[r["view"]] = float(r["mean"])
    return out


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
    ap.add_argument("--rtol", type=float, default=1e-4)
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
    fft = "cufft" if a.fft == "cufft" or (a.fft == "auto" and device.type == "cpu") else "native"
    report = {"run": a.run, "suite": a.suite, "recipe": recipe, "overrides": overrides, "test_receipt": receipt, "device": str(device),
              "torch": torch.__version__, "fft": fft if device.type == "cpu" else "native (cuda)", "views": {}}
    with torch.no_grad(), (cufft_c2r_on_cpu() if fft == "cufft" else contextlib.nullcontext()):
        for proto, rep, label in views:
            apply_layout(test, proto, pde=recipe["pde"], rep=rep)
            errs = []
            for b in DataLoader(test, batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate_benchmark):
                inputs, mask, target, geometry, metadata = prepare_batch_with_context(b, tconf, device)
                pred = _forward(model, inputs, mask, target, tconf, False, geometry, metadata)
                p = pred.double().cpu().numpy().reshape(len(metadata), -1)
                t = target.double().cpu().numpy().reshape(len(metadata), -1)
                errs.append(np.sqrt(((p - t) ** 2).sum(1)) / np.maximum(np.sqrt((t ** 2).sum(1)), 1e-12))
            e = np.concatenate(errs)
            ref = stored.get(label)
            report["views"][label] = {"replay_mean": float(e.mean()), "stored_mean": ref, "records": int(len(e)),
                                      "rel_diff": (abs(float(e.mean()) - ref) / max(abs(ref), 1e-300)) if ref is not None else None}
    diffs = [v["rel_diff"] for v in report["views"].values() if v["rel_diff"] is not None]
    report["max_rel_diff"] = max(diffs) if diffs else None
    report["compared_views"] = len(diffs)
    report["agrees"] = bool(diffs) and max(diffs) <= a.rtol
    report["seconds"] = round(time.time() - t0, 1)
    txt = json.dumps(report, indent=1, default=str)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt)
    print(txt)
    return 0 if report["agrees"] else 1


if __name__ == "__main__":
    sys.exit(main())
