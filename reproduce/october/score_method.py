"""Score any predictor on the October 2026 V2 views (and on the converted RPB data) with the revision's protocol.

This is the entry point for a method of your own: it builds the test split of a V2 view with the same per-parent sensor
layouts, input representation and float64 full-field scorer as `replay_inference.py`, but takes its predictions from a
callable instead of a checkpoint. Three built-in predictors come with it:

    zero         the masked field itself (hidden cells 0; temporal families: persistence of the last observed frame)
    nearest      the nearest-observation fill (`pdeobs.revision.fill`), the input representation of the Voronoi-CNN
    covariance   the covariance-weighted linear reconstruction (`pdeobs.revision.covariance`), fitted on the training
                 split, regularisation chosen on the validation split with the frozen candidate list of
                 configs/paper/october/classical.yaml (static families only)

    python reproduce/october/score_method.py --dataset-id v2_poisson_ell009_ref128 --data-root <V2 store> --method nearest
    python reproduce/october/score_method.py --dataset-id v2_poisson_ell009_ref128 --data-root <V2 store> --method covariance
    python reproduce/october/score_method.py --dataset-id v2_poisson_rpb_ref128 --data-root <store> --part test --method nearest
    python reproduce/october/score_method.py --dataset-id v2_ns_ell009_ref128 --data-root <V2 store> --method my_pkg.my_module:predict

A custom predictor is `module:callable` with the signature

    predict(inputs, mask, geometry, metadata) -> prediction

where every argument is what the October models receive (`pdeobs.training.prepare_batch_with_context`): static families
`inputs` [B, C, H, W] (observed values, hidden cells 0 or nearest-filled with --fill nearest), `mask` [B, 1, H, W] (1 =
observed), `geometry` [B, G, H, W]; temporal families `inputs` [B, T_hist, C, H, W] (the observed initial state) and the
prediction must hold the next `horizon` frames [B, horizon, C, H, W]. Tensors are torch float32 on --device; the
prediction must have the target's shape. The callable never sees the target. Scores are the mean over the split of the
per-record relative L2 of the complete predicted field (float64; temporal families: all predicted frames jointly), the
same statistic as `mean` in results/revision_october.

The V2 store is the directory that holds the views (`<dataset_id>[__t4]/`, from huggingface.co/datasets/ru1ch3n/PDE-OBS,
`october2026/v2/`). The RPB store is the output of `python -m pdeobs.revision.external.rpb` (parts `test` and `ood`): its
view is 64^2 (`v2_poisson_rpb_ref128__grid64/`), which the script selects for the rpb slice; `--grid` picks another side
(the N8 resolution views `<dataset_id>__grid64/` and `__grid256/`).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_inference import V2EVAL7  # noqa: E402

STATIC = ("poisson", "darcy", "helmholtz")
RPB_VIEWS = ("U256", "C256", "U1024", "C1024", "G256")


def parse_dataset_id(dataset_id: str) -> tuple[str, str]:
    p = dataset_id.split("_")
    if len(p) < 4 or p[0] != "v2":
        raise SystemExit(f"dataset id {dataset_id!r} is not v2_<pde>_<slice>_ref128")
    return {"ns": "navier_stokes"}.get(p[1], p[1]), p[2]


def recipe_for(dataset_id: str, fill: str | None, grid: int | None = None) -> dict:
    """A minimal recipe: the frozen FNO settings of the slice give the data configuration; no network is built.

    `grid` is the side of the view's grid: 128 for the V2 views (default), 64 for the converted RPB data (default when
    the slice is rpb; its view directory is v2_poisson_rpb_ref128__grid64), 64 or 256 for the N8 resolution views
    (<dataset_id>__grid<N>)."""
    pde, ell = parse_dataset_id(dataset_id)
    if grid is None:
        grid = 64 if ell == "rpb" else 128
    rec = {"id": f"score_method__{dataset_id}", "pde": pde, "dataset_id": dataset_id, "method": "fno", "seed": 0, "grid": int(grid),
           "protocol": "U256", "n_train": 9000}
    if fill == "nearest":
        rec["fill_input"] = "nearest"
    return rec


def load_predictor(spec: str):
    if spec in ("zero", "nearest", "covariance"):
        return spec
    mod, _, fn = spec.partition(":")
    if not fn:
        raise SystemExit("--method must be zero, nearest, covariance or module:callable")
    return getattr(importlib.import_module(mod), fn)


def persistence(inputs, target):
    """Static: the (filled) observation itself; temporal: the last observed frame repeated over the horizon."""
    if target.ndim == 5:
        last = inputs[:, -1:]
        return last.expand(-1, target.shape[1], -1, -1, -1).clone()
    return inputs.clone()


def fit_covariance(cfg, recipe, pde, repo_root, regs, views, workers, log):
    """Fit on the training split, choose reg on the validation split (mean relative L2 over the given views)."""
    from pdeobs.revision.covariance import CovarianceReconstructor
    from pdeobs.revision.data import apply_layout, v2_split
    train, _ = v2_split(cfg, recipe, "train")
    t0 = time.time()
    X = np.stack([np.asarray(train[i]["target"], dtype=np.float64).reshape(-1) for i in range(len(train))])
    rec = CovarianceReconstructor(rank=2048).fit(X)
    del X
    log(f"covariance fitted on {len(train)} training records in {time.time() - t0:.0f} s (modes {rec.U.shape[1]})")
    val, _ = v2_split(cfg, recipe, "validation")
    scores = np.zeros(len(regs))
    n = 0
    for v in views:
        apply_layout(val, v, pde=pde, rep=0)
        for i in range(len(val)):
            item = val[i]
            t = np.asarray(item["target"], dtype=np.float64).reshape(-1)
            m = np.asarray(item["mask"]).reshape(-1) > 0.5
            for k, c in enumerate(rec.coefficients(t[m], m, regs)):
                u = rec.mu + rec.U @ c
                scores[k] += float(np.linalg.norm(u - t) / max(np.linalg.norm(t), 1e-12))
            n += 1
    scores /= max(n, 1)
    best = int(np.argmin(scores))
    rec.reg = float(regs[best])
    log(f"reg chosen on validation ({n} records over {views}): {rec.reg} (scores {dict(zip(map(str, regs), map(float, scores)))})")
    return rec, {"rank": int(rec.U.shape[1]), "reg": rec.reg, "validation_scores": {str(r): float(s) for r, s in zip(regs, scores)},
                 "selection_views": list(views), "validation_records": n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-id", required=True, help="v2_<pde>_<slice>_ref128, e.g. v2_poisson_ell009_ref128 or v2_poisson_rpb_ref128")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--method", default="nearest", help="zero | nearest | covariance | module:callable")
    ap.add_argument("--fill", default=None, choices=(None, "nearest"), help="nearest-fill the observation channel before a custom predictor")
    ap.add_argument("--part", default="test", choices=("test", "confirm", "ood"))
    ap.add_argument("--grid", type=int, default=None,
                    help="grid side of the view: 128 for the V2 views (default), 64 for the converted RPB data (default for the rpb "
                         "slice), 64 or 256 for the N8 resolution views <dataset_id>__grid<N>")
    ap.add_argument("--views", nargs="*", default=None, help="layouts to score (default: the seven v2eval7 views that exist on the grid)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-records", type=int, default=None, help="smoke tests only: score the first N records of the split")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out", default=None, help="write the report JSON here")
    a = ap.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from pdeobs.dataset import collate_benchmark
    from pdeobs.one_setting import subset_dataset
    from pdeobs.runner import _training_config
    from pdeobs.training import prepare_batch_with_context
    from pdeobs.revision.data import apply_layout, input_setup, v2_config, v2_split
    from pdeobs.revision.v2masks import protocol_mask

    t0 = time.time()
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    pde, ell = parse_dataset_id(a.dataset_id)
    predictor = load_predictor(a.method)
    fill = "nearest" if (predictor == "nearest" or a.fill == "nearest") else None
    if predictor == "covariance" and pde not in STATIC:
        raise SystemExit("the covariance reconstruction is a static reference; use it on Poisson / Darcy views")
    recipe = recipe_for(a.dataset_id, fill, a.grid)
    cfg = v2_config(recipe, a.data_root, a.repo_root)
    if a.device == "cpu":
        cfg["training"]["device"] = "cpu"
    views = a.views
    if not views:
        views = []
        for v in (RPB_VIEWS if ell == "rpb" else V2EVAL7):
            try:
                protocol_mask(v, pde, "feasibility", rep=0, n=int(recipe["grid"]))
            except ValueError:
                continue
            views.append(v)
    cov_info = None
    if predictor == "covariance":
        classical = yaml.safe_load(open(os.path.join(a.repo_root, "configs/paper/october/classical.yaml"), encoding="utf-8"))["covariance"]
        sel_views = [v for v in ("U256", "C256", "L256") if v in views] or views[:1]
        predictor, cov_info = fit_covariance(cfg, recipe, pde, a.repo_root, [float(r) for r in classical["regs"]], sel_views, a.workers, log)
    test, receipt = v2_split(cfg, recipe, a.part)
    input_setup(recipe, test)
    if a.max_records:
        test = subset_dataset(test, list(range(min(a.max_records, len(test)))))
    tconf = _training_config(cfg, Path(tempfile.gettempdir()) / "pdeobs-score-method", None)
    device = torch.device(a.device)
    report = {"dataset_id": a.dataset_id, "part": a.part, "grid": int(recipe["grid"]), "method": a.method, "fill": fill, "receipt": receipt,
              "covariance": cov_info, "device": str(device), "torch": torch.__version__, "views": {}}
    with torch.no_grad():
        for v in views:
            apply_layout(test, v, pde=pde, rep=0)
            errs = []
            for b in DataLoader(test, batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate_benchmark):
                inputs, mask, target, geometry, metadata = prepare_batch_with_context(b, tconf, device)
                if predictor in ("zero", "nearest"):
                    pred = persistence(inputs, target)
                elif hasattr(predictor, "coefficients"):
                    out = []
                    for i in range(target.shape[0]):
                        m = mask[i, 0].double().cpu().numpy().reshape(-1) > 0.5
                        z = target[i].double().cpu().numpy().reshape(-1)[m]          # the observed values only
                        out.append(predictor.predict(z, m).reshape(target.shape[1:]))
                    pred = torch.as_tensor(np.stack(out), dtype=torch.float32, device=device)
                else:
                    pred = predictor(inputs, mask, geometry, metadata)
                    pred = torch.as_tensor(pred, dtype=torch.float32, device=device)
                if tuple(pred.shape) != tuple(target.shape):
                    raise SystemExit(f"prediction shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
                p = pred.double().cpu().numpy().reshape(len(metadata), -1)
                t = target.double().cpu().numpy().reshape(len(metadata), -1)
                errs.append(np.sqrt(((p - t) ** 2).sum(1)) / np.maximum(np.sqrt((t ** 2).sum(1)), 1e-12))
            e = np.concatenate(errs)
            report["views"][f"{v}__rep0"] = {"mean": float(e.mean()), "median": float(np.median(e)), "p90": float(np.quantile(e, 0.9)),
                                            "max": float(e.max()), "n": int(len(e))}
            log(f"{v}: mean {e.mean():.4f} median {np.median(e):.4f} (n={len(e)})")
    report["seconds"] = round(time.time() - t0, 1)
    txt = json.dumps(report, indent=1, default=str)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
