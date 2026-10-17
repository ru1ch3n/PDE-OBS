"""Training reproduction of an October 2026 run from the public package and the public configs (reproduction path D).

The script:
1. builds the run's experiment configuration from its frozen recipe (configs/paper/october/recipes.yaml) and the V2 store;
2. selects the training split (3,000 parents per regime) and the run's fixed per-parent training layout (U / C / L / G);
3. for method-tuned (V4) recipes, computes the training-set scalar normaliser (mean / std of every training target value);
4. seeds and builds the network and trains it with pdeobs.training.Trainer for the recipe's epochs, without validation data
   and without early stopping, as the October runs did;
5. applies the October completion gate: every epoch, the expected number of optimizer steps, no training pathology,
   finite losses. A training pathology (e.g. constant-output collapse) ends the run as "failed", as in the October campaign.

    python reproduce/october/train_run.py --run V4C__poisson__ell009__fno__U256__s1101 --data-root <V2 store> --out <dir> \
        [--device cuda]

Evaluate the trained model with reproduce/october/replay_inference.py --checkpoint <dir>/checkpoints/last.pt (same run id).
The stored scores of the original run are then a reference, not a target: retraining is not bit-identical across hardware
or library versions. A run whose <dir>/checkpoints/last.pt exists is resumed from it (exact state: weights, optimizer,
scheduler, random generators). Mixture, jitter-augmented and parameter-channel recipes are not covered by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

from replay_inference import recipe_of


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=2, help="data-loader workers (no effect on the numbers)")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--max-train", type=int, default=None, help="smoke tests only: first N training records")
    ap.add_argument("--epochs", type=int, default=None, help="smoke tests only: override the recipe's epochs")
    a = ap.parse_args()
    import torch
    import pdeobs.training as T
    from pdeobs.one_setting import subset_dataset
    from pdeobs.runner import _loader, _training_config
    from pdeobs.revision.data import apply_layout, input_setup, train_scalar_stats, v2_config, v2_split
    from pdeobs.revision.models import build_model

    t0 = time.time()
    out = Path(a.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    recipe = recipe_of(a.run, os.path.join(a.repo_root, "configs/paper/october/recipes.yaml"))
    proto = recipe["protocol"]
    if proto.startswith(("MX", "GJ")) or recipe.get("param_channel") or recipe.get("records_per_epoch"):
        raise SystemExit(f"{a.run}: mixture / jitter / parameter-channel / pass-sampled recipes are not covered by this script")
    cfg = v2_config(recipe, a.data_root, a.repo_root)
    tr = cfg["training"]
    tr["num_workers"] = int(a.workers)
    tr["persistent_workers"] = a.workers > 0
    if a.epochs is not None:
        tr["epochs"] = int(a.epochs)
    if a.device == "cpu":
        tr["device"] = "cpu"
    train, receipt = v2_split(cfg, recipe, "train")
    norm = None
    if recipe.get("v4_norm"):
        norm = train_scalar_stats(train, recipe["pde"])              # all training records, before any smoke subset
        (out / "normaliser.json").write_text(json.dumps(norm, indent=1))
    if a.max_train:
        train = subset_dataset(train, list(range(min(a.max_train, len(train)))))
    apply_layout(train, proto, pde=recipe["pde"], rep=0)              # the parent's fixed layout, replicate 0
    input_setup(recipe, train)
    loader = _loader(train, cfg, shuffle=True)
    resume = out / "checkpoints/last.pt"
    resume = resume if resume.exists() and resume.stat().st_size > 0 else None
    if resume is not None:                                            # load on the CPU: exact resume on a GPU device
        frozen = T.load_checkpoint_payload
        T.load_checkpoint_payload = lambda path, *, map_location="cpu": frozen(path, map_location="cpu")
    T.seed_everything(int(recipe["seed"]), True)                      # initialization (a resume restores the saved state)
    model = build_model(recipe, cfg, norm={"mu": norm["mu"], "sigma": norm["sigma"]} if norm else None)
    tconf = _training_config(cfg, out, resume)
    trainer = T.Trainer(model, tconf)
    status, error = "failed", None
    try:
        trainer.fit(loader, None)
        bs = int(tr["batch_size"])
        expected = math.ceil(len(train) / bs) * int(tr["epochs"])
        hist = trainer.history
        ok = (len(hist) == int(tr["epochs"]) and trainer._optimizer_steps == expected and not trainer.health_events
              and all(math.isfinite(float(r["train_loss"])) for r in hist))
        status = "completed" if ok else "failed_gate"
    except RuntimeError as exc:                                       # training pathology: kept as a failure
        error = f"{type(exc).__name__}: {exc}"
        expected = None
    hist = trainer.history
    (out / "history.json").write_text(json.dumps({"history": hist, "health_events": trainer.health_events,
                                                  "health_warnings": trainer.health_warnings}, indent=1, default=str))
    last = out / "checkpoints/last.pt"
    rec = {"run": a.run, "status": status, "error": error, "epochs_done": len(hist), "epochs": int(tr["epochs"]),
           "optimizer_steps": trainer._optimizer_steps, "expected_optimizer_steps": expected,
           "final_train_loss": float(hist[-1]["train_loss"]) if hist else None, "train_records": len(train),
           "train_receipt": receipt, "normaliser": norm, "resumed": resume is not None, "smoke": bool(a.max_train or a.epochs),
           "checkpoint_sha256": sha256_file(last) if last.exists() else None, "recipe": recipe, "torch": torch.__version__,
           "training_config": {k: getattr(tconf, k, None) for k in ("epochs", "learning_rate", "batch_size", "amp", "loss",
                                                                     "health_policy", "early_stopping_patience", "deterministic",
                                                                     "seed", "checkpoint_every", "scheduler")},
           "device": torch.cuda.get_device_name(0) if a.device != "cpu" and torch.cuda.is_available() else "cpu",
           "seconds": round(time.time() - t0, 1)}
    (out / "completion.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: rec[k] for k in ("run", "status", "error", "epochs_done", "final_train_loss", "seconds")}))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
