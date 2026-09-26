"""PDE-OBS v0.2.0 full-coverage sweep on one GPU node.

Not the cartesian product (that is ~10^5 runs); every dimension VALUE and every declared feature and
error path is exercised at least once, minimal depth (a couple of optimizer steps, tiny grids), and
the result is a supported/unsupported table. Each tier writes qa_full/T<n>.json immediately so a
preempted (interruptible) node can resume with --tiers.

usage: python remote_full_coverage.py <workdir> [--tiers T1,T2,...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

WORK = Path(sys.argv[1])
QA = WORK / "qa_full"
QA.mkdir(parents=True, exist_ok=True)
ROOT = WORK / "work" / "full"
ROOT.mkdir(parents=True, exist_ok=True)

PDES = ("darcy", "poisson", "helmholtz", "heat", "reaction_diffusion", "burgers", "navier_stokes")
STATIC = ("darcy", "poisson", "helmholtz")
TEMPORAL = ("heat", "reaction_diffusion", "burgers", "navier_stokes")
BOUNDARIES = ("dirichlet", "neumann", "periodic", "robin_obstacle")
SETTINGS = ("smooth_grf", "medium_grf", "rough_grf", "low_frequency_fourier", "multi_frequency_fourier",
            "gaussian_blobs", "piecewise_blocks", "threshold_level_set", "dipole_vortex_pair", "front_ring_shock")
REGIMES = ("low", "medium", "high")
MAIN = ("fno", "pino", "ufno", "cno", "deeponet", "gnot", "transolver")
OTHER_NEURAL = ("unet", "unet_paper_guided", "convlstm", "mae_small")
CLASSICAL = ("zero", "mean", "nearest", "bilinear", "rbf", "persistence", "gappy_pod")
GENERAL_PROTOCOLS = ("random", "grid", "block", "line", "horizontal", "vertical", "boundary", "clustered", "full")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write(name, payload):
    (QA / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attempt(fn, *a, **k):
    """Run fn; return (status, value_or_error)."""
    try:
        return "ok", fn(*a, **k)
    except Exception as exc:  # noqa: BLE001
        return "error", f"{type(exc).__name__}: {str(exc)[:300]}"


def expect_error(fn, *a, **k):
    try:
        fn(*a, **k)
        return {"status": "FAILED_no_error"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "rejected_ok", "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


# --------------------------------------------------------------------------- T1
def t1_generation():
    """Every PDE x boundary x setting x regime at 16^2 (1 sample) + user-array route."""
    from pdeobs import api
    out = {"tier": "T1", "what": "numerical generation coverage", "utc": now(), "combos": {}, "arrays": {}}
    root = ROOT / "t1"
    t0 = time.perf_counter()
    for pde in PDES:
        for b in BOUNDARIES:
            for s in SETTINGS:
                for r in REGIMES:
                    key = f"{pde}/{b}/{s}/{r}"
                    d = root / pde / b / s / r
                    st, v = attempt(api.create_dataset, pde=pde, out=d, boundary=b, setting=s, regime=r, num_samples=1,
                                    resolution=16, seed=1, time_steps=3 if pde in TEMPORAL else None, overwrite=True)
                    out["combos"][key] = {"status": "supported" if st == "ok" else "unsupported_or_error",
                                          **({"shape": [v.time_steps, *v.resolution, v.state_channels]} if st == "ok" else {"error": v})}
        write("T1", out)
        print(f"T1 {pde} done", flush=True)
    # user-supplied arrays through the existing kernels
    n = 16
    ax = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(ax, ax)
    src = np.stack([np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * (k + 1) for k in range(2)])
    for pde, arrays, physical in (("poisson", {"source": src}, {}), ("helmholtz", {"source": src}, {"reaction": 2.0}),
                                  ("darcy", {"source": src, "coefficient": 1 + 0.3 * np.cos(2 * np.pi * x)[None].repeat(2, 0)}, {}),
                                  ("heat", {"initial_state": src}, {"diffusivity": 0.02, "dt": 0.1, "steps": 3})):
        st, v = attempt(api.create_dataset_from_arrays, pde=pde, out=root / "arrays" / pde, boundary="periodic", arrays=arrays,
                        physical=physical, overwrite=True)
        out["arrays"][pde] = {"status": st, **({"samples": v.sample_count, "T": v.time_steps} if st == "ok" else {"error": v})}
    out["arrays"]["burgers_rejected"] = expect_error(api.create_dataset_from_arrays, pde="burgers", out=root / "arrays" / "x",
                                                     boundary="periodic", arrays={"initial_state": src}, physical={}, overwrite=True)
    out["arrays"]["darcy_without_coefficient_rejected"] = expect_error(api.create_dataset_from_arrays, pde="darcy", out=root / "arrays" / "y",
                                                                       boundary="periodic", arrays={"source": src}, physical={}, overwrite=True)
    out["download_refused"] = expect_error(api.load_dataset, "nowhere", source="download")
    sup = sum(1 for c in out["combos"].values() if c["status"] == "supported")
    out["summary"] = {"combos": len(out["combos"]), "supported": sup, "unsupported_or_error": len(out["combos"]) - sup,
                      "seconds": round(time.perf_counter() - t0, 1)}
    write("T1", out)
    return out["summary"]


# --------------------------------------------------------------------------- T2
def t2_observations():
    from pdeobs import api
    from pdeobs.api import specs as S
    from pdeobs.masks import generate_mask
    out = {"tier": "T2", "what": "observation protocol coverage", "utc": now(), "general": {}, "paper": {}, "errors": {}, "invariants": {}}
    variants = {"random": [{}, {"ratio": 0.5}, {"count": 37}], "grid": [{}, {"ratio": 0.1}, {"spacing": 4, "random_offset": True}],
                "block": [{}, {"missing_fraction": 0.49}, {"block_shape": 6}], "line": [{}, {"num_lines": 6, "orientation": "both"}],
                "horizontal": [{}, {"num_lines": 4}], "vertical": [{}, {"num_lines": 4}], "boundary": [{}, {"width": 3}, {"width": 2, "count": 40}],
                "clustered": [{}, {"ratio": 0.2, "clusters": 2, "spread": 0.1}], "full": [{}]}
    for res in (16, 32, 64, 128):
        for name, vs in variants.items():
            for params in vs:
                key = f"{name}@{res}:{json.dumps(params, sort_keys=True)}"
                try:
                    spec = api.make_observation(name, **params)
                    c1 = api.realized_count(spec, (res, res), seed=1)
                    c2 = api.realized_count(spec, (res, res), seed=1)
                    c3 = api.realized_count(spec, (res, res), seed=2)
                    out["general"][key] = {"status": "ok", "count": c1, "ratio": round(c1 / res / res, 4), "deterministic": c1 == c2,
                                           "seed_sensitive_count": c3 != c1, "mask_config": spec.mask_config}
                except Exception as exc:  # noqa: BLE001
                    out["general"][key] = {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    for view, info in S.PAPER_VIEWS.items():
        spec = api.make_observation(f"paper:{view}")
        c = api.realized_count(spec, (128, 128), seed=0)
        out["paper"][view] = {"label": info["label"], "expected_128": info["expected_count_128"], "realized_128": c,
                                 "exact": c == info["expected_count_128"]}
    out["errors"]["conflict_ratio_count"] = expect_error(api.make_observation, "random", ratio=0.5, count=10)
    out["errors"]["unknown_param"] = expect_error(api.make_observation, "random", clusters=3)
    out["errors"]["full_with_param"] = expect_error(api.make_observation, "full", ratio=0.5)
    out["errors"]["paper_view_in_general_ns"] = expect_error(api.make_observation, "random_65pct")
    out["errors"]["paper_view_with_param"] = expect_error(api.make_observation, "paper:R65", ratio=0.1)
    out["errors"]["boundary_width_too_large"] = expect_error(api.realized_count, api.make_observation("boundary", width=40), (16, 16))
    # custom registered factory + stored namespace
    from pdeobs.registry import MASK_REGISTRY
    MASK_REGISTRY.register("cov_checker", replace=True)(lambda shape, rng, **k: np.indices(shape).sum(0) % 2 == 0)
    cs = api.make_observation("cov_checker", namespace="custom")
    out["general"]["custom:cov_checker@16"] = {"count": api.realized_count(cs, (16, 16)), "status": "ok"}
    out["general"]["stored"] = {"status": "ok", **api.make_observation("stored").to_dict()}
    # invariants: observation does not mutate the record; switching views does not re-solve; mask != geometry
    from pdeobs.dataset import BenchmarkDataset
    h = api.load_dataset(ROOT / "t1" / "poisson" / "periodic" / "smooth_grf" / "low")
    from pdeobs.storage import LazyHDF5Dataset
    rd = LazyHDF5Dataset(h.shards)
    before = rd[0].trajectory.copy(); rd.close()
    for spec in (api.make_observation("random", ratio=0.3), api.make_observation("block", missing_fraction=0.4)):
        ds = BenchmarkDataset(h.shards, task="recovery", mask=dict(spec.mask_config), seed=3)
        row = ds[0]; ds.close()
        obs_ok = np.array_equal(row["target"], before[0]) and np.allclose(row["observations"][row["mask"][..., 0] > 0], row["target"][row["mask"][..., 0] > 0])
        out["invariants"][spec.name] = {"record_unchanged": bool(np.array_equal(row["target"], before[0])), "observed_equals_target": bool(obs_ok),
                                        "mask_is_not_geometry": bool(not np.array_equal(row["mask"], row["geometry"]))}
    rd2 = LazyHDF5Dataset(h.shards); after = rd2[0].trajectory.copy(); rd2.close()
    out["invariants"]["stored_record_bytes_unchanged"] = bool(np.array_equal(before, after))
    out["summary"] = {"general_ok": sum(1 for v in out["general"].values() if v.get("status") == "ok"), "general_total": len(out["general"]),
                      "paper_exact": all(v["exact"] for v in out["paper"].values()),
                      "errors_rejected": all(v.get("status") == "rejected_ok" for v in out["errors"].values())}
    write("T2", out)
    return out["summary"]


# --------------------------------------------------------------------------- helpers for training tiers
def _ds(api, pde, res=16, samples=4, boundary=None, setting="smooth_grf", regime="low", frames=4, seed=11, tag=""):
    task = "rollout" if pde in TEMPORAL else "recovery"
    boundary = boundary or ("periodic" if pde in TEMPORAL else "dirichlet")
    d = ROOT / "data" / f"{pde}-{boundary}-{setting}-{regime}-{res}-{samples}{tag}"
    if (d / "dataset.json").is_file():
        return api.load_dataset(d), task
    h = api.create_dataset(pde=pde, out=d, boundary=boundary, setting=setting, regime=regime, num_samples=samples, resolution=res,
                           seed=seed, time_steps=frames if pde in TEMPORAL else None, overwrite=True)
    return h, task


def _flow(api, handle, task, model, out, *, preset="smoke", params=None, steps=2, device="cuda", obs=None, bs=2, views=1,
          horizon=None, state_representation=None, training=None):
    obs = obs or api.make_observation("random", ratio=0.5)
    horizon = horizon if horizon is not None else (3 if task == "rollout" else None)
    run = api.train(handle, task=task, model=model, preset=preset, model_params=params, observation=obs, out=out,
                    budget={"max_steps": steps}, batch_size=bs, device=device, history_steps=1, horizon=horizon,
                    state_representation=state_representation, training=training, overwrite=True)
    pred = api.load_predictor(run.artifact.path, device=device)
    package, targets = api.inference_input_from_dataset(handle, obs, task=task, history_steps=1, state_representation=state_representation)
    b = api.predict(pred, package, horizon=horizon)
    rep = api.evaluate(b, targets)
    e = {"status": "passed" if rep.get("status") == "valid" and run.resolved.get("loss_finite", True) else "failed", "params": run.summary["parameters"]["total"],
         "steps": run.optimizer_steps, "pred_shape": list(b.predictions.shape), "score_status": rep.get("status"),
         "rel_l2": (rep.get("summary") or {}).get("rel_l2_joint_mean"), "peak_gpu_mb": round((run.resolved.get("peak_gpu_memory_bytes") or 0) / 1e6, 1),
         "train_s": run.resolved.get("training_seconds"), "artifact": str(run.artifact.path)}
    if views > 1:
        vs = [api.make_observation("random", ratio=0.5), api.make_observation("block", missing_fraction=0.5)]
        e["views"] = {v.name: api.evaluate_records(pred, handle, v, out=Path(out) / "eval" / v.name, batch_size=1, device=device).get("status") for v in vs[:views]}
    return e


# --------------------------------------------------------------------------- T3
def t3_tasks():
    from pdeobs import api
    out = {"tier": "T3", "what": "task interfaces x models (neural + classical)", "utc": now(), "flows": {}}
    hp, _ = _ds(api, "poisson"); hd, _ = _ds(api, "darcy"); hh, _ = _ds(api, "heat")
    cases = [("recovery", hp, "fno"), ("forward", hp, "fno"), ("inverse", hd, "fno"), ("rollout", hh, "fno"),
             ("recovery", hp, "pino"), ("forward", hp, "pino"), ("rollout", hh, "pino")]
    for task, h, model in cases:
        k = f"{model}:{task}"
        st, v = attempt(_flow, api, h, task, model, ROOT / "t3" / k, steps=2)
        out["flows"][k] = v if st == "ok" else {"status": "failed", "error": v}
        print("T3", k, out["flows"][k].get("status"), flush=True)
    out["flows"]["pino:inverse_rejected"] = expect_error(api.resolve_model, "pino", task="inverse")
    out["flows"]["convlstm:recovery_rejected"] = expect_error(api.resolve_model, "convlstm", task="recovery")
    for m in ("zero", "mean", "nearest", "bilinear", "rbf"):
        for task, h in (("recovery", hp), ("forward", hp), ("inverse", hd)):
            k = f"{m}:{task}"
            st, v = attempt(_flow, api, h, task, m, ROOT / "t3" / k, steps=1, device="cpu", preset=None)
            out["flows"][k] = v if st == "ok" else {"status": "failed", "error": v}
    st, v = attempt(_flow, api, hh, "rollout", "persistence", ROOT / "t3" / "persistence", steps=1, device="cpu", preset=None)
    out["flows"]["persistence:rollout"] = v if st == "ok" else {"status": "failed", "error": v}
    st, v = attempt(_flow, api, hp, "recovery", "gappy_pod", ROOT / "t3" / "gappy", steps=1, device="cpu", preset="smoke")
    out["flows"]["gappy_pod:recovery(fitted)"] = v if st == "ok" else {"status": "failed", "error": v}
    out["summary"] = {"passed": sum(1 for e in out["flows"].values() if e.get("status") in ("passed", "rejected_ok")), "total": len(out["flows"])}
    write("T3", out)
    return out["summary"]


# --------------------------------------------------------------------------- T4
def t4_models():
    from pdeobs import api
    from pdeobs.api import specs as S
    out = {"tier": "T4", "what": "every public model: presets, custom, illegal, geometry variants, multichannel", "utc": now(), "models": {}}
    hp, _ = _ds(api, "poisson", res=32); hh, _ = _ds(api, "heat", res=32)
    hp64, _ = _ds(api, "poisson", res=64); hh64, _ = _ds(api, "heat", res=64)  # paper structures need >= 64^2 (pino modes=20)
    illegal = {"fno": {"width": 36}, "pino": {"width": 36}, "gnot": {"hidden": 30, "heads": 4}, "transolver": {"hidden": 30, "heads": 4},
               "cno": {"depth": 3}, "ufno": {"layers": 4}, "deeponet": {"width": 5}, "unet": {"levels": 2}, "unet_paper_guided": {"modes": 4},
               "convlstm": {"width": 8}, "mae_small": {"modes": 4}}
    for m in MAIN + OTHER_NEURAL:
        spec = S.MODEL_SPECS[m]
        task = "rollout" if "rollout" in spec.tasks and ("recovery" not in spec.tasks) else "recovery"
        h = hh if task == "rollout" else hp
        e = {"tasks": list(spec.tasks), "presets": {}}
        for preset in [p for p in ("smoke", "paper") if p in spec.presets]:
            hsel = (hh64 if task == "rollout" else hp64) if preset == "paper" else h
            st, v = attempt(_flow, api, hsel, task, m, ROOT / "t4" / f"{m}-{preset}", preset=preset, steps=1, bs=2)
            e["presets"][preset] = v if st == "ok" else {"status": "failed", "error": v}
        e["illegal"] = expect_error(api.resolve_model, m, task=task, params=illegal[m])
        # custom non-default structure changes parameter count
        cfg_s = api.resolve_model(m, task=task, preset="smoke") if "smoke" in spec.presets else api.resolve_model(m, task=task)
        alt = {"fno": {"width": 24, "modes": 4, "layers": 3}, "pino": {"width": 24, "modes": 4, "layers": 3}, "ufno": {"width": 20, "modes": 4},
               "cno": {"width": 12}, "deeponet": {"hidden": 24, "latent": 24}, "gnot": {"hidden": 24, "layers": 2, "heads": 2},
               "transolver": {"hidden": 24, "layers": 2, "heads": 2, "slices": 4}, "unet": {"width": 6}, "unet_paper_guided": {"width": 6, "levels": 3},
               "convlstm": {"hidden_channels": 6}, "mae_small": {"width": 6, "latent_channels": 12, "patch_size": 4}}[m]
        cfg_a = api.resolve_model(m, task=task, params={**cfg_s.kwargs, **alt})
        e["param_count_smoke"] = api.count_parameters(api.build_model(cfg_s))["total"]
        e["param_count_custom"] = api.count_parameters(api.build_model(cfg_a))["total"]
        e["custom_changes_structure"] = e["param_count_smoke"] != e["param_count_custom"]
        e["status"] = "passed" if all(p.get("status") == "passed" for p in e["presets"].values()) and e["illegal"]["status"] == "rejected_ok" and e["custom_changes_structure"] else "failed"
        out["models"][m] = e
        print("T4", m, e["status"], flush=True)
    # geometry_channels=0 variant, and rollout through the autoregressive wrapper for a non-main model
    st, v = attempt(_flow, api, hp, "recovery", "fno", ROOT / "t4" / "fno-nogeom", preset="smoke", params={"geometry_channels": 0}, steps=1)
    out["models"]["fno[geometry_channels=0]"] = v if st == "ok" else {"status": "failed", "error": v}
    st, v = attempt(_flow, api, hh, "rollout", "unet", ROOT / "t4" / "unet-rollout", preset="smoke", steps=1)
    out["models"]["unet:rollout(autoregressive)"] = v if st == "ok" else {"status": "failed", "error": v}
    # NS velocity (2 channels) and vorticity representations
    hn, _ = _ds(api, "navier_stokes", res=32)
    for rep in ("vorticity", "velocity"):
        st, v = attempt(_flow, api, hn, "rollout", "fno", ROOT / "t4" / f"ns-{rep}", preset="smoke", steps=1, state_representation=rep)
        out["models"][f"fno:navier_stokes[{rep}]"] = v if st == "ok" else {"status": "failed", "error": v}
    out["models"]["upstream_wrapper_gated"] = expect_error(api.resolve_model, "paper_fno", task="recovery")
    out["summary"] = {"passed": sum(1 for e in out["models"].values() if e.get("status") in ("passed", "rejected_ok")), "total": len(out["models"])}
    write("T4", out)
    return out["summary"]


# --------------------------------------------------------------------------- T5
def t5_views_gpu():
    from pdeobs import api
    from pdeobs.api import specs as S
    out = {"tier": "T5", "what": "7 main models x all observation views at 128^2 (paper nine + general)", "utc": now(), "flows": {}}
    hp, _ = _ds(api, "poisson", res=128, samples=4); hh, _ = _ds(api, "heat", res=128, samples=4)
    general = [api.make_observation(n, **p) for n, p in (("random_1pct", {}), ("random", {"ratio": 0.05}), ("random", {"ratio": 0.10}),
                                                          ("grid", {"ratio": 0.03}), ("line", {"ratio": 0.03}), ("boundary", {"width": 2}),
                                                          ("clustered", {"ratio": 0.03}), ("full", {}))]
    for m in MAIN:
        for h, task in ((hp, "recovery"), (hh, "rollout")):
            k = f"{m}:{task}@128"
            try:
                run = api.train(h, task=task, model=m, preset="paper", observation=api.make_observation("random", ratio=0.5),
                                out=ROOT / "t5" / k, budget={"max_steps": 2}, batch_size=2, device="cuda", history_steps=1,
                                horizon=3 if task == "rollout" else None, overwrite=True)
                pred = api.load_predictor(run.artifact.path, device="cuda")
                paper = api.evaluate_views(pred, h, out=ROOT / "t5" / k / "paper", batch_size=1, device="cuda")
                gen = api.evaluate_views(pred, h, general, out=ROOT / "t5" / k / "general", batch_size=1, device="cuda", namespace="general")
                out["flows"][k] = {"status": "passed" if paper["all_valid"] and gen["all_valid"] else "failed",
                                   "paper": {n: v["status"] for n, v in paper["views"].items()},
                                   "general": {n: v["status"] for n, v in gen["views"].items()},
                                   "peak_gpu_mb": round((run.resolved.get("peak_gpu_memory_bytes") or 0) / 1e6, 1)}
            except Exception as exc:  # noqa: BLE001
                out["flows"][k] = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:300]}", "trace": traceback.format_exc()[-800:]}
            print("T5", k, out["flows"][k]["status"], flush=True)
            write("T5", out)
    out["summary"] = {"passed": sum(1 for e in out["flows"].values() if e.get("status") == "passed"), "total": len(out["flows"])}
    write("T5", out)
    return out["summary"]


# --------------------------------------------------------------------------- T6
def t6_boundary_regime():
    from pdeobs import api
    out = {"tier": "T6", "what": "fno smoke across boundary x regime for a static and a temporal PDE", "utc": now(), "flows": {}}
    for pde in ("poisson", "heat"):
        for b in BOUNDARIES:
            for r in REGIMES:
                k = f"{pde}/{b}/{r}"
                st, h = attempt(_ds, api, pde, 32, 4, b, "smooth_grf", r)
                if st != "ok":
                    out["flows"][k] = {"status": "unsupported_data", "error": h}
                    continue
                handle, task = h
                st, v = attempt(_flow, api, handle, task, "fno", ROOT / "t6" / k.replace("/", "-"), preset="smoke", steps=1)
                out["flows"][k] = v if st == "ok" else {"status": "failed", "error": v}
                print("T6", k, out["flows"][k].get("status"), flush=True)
    write("T6", out)
    out["summary"] = {"passed": sum(1 for e in out["flows"].values() if e.get("status") == "passed"), "total": len(out["flows"])}
    write("T6", out)
    return out["summary"]


# --------------------------------------------------------------------------- T7
def t7_splits_budgets():
    from pdeobs import api
    from pdeobs.dataset import BenchmarkDataset
    out = {"tier": "T7", "what": "splits (paper 2000-record, holdout, explicit, stored), budgets, identities, overwrite", "utc": now()}
    d = ROOT / "data" / "poisson-2000"
    if not (d / "high" / "dataset.json").is_file():  # 667 + 667 + 666 records over the three regimes
        for regime, n in (("low", 667), ("medium", 667), ("high", 666)):
            api.create_dataset(pde="poisson", out=d / regime, boundary="dirichlet", setting="smooth_grf", regime=regime, num_samples=n,
                               resolution=16, seed=20260804, shard_size=500, overwrite=True)
    h = api.load_dataset(d)
    out["records"] = h.sample_count
    ds = BenchmarkDataset(h.shards, task="recovery", mask={"protocol": "random_3pct", "ratio": 0.5}, seed=0)
    try:
        meta = list(ds.metadata)
    finally:
        ds.close()
    ident = "poisson|dirichlet|smooth_grf"
    p = api.plan_split(meta, "paper", seed=20260804, identity=ident)
    out["paper_split"] = {"train": len(p.train_ids), "test": len(p.test_ids), "overlap": len(set(p.train_ids) & set(p.test_ids)),
                             "receipt_schema": p.receipt.get("schema_version"), "regime_counts": p.receipt.get("regime_counts")}
    ho = api.plan_split(meta, "holdout:0.2", seed=1, identity=ident)
    out["holdout"] = {"train": len(ho.train_ids), "test": len(ho.test_ids), "overlap": len(set(ho.train_ids) & set(ho.test_ids))}
    stv = api.plan_split(meta, "stored", seed=1, identity=ident)
    out["stored"] = {"train": len(stv.train_ids), "test": len(stv.test_ids)}
    ex = api.plan_split(meta, {"train_ids": [m["sample_id"] for m in meta[:5]], "test_ids": [m["sample_id"] for m in meta[5:7]]}, seed=1, identity=ident)
    out["explicit"] = {"train": len(ex.train_ids), "test": len(ex.test_ids)}
    out["explicit_overlap_rejected"] = expect_error(api.plan_split, meta, {"train_ids": [meta[0]["sample_id"]], "test_ids": [meta[0]["sample_id"]]}, seed=1, identity=ident)
    out["paper_split_wrong_count_rejected"] = expect_error(api.plan_split, meta[:100], "paper", seed=1, identity=ident)
    obs = api.make_observation("random", ratio=0.5)
    r1 = api.train(h, task="recovery", model="fno", preset="smoke", observation=obs, out=ROOT / "t7" / "paper-split", budget={"max_steps": 2},
                   split="paper", batch_size=4, device="cuda", overwrite=True)
    out["train_on_paper_split"] = {"train_records": r1.resolved["split"]["train_records"], "test_records": r1.resolved["split"]["test_records"], "steps": r1.optimizer_steps}
    r2 = api.train(h, task="recovery", model="fno", preset="smoke", observation=obs, out=ROOT / "t7" / "epochs", budget={"epochs": 1},
                   batch_size=64, device="cuda", max_samples=128, overwrite=True)
    out["epoch_budget"] = {"epochs": len(r2.history), "steps": r2.optimizer_steps, "expected_steps": 2}
    # paper structures (fno modes=12) need a grid >= 24^2; identity/origin checks run on the 64^2 poisson records
    h64, _ = _ds(api, "poisson", res=64)
    r3 = api.train(h64, task="recovery", model="fno", preset="paper", observation=obs, out=ROOT / "t7" / "paper-preset", budget={"max_steps": 1},
                   batch_size=4, device="cuda", max_samples=8, overwrite=True)
    r4 = api.train(h64, task="recovery", model="fno", preset="paper", model_params={"width": 16}, observation=obs, out=ROOT / "t7" / "custom",
                   budget={"max_steps": 1}, batch_size=4, device="cuda", max_samples=8, overwrite=True)
    out["identity"] = {"paper_label": r3.resolved["label"], "paper_origin": r3.resolved["model"]["origin"], "custom_origin": r4.resolved["model"]["origin"],
                       "recipe_hashes_differ": r3.resolved["training_recipe_sha256"] != r4.resolved["training_recipe_sha256"]}
    out["overwrite_refused"] = expect_error(api.train, h, task="recovery", model="fno", preset="smoke", observation=obs, out=ROOT / "t7" / "custom",
                                            budget={"max_steps": 1}, max_samples=8)
    out["budget_required"] = expect_error(api.train, h, task="recovery", model="fno", preset="smoke", observation=obs, out=ROOT / "t7" / "nb", budget=None)
    out["budget_conflict"] = expect_error(api.train, h, task="recovery", model="fno", preset="smoke", observation=obs, out=ROOT / "t7" / "nb2", budget={"epochs": 1, "max_steps": 1})
    out["paper_view_on_16grid_rejected"] = expect_error(api.train, h, task="recovery", model="fno", preset="smoke", observation=api.make_observation("paper:R50"),
                                                        out=ROOT / "t7" / "nb3", budget={"max_steps": 1})
    out["summary"] = {"paper_split_1800_200": out["paper_split"]["train"] == 1800 and out["paper_split"]["test"] == 200 and out["paper_split"]["overlap"] == 0,
                      "all_rejections_ok": all(out[k]["status"] == "rejected_ok" for k in ("explicit_overlap_rejected", "paper_split_wrong_count_rejected", "overwrite_refused", "budget_required", "budget_conflict", "paper_view_on_16grid_rejected"))}
    write("T7", out)
    return out["summary"]


# --------------------------------------------------------------------------- T8
def t8_artifacts_inference():
    from pdeobs import api
    out = {"tier": "T8", "what": "artifact / inference / evaluation edge cases and refusals", "utc": now()}
    hp, _ = _ds(api, "poisson", res=32); hh, _ = _ds(api, "heat", res=32)
    obs = api.make_observation("random", ratio=0.5)
    r = api.train(hp, task="recovery", model="fno", preset="smoke", observation=obs, out=ROOT / "t8" / "base", budget={"max_steps": 2}, batch_size=2, device="cuda", overwrite=True)
    art = r.artifact.path
    pred = api.load_predictor(art, device="cuda")
    package, targets = api.inference_input_from_dataset(hp, obs, task="recovery")
    pkg_path = package.save(ROOT / "t8" / "inputs.npz")
    b1 = api.predict(pred, pkg_path)  # observation="stored"
    out["npz_package_roundtrip"] = {"status": "passed" if b1.predictions.shape == targets.shape else "failed", "targets_used": b1.provenance["targets_used"]}
    out["stored_package_with_new_obs_refused"] = expect_error(api.predict, pred, pkg_path, observation="random")
    out["records_with_stored_refused"] = expect_error(api.predict, pred, hp, observation="stored")
    nogeo = api.InferenceInput(observations=package.observations, mask=package.mask, sample_ids=package.sample_ids)
    out["geometry_required_refused"] = expect_error(api.predict, pred, nogeo)
    bfree = api.predict(pred, nogeo, free_geometry=True)
    out["free_geometry_declared"] = {"status": "passed", "warning": bfree.provenance["warnings"]}
    h64, _ = _ds(api, "poisson", res=64)
    p64, _ = api.inference_input_from_dataset(h64, obs, task="recovery")
    b64 = api.predict(pred, p64)
    out["resolution_mismatch_warned_but_runs"] = {"shape": list(b64.predictions.shape), "warnings": b64.provenance["warnings"]}
    out["evaluate_without_target_refused"] = expect_error(api.evaluate, b1)
    bad = api.evaluate(b1, targets[::-1] * 0 + 1.0)  # wrong values still valid arrays -> valid but poor score
    out["evaluate_bad_prediction_still_scored"] = {"status": bad.get("status"), "rel_l2": (bad.get("summary") or {}).get("rel_l2_joint_mean")}
    shp = api.evaluate(b1, targets[:, :16, :16])
    out["evaluate_shape_mismatch_invalid"] = {"status": shp.get("status"), "errors": shp.get("errors", [])[:1]}
    ids = api.evaluate(b1, targets, target_ids=["x" + s for s in b1.sample_ids])
    out["evaluate_identity_mismatch_invalid"] = {"status": ids.get("status"), "errors": ids.get("errors", [])[:1]}
    nan = np.array(targets); nan[0, 0, 0, 0] = np.nan
    out["evaluate_nonfinite_target_invalid"] = {"status": api.evaluate(b1, nan).get("status")}
    out["untrained_predictor_refused"] = expect_error(api.predict, api.load_predictor(art, allow_untrained=True), package)
    out["structure_mismatch_refused"] = expect_error(api.load_predictor, art, params={"width": 40})
    import shutil
    tampered = ROOT / "t8" / "tampered"
    shutil.rmtree(tampered, ignore_errors=True); shutil.copytree(art, tampered)
    with open(tampered / "weights.pt", "ab") as fh:
        fh.write(b"\x00")
    out["tampered_weights_refused"] = expect_error(api.load_predictor, tampered)
    leg = api.load_legacy_checkpoint(r.out / "checkpoints" / "last.pt", model="fno", task="recovery", params=r.resolved["model"]["kwargs"], device="cuda")
    bl = api.predict(leg, package)
    out["legacy_checkpoint_adapter"] = {"status": "passed" if np.allclose(bl.predictions, b1.predictions, atol=1e-5) else "failed",
                                       "unknown_fields": leg.provenance["unknown_fields"]}
    out["legacy_wrong_task_refused"] = expect_error(api.load_legacy_checkpoint, art / "checkpoints" / "last.pt", model="fno", task="forward", params=r.resolved["model"]["kwargs"])
    rr = api.train(hh, task="rollout", model="fno", preset="smoke", observation=obs, out=ROOT / "t8" / "roll", budget={"max_steps": 1}, batch_size=2, device="cuda", overwrite=True)
    pr = api.load_predictor(rr.artifact.path, device="cuda")
    ph, th = api.inference_input_from_dataset(hh, obs, task="rollout")
    b5 = api.predict(pr, ph, horizon=5)
    out["horizon_beyond_trained_warned"] = {"shape": list(b5.predictions.shape), "warnings": b5.provenance["warnings"],
                                           "time_indices": b5.time_indices}
    out["static_input_to_rollout_refused"] = expect_error(api.predict, pr, package)
    out["summary"] = {"refusals_ok": all(out[k]["status"] == "rejected_ok" for k in out if k.endswith("_refused")),
                      "invalid_scores_flagged": all(out[k]["status"] == "invalid" for k in out if k.endswith("_invalid"))}
    write("T8", out)
    return out["summary"]


# --------------------------------------------------------------------------- T9
def t9_cli():
    out = {"tier": "T9", "what": "every easy CLI subcommand + legacy CLI entry points from the installed package", "utc": now(), "commands": {}}
    base = ROOT / "t9"; base.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    def run(name, args, cwd=None, timeout=900):
        t0 = time.perf_counter()
        try:
            p = subprocess.run(["pdeobs", *args], capture_output=True, text=True, timeout=timeout, cwd=cwd or str(base), env=env)
            out["commands"][name] = {"rc": p.returncode, "seconds": round(time.perf_counter() - t0, 1), "stdout_tail": p.stdout[-300:], "stderr_tail": p.stderr[-300:]}
        except subprocess.TimeoutExpired:
            out["commands"][name] = {"rc": "timeout", "seconds": timeout}
        write("T9", out)
        print("T9", name, out["commands"][name]["rc"], flush=True)

    run("easy list models", ["easy", "list", "models"])
    run("easy list observations", ["easy", "list", "observations"])
    run("easy list pdes", ["easy", "list", "pdes"])
    run("easy list presets", ["easy", "list", "presets"])
    run("easy describe transolver", ["easy", "describe", "transolver"])
    run("easy describe bad", ["easy", "describe", "nope"])  # expect rc 2
    run("easy data generate", ["easy", "data", "--source", "generate", "--pde", "heat", "--resolution", "16", "--samples", "6", "--frames", "4", "--out", "d/heat"])
    run("easy data local", ["easy", "data", "--source", "local", "--path", "d/heat"])
    run("easy train", ["easy", "train", "--data", "d/heat", "--task", "rollout", "--model", "fno", "--model-param", "width=16", "--model-param", "modes=4",
                       "--model-param", "layers=2", "--obs", "random", "--obs-param", "ratio=0.5", "--max-steps", "3", "--batch-size", "2", "--device", "cuda", "--out", "r/fno"])
    run("easy train bad param", ["easy", "train", "--data", "d/heat", "--task", "rollout", "--model", "fno", "--model-param", "widht=16", "--obs", "random",
                                 "--max-steps", "1", "--out", "r/bad"])
    run("easy infer records", ["easy", "infer", "--model-from", "r/fno/model", "--data", "d/heat", "--obs", "random", "--obs-param", "ratio=0.5", "--horizon", "3", "--out", "p/preds.h5"])
    # build a target-free package and infer from it, then eval with a targets file
    from pdeobs import api
    h = api.load_dataset(base / "d" / "heat")
    pkg, tg = api.inference_input_from_dataset(h, api.make_observation("random", ratio=0.5), task="rollout")
    pkg.save(base / "p" / "inputs.npz")
    np.savez(base / "p" / "targets.npz", target=tg, target_ids=np.asarray(pkg.sample_ids), target_time_indices=np.asarray([1, 2, 3]))
    run("easy infer package", ["easy", "infer", "--model-from", "r/fno/model", "--input", "p/inputs.npz", "--horizon", "3", "--out", "p/preds2.npz"])
    run("easy eval predictions", ["easy", "eval", "--predictions", "p/preds2.npz", "--targets", "p/targets.npz", "--out", "s/bundle"])
    run("easy eval records", ["easy", "eval", "--model-from", "r/fno/model", "--data", "d/heat", "--obs", "block", "--obs-param", "missing_fraction=0.5", "--out", "s/records"])
    (base / "pipe.yaml").write_text("""schema_version: pdeobs-pipeline/v1
name: t9
task: recovery
seed: 3
out: pipe-out
data: {source: generate, pde: poisson, out: pipe-out/data, num_samples: 4, resolution: 16}
observation: {protocol: random, ratio: 0.5}
model: {name: cno, preset: smoke}
train: {budget: {max_steps: 2}, batch_size: 2}
predict: {out: predictions.npz}
evaluate: {mode: records}
""", encoding="utf-8")
    (base / "pipe.json").write_text(json.dumps({"schema_version": "pdeobs-pipeline/v1", "task": "recovery", "out": "pipe-json",
                                                "data": {"source": "generate", "pde": "poisson", "out": "pipe-json/data", "num_samples": 4, "resolution": 16},
                                                "observation": {"protocol": "grid", "ratio": 0.2}, "model": {"name": "unet", "preset": "smoke"},
                                                "train": {"budget": {"epochs": 1}, "batch_size": 2}, "predict": {}, "evaluate": {}}), encoding="utf-8")
    run("easy validate", ["easy", "validate", "--config", "pipe.yaml"])
    run("easy plan", ["easy", "plan", "--config", "pipe.yaml"])
    run("easy run dry", ["easy", "run", "--config", "pipe.yaml", "--dry-run"])
    run("easy run yaml", ["easy", "run", "--config", "pipe.yaml"])
    run("easy run json", ["easy", "run", "--config", "pipe.json"])
    # legacy entry points (documented in README)
    for ex in ("E1", "E2", "E3", "E4", "E5", "E6"):
        run(f"demo {ex}", ["demo", "--example", ex, "--data-root", f"demo-data-{ex}", "--output-root", f"demo-runs-{ex}", "--threads", "1"], timeout=1200)
    run("generate-case", ["generate-case", "--pde", "poisson", "--boundary", "periodic", "--setting", "smooth_grf", "--param-regime", "low", "--num-samples", "3", "--resolution", "16", "--seed", "17", "--root", "gc"])
    run("list", ["list", "--kind", "all", "--json"])
    run("doctor", ["doctor", "--offline"])
    run("protocol --check", ["protocol", "--check"])
    run("paper-row-demo-data", ["paper-row-demo-data", "--pde", "poisson", "--output", "prd"])
    src = Path(__import__("pdeobs").__file__).resolve().parent
    cfg = [p for p in [WORK / "work" / "exact" / "PDE_OBS_code" / "configs" / "demo" / "paper_row_static.yaml"] if p.is_file()]
    if cfg:
        run("identity-manifest", ["identity-manifest", "--data-root", "prd", "--shards", "low.h5", "medium.h5", "high.h5", "--output", "prd/manifest.json"])
        run("paper-row demo", ["paper-row", "--config", str(cfg[0]), "--data-root", "prd", "--manifest", "prd/manifest.json", "--output", "prd-out", "--device", "cuda"], timeout=1200)
    ok = sum(1 for k, v in out["commands"].items() if (v["rc"] == 0) != (k in ("easy describe bad", "easy train bad param")))
    out["summary"] = {"as_expected": ok, "total": len(out["commands"])}
    write("T9", out)
    return out["summary"]


# --------------------------------------------------------------------------- T10 / T11
def t10_device_parity():
    from pdeobs import api
    out = {"tier": "T10", "what": "CPU vs CUDA prediction parity for the same weights", "utc": now(), "models": {}}
    hp, _ = _ds(api, "poisson", res=32)
    obs = api.make_observation("random", ratio=0.5)
    package, _ = api.inference_input_from_dataset(hp, obs, task="recovery")
    for m in MAIN:
        try:
            r = api.train(hp, task="recovery", model=m, preset="smoke", observation=obs, out=ROOT / "t10" / m, budget={"max_steps": 1}, batch_size=2, device="cuda", overwrite=True)
            a = api.predict(api.load_predictor(r.artifact.path, device="cuda"), package).predictions
            b = api.predict(api.load_predictor(r.artifact.path, device="cpu"), package).predictions
            diff = float(np.abs(a - b).max()); scale = float(np.abs(b).max()) + 1e-12
            out["models"][m] = {"max_abs_diff": diff, "rel_to_scale": diff / scale, "within_1e-3_rel": diff / scale < 1e-3}
        except Exception as exc:  # noqa: BLE001
            out["models"][m] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    out["summary"] = {"all_within_tol": all(v.get("within_1e-3_rel") for v in out["models"].values())}
    write("T10", out)
    return out["summary"]


def t11_determinism():
    from pdeobs import api
    out = {"tier": "T11", "what": "same seed -> same losses and weights (deterministic mode); masks deterministic", "utc": now()}
    hp, _ = _ds(api, "poisson", res=32)
    obs = api.make_observation("random", ratio=0.5)
    runs = []
    for i in range(2):
        r = api.train(hp, task="recovery", model="fno", preset="smoke", observation=obs, out=ROOT / "t11" / f"run{i}", budget={"max_steps": 3},
                      batch_size=2, device="cuda", seed=7, overwrite=True)
        runs.append(r)
    losses = [[h["train_loss"] for h in r.history] for r in runs]
    w = [sha(r.artifact.weights) for r in runs]
    out["gpu"] = {"losses": losses, "losses_equal": losses[0] == losses[1], "weights_sha_equal": w[0] == w[1]}
    cpu = []
    for i in range(2):
        r = api.train(hp, task="recovery", model="fno", preset="smoke", observation=obs, out=ROOT / "t11" / f"cpu{i}", budget={"max_steps": 3},
                      batch_size=2, device="cpu", seed=7, overwrite=True)
        cpu.append(sha(r.artifact.weights))
    out["cpu"] = {"weights_sha_equal": cpu[0] == cpu[1]}
    out["summary"] = {"gpu_bitwise": out["gpu"]["weights_sha_equal"], "cpu_bitwise": out["cpu"]["weights_sha_equal"]}
    write("T11", out)
    return out["summary"]


# --------------------------------------------------------------------------- T12 / T13 / T14
def t12_regression():
    out = {"tier": "T12", "what": "frozen 521 portable + updater tests and the 38 API tests", "utc": now()}
    repo = WORK / "work" / "exact" / "PDE_OBS_code"
    for d in (ROOT / "t12-core", ROOT / "t12-temp", ROOT / "t12-api"):
        import shutil; shutil.rmtree(d, ignore_errors=True)
    p = subprocess.run([sys.executable, str(repo / "release" / "run_core_tests.py"), "--source-root", str(repo), "--output-dir", str(ROOT / "t12-core"),
                        "--basetemp", str(ROOT / "t12-temp")], capture_output=True, text=True)
    ex = ROOT / "t12-core" / "execution.json"
    out["core"] = json.loads(ex.read_text())["counts"] if ex.is_file() else {"rc": p.returncode, "tail": p.stdout[-400:]}
    q = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(repo / "tests" / "test_api_specs.py"), str(repo / "tests" / "test_api_workflow.py"),
                        "--basetemp", str(ROOT / "t12-api")], capture_output=True, text=True, cwd=str(repo))
    out["api"] = {"rc": q.returncode, "tail": q.stdout.strip().splitlines()[-1] if q.stdout.strip() else q.stderr[-200:]}
    out["summary"] = {"core_passed": out["core"].get("passed"), "core_failed": out["core"].get("failed"), "api": out["api"]["tail"]}
    write("T12", out)
    return out["summary"]


def t13_paper_resource_envelope():
    from pdeobs import api
    from pdeobs.api import specs as S
    import torch
    out = {"tier": "T13", "what": "paper structures at 128^2 with the paper batch sizes: peak memory + per-step timing -> epoch / 500-epoch estimate, then stop (no training)", "utc": now(), "runs": {}}
    hp, _ = _ds(api, "poisson", res=128, samples=8, tag="-env"); hh, _ = _ds(api, "heat", res=128, samples=8, tag="-env")
    for m in MAIN:
        bs = S.PAPER_OPTIMIZER[m]["batch_size"]
        for h, task in ((hp, "recovery"), (hh, "rollout")):
            b = S.PAPER_TEMPORAL_OPTIMIZER.get(m, {}).get("batch_size", bs) if task == "rollout" else bs
            k = f"{m}:{task}@128 bs={b}"
            try:
                torch.cuda.empty_cache()
                r = api.train(h, task=task, model=m, preset="paper", observation=api.make_observation("random", ratio=0.5), out=ROOT / "t13" / k.replace(" ", "_").replace("=", ""),
                              budget={"max_steps": 1}, batch_size=b, device="cuda", history_steps=1, horizon=3 if task == "rollout" else None, overwrite=True)
                # time N more steps in-process to get a clean per-step estimate (first step carries warm-up)
                import math
                from pdeobs.runner import _loader
                from pdeobs.one_setting import subset_dataset
                from pdeobs.dataset import BenchmarkDataset
                ds = BenchmarkDataset(h.shards, task=task, mask={"protocol": "random_3pct", "ratio": 0.5}, history_steps=1, horizon=3 if task == "rollout" else 8, seed=0)
                loader = _loader(ds, {"training": {"batch_size": b, "num_workers": 0, "pin_memory": False}, "seed": 0}, shuffle=False)
                from pdeobs.training import Trainer, TrainingConfig
                tcfg = TrainingConfig(task=task, epochs=1, amp=False, data_layout="channels_last", horizon=3 if task == "rollout" else 1, rollout_target_offset=0,
                                      device="cuda", physics_loss=r.resolved["training"].get("physics_loss", "none"), physics_loss_weight=r.resolved["training"].get("physics_loss_weight", 0.0),
                                      data_loss_weight=r.resolved["training"].get("data_loss_weight", 1.0), checkpoint_dir=str(ROOT / "t13" / "tmp"))
                model = api.load_predictor(r.artifact.path, device="cuda").model
                model.train()
                trainer = Trainer(model, tcfg)
                batches = list(loader)[:2]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(3):
                    trainer.run_epoch(batches, training=True)
                torch.cuda.synchronize(); step_s = (time.perf_counter() - t0) / (3 * len(batches))
                ds.close()
                steps_per_epoch = math.ceil(1800 / b)
                epoch_s = step_s * steps_per_epoch
                out["runs"][k] = {"status": "passed", "params": r.summary["parameters"]["total"], "peak_gpu_mb": round((r.resolved.get("peak_gpu_memory_bytes") or 0) / 1e6, 1),
                                  "first_step_s": r.resolved.get("training_seconds"), "loss": r.summary.get("final_train_loss"), "scheduler": r.resolved["training"].get("scheduler"),
                                  "optimizer": r.resolved["training"].get("optimizer"), "physics_loss": r.resolved["training"].get("physics_loss"),
                                  "estimate": {"step_s": round(step_s, 4), "steps_per_epoch_1800": steps_per_epoch, "epoch_s": round(epoch_s, 1),
                                               "epoch_min": round(epoch_s / 60, 2), "epochs_500_h": round(500 * epoch_s / 3600, 2),
                                               "note": "extrapolated from 3x2 timed steps at 128^2 on this GPU; no training performed beyond that"}}
                del trainer, model; torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                out["runs"][k] = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            print("T13", k, out["runs"][k]["status"], out["runs"][k].get("peak_gpu_mb"), flush=True)
            write("T13", out)
    out["summary"] = {"passed": sum(1 for e in out["runs"].values() if e["status"] == "passed"), "total": len(out["runs"]),
                      "max_peak_gpu_mb": max((e.get("peak_gpu_mb") or 0) for e in out["runs"].values()),
                      "epochs_500_hours": {k: e.get("estimate", {}).get("epochs_500_h") for k, e in out["runs"].items()}}
    write("T13", out)
    return out["summary"]


def t14_optional_backends():
    from pdeobs import api
    from pdeobs.api import specs as S
    out = {"tier": "T14", "what": "optional exact upstream wrappers: dependency-gated, never substituted", "utc": now(), "wrappers": {}}
    for m in ("paper_unet", "paper_fno", "paper_cno"):
        out["wrappers"][m] = {"easy_api": expect_error(api.resolve_model, m, task="recovery"), "status": "dependency_blocked"}
    from pdeobs.methods import create_method
    out["direct_registry_requires_attestation"] = expect_error(create_method, "paper_fno")
    out["summary"] = {"gated": all(v["easy_api"]["status"] == "rejected_ok" for v in out["wrappers"].values())}
    write("T14", out)
    return out["summary"]


TIERS = {"T1": t1_generation, "T2": t2_observations, "T3": t3_tasks, "T4": t4_models, "T5": t5_views_gpu, "T6": t6_boundary_regime,
         "T7": t7_splits_budgets, "T8": t8_artifacts_inference, "T9": t9_cli, "T10": t10_device_parity, "T11": t11_determinism,
         "T12": t12_regression, "T13": t13_paper_resource_envelope, "T14": t14_optional_backends}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("workdir"); ap.add_argument("--tiers", default=",".join(TIERS))
    a = ap.parse_args()
    cov_path = QA / "coverage.json"
    cov = json.loads(cov_path.read_text()) if cov_path.is_file() else {"utc": now(), "tiers": {}}
    import torch
    cov["env"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None}
    for t in a.tiers.split(","):
        t = t.strip()
        if t not in TIERS:
            continue
        if cov["tiers"].get(t, {}).get("status") == "done":
            print(f"skip {t} (done)", flush=True); continue
        t0 = time.perf_counter()
        print(f"===== {t} start", flush=True)
        try:
            s = TIERS[t]()
            cov["tiers"][t] = {"status": "done", "summary": s, "seconds": round(time.perf_counter() - t0, 1)}
        except Exception as exc:  # noqa: BLE001
            cov["tiers"][t] = {"status": "crashed", "error": f"{type(exc).__name__}: {str(exc)[:300]}", "trace": traceback.format_exc()[-1500:],
                               "seconds": round(time.perf_counter() - t0, 1)}
        cov_path.write_text(json.dumps(cov, indent=2, default=str), encoding="utf-8")
        print(f"===== {t} {cov['tiers'][t]['status']} {cov['tiers'][t].get('summary') or cov['tiers'][t].get('error')} ({cov['tiers'][t]['seconds']}s)", flush=True)
    print("ALL TIERS PROCESSED", flush=True)
    (QA / "COVERAGE_DONE").touch()


if __name__ == "__main__":
    main()
