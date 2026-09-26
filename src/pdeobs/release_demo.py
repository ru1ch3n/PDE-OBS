"""Six small CPU-only, executable release workflows.

The examples are interface demonstrations, not benchmark performance evidence.
Every destination is explicit; no author filesystem or cluster is required.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

from . import __version__
from .strict_score import CONTRACT_VERSION, write_report


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _peak_memory() -> int | None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess
        handle.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if query(handle(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _environment(threads: int) -> dict[str, Any]:
    versions = {}
    for package in ("pdeobs", "numpy", "scipy", "h5py", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "package_version": __version__, "dependencies": versions,
            "device": "cpu", "cpu_threads": threads, "gpu_requested": 0,
            "demo_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _generate(data_dir: Path, family: str, count: int = 3) -> Path:
    from .generation import GenerationJob, generate_job

    path = data_dir / f"{family}.h5"
    job = GenerationJob(pde=family, boundary="periodic", setting="smooth_grf", regime="low",
                        sample_start=0, sample_count=count, shard_index=0,
                        output_path=str(path), resolution=16, seed=1729, macro_size=30,
                        time_steps=4 if family == "heat" else 1,
                        tier="custom", quality={"profile": "report"})
    result = generate_job(job, resume=False, overwrite=False)
    _json(data_dir / "generation.json", {"job": job.to_dict(), "result": asdict(result)})
    return path


def _dataset(shard: Path, task: str = "recovery", mask: dict[str, Any] | None = None):
    from .dataset import BenchmarkDataset
    return BenchmarkDataset([shard], task=task, mask=mask or {"protocol": "random_3pct", "ratio": 0.5},
                            history_steps=1, horizon=3, verify=True, seed=42)


def _predict(method: Any, rows: list[dict[str, Any]], task: str):
    from .dataset import collate_benchmark
    from .evaluation import EvaluationConfig, predict_batch

    config = EvaluationConfig(task=task, device="cpu", data_layout="channels_last",
                              history_steps=1, horizon=3 if task == "rollout" else 1,
                              horizons=(1, 2, 3) if task == "rollout" else (1,),
                              rollout_target_offset=0)
    predictions, targets, sizes = [], [], []
    for start in range(0, len(rows), 2):
        batch = collate_benchmark(rows[start:start + 2])
        pred, truth = predict_batch(method, batch, config)
        predictions.append(pred)
        targets.append(truth)
        sizes.append(len(pred))
    return np.concatenate(predictions), np.concatenate(targets), sizes


def _score_output(out: Path, rows: list[dict[str, Any]], prediction: np.ndarray,
                  target: np.ndarray, task: str, checkpoint_id: str, *, projection: bool = False):
    from .runner import run_strict_score

    ids = [str(row["metadata"]["sample_id"]) for row in rows]
    times = [1, 2, 3] if task == "rollout" else [0]
    contract = {"schema_version": CONTRACT_VERSION, "task": task, "split": "demo",
                "expected_ids": ids, "expected_shape": list(target.shape[1:]),
                "target_time_indices": times, "observation_id": str(rows[0]["metadata"]["mask_id"]),
                "checkpoint_id": checkpoint_id, "projection": projection,
                "scope": "workflow demonstration, not a paper performance estimate"}
    arrays = {"prediction": prediction, "target": target,
              "prediction_ids": np.asarray(ids), "target_ids": np.asarray(ids),
              "prediction_time_indices": np.asarray(times), "target_time_indices": np.asarray(times)}
    if task == "recovery":
        arrays.update(mask=np.stack([row["mask"] for row in rows]),
                      observation=np.stack([row["observations"] for row in rows]))
    np.savez_compressed(out / "predictions.npz", **arrays)
    _json(out / "score-contract.json", contract)
    result = run_strict_score(out / "predictions.npz", out / "score-contract.json", out / "strict-score.json")
    if result["status"] != "valid":
        raise RuntimeError("strict score rejected demo predictions: " + "; ".join(result["errors"]))
    return result


def _e1(data: Path, out: Path) -> dict[str, Any]:
    from .masks import apply_mask, generate_mask
    from .storage import LazyHDF5Dataset

    shard = _generate(data, "poisson")
    reader = LazyHDF5Dataset([shard], verify=True)
    try:
        sample = reader[0]
        field = sample.trajectory[0]
        original = field.copy()
        masks = {"random": generate_mask("random_3pct", (16, 16), seed=8, count=128),
                 "block": generate_mask("block_missing", (16, 16), seed=8, missing_fraction=0.5),
                 "horizontal": generate_mask("line_sensors", (16, 16), seed=8, ratio=0.5, orientation="horizontal")}
        arrays = {"field": field}
        for name, mask in masks.items():
            arrays[name + "_mask"] = mask
            arrays[name + "_observed"] = apply_mask(field, mask)
        if not np.array_equal(original, field):
            raise RuntimeError("observation construction changed the physical record")
        np.savez_compressed(out / "observations.npz", **arrays)
        return {"identities": len(reader), "field_shape": list(field.shape),
                "observed_counts": {name: int(mask.sum()) for name, mask in masks.items()},
                "same_record_without_resolve": True}
    finally:
        reader.close()


def _classic(data: Path, out: Path, family: str, task: str, method_name: str):
    from .methods import create_method

    dataset = _dataset(_generate(data, family), task)
    try:
        rows = [dataset[i] for i in range(len(dataset))]
        prediction, target, sizes = _predict(create_method(method_name), rows, task)
        result = _score_output(out, rows, prediction, target, task, "untrained:" + method_name)
        return {"method": method_name, "test_role": "untrained workflow demonstration", "batch_sizes": sizes,
                "strict_status": result["status"], "summary": result["summary"]}
    finally:
        dataset.close()


def _e4(data: Path, out: Path):
    from .physical_inputs import advance_custom_heat, solve_custom_elliptic

    n = 16
    axis = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(axis, axis)
    source = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    coefficient = 1.0 + 0.2 * np.cos(2 * np.pi * x)
    solution, elliptic_info = solve_custom_elliptic(source, coefficient=coefficient,
                                                   boundary="periodic", dx=1/n, dy=1/n)
    initial = np.cos(2 * np.pi * x) * np.sin(4 * np.pi * y)
    advanced, heat_info = advance_custom_heat(initial, diffusivity=0.02, dt=0.125,
                                               boundary="periodic", dx=1/n, dy=1/n)
    reference = initial * np.exp(-0.02 * 0.125 * (2 * np.pi)**2 * (1 + 4))
    error = float(np.max(np.abs(advanced - reference)))
    if error > 1e-12:
        raise RuntimeError(f"periodic Heat analytic reference failed: {error}")
    np.savez_compressed(data / "custom-physical-arrays.npz", source=source, coefficient=coefficient,
                        solution=solution, initial_state=initial, advanced=advanced, analytic_reference=reference)
    def solver_receipt(info):
        # Existing kernels can return np.bool_; receipt JSON uses native types.
        return {"converged": bool(info.converged), "iterations": int(info.iterations),
                "relative_residual": float(info.relative_residual)}

    return {"elliptic_solver": solver_receipt(elliptic_info), "heat_solver": solver_receipt(heat_info),
            "heat_analytic_max_absolute_error": error,
            "scope": "one resolved periodic Fourier mode; not a seven-family convergence study"}


def _e5(data: Path, out: Path, threads: int):
    import torch
    from .dataset import collate_benchmark
    from .methods import create_method
    from .training import Trainer, TrainingConfig

    torch.set_num_threads(threads)
    torch.manual_seed(19)
    shard = _generate(data, "poisson", count=7)
    dataset = _dataset(shard)
    try:
        # Explicit demo partition; never relabel production train records as test.
        train_rows = [dataset[i] for i in range(4)]
        train_ids = [row["metadata"]["sample_id"] for row in train_rows]
        # Read test metadata only here. Test field arrays first enter the fresh
        # evaluation process after the final training checkpoint is closed.
        test_ids = [row["sample_id"] for row in dataset.metadata[4:]]
        if set(train_ids) & set(test_ids):
            raise RuntimeError("demo train/test identity overlap")
        split = {"scope": "explicit independent demo partition (not the paper split)",
                 "train_ids": train_ids, "test_ids": test_ids, "validation_ids": []}
        _json(out / "demo-split.json", split)
        model = create_method("unet", width=4)
        configuration = TrainingConfig(epochs=2, loss="mse", device="cpu", amp=False,
                                       data_layout="channels_last", learning_rate=1e-3,
                                       checkpoint_dir=str(out / "checkpoints"), seed=19)
        trainer = Trainer(model, configuration)
        loader = [collate_benchmark(train_rows[:2]), collate_benchmark(train_rows[2:])]
        history = trainer.fit(loader, val_loader=None)
        _json(out / "training-history.json", history)
        checkpoint = out / "checkpoints" / "last.pt"
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        reload_spec = {"shard": str(shard.resolve()), "checkpoint": str(checkpoint.resolve()),
                       "checkpoint_sha256": checkpoint_hash, "test_ids": test_ids,
                       "train_ids": train_ids, "threads": threads, "output": str((out / "reloaded").resolve())}
        _json(out / "reload-spec.json", reload_spec)
        command = [sys.executable, "-B", "-m", "pdeobs.release_demo", "--reload-neural", str(out / "reload-spec.json")]
        child = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
        (out / "reload.stdout.log").write_text(child.stdout, encoding="utf-8")
        (out / "reload.stderr.log").write_text(child.stderr, encoding="utf-8")
        if child.returncode:
            raise RuntimeError(f"fresh-process checkpoint reload failed with exit {child.returncode}; see reload logs")
        receipt = json.loads((out / "reloaded" / "reload-receipt.json").read_text(encoding="utf-8"))
        if receipt["pid"] == os.getpid() or receipt["checkpoint_sha256"] != checkpoint_hash:
            raise RuntimeError("checkpoint reload process/hash verification failed")
        return {"epochs": len(history), "optimizer_steps": int(trainer._optimizer_steps),
                "train_identities": len(train_ids), "test_identities": len(test_ids),
                "overlap": 0, "checkpoint_sha256": checkpoint_hash, "reload_command": command,
                "reload_exit_code": child.returncode, "reload_pid": receipt["pid"],
                "reload_peak_process_rss_bytes": receipt["peak_process_rss_bytes"],
                "strict_status": receipt["strict_status"], "scope": "interface validation only; not convergence"}
    finally:
        dataset.close()


def _reload_neural(spec_path: Path) -> int:
    import torch
    from .methods import create_method
    from .training import load_checkpoint_payload

    started = time.perf_counter()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    torch.set_num_threads(int(spec["threads"]))
    destination = Path(spec["output"])
    destination.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(spec["checkpoint"])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != spec["checkpoint_sha256"]:
        raise ValueError("checkpoint bytes changed before fresh-process reload")
    dataset = _dataset(Path(spec["shard"]))
    try:
        rows = [dataset[i] for i, metadata in enumerate(dataset.metadata) if metadata["sample_id"] in spec["test_ids"]]
        ids = [row["metadata"]["sample_id"] for row in rows]
        if ids != spec["test_ids"] or set(ids) & set(spec["train_ids"]):
            raise ValueError("fresh-process test identity selection differs")
        model = create_method("unet", width=4)
        state = load_checkpoint_payload(checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state"], strict=True)
        prediction, target, sizes = _predict(model, rows, "recovery")
        result = _score_output(destination, rows, prediction, target, "recovery", digest)
        _json(destination / "reload-receipt.json", {"pid": os.getpid(), "checkpoint_sha256": digest,
              "strict_status": result["status"], "batch_sizes": sizes,
              "duration_seconds": time.perf_counter() - started, "peak_process_rss_bytes": _peak_memory(),
              "import_path": str(Path(__file__).resolve()), "environment": _environment(spec["threads"])})
        return 0
    finally:
        dataset.close()


def _e6(data: Path, out: Path):
    from .methods import create_method
    from .methods.base import MethodCapabilities, register_method
    from .registry import MASK_REGISTRY

    def checkerboard(shape, rng, **kwargs):
        del rng, kwargs
        return np.indices(shape).sum(axis=0) % 2 == 0

    class VisibleIdentity:
        name = "demo_visible_identity"
        capabilities = MethodCapabilities(requires_mask=True)

        def predict(self, observations, mask=None, **kwargs):
            del kwargs
            if mask is None:
                raise ValueError("custom demo model requires a mask")
            return np.where(np.asarray(mask, dtype=bool), observations, 0.0)

    MASK_REGISTRY.register("demo_checkerboard", replace=True)(checkerboard)
    register_method("demo_visible_identity", replace=True)(VisibleIdentity)
    dataset = _dataset(_generate(data, "poisson"), mask={"protocol": "demo_checkerboard"})
    try:
        rows = [dataset[i] for i in range(len(dataset))]
        prediction, target, sizes = _predict(create_method("demo_visible_identity"), rows, "recovery")
        result = _score_output(out, rows, prediction, target, "recovery", "untrained:demo_visible_identity")
        return {"extension": "registry factory without changing the PDE kernel", "batch_sizes": sizes,
                "strict_status": result["status"], "summary": result["summary"]}
    finally:
        dataset.close()


def run_demo(example: str, *, data_root: str | Path, output_root: str | Path,
             threads: int = 1) -> dict[str, Any]:
    if threads not in (1, 2) or example not in {"E1", "E2", "E3", "E4", "E5", "E6"}:
        raise ValueError("select E1-E6 and one or two CPU threads")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    data, out = Path(data_root).resolve() / example, Path(output_root).resolve() / example
    if data == out or data in out.parents or out in data.parents:
        raise ValueError("data and output roots must be separate non-nested locations")
    # Exclusive directories prevent accidental replacement of prior evidence.
    data.mkdir(parents=True, exist_ok=False)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = {"example": example, "status": "running", "environment": _environment(threads),
               "command": sys.argv, "data_directory": str(data), "output_directory": str(out),
               "device": "cpu", "gpu_requested": 0, "scope": "small workflow demonstration"}
    try:
        if example == "E1":
            result = _e1(data, out)
        elif example == "E2":
            result = _classic(data, out, "poisson", "recovery", "nearest")
        elif example == "E3":
            result = _classic(data, out, "heat", "rollout", "persistence")
        elif example == "E4":
            result = _e4(data, out)
        elif example == "E5":
            result = _e5(data, out, threads)
        else:
            result = _e6(data, out)
        receipt.update(status="passed", exit_code=0, details=result)
    except Exception as exc:
        # Record, then re-raise. Never convert a failed acceptance into success.
        receipt.update(status="failed", exit_code=2, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt.update(duration_seconds=time.perf_counter() - started,
                       peak_process_rss_bytes=_peak_memory())
        _json(out / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fresh-process checkpoint verification helper")
    parser.add_argument("--reload-neural", required=True, type=Path)
    arguments = parser.parse_args()
    raise SystemExit(_reload_neural(arguments.reload_neural))
