"""Fail-closed execution and promotion for the all-factor campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .campaign import read_training_plan_row, sha256_file, write_training_plan
from .full_to_partial import EXPECTED_VIEWS, PLAN_SCHEMA

PREFLIGHT_SCHEMA = "pdeobs.full-to-partial-preflight/v1"
REQUIRED_PREFLIGHT_DOMAINS = ("a100", "b40")
REQUIRED_REPLICAS = (0, 1, 2, 3)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _safe_repository_file(root: Path, relative: Any, expected_sha: Any) -> Path:
    path = (root / str(relative)).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"repository artifact is missing or unsafe: {relative!r}")
    actual = sha256_file(path)
    if actual != str(expected_sha):
        raise ValueError(f"repository artifact changed after planning: {relative!r}")
    return path


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _setting_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["method"]), str(row["pde"]), str(row["training_case"])


def verify_preflight_bundle(
    attestations: Mapping[str, Sequence[str | Path]],
    *,
    expected_code_commit: str,
    expected_dataset_summary_sha256: str,
    expected_candidate_plan_sha256: str,
) -> dict[str, Any]:
    """Require four complete, pathology-free replicas on A100 and B40."""

    if tuple(sorted(attestations)) != tuple(sorted(REQUIRED_PREFLIGHT_DOMAINS)):
        raise ValueError("preflight bundle must contain exactly the a100 and b40 domains")
    expected_settings = {
        (method, pde, training_case)
        for method in ("unet", "fno", "cno", "deeponet", "transolver", "gnot")
        for pde in (
            "darcy",
            "poisson",
            "helmholtz",
            "heat",
            "reaction_diffusion",
            "burgers",
            "navier_stokes",
        )
        for training_case in ("full_train", "mixed_partial_train")
    }
    expected_classical = {
        (method, pde)
        for method in ("rbf", "gappy_pod")
        for pde in (
            "darcy",
            "poisson",
            "helmholtz",
            "heat",
            "reaction_diffusion",
            "burgers",
            "navier_stokes",
        )
    }
    verified: dict[str, Any] = {}
    for domain in REQUIRED_PREFLIGHT_DOMAINS:
        paths = tuple(Path(path).resolve() for path in attestations[domain])
        if len(paths) != 4:
            raise ValueError(f"{domain} requires exactly four preflight replicas")
        replicas: dict[int, dict[str, Any]] = {}
        setting_sets: list[set[tuple[str, str, str]]] = []
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != PREFLIGHT_SCHEMA:
                raise ValueError(f"invalid preflight schema: {path}")
            if payload.get("status") != "passed" or payload.get("domain") != domain:
                raise ValueError(f"preflight did not pass for {domain}: {path}")
            if payload.get("code_commit") != expected_code_commit:
                raise ValueError(f"preflight code commit mismatch: {path}")
            if payload.get("dataset_summary_sha256") != expected_dataset_summary_sha256:
                raise ValueError(f"preflight dataset hash mismatch: {path}")
            if payload.get("candidate_plan_sha256") != expected_candidate_plan_sha256:
                raise ValueError(f"preflight candidate-plan hash mismatch: {path}")
            replica = int(payload.get("replica", -1))
            if replica in replicas:
                raise ValueError(f"duplicate {domain} replica {replica}")
            settings = payload.get("settings")
            if not isinstance(settings, list) or len(settings) != len(expected_settings):
                raise ValueError(f"preflight setting count mismatch: {path}")
            keys = {_setting_key(row) for row in settings}
            if keys != expected_settings:
                raise ValueError(f"preflight setting identities are incomplete: {path}")
            for row in settings:
                if row.get("pathology_events"):
                    raise ValueError(f"preflight contains a pathology event: {path}")
                if set(row.get("view_relative_l2", {})) != set(EXPECTED_VIEWS):
                    raise ValueError(f"preflight observation views are incomplete: {path}")
                if not _finite(row):
                    raise ValueError(f"preflight contains non-finite evidence: {path}")
            if int(payload.get("evaluation_view_count", -1)) != len(expected_settings) * len(
                EXPECTED_VIEWS
            ):
                raise ValueError(f"preflight evaluation-view count mismatch: {path}")
            classical = payload.get("classical_settings")
            if not isinstance(classical, list) or {
                (str(row.get("method")), str(row.get("pde"))) for row in classical
            } != expected_classical:
                raise ValueError(f"classical preflight settings are incomplete: {path}")
            if any(
                set(row.get("view_relative_l2", {})) != set(EXPECTED_VIEWS)
                or not _finite(row)
                for row in classical
            ):
                raise ValueError(f"classical preflight views are invalid: {path}")
            replicas[replica] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "device": payload.get("device"),
                "slurm": payload.get("slurm"),
            }
            setting_sets.append(keys)
        if tuple(sorted(replicas)) != REQUIRED_REPLICAS or any(
            keys != setting_sets[0] for keys in setting_sets[1:]
        ):
            raise ValueError(f"{domain} replicas are not a consistent 0..3 set")
        verified[domain] = {"replicas": replicas, "passed_settings": len(expected_settings)}
    digest_payload = json.dumps(verified, sort_keys=True, separators=(",", ":")).encode()
    return {
        "status": "passed",
        "domains": verified,
        "bundle_sha256": hashlib.sha256(digest_payload).hexdigest(),
    }


def promote_candidate_plan(
    candidate_plan: str | Path,
    destination: str | Path,
    *,
    attestations: Mapping[str, Sequence[str | Path]],
    repository_root: str | Path,
    code_commit: str,
) -> Path:
    """Promote candidate rows only after both replicated GPU domains pass."""

    repo = Path(repository_root).resolve()
    candidate = Path(candidate_plan).resolve()
    candidate_sha = sha256_file(candidate)
    rows: list[dict[str, Any]] = []
    with candidate.open("r", encoding="utf-8") as handle:
        for expected_index, line in enumerate(handle):
            row = json.loads(line)
            if row.get("schema_version") != PLAN_SCHEMA or row.get("index") != expected_index:
                raise ValueError("candidate plan schema or contiguous index is invalid")
            if not row.get("candidate_preflight_only") or row.get("release_eligible"):
                raise ValueError("input plan is not an unreleased candidate")
            _safe_repository_file(repo, row["config"], row["config_sha256"])
            _safe_repository_file(repo, row["campaign_file"], row["campaign_sha256"])
            _safe_repository_file(
                repo,
                row["model_setting_registry"],
                row["model_setting_registry_sha256"],
            )
            rows.append(row)
    if len(rows) != 98:
        raise ValueError("candidate plan must contain exactly 98 rows")
    if _git(repo, "rev-parse", "HEAD") != code_commit:
        raise ValueError("requested execution commit is not checked out")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("plan promotion requires a clean private checkout")
    dataset_sha = str(rows[0]["dataset"]["summary_sha256"])
    bundle = verify_preflight_bundle(
        attestations,
        expected_code_commit=code_commit,
        expected_dataset_summary_sha256=dataset_sha,
        expected_candidate_plan_sha256=candidate_sha,
    )
    promoted = []
    for row in rows:
        materialized = dict(row)
        materialized.update(
            candidate_preflight_only=False,
            release_eligible=True,
            code_commit=code_commit,
            candidate_plan_sha256=candidate_sha,
            preflight_bundle=bundle,
        )
        promoted.append(materialized)
    return write_training_plan(promoted, destination)


def _verify_dataset(root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    expected = dict(row["dataset"])
    summary = root / "summary.json"
    if not summary.is_file() or sha256_file(summary) != expected.get("summary_sha256"):
        raise ValueError("immutable dataset summary is missing or has the wrong SHA256")
    shards = tuple(root.rglob("*.h5"))
    if len(shards) != int(expected["shards"]):
        raise ValueError(f"immutable dataset shard count differs: {len(shards)}")
    return {
        "root": str(root),
        "summary": str(summary),
        "summary_sha256": sha256_file(summary),
        "shards": len(shards),
        "records": int(expected["records"]),
        "strict_max_pde_loss": float(expected["pde_loss_normalized_max_observed"]),
    }


def _identity_manifest(config_path: Path, overrides: Sequence[str]) -> dict[str, Any]:
    from .runner import _dataset, _load_runner_config

    config = _load_runner_config(config_path, overrides)
    dataset = _dataset(config, "test", mask_override={"protocol": "full"})
    assert dataset is not None
    try:
        identities = [str(row.get("sample_id", "")) for row in dataset.metadata]
        if len(identities) != 12000 or any(not value for value in identities):
            raise ValueError("test split must contain 12,000 non-empty identities per PDE")
        if len(set(identities)) != len(identities):
            raise ValueError("test split contains duplicate sample identities")
        digest = hashlib.sha256(
            json.dumps(identities, separators=(",", ":")).encode()
        ).hexdigest()
        return {"samples": len(identities), "sha256": digest}
    finally:
        dataset.close()


def _example_manifest(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        identities = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["sample_id"][:]
        ]
        prediction = np.asarray(handle["prediction"][:])
        target = np.asarray(handle["target"][:])
    if not identities or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise FloatingPointError("prediction examples are empty or non-finite")
    target_std = float(np.std(target))
    prediction_std = float(np.std(prediction))
    return {
        "samples": len(identities),
        "identity_sha256": hashlib.sha256(
            json.dumps(identities, separators=(",", ":")).encode()
        ).hexdigest(),
        "prediction_sha256": sha256_file(path),
        "prediction_std": prediction_std,
        "target_std": target_std,
        "output_std_ratio": prediction_std / max(target_std, 1.0e-12),
    }


def _compute_receipt() -> dict[str, Any]:
    receipt = {
        "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        receipt["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "count": int(torch.cuda.device_count()),
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "uuid": (
                str(torch.cuda.get_device_properties(0).uuid)
                if torch.cuda.is_available()
                and getattr(torch.cuda.get_device_properties(0), "uuid", None)
                else None
            ),
        }
    except ImportError:
        receipt["cuda"] = {"available": False, "count": 0}
    return receipt


def execute_full_to_partial_plan_row(
    plan_path: str | Path,
    index: int,
    *,
    dataset_root: str | Path,
    runs_root: str | Path,
    repository_root: str | Path | None = None,
    resume_incomplete: bool = False,
    stale_lock_job_id: str | None = None,
    prediction_example_samples: int = 64,
) -> dict[str, Any]:
    """Train or fit one released row and score the same test set in ten views."""

    from .runner import run_evaluate, run_infer, run_train

    repo = Path(repository_root or Path.cwd()).resolve()
    row = read_training_plan_row(plan_path, index)
    if row.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("not a full-to-partial plan row")
    if row.get("candidate_preflight_only") or not row.get("release_eligible"):
        raise ValueError("candidate rows cannot execute before replicated preflight promotion")
    if row.get("code_commit") != _git(repo, "rev-parse", "HEAD"):
        raise ValueError("execution checkout differs from the released plan commit")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("formal execution requires a clean private checkout")
    config_path = _safe_repository_file(repo, row["config"], row["config_sha256"])
    _safe_repository_file(repo, row["campaign_file"], row["campaign_sha256"])
    _safe_repository_file(
        repo, row["model_setting_registry"], row["model_setting_registry_sha256"]
    )
    dataset_attestation = _verify_dataset(Path(dataset_root).resolve(), row)
    relative = Path(str(row["run_relpath"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("plan run path is unsafe")
    root = Path(runs_root).resolve()
    run_dir = (root / relative).resolve()
    if root not in run_dir.parents:
        raise ValueError("plan run path escaped the run root")
    complete = run_dir / "status.json"
    if complete.is_file():
        payload = json.loads(complete.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and payload.get("row") == row:
            return {**payload, "skipped": True}
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = run_dir / "row_receipt.json"
    receipt = {
        "plan": str(Path(plan_path).resolve()),
        "plan_sha256": sha256_file(plan_path),
        "index": index,
        "row": row,
    }
    existing = [path for path in run_dir.iterdir() if path.name != ".active.lock"]
    if existing and not resume_incomplete:
        raise FileExistsError("incomplete row output exists; diagnose before explicit continuation")
    if resume_incomplete:
        if not receipt_path.is_file() or json.loads(
            receipt_path.read_text(encoding="utf-8")
        ) != receipt:
            raise ValueError("incomplete continuation has no matching immutable row receipt")
    else:
        _atomic_json(receipt_path, receipt)
    lock = run_dir / ".active.lock"
    if lock.exists():
        prior = json.loads(lock.read_text(encoding="utf-8"))
        if (
            not resume_incomplete
            or not stale_lock_job_id
            or str(prior.get("slurm_job_id") or "") != stale_lock_job_id
        ):
            raise FileExistsError("row has an active or unverified stale owner")
        lock.unlink()
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(
        descriptor,
        json.dumps(
            {
                "pid": os.getpid(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
                "index": index,
                "output_identity_sha256": row["output_identity_sha256"],
            },
            sort_keys=True,
        ).encode(),
    )
    os.close(descriptor)
    overrides = list(row["overrides"])
    started = time.perf_counter()

    def load_stage(name: str) -> dict[str, Any] | None:
        marker = run_dir / "_stages" / f"{name}.json"
        if not marker.is_file():
            return None
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or not _finite(payload.get("result")):
            raise ValueError(f"invalid completed stage marker: {name}")
        return dict(payload["result"])

    def save_stage(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
        materialized = dict(result)
        if not _finite(materialized):
            raise FloatingPointError(f"stage contains non-finite values: {name}")
        _atomic_json(run_dir / "_stages" / f"{name}.json", {"status": "complete", "result": materialized})
        return materialized

    try:
        training = load_stage("train")
        checkpoint: Path | None = None
        if row["mode"] == "train_eval":
            if training is None:
                train_dir = run_dir / "train"
                existing_train = list(train_dir.iterdir()) if train_dir.is_dir() else []
                resume_checkpoint = train_dir / "checkpoints" / "last.pt"
                if existing_train and not resume_checkpoint.is_file():
                    raise FileExistsError("partial training lacks a resumable neural checkpoint")
                training = run_train(
                    config_path=config_path,
                    overrides=overrides,
                    output=train_dir,
                    resume=resume_checkpoint if resume_checkpoint.is_file() else None,
                )
                training = save_stage("train", training)
            checkpoint = Path(str(training["checkpoint"])).resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError("training checkpoint is missing")
            health = json.loads(Path(str(training["health_report"])).read_text(encoding="utf-8"))
            if health.get("status") != "completed" or not health.get("accepted"):
                raise RuntimeError("training health gate did not accept the checkpoint")
            if row["training_case"] == "full_train" and health.get("events"):
                raise RuntimeError("full-observation training has an unexplained pathology event")
        elif row["mode"] != "eval":
            raise ValueError(f"unsupported row mode: {row['mode']!r}")

        test_manifest = _identity_manifest(config_path, overrides)
        evaluation = load_stage("evaluate_all_views")
        metrics_path = run_dir / "evaluation" / "metrics.json"
        if evaluation is None:
            if metrics_path.exists():
                raise FileExistsError("evaluation metrics exist without a completed stage marker")
            evaluation = run_evaluate(
                config_path=config_path,
                overrides=overrides,
                output=metrics_path,
                checkpoint=checkpoint,
            )
            evaluation = save_stage("evaluate_all_views", evaluation)
        views = evaluation.get("observation_views")
        if not isinstance(views, Mapping) or tuple(views) != EXPECTED_VIEWS:
            raise ValueError("evaluation did not return the exact frozen observation-view order")
        if any(int(result.get("samples", -1)) != 12000 for result in views.values()):
            raise ValueError("an evaluation view did not score all 12,000 PDE test identities")

        examples: dict[str, Any] = {}
        example_identity: str | None = None
        for protocol in EXPECTED_VIEWS:
            stage_name = f"examples_{protocol}"
            result = load_stage(stage_name)
            output = run_dir / "evaluation" / protocol / "prediction_examples.h5"
            if result is None:
                if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
                    raise FileExistsError(f"{protocol} examples exist without a stage marker")
                result = run_infer(
                    config_path=config_path,
                    overrides=[
                        *overrides,
                        f'data.mask.protocol="{protocol}"',
                        "evaluation.observation_protocols=[]",
                        f"data.max_samples={int(prediction_example_samples)}",
                    ],
                    output=output,
                    checkpoint=checkpoint,
                )
                result = {**result, "manifest": _example_manifest(output)}
                result = save_stage(stage_name, result)
            manifest = dict(result["manifest"])
            if example_identity is None:
                example_identity = str(manifest["identity_sha256"])
            elif manifest["identity_sha256"] != example_identity:
                raise RuntimeError("observation views used different prediction-example identities")
            if protocol == "full" and float(manifest["output_std_ratio"]) <= 1.0e-4:
                raise RuntimeError("full-observation prediction examples collapsed to a constant")
            examples[protocol] = result

        checkpoint_sha = sha256_file(checkpoint) if checkpoint is not None else None
        control = {
            "role": "separate_full_observation_control",
            "checkpoint_sha256": checkpoint_sha,
            "test_identity_sha256": test_manifest["sha256"],
            "example_identity_sha256": example_identity,
            "metrics": views["full"],
            "prediction_examples": examples["full"],
            "partial_metrics_modified": False,
            "post_hoc_cutoff_applied": False,
        }
        _atomic_json(run_dir / "evaluation" / "full" / "control_receipt.json", control)
        result = {
            "status": "complete",
            "row": row,
            "dataset_attestation": dataset_attestation,
            "compute": _compute_receipt(),
            "training": training,
            "checkpoint_sha256": checkpoint_sha,
            "test_identity_manifest": test_manifest,
            "evaluation": evaluation,
            "prediction_examples": examples,
            "full_observation_control": control,
            "seconds": time.perf_counter() - started,
        }
        _atomic_json(complete, result)
        return result
    except BaseException as exc:
        failure = {
            "status": "failed",
            "index": index,
            "output_identity_sha256": row.get("output_identity_sha256"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "seconds": time.perf_counter() - started,
        }
        _atomic_json(
            run_dir
            / "_failures"
            / f"failure-{int(time.time() * 1_000_000)}-{os.environ.get('SLURM_JOB_ID', 'local')}.json",
            failure,
        )
        _atomic_json(run_dir / "failure.json", failure)
        raise
    finally:
        lock.unlink(missing_ok=True)


__all__ = [
    "PREFLIGHT_SCHEMA",
    "execute_full_to_partial_plan_row",
    "promote_candidate_plan",
    "verify_preflight_bundle",
]
