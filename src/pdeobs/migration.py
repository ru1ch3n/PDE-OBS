"""Fail-closed planning for cross-domain GPU load balancing.

This module never calls Slurm.  It turns independently captured scheduler,
artifact, and compatibility attestations into an immutable transaction plan.
An operator must advance each transaction one state at a time and re-observe
both control planes between state changes.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .campaign import PAPER_SETTING_CAMPAIGN_SCHEMA, sha256_file

DOMAIN_ATTESTATION_SCHEMA = "pdeobs.paper-domain-attestation/v1"
DOMAIN_SNAPSHOT_SCHEMA = "pdeobs.gpu-fill-domain-snapshot/v1"
GPU_FILL_SNAPSHOT_SCHEMA = "pdeobs.gpu-fill-snapshot/v1"
GPU_FILL_PLAN_SCHEMA = "pdeobs.gpu-fill-plan/v1"
SLURM_OBSERVATION_SCHEMA = "pdeobs.slurm-job-observation/v1"
FINAL_DATASET_SUMMARY_SHA256 = (
    "b0141eec2199615a27facc75e2517ec4ec7e00913638e48c84ab2c88a2be77c1"
)

_RUNNING_STATES = {"RUNNING", "COMPLETING", "CONFIGURING"}
_PENDING_STATES = {"PENDING"}
_ACTIVE_STATES = _RUNNING_STATES | _PENDING_STATES | {"SUSPENDED"}


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must be a mapping: {source}")
    return payload


def _hex(value: Any, length: int, name: str) -> str:
    token = str(value)
    if len(token) != length or any(character not in "0123456789abcdef" for character in token):
        raise ValueError(f"{name} must be a lowercase {length}-character hexadecimal digest")
    return token


def _row_key(plan_sha256: str, index: int) -> str:
    return f"{plan_sha256}:{index}"


def verify_domain_attestation(
    path: str | Path,
    *,
    expected_domain: str,
    expected_sha256: str,
    launcher_sha256: str,
    required_plan_hashes: set[str],
) -> dict[str, Any]:
    source = Path(path).resolve()
    if sha256_file(source) != expected_sha256:
        raise ValueError(f"domain attestation SHA256 changed: {source}")
    payload = _load_json(source)
    if payload.get("schema_version") != DOMAIN_ATTESTATION_SCHEMA:
        raise ValueError(f"unsupported domain attestation schema: {source}")
    if payload.get("status") != "passed" or payload.get("domain") != expected_domain:
        raise ValueError(f"domain attestation did not pass for {expected_domain!r}")
    if payload.get("dataset_summary_sha256") != FINAL_DATASET_SUMMARY_SHA256:
        raise ValueError(f"domain {expected_domain!r} did not attest the immutable dataset")
    repository_commit = _hex(payload.get("repository_commit"), 40, "repository_commit")
    if payload.get("environment_commit") != repository_commit:
        raise ValueError(f"domain {expected_domain!r} environment/checkout mismatch")
    device = payload.get("device")
    if not isinstance(device, Mapping) or device.get("type") != "cuda":
        raise ValueError(f"domain {expected_domain!r} did not pass a CUDA preflight")
    attested_plans = {
        str(item.get("sha256"))
        for item in payload.get("plans", [])
        if isinstance(item, Mapping)
    }
    if not required_plan_hashes.issubset(attested_plans):
        raise ValueError(f"domain {expected_domain!r} did not attest every plan hash")
    launchers = payload.get("launchers")
    if not isinstance(launchers, Sequence) or isinstance(launchers, (str, bytes)):
        raise ValueError(f"domain {expected_domain!r} has no launcher attestation")
    if launcher_sha256 not in {
        str(item.get("sha256")) for item in launchers if isinstance(item, Mapping)
    }:
        raise ValueError(f"domain {expected_domain!r} did not attest its exact launcher")
    settings = payload.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError(f"domain {expected_domain!r} has no per-setting compatibility results")
    compatible = {
        str(setting_id)
        for setting_id, result in settings.items()
        if isinstance(result, Mapping) and result.get("status") == "passed"
    }
    return {**payload, "compatible_setting_ids": sorted(compatible)}


def _load_plans(snapshot: Mapping[str, Any]) -> tuple[dict[str, Path], dict[str, dict[int, dict[str, Any]]]]:
    paths: dict[str, Path] = {}
    rows: dict[str, dict[int, dict[str, Any]]] = {}
    raw_plans = snapshot.get("plans")
    if not isinstance(raw_plans, Sequence) or isinstance(raw_plans, (str, bytes)) or not raw_plans:
        raise ValueError("GPU fill snapshot needs at least one materialized plan")
    for item in raw_plans:
        if not isinstance(item, Mapping):
            raise ValueError("plan entries must be mappings")
        source = Path(str(item.get("path", ""))).resolve()
        expected = _hex(item.get("sha256"), 64, "plan SHA256")
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"materialized plan missing or changed: {source}")
        if expected in paths:
            raise ValueError(f"duplicate plan SHA256 in snapshot: {expected}")
        paths[expected] = source
        plan_rows: dict[int, dict[str, Any]] = {}
        for index, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("schema_version") != PAPER_SETTING_CAMPAIGN_SCHEMA:
                raise ValueError("GPU fill accepts only formal v2 paper-setting rows")
            row_index = int(row.get("index", index))
            if row_index in plan_rows:
                raise ValueError(f"duplicate plan index {row_index} in {source}")
            if not isinstance(row.get("model_setting"), Mapping):
                raise ValueError(f"v2 plan row {row_index} lacks model-setting provenance")
            plan_rows[row_index] = row
        rows[expected] = plan_rows
    return paths, rows


def _validate_locations(
    raw: Any,
    *,
    domains: set[str],
) -> dict[str, dict[str, bool]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("row locations must be a sequence")
    locations: dict[str, dict[str, bool]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("row location entries must be mappings")
        domain = str(item.get("domain", ""))
        if domain not in domains or domain in locations:
            raise ValueError(f"unknown or duplicate row location domain: {domain!r}")
        locations[domain] = {
            field: bool(item.get(field, False))
            for field in ("complete", "incomplete_artifacts", "failure", "lock")
        }
    if set(locations) != domains:
        raise ValueError("every row needs an artifact snapshot from every target domain")
    return locations


def build_gpu_fill_plan(
    snapshot_path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Build non-overlapping held-submission transactions from a fresh snapshot."""

    source = Path(snapshot_path).resolve()
    snapshot = _load_json(source)
    if snapshot.get("schema_version") != GPU_FILL_SNAPSHOT_SCHEMA:
        raise ValueError("unsupported GPU fill snapshot schema")
    snapshot_sha = sha256_file(source)
    repo = Path(repository_root).resolve()
    plan_paths, plan_rows = _load_plans(snapshot)
    required_plan_hashes = set(plan_paths)

    raw_domains = snapshot.get("domains")
    if not isinstance(raw_domains, Sequence) or isinstance(raw_domains, (str, bytes)):
        raise ValueError("snapshot domains must be a sequence")
    domains: dict[str, dict[str, Any]] = {}
    for raw in raw_domains:
        if not isinstance(raw, Mapping):
            raise ValueError("domain entries must be mappings")
        domain = str(raw.get("name", ""))
        if not domain or domain in domains:
            raise ValueError(f"invalid or duplicate domain name: {domain!r}")
        idle = int(raw.get("idle_allocatable_gpus", -1))
        max_bundle = int(raw.get("max_gpus_per_job", 0))
        cpus = int(raw.get("cpus_per_gpu", 0))
        memory = int(raw.get("memory_gib_per_gpu", 0))
        if idle < 0 or not 1 <= max_bundle <= 8 or cpus < 1 or memory < 1:
            raise ValueError(f"invalid capacity/resource declaration for domain {domain!r}")
        launcher = (repo / str(raw.get("launcher", ""))).resolve()
        if repo not in launcher.parents or not launcher.is_file():
            raise ValueError(f"launcher is missing or outside the repository: {launcher}")
        launcher_sha = sha256_file(launcher)
        if launcher_sha != raw.get("launcher_sha256"):
            raise ValueError(f"launcher SHA256 changed for domain {domain!r}")
        attestation = verify_domain_attestation(
            raw.get("attestation", ""),
            expected_domain=domain,
            expected_sha256=str(raw.get("attestation_sha256", "")),
            launcher_sha256=launcher_sha,
            required_plan_hashes=required_plan_hashes,
        )
        remote_plans = raw.get("remote_plan_paths")
        if not isinstance(remote_plans, Mapping) or set(remote_plans) != required_plan_hashes:
            raise ValueError(f"domain {domain!r} needs exact remote paths for every plan")
        domains[domain] = {
            **dict(raw),
            "idle_allocatable_gpus": idle,
            "max_gpus_per_job": max_bundle,
            "cpus_per_gpu": cpus,
            "memory_gib_per_gpu": memory,
            "launcher": launcher.relative_to(repo).as_posix(),
            "launcher_sha256": launcher_sha,
            "compatible_setting_ids": set(attestation["compatible_setting_ids"]),
        }
    domain_names = set(domains)

    jobs_by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_jobs = snapshot.get("jobs", [])
    if not isinstance(raw_jobs, Sequence) or isinstance(raw_jobs, (str, bytes)):
        raise ValueError("snapshot jobs must be a sequence")
    for raw in raw_jobs:
        if not isinstance(raw, Mapping):
            raise ValueError("job entries must be mappings")
        job = dict(raw)
        state = str(job.get("state", "")).upper()
        if state not in _ACTIVE_STATES:
            continue
        domain = str(job.get("domain", ""))
        plan_sha = str(job.get("plan_sha256", ""))
        index = int(job.get("index", -1))
        if domain not in domains or plan_sha not in plan_rows or index not in plan_rows[plan_sha]:
            raise ValueError("active job does not own a known plan row/domain")
        job["state"] = state
        job["job_id"] = str(job.get("job_id", ""))
        if not job["job_id"]:
            raise ValueError("active jobs need exact job_id values")
        jobs_by_row[_row_key(plan_sha, index)].append(job)
    duplicates = {key: jobs for key, jobs in jobs_by_row.items() if len(jobs) > 1}
    if duplicates:
        raise ValueError(f"active duplicate row owners detected: {sorted(duplicates)}")

    raw_status = snapshot.get("rows")
    if not isinstance(raw_status, Sequence) or isinstance(raw_status, (str, bytes)):
        raise ValueError("snapshot rows must be a sequence")
    status_by_key: dict[str, dict[str, Any]] = {}
    for raw in raw_status:
        if not isinstance(raw, Mapping):
            raise ValueError("row status entries must be mappings")
        plan_sha = str(raw.get("plan_sha256", ""))
        index = int(raw.get("index", -1))
        if plan_sha not in plan_rows or index not in plan_rows[plan_sha]:
            raise ValueError("row status does not identify a materialized plan row")
        key = _row_key(plan_sha, index)
        if key in status_by_key:
            raise ValueError(f"duplicate row status: {key}")
        status_by_key[key] = {
            **dict(raw),
            "locations": _validate_locations(raw.get("locations"), domains=domain_names),
        }
    expected_keys = {
        _row_key(plan_sha, index)
        for plan_sha, rows in plan_rows.items()
        for index in rows
    }
    if set(status_by_key) != expected_keys:
        missing = sorted(expected_keys - set(status_by_key))
        extra = sorted(set(status_by_key) - expected_keys)
        raise ValueError(f"row snapshot coverage is not exact; missing={missing}, extra={extra}")

    candidates: list[dict[str, Any]] = []
    blocked = Counter()
    for key in sorted(expected_keys):
        plan_sha, raw_index = key.split(":", 1)
        index = int(raw_index)
        row = plan_rows[plan_sha][index]
        locations = status_by_key[key]["locations"]
        job = jobs_by_row.get(key, [])
        active = job[0] if job else None
        if any(location["complete"] for location in locations.values()):
            blocked["already_complete"] += 1
            continue
        if any(location["failure"] for location in locations.values()):
            blocked["failure_requires_diagnosis"] += 1
            continue
        if any(location["incomplete_artifacts"] for location in locations.values()):
            blocked["incomplete_artifacts_require_diagnosis"] += 1
            continue
        if any(location["lock"] for location in locations.values()):
            blocked["active_or_stale_lock"] += 1
            continue
        if active and active["state"] in _RUNNING_STATES | {"SUSPENDED"}:
            blocked["healthy_or_nonmigratable_active_job"] += 1
            continue
        if active and int(active.get("fresh_observations", 0)) < 2:
            blocked["pending_job_lacks_two_fresh_observations"] += 1
            continue
        setting = str(row["model_setting"]["setting_id"])
        candidates.append(
            {
                "key": key,
                "plan_sha256": plan_sha,
                "index": index,
                "row": row,
                "setting_id": setting,
                "source_job": active,
                "priority": 0 if active is None else 1,
                "start_estimate": str((active or {}).get("start_estimate", "9999-12-31T23:59:59Z")),
            }
        )
    unowned = sorted(
        (item for item in candidates if item["source_job"] is None),
        key=lambda item: item["key"],
    )
    delayed = sorted(
        (item for item in candidates if item["source_job"] is not None),
        key=lambda item: (item["start_estimate"], item["key"]),
        reverse=True,
    )
    candidates = [*unowned, *delayed]

    assignments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    remaining = {name: int(domain["idle_allocatable_gpus"]) for name, domain in domains.items()}
    for candidate in candidates:
        eligible = []
        for name, domain in domains.items():
            if remaining[name] < 1 or candidate["setting_id"] not in domain["compatible_setting_ids"]:
                continue
            source_job = candidate["source_job"]
            if source_job and source_job["domain"] == name:
                source_shape = str(source_job.get("resource_shape_id", ""))
                if source_shape == str(domain.get("resource_shape_id", "")):
                    continue
            eligible.append(name)
        if not eligible:
            blocked["no_idle_compatible_target"] += 1
            continue
        target = sorted(eligible, key=lambda name: (-remaining[name], name))[0]
        assignments[target].append(candidate)
        remaining[target] -= 1

    transactions: list[dict[str, Any]] = []
    assigned_keys: set[str] = set()
    for domain_name in sorted(assignments):
        domain = domains[domain_name]
        rows = assignments[domain_name]
        bundle_size = int(domain["max_gpus_per_job"])
        for offset in range(0, len(rows), bundle_size):
            bundle = rows[offset : offset + bundle_size]
            keys = [item["key"] for item in bundle]
            if assigned_keys.intersection(keys):
                raise RuntimeError("planner attempted to assign a row twice")
            assigned_keys.update(keys)
            plan_hashes = {item["plan_sha256"] for item in bundle}
            if len(plan_hashes) != 1:
                raise ValueError("one Slurm bundle may not mix immutable plan files")
            plan_sha = next(iter(plan_hashes))
            transaction_seed = json.dumps(
                {"snapshot": snapshot_sha, "target": domain_name, "rows": keys},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            transaction_id = hashlib.sha256(transaction_seed).hexdigest()[:20]
            migrating = any(item["source_job"] is not None for item in bundle)
            steps = [
                "stage_target_held",
                "verify_target_receipt_resources_and_global_nonoverlap",
            ]
            if migrating:
                steps.append("hold_each_pending_source")
            steps.extend(
                [
                    "release_target_once",
                    "confirm_target_running_with_exact_gpu_cpu_memory",
                ]
            )
            if migrating:
                steps.append("cancel_only_the_replaced_pending_sources")
            transactions.append(
                {
                    "transaction_id": transaction_id,
                    "kind": "migrate_pending" if migrating else "assign_unowned",
                    "state": "planned",
                    "target_domain": domain_name,
                    "target_partition": domain.get("partition"),
                    "launcher": domain["launcher"],
                    "launcher_sha256": domain["launcher_sha256"],
                    "remote_plan_path": domain["remote_plan_paths"][plan_sha],
                    "plan_sha256": plan_sha,
                    "indices": [item["index"] for item in bundle],
                    "setting_ids": [item["setting_id"] for item in bundle],
                    "resources": {
                        "nodes": 1,
                        "gpus": len(bundle),
                        "tasks": len(bundle),
                        "gpus_per_task": 1,
                        "cpus_per_task": domain["cpus_per_gpu"],
                        "memory_gib": len(bundle) * domain["memory_gib_per_gpu"],
                    },
                    "held_submission": {
                        "sbatch_flags": [
                            "--hold",
                            "--parsable",
                            "--nodes=1",
                            f"--ntasks={len(bundle)}",
                            f"--cpus-per-task={domain['cpus_per_gpu']}",
                            f"--gres=gpu:{len(bundle)}",
                            f"--mem={len(bundle) * domain['memory_gib_per_gpu']}G",
                            f"--partition={domain.get('partition')}",
                            f"--comment=pdeobs-v2:{transaction_id}",
                        ],
                        "export": {
                            "PDEOBS_DOMAIN": domain_name,
                            "PDEOBS_DOMAIN_ATTESTATION": domain["attestation"],
                            "PDEOBS_DOMAIN_ATTESTATION_SHA256": domain[
                                "attestation_sha256"
                            ],
                            "PDEOBS_PLAN_SHA256": plan_sha,
                        },
                        "launcher_arguments": [
                            domain["remote_plan_paths"][plan_sha],
                            *[str(item["index"]) for item in bundle],
                        ],
                    },
                    "source_jobs": [
                        item["source_job"] for item in bundle if item["source_job"] is not None
                    ],
                    "required_steps": steps,
                    "rollback_before_target_running": (
                        "cancel_held_target_and_release_every_held_source"
                        if migrating
                        else "cancel_held_target"
                    ),
                    "prohibitions": [
                        "never_cancel_a_running_source",
                        "never_release_twice",
                        "never_create_a_second_target_for_the_same_row",
                    ],
                }
            )

    unfilled = [
        {
            "domain": name,
            "idle_allocatable_gpus": count,
            "reason": "no_remaining_verified_nonoverlapping_compatible_rows",
        }
        for name, count in sorted(remaining.items())
        if count > 0
    ]
    return {
        "schema_version": GPU_FILL_PLAN_SCHEMA,
        "snapshot": str(source),
        "snapshot_sha256": snapshot_sha,
        "dataset_summary_sha256": FINAL_DATASET_SUMMARY_SHA256,
        "transactions": transactions,
        "assigned_rows": len(assigned_keys),
        "unfilled_capacity": unfilled,
        "blocked_row_counts": dict(sorted(blocked.items())),
        "policy": {
            "healthy_running_jobs_moved": 0,
            "formal_v2_only": True,
            "target_submission_initial_state": "HELD",
            "cancel_source_only_after_target_running": True,
        },
    }


__all__ = [
    "DOMAIN_ATTESTATION_SCHEMA",
    "DOMAIN_SNAPSHOT_SCHEMA",
    "GPU_FILL_PLAN_SCHEMA",
    "GPU_FILL_SNAPSHOT_SCHEMA",
    "SLURM_OBSERVATION_SCHEMA",
    "FINAL_DATASET_SUMMARY_SHA256",
    "build_gpu_fill_plan",
    "verify_domain_attestation",
]
