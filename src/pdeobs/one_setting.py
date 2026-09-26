"""Frozen helpers for the all-PDE one-setting cross-observation campaign.

This module contains only protocol materialization: it does not submit jobs or
silently promote a candidate.  Keeping the row identity builder in the package
makes portable preflight and later Slurm workers resolve exactly the same
models, masks, temporal recurrence, and optimizer settings.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

CAMPAIGN_SCHEMA = "pdeobs.one-setting-cross-observation-campaign/v1"
DATASET_SUMMARY_SHA256 = (
    "b0141eec2199615a27facc75e2517ec4ec7e00913638e48c84ab2c88a2be77c1"
)
PDE_ORDER = (
    "darcy",
    "poisson",
    "helmholtz",
    "heat",
    "reaction_diffusion",
    "burgers",
    "navier_stokes",
)
VIEW_ORDER = (
    "random_50pct",
    "random_65pct",
    "random_80pct",
    "block_observed_50pct",
    "line_sensors_50pct",
    "horizontal_lines_50pct",
    "vertical_lines_50pct",
    "boundary_band_50pct",
    "clustered_50pct",
)
LEARNED_METHOD_ORDER = (
    "ufno_2d",
    "fno",
    "cno",
    "deeponet",
    "gnot",
    "transolver",
    "jeno",
    "pino",
)
PUBLIC_LEARNED_METHODS = tuple(value for value in LEARNED_METHOD_ORDER if value != "jeno")
CLASSICAL_METHOD_ORDER = ("gaussian_rbf", "gappy_pod")
TEMPORAL_PDES = frozenset({"heat", "reaction_diffusion", "burgers", "navier_stokes"})


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    return hashlib.sha256(source.read_bytes()).hexdigest()


def load_campaign(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != CAMPAIGN_SCHEMA:
        raise ValueError(f"campaign must use {CAMPAIGN_SCHEMA}")
    campaign = dict(payload)
    validate_campaign(campaign)
    return campaign


def validate_campaign(campaign: Mapping[str, Any]) -> None:
    if campaign.get("status") != "candidate_preflight_only":
        raise ValueError("the campaign must remain candidate_preflight_only before audit")
    if campaign.get("release_eligible") is not False or int(campaign.get("formal_credit", -1)) != 0:
        raise ValueError("an unaudited candidate must have zero formal credit")
    immutable = campaign.get("immutable_inputs", {})
    if immutable.get("dataset_summary_sha256") != DATASET_SUMMARY_SHA256:
        raise ValueError("dataset summary anchor changed")
    problems = campaign.get("problem_settings", {})
    if tuple(problems) != PDE_ORDER:
        raise ValueError("problem settings must use the frozen seven-PDE order")
    for pde, problem in problems.items():
        expected_task = "rollout" if pde in TEMPORAL_PDES else "recovery"
        if problem.get("task") != expected_task:
            raise ValueError(f"{pde} must use task={expected_task}")
        expected_boundary = "periodic" if pde in TEMPORAL_PDES else "dirichlet"
        if problem.get("boundary") != expected_boundary or problem.get("setting") != "smooth_grf":
            raise ValueError(f"{pde} representative macrodomain changed")
        expected_state = "vorticity" if pde == "navier_stokes" else "native"
        if problem.get("state_representation") != expected_state:
            raise ValueError(f"{pde} must use state_representation={expected_state}")
    if tuple(campaign.get("views", {})) != VIEW_ORDER:
        raise ValueError("observation views differ from the frozen nine-view order")
    methods = campaign.get("methods", {})
    if tuple(methods.get("classical", ())) != CLASSICAL_METHOD_ORDER:
        raise ValueError("classical method slots changed")
    if tuple(methods.get("learned", ())) != LEARNED_METHOD_ORDER:
        raise ValueError("learned method slots changed")
    temporal = campaign.get("temporal_contract", {})
    if (
        temporal.get("input_frames") != ["t0"]
        or temporal.get("prediction_frames") != ["t1", "t2", "t3"]
        or float(temporal.get("teacher_forcing_ratio", -1.0)) != 0.0
        or temporal.get("ground_truth_future_as_recurrent_input") != "forbidden"
    ):
        raise ValueError("free-rollout contract changed")
    evaluation = campaign.get("evaluation", {})
    if (
        evaluation.get("learned", {}).get("training_rows") != 504
        or evaluation.get("learned", {}).get("unique_blocks") != 4536
        or evaluation.get("classical", {}).get("unique_blocks") != 126
        or evaluation.get("unique_blocks_total") != 4662
    ):
        raise ValueError("campaign accounting changed")


def stable_split(
    metadata: Sequence[Mapping[str, Any]], seed: int, identity: str
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Make the exact 90/10 split while preserving all three regimes."""

    if len(metadata) != 2000:
        raise ValueError(f"production split requires exactly 2000 rows, found {len(metadata)}")
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        groups.setdefault(str(row.get("regime")), []).append(index)
    if set(groups) != {"low", "medium", "high"}:
        raise ValueError(f"production data have wrong regimes: {sorted(groups)}")
    test_total = len(metadata) // 10
    test_counts = {regime: len(indices) // 10 for regime, indices in groups.items()}
    remainder = test_total - sum(test_counts.values())
    order = sorted(groups, key=lambda name: (-(len(groups[name]) % 10), name))
    for regime in order[:remainder]:
        test_counts[regime] += 1
    train: list[int] = []
    test: list[int] = []
    group_receipt: dict[str, Any] = {}
    for regime, indices in sorted(groups.items()):
        ranked = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"{seed}|{identity}|{metadata[index]['sample_id']}".encode()
            ).hexdigest(),
        )
        count = test_counts[regime]
        test.extend(ranked[:count])
        train.extend(ranked[count:])
        group_receipt[regime] = {
            "records": len(ranked),
            "train": len(ranked) - count,
            "test": count,
        }
    if len(train) != 1800 or len(test) != 200 or set(train) & set(test):
        raise RuntimeError("failed to construct the exact 1800/200 split")
    train_ids = sorted(str(metadata[index]["sample_id"]) for index in train)
    test_ids = sorted(str(metadata[index]["sample_id"]) for index in test)
    return train, test, {
        "schema_version": "pdeobs.one-setting-split/v1",
        "algorithm": "sha256_rank_within_regime_largest_remainder_exact_ten_percent_test",
        "seed": int(seed),
        "macrodomain_identity": identity,
        "records": 2000,
        "train_records": 1800,
        "validation_records": 0,
        "test_records": 200,
        "regime_counts": group_receipt,
        "train_sample_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "test_sample_ids_sha256": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
    }


def preflight_subset(
    metadata: Sequence[Mapping[str, Any]], seed: int, identity: str, *, per_regime: int = 30
) -> tuple[list[int], dict[str, Any]]:
    """Select a deterministic canonical-train diagnostic subset.

    The diagnostic subset intentionally needs only immutable canonical-train
    shards.  It therefore never needs the held-out test identities merely
    to verify three-epoch optimization behavior.
    """

    selected: list[int] = []
    counts: dict[str, int] = {}
    for regime in ("low", "medium", "high"):
        candidates = [
            index
            for index, row in enumerate(metadata)
            if str(row.get("regime")) == regime and str(row.get("split")) == "train"
        ]
        ranked = sorted(
            candidates,
            key=lambda index: hashlib.sha256(
                f"preflight|{seed}|{identity}|{metadata[index]['sample_id']}".encode()
            ).hexdigest(),
        )
        if len(ranked) < per_regime:
            raise ValueError(
                f"preflight needs {per_regime} canonical-train {regime} rows, found {len(ranked)}"
            )
        chosen = ranked[:per_regime]
        selected.extend(chosen)
        counts[regime] = len(chosen)
    ids = sorted(str(metadata[index]["sample_id"]) for index in selected)
    return selected, {
        "schema_version": "pdeobs.one-setting-train-subset/v1",
        "selection": "sha256_rank_from_canonical_train_balanced_by_regime",
        "seed": int(seed),
        "identity": identity,
        "records": len(selected),
        "regime_counts": counts,
        "sample_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    }


def subset_dataset(dataset: Any, positions: Sequence[int]) -> Any:
    selected = copy.copy(dataset)
    selected.indices = [dataset.indices[index] for index in positions]
    selected.metadata = [dataset.metadata[index] for index in positions]
    return selected


def _method_spec(registry: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    try:
        spec = registry["settings"][method]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"method {method!r} is missing from the method registry") from exc
    if spec.get("category") != "learned":
        raise ValueError(f"method {method!r} is not a learned method")
    return spec


def build_experiment_config(
    campaign: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    dataset_root: str | Path,
    pde: str,
    method: str,
    view: str,
    epochs: int,
) -> dict[str, Any]:
    """Resolve one public learned-method row without accessing test data."""

    if pde not in PDE_ORDER or method not in PUBLIC_LEARNED_METHODS or view not in VIEW_ORDER:
        raise ValueError("unknown PDE, public method, or view")
    if int(epochs) not in {3, 500}:
        raise ValueError("only the 3-epoch preflight or 500-epoch production is supported")
    problem = campaign["problem_settings"][pde]
    spec = _method_spec(registry, method)
    architecture = dict(spec.get("architecture", {}))
    optimizer = dict(spec.get("optimizer", {}))
    scheduler = dict(spec.get("scheduler", {}))
    health = dict(spec.get("health", {}))
    if pde in TEMPORAL_PDES:
        architecture.update(dict(spec.get("temporal_architecture", {})))
        optimizer.update(dict(spec.get("temporal_optimizer", {})))
        scheduler.update(dict(spec.get("temporal_scheduler", {})))
        health.update(dict(spec.get("temporal_health", {})))
    geometry_channels = 1
    model_names = {
        "ufno_2d": "ufno",
        "fno": "fno",
        "cno": "cno",
        "deeponet": "deeponet",
        "gnot": "gnot",
        "transolver": "transolver",
        "pino": "pino",
    }
    model_kwargs: dict[str, Any] = {
        "in_channels": 1,
        "out_channels": 1,
        "geometry_channels": geometry_channels,
    }
    if method == "ufno_2d":
        model_kwargs.update(
            width=architecture["width"],
            modes=architecture["modes"],
            padding=architecture["padding"],
            dropout=0.0,
        )
    elif method == "fno":
        model_kwargs.update(
            width=architecture["width"],
            modes=architecture["modes"],
            layers=architecture["layers"],
        )
    elif method == "cno":
        model_kwargs.update(width=architecture["width"])
    elif method == "deeponet":
        model_kwargs.update(
            hidden=architecture["hidden"],
            latent=architecture["latent"],
            branch_layers=architecture["branch_layers"],
            trunk_layers=architecture["trunk_layers"],
            branch_mode=architecture.get("branch_mode", "sensor_set"),
            branch_grid=architecture.get("branch_grid", 4),
            branch_width=architecture.get("branch_width"),
        )
    elif method == "gnot":
        model_kwargs.update(
            hidden=architecture["hidden"],
            layers=architecture["layers"],
            heads=architecture["heads"],
            experts=architecture["experts"],
        )
    elif method == "transolver":
        model_kwargs.update(
            hidden=architecture["hidden"],
            layers=architecture["layers"],
            heads=architecture["heads"],
            slices=architecture["physics_slices"],
            mlp_ratio=architecture["mlp_ratio"],
        )
    else:
        model_kwargs.update(
            width=architecture["width"],
            modes=architecture["modes"],
            layers=architecture["layers"],
            physics_contract=(
                "pino_rollout_spectral_v1" if pde in TEMPORAL_PDES else "pino_static_fd_v1"
            ),
        )

    base = {"name": model_names[method], "kwargs": model_kwargs}
    method_config = (
        {"name": "autoregressive", "base": base}
        if pde in TEMPORAL_PDES
        else base
    )
    physics_loss = "none"
    physics_weight = 0.0
    data_weight = 1.0
    if method == "pino":
        physics_loss = (
            "pino_rollout_spectral_v1" if pde in TEMPORAL_PDES else "pino_static_fd_v1"
        )
        data_weight = float(spec["objective"].get("data_loss_weight", 1.0))
        physics_weight = float(spec["objective"]["physics_loss_weight"])
    scheduler_name = str(scheduler.get("name", "step"))
    expected_records = 90 if int(epochs) == 3 else 1800
    scheduler_steps_per_epoch = math.ceil(
        expected_records / int(optimizer["batch_size"])
    )
    config: dict[str, Any] = {
        "schema_version": "pdeobs.experiment/v1",
        "name": f"{pde}-{method}-{view}-{epochs}epoch",
        "seed": int(campaign["seed"]),
        "task": problem["task"],
        "data": {
            "root": str(Path(dataset_root)),
            "train_glob": f"{pde}/{problem['boundary']}/{problem['setting']}/**/*.h5",
            "verify_shards": False,
            "split": "iid",
            "filters": {
                "pde": pde,
                "boundary": problem["boundary"],
                "setting": problem["setting"],
            },
            "mask": dict(campaign["views"][view]["mask"]),
            "state_representation": problem["state_representation"],
            "input_horizon": 1,
            "training_horizons": [3],
        },
        "method": method_config,
        "training": {
            "epochs": int(epochs),
            "batch_size": int(optimizer["batch_size"]),
            "optimizer": str(optimizer["name"]),
            "learning_rate": float(optimizer["learning_rate"]),
            "weight_decay": float(optimizer["weight_decay"]),
            "loss": "relative_l2",
            "data_loss_weight": data_weight,
            "physics_loss": physics_loss,
            "physics_loss_weight": physics_weight,
            "physics_loss_epsilon": 1.0e-12,
            "scheduler": scheduler_name,
            "scheduler_step_size": int(scheduler.get("step_size_epochs", 100)),
            "scheduler_gamma": float(scheduler.get("gamma", 0.5)),
            "scheduler_steps_per_epoch": scheduler_steps_per_epoch,
            "scheduler_pct_start": float(scheduler.get("pct_start", 0.3)),
            "scheduler_div_factor": float(scheduler.get("div_factor", 25.0)),
            "scheduler_final_div_factor": float(
                scheduler.get("final_div_factor", 1.0e4)
            ),
            "scheduler_anneal_strategy": str(
                scheduler.get("anneal_strategy", "cos")
            ),
            "checkpoint_every": 1,
            "early_stopping_patience": None,
            "teacher_forcing_ratio": 0.0,
            "mixed_precision": False,
            "grad_clip": health.get("grad_clip", 1.0),
            "deterministic": True,
            "health_policy": "fail",
            "loss_spike_scope": "epoch",
            "loss_spike_ratio": 10.0,
            "loss_spike_consecutive_windows": 3,
            "gradient_norm_warning": float(
                health.get("gradient_norm_warning", 1000.0)
            ),
            "gradient_warning_consecutive_windows": int(
                health.get("gradient_warning_consecutive_windows", 3)
            ),
            "gradient_warning_fatal": bool(
                health.get("gradient_warning_fatal", True)
            ),
            "output_std_ratio_collapse_max": 1.0e-4,
            "collapse_consecutive_windows": 20,
            "validation_improvement_min_fraction": None,
            "num_workers": 0 if epochs == 3 else 4,
            "pin_memory": True,
            "persistent_workers": epochs == 500,
        },
        "evaluation": {"horizons": [1, 2, 3] if pde in TEMPORAL_PDES else []},
    }
    return config


def row_identity(
    *, campaign_sha256: str, registry_sha256: str, pde: str, method: str, view: str
) -> dict[str, Any]:
    payload = {
        "campaign_sha256": campaign_sha256,
        "registry_sha256": registry_sha256,
        "pde": pde,
        "method": method,
        "training_view": view,
        "seed": 20260804,
        "epochs": 500,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "output_identity_sha256": hashlib.sha256(encoded).hexdigest()}


__all__ = [
    "CAMPAIGN_SCHEMA",
    "CLASSICAL_METHOD_ORDER",
    "DATASET_SUMMARY_SHA256",
    "LEARNED_METHOD_ORDER",
    "PDE_ORDER",
    "PUBLIC_LEARNED_METHODS",
    "TEMPORAL_PDES",
    "VIEW_ORDER",
    "build_experiment_config",
    "load_campaign",
    "preflight_subset",
    "row_identity",
    "sha256_file",
    "stable_split",
    "subset_dataset",
    "validate_campaign",
]
