"""Materialize the all-factor full-to-partial campaign without implicit rows."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .campaign import sha256_file
from .config import load_config

FULL_TO_PARTIAL_SCHEMA = "pdeobs.full-to-partial-campaign/v1"
PLAN_SCHEMA = "pdeobs.full-to-partial-plan/v1"
EXPECTED_BOUNDARIES = ("dirichlet", "neumann", "periodic", "robin_obstacle")
EXPECTED_SETTINGS = (
    "smooth_grf",
    "medium_grf",
    "rough_grf",
    "low_frequency_fourier",
    "multi_frequency_fourier",
    "gaussian_blobs",
    "piecewise_blocks",
    "threshold_level_set",
    "dipole_vortex_pair",
    "front_ring_shock",
)
EXPECTED_REGIMES = ("low", "medium", "high")
EXPECTED_VIEWS = (
    "full",
    "random_1pct",
    "random_3pct",
    "random_5pct",
    "random_10pct",
    "regular_grid",
    "block_missing",
    "line_sensors",
    "boundary_sensors",
    "clustered_sensors",
)


def _sequence(value: Any, *, field: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence")
    result = tuple(value)
    identities = [
        json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)
        for item in result
    ]
    if not result or len(identities) != len(set(identities)):
        raise ValueError(f"{field} must be non-empty and unique")
    return result


def _safe_repo_file(root: Path, value: Any, *, field: str) -> Path:
    path = (root / str(value)).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"{field} is missing or outside the repository: {value!r}")
    return path


def _identity_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_full_to_partial_plan(
    campaign_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build 84 neural, 7 POD, and 7 RBF scheduler-row identities."""

    source = Path(campaign_path).resolve()
    root = Path(repository_root).resolve() if repository_root else source.parents[2]
    campaign = load_config(source)
    if campaign.get("schema_version") != FULL_TO_PARTIAL_SCHEMA:
        raise ValueError(f"campaign schema must be {FULL_TO_PARTIAL_SCHEMA!r}")
    if campaign.get("status") != "candidate_preflight_only":
        raise ValueError("only a candidate_preflight_only campaign may be materialized here")
    name = str(campaign.get("name", "")).strip()
    if not name:
        raise ValueError("campaign name is required")
    seed = int(campaign.get("seed"))

    factors = dict(campaign.get("dataset", {}).get("factors", {}))
    pdes = tuple(str(value) for value in _sequence(factors.get("pdes"), field="pdes"))
    if len(pdes) != 7:
        raise ValueError("the campaign must contain exactly seven PDE families")
    boundaries = tuple(
        str(value) for value in _sequence(factors.get("boundaries"), field="boundaries")
    )
    settings = tuple(
        str(value) for value in _sequence(factors.get("settings"), field="settings")
    )
    regimes = tuple(
        str(value) for value in _sequence(factors.get("regimes"), field="regimes")
    )
    if boundaries != EXPECTED_BOUNDARIES:
        raise ValueError("boundary factors differ from the frozen four-protocol order")
    if settings != EXPECTED_SETTINGS:
        raise ValueError("condition settings differ from the frozen ten-setting order")
    if regimes != EXPECTED_REGIMES:
        raise ValueError("regimes differ from the frozen three-regime order")

    partial = tuple(
        str(row["id"])
        for row in _sequence(
            campaign.get("observation_protocols", {}).get("partial"),
            field="partial observation protocols",
        )
    )
    views = (str(campaign.get("observation_protocols", {}).get("full", {}).get("id")), *partial)
    if views != EXPECTED_VIEWS:
        raise ValueError("evaluation views differ from the frozen full-plus-nine order")

    registry = _safe_repo_file(
        root, campaign.get("model_setting_registry"), field="model_setting_registry"
    )
    registry_sha = sha256_file(registry)
    method_rows = _sequence(campaign.get("methods"), field="methods")
    if len(method_rows) != 8:
        raise ValueError("the comparison must contain exactly eight method slots")
    method_ids = tuple(str(row["id"]) for row in method_rows)
    if method_ids != (
        "rbf",
        "gappy_pod",
        "unet",
        "fno",
        "cno",
        "deeponet",
        "gnot",
        "transolver",
    ):
        raise ValueError("method slots differ from the frozen eight-method order")

    campaign_sha = sha256_file(source)
    dataset = dict(campaign["dataset"])
    rows: list[dict[str, Any]] = []
    output_paths: set[str] = set()
    for method in method_rows:
        method = dict(method)
        method_id = str(method["id"])
        method_name = str(method["method_name"])
        config = _safe_repo_file(root, method.get("config"), field=f"{method_id}.config")
        if method.get("execution_status") != "implemented_unattested":
            raise ValueError(f"{method_id} must remain implemented_unattested before preflight")
        if method_id == "rbf":
            cases = (("no_fit", "eval"),)
        elif method_id == "gappy_pod":
            cases = (("full_basis", "train_eval"),)
        else:
            cases = (("full_train", "train_eval"), ("mixed_partial_train", "train_eval"))
        case_names = {case for case, _mode in cases}
        raw_case_overrides = method.get("training_case_overrides", {})
        if not isinstance(raw_case_overrides, Mapping):
            raise ValueError(f"{method_id}.training_case_overrides must be a mapping")
        unexpected_cases = set(raw_case_overrides) - case_names
        if unexpected_cases:
            raise ValueError(
                f"{method_id}.training_case_overrides contains unsupported cases: "
                f"{sorted(unexpected_cases)}"
            )
        case_overrides: dict[str, tuple[str, ...]] = {}
        for case_name, values in raw_case_overrides.items():
            overrides = tuple(
                str(value).strip()
                for value in _sequence(
                    values,
                    field=f"{method_id}.training_case_overrides.{case_name}",
                )
            )
            if any(not value or "\n" in value or "\r" in value for value in overrides):
                raise ValueError(
                    f"{method_id}.training_case_overrides.{case_name} "
                    "must contain non-empty single-line overrides"
                )
            case_overrides[str(case_name)] = overrides
        for pde, (training_case, mode) in itertools.product(pdes, cases):
            mask_protocol = "mixed_partial" if training_case == "mixed_partial_train" else "full"
            run_relpath = f"{name}/{method_id}/{pde}/{training_case}/seed_{seed}"
            if run_relpath in output_paths:
                raise ValueError(f"duplicate output identity: {run_relpath}")
            output_paths.add(run_relpath)
            identity = {
                "campaign_sha256": campaign_sha,
                "registry_sha256": registry_sha,
                "method": method_id,
                "method_name": method_name,
                "pde": pde,
                "training_case": training_case,
                "seed": seed,
                "evaluation_views": list(views),
            }
            row = {
                "schema_version": PLAN_SCHEMA,
                "index": len(rows),
                "campaign": name,
                "campaign_file": source.relative_to(root).as_posix(),
                "campaign_sha256": campaign_sha,
                "candidate_preflight_only": True,
                "release_eligible": False,
                "method": method_id,
                "method_name": method_name,
                "mode": mode,
                "pde": pde,
                "training_case": training_case,
                "seed": seed,
                "config": config.relative_to(root).as_posix(),
                "config_sha256": sha256_file(config),
                "model_setting_registry": registry.relative_to(root).as_posix(),
                "model_setting_registry_sha256": registry_sha,
                "run_relpath": run_relpath,
                "output_identity_sha256": _identity_sha256(identity),
                "dataset": dataset,
                "factor_product": {
                    "pde": [pde],
                    "boundary": list(boundaries),
                    "setting": list(settings),
                    "regime": list(regimes),
                    "expected_strata": len(boundaries) * len(settings) * len(regimes),
                },
                "evaluation_views": list(views),
                "overrides": [
                    f"name={json.dumps(f'{name}-{method_id}-{pde}-{training_case}')}",
                    f"seed={seed}",
                    f"data.train_glob={json.dumps(f'{pde}/**/*.h5')}",
                    f"data.filters.pde={json.dumps(pde)}",
                    f"data.required_factor_coverage.pde={json.dumps([pde])}",
                    f"data.mask.protocol={json.dumps(mask_protocol)}",
                    *(
                        [f"data.mask.protocols={json.dumps(list(partial))}"]
                        if mask_protocol == "mixed_partial"
                        else []
                    ),
                    *case_overrides.get(training_case, ()),
                ],
            }
            rows.append(row)

    if len(rows) != 98:
        raise RuntimeError(f"expected 98 scheduler-row identities, built {len(rows)}")
    if len({row["output_identity_sha256"] for row in rows}) != len(rows):
        raise RuntimeError("output identity hashes are not globally unique")
    return rows


__all__ = [
    "FULL_TO_PARTIAL_SCHEMA",
    "PLAN_SCHEMA",
    "build_full_to_partial_plan",
]
