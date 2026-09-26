from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdeobs.full_to_partial import EXPECTED_VIEWS
from pdeobs.full_to_partial_execution import (
    execute_full_to_partial_plan_row,
    verify_preflight_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "configs/campaign/full_to_partial_7x4x10_one_seed_v1.candidate.jsonl"


def _settings() -> list[dict[str, object]]:
    return [
        {
            "method": method,
            "pde": pde,
            "training_case": case,
            "pathology_events": [],
            "training_loss_one_step": 1.0,
            "view_relative_l2": {view: 1.0 for view in EXPECTED_VIEWS},
        }
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
        for case in ("full_train", "mixed_partial_train")
    ]


def _attestations(tmp_path: Path) -> dict[str, list[Path]]:
    paths: dict[str, list[Path]] = {"a100": [], "b40": []}
    for domain in paths:
        for replica in range(4):
            path = tmp_path / f"{domain}-{replica}.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "pdeobs.full-to-partial-preflight/v1",
                        "status": "passed",
                        "domain": domain,
                        "replica": replica,
                        "code_commit": "a" * 40,
                        "dataset_summary_sha256": "b" * 64,
                        "candidate_plan_sha256": "c" * 64,
                        "settings": _settings(),
                        "classical_settings": [
                            {
                                "method": method,
                                "pde": pde,
                                "view_relative_l2": {view: 1.0 for view in EXPECTED_VIEWS},
                            }
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
                        ],
                        "evaluation_view_count": 840,
                        "device": {"name": domain},
                        "slurm": {"job_id": str(replica)},
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            paths[domain].append(path)
    return paths


def test_preflight_bundle_requires_four_consistent_pathology_free_replicas(
    tmp_path: Path,
) -> None:
    paths = _attestations(tmp_path)
    result = verify_preflight_bundle(
        paths,
        expected_code_commit="a" * 40,
        expected_dataset_summary_sha256="b" * 64,
        expected_candidate_plan_sha256="c" * 64,
    )
    assert result["status"] == "passed"
    assert result["domains"]["a100"]["passed_settings"] == 84
    assert len(result["bundle_sha256"]) == 64

    broken = json.loads(paths["b40"][0].read_text(encoding="utf-8"))
    broken["settings"][0]["pathology_events"] = [{"kind": "nonfinite"}]
    paths["b40"][0].write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="pathology"):
        verify_preflight_bundle(
            paths,
            expected_code_commit="a" * 40,
            expected_dataset_summary_sha256="b" * 64,
            expected_candidate_plan_sha256="c" * 64,
        )


