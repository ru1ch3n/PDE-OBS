from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from pdeobs.campaign import (
    _observation_diagnostic_summary,
    build_training_plan,
    read_training_plan_row,
    verify_dataset_attestation,
    verify_model_setting_attestation,
    write_training_plan,
)

ROOT = Path(__file__).parents[1]


def test_core_pilot_expands_to_27_unique_owned_outputs(tmp_path: Path) -> None:
    rows = build_training_plan(ROOT / "configs/campaign/core_pilot_t15.yaml", repository_root=ROOT)
    assert len(rows) == 27
    assert [row["index"] for row in rows] == list(range(27))
    assert len({row["run_relpath"] for row in rows}) == 27
    assert {row["method"] for row in rows} == {"unet", "fno", "cno"}
    assert {row["pde"] for row in rows} == {"poisson", "burgers", "navier_stokes"}
    assert {row["observation_protocol"] for row in rows} == {
        "random_3pct",
        "block_missing",
        "boundary_sensors",
    }
    for row in rows:
        expected = f'data.train_glob="{row["pde"]}/**/*.h5"'
        assert expected in row["overrides"]
    plan = write_training_plan(rows, tmp_path / "plan.jsonl")
    assert read_training_plan_row(plan, 26) == rows[26]
    with pytest.raises(IndexError):
        read_training_plan_row(plan, 27)


def test_core_rbf_baseline_expands_to_nine_evaluation_rows() -> None:
    rows = build_training_plan(
        ROOT / "configs/campaign/core_baseline_t15.yaml", repository_root=ROOT
    )
    assert len(rows) == 9
    assert {row["mode"] for row in rows} == {"eval"}
    assert {row["method"] for row in rows} == {"rbf"}
    assert all(
        f'data.train_glob="{row["pde"]}/**/*.h5"' in row["overrides"] for row in rows
    )


def test_core_medium_executable_expands_to_189_nonoverlapping_rows() -> None:
    rows = build_training_plan(
        ROOT / "configs/campaign/core_medium_executable.yaml", repository_root=ROOT
    )
    assert len(rows) == 189
    assert [row["index"] for row in rows] == list(range(189))
    assert len({row["run_relpath"] for row in rows}) == 189
    assert {row["method"] for row in rows} == {"unet", "fno", "cno"}
    assert len({row["pde"] for row in rows}) == 7
    assert len({row["observation_protocol"] for row in rows}) == 9
    assert all(row["dataset"]["selected_release_tier"] == "medium" for row in rows)
    assert all(
        "training.mixed_precision=false" in row["overrides"]
        for row in rows
        if row["method"] == "fno"
    )
    assert all(
        "training.mixed_precision=false" not in row["overrides"]
        for row in rows
        if row["method"] != "fno"
    )


def test_core_full_anchors_expand_to_112_executable_rows() -> None:
    rows = build_training_plan(
        ROOT / "configs/campaign/core_full_anchors_executable.yaml", repository_root=ROOT
    )
    assert len(rows) == 112
    assert [row["index"] for row in rows] == list(range(112))
    assert len({row["run_relpath"] for row in rows}) == 112
    assert {row["method"] for row in rows} == {"rbf", "unet", "fno", "cno"}
    assert {row["mode"] for row in rows if row["method"] == "rbf"} == {"eval"}
    assert len({row["pde"] for row in rows}) == 7
    assert len({row["observation_protocol"] for row in rows}) == 4










def test_dataset_attestation_requires_exact_summary_digest(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"dataset": {"valid": True}}) + "\n", encoding="utf-8")
    digest = hashlib.sha256(summary.read_bytes()).hexdigest()
    row = {
        "dataset": {
            "summary_file": "summary.json",
            "summary_sha256": digest,
            "expected_shards": 3360,
            "expected_samples": 560000,
            "strict_max_pde_loss": 0.05,
        }
    }
    attestation = verify_dataset_attestation(tmp_path, row)
    assert attestation["summary_sha256"] == digest
    summary.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_dataset_attestation(tmp_path, row)


def test_full_observation_diagnosis_requires_a_preregistered_cutoff() -> None:
    matched = {"metrics": {"relative_l2": 0.8}}
    full = {"metrics": {"relative_l2": 0.02}}

    unregistered = _observation_diagnostic_summary(matched, full, {})
    assert unregistered["classification"] == "unclassified_no_preregistered_threshold"
    registered = _observation_diagnostic_summary(
        matched,
        full,
        {"full_observation_relative_l2_good_max": 0.05},
    )
    assert registered["classification"] == "observation_limited"
    assert registered["action"] == "do_not_retry_for_partial_observation_score_alone"


def test_v2_campaign_requires_and_attests_exact_verified_paper_setting(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configs" / "experiment" / "paper_unet.yaml"
    registry_path = tmp_path / "configs" / "method" / "registry.yaml"
    campaign_path = tmp_path / "configs" / "campaign" / "formal.yaml"
    for path in (config_path, registry_path, campaign_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "method:\n"
        "  name: unet\n"
        "  kwargs:\n"
        f"    upstream_revision: {'a' * 40}\n"
        f"    source_file_sha256: {'b' * 64}\n",
        encoding="utf-8",
    )
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    record = {
        "method": "unet",
        "status": "verified",
        "paper_url": "https://arxiv.org/abs/1505.04597",
        "official_code_url": "https://example.edu/authors/unet-code",
        "official_code_revision": "a" * 40,
        "official_source_file": "models/unet.py",
        "official_source_sha256": "b" * 64,
        "source_setting_locator": "paper section 2 and author config original.prototxt",
        "local_config": "configs/experiment/paper_unet.yaml",
        "local_config_sha256": config_sha,
        "supported_pdes": ["poisson"],
        "reproduction_scope": "paper_architecture_training_with_declared_task_adapter",
        "task_adapter": {
            "input_policy": "zero_filled_visible_target_field",
            "spatial_adapter": "deterministic_area_average_128_to_64_before_mask_v1",
            "official_architecture_unchanged": True,
            "full_observation_control": "required",
            "normalization": "declared_test_units",
            "claim_limit": "adapted_recovery_task_only",
        },
    }
    registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "pdeobs.paper-settings/v1",
                "settings": {"unet_original": record},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    campaign = {
        "schema_version": "pdeobs.training-campaign/v2",
        "name": "formal",
        "task": "recovery",
        "model_setting_registry": "configs/method/registry.yaml",
        "dataset": {
            "summary_file": "summary.json",
            "summary_sha256": "0" * 64,
            "expected_shards": 1,
            "expected_samples": 1,
            "strict_max_pde_loss": 0.05,
        },
        "matrix": {
            "pdes": ["poisson"],
            "observation_protocols": ["random_3pct"],
            "seeds": [1],
            "methods": [
                {
                    "id": "unet",
                    "config": "configs/experiment/paper_unet.yaml",
                    "setting_id": "unet_original",
                }
            ],
        },
    }
    campaign_path.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")

    rows = build_training_plan(campaign_path, repository_root=tmp_path)
    assert rows[0]["schema_version"] == "pdeobs.training-campaign/v2"
    assert rows[0]["model_setting"]["setting_id"] == "unet_original"
    attestation = verify_model_setting_attestation(tmp_path, rows[0])
    assert attestation["formal_paper_setting"] is True
    assert attestation["exact_paper_benchmark_reproduction"] is False

    mismatched_config = config_path.read_text(encoding="utf-8").replace(
        "b" * 64, "c" * 64
    )
    config_path.write_text(mismatched_config, encoding="utf-8")
    record["local_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "pdeobs.paper-settings/v1",
                "settings": {"unet_original": record},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config source SHA256 differs"):
        build_training_plan(campaign_path, repository_root=tmp_path)

    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("c" * 64, "b" * 64),
        encoding="utf-8",
    )
    record["local_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()

    record["status"] = "reference_only"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "pdeobs.paper-settings/v1",
                "settings": {"unet_original": record},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only verified"):
        build_training_plan(campaign_path, repository_root=tmp_path)
    with pytest.raises(ValueError, match="SHA256 changed"):
        verify_model_setting_attestation(tmp_path, rows[0])


def test_paper_campaigns_cover_only_verified_poisson_and_navier_stokes() -> None:
    poisson = build_training_plan(
        ROOT / "configs/campaign/core_paper_poisson_t15.yaml", repository_root=ROOT
    )
    navier_stokes = build_training_plan(
        ROOT / "configs/campaign/core_paper_navier_stokes_t15.yaml", repository_root=ROOT
    )
    rows = [*poisson, *navier_stokes]
    assert len(poisson) == 9
    assert len(navier_stokes) == 9
    assert {row["pde"] for row in rows} == {"poisson", "navier_stokes"}
    assert "burgers" not in {row["pde"] for row in rows}
    assert len({row["run_relpath"] for row in rows}) == 18
    assert {row["method"] for row in rows} == {"paper_unet", "paper_fno", "paper_cno"}
    assert all(row["schema_version"] == "pdeobs.training-campaign/v2" for row in rows)
    assert all(
        row["model_setting"]["reproduction_scope"]
        == "paper_architecture_training_with_declared_task_adapter"
        for row in rows
    )
