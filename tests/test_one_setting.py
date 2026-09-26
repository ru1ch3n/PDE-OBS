from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pdeobs.one_setting import (
    LEARNED_METHOD_ORDER,
    PDE_ORDER,
    VIEW_ORDER,
    build_experiment_config,
    load_campaign,
    preflight_subset,
    stable_split,
    validate_campaign,
)




def test_campaign_rejects_noncanonical_scalar_representation() -> None:
    campaign = load_campaign(CAMPAIGN)
    campaign["problem_settings"]["darcy"]["state_representation"] = "scalar"
    with pytest.raises(ValueError, match="state_representation=native"):
        validate_campaign(campaign)

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs/campaign/all_pde_one_setting_10method_9x9.yaml"
REGISTRY = ROOT / "configs/method/all_pde_source_faithful_registry.yaml"


def test_campaign_has_unique_non_pseudoreplicated_accounting() -> None:
    campaign = load_campaign(CAMPAIGN)

    assert tuple(campaign["problem_settings"]) == PDE_ORDER
    assert tuple(campaign["views"]) == VIEW_ORDER
    assert tuple(campaign["methods"]["learned"]) == LEARNED_METHOD_ORDER
    assert campaign["methods"]["forbidden_slot"] == "generic_unet"
    assert campaign["evaluation"]["learned"]["training_rows"] == 7 * 8 * 9
    assert campaign["evaluation"]["learned"]["unique_blocks"] == 7 * 8 * 9 * 9
    assert campaign["evaluation"]["classical"]["unique_blocks"] == 7 * 2 * 9
    assert campaign["evaluation"]["unique_blocks_total"] == 4662
    assert campaign["preflight"]["loss_gate"][
        "minimum_final_output_std_ratio"
    ] == pytest.approx(1.0e-3)
    assert 7.272319859567991e-6 < campaign["preflight"]["loss_gate"][
        "minimum_final_output_std_ratio"
    ]


def test_public_configs_preserve_static_and_free_rollout_contracts() -> None:
    campaign = load_campaign(CAMPAIGN)
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))

    static = build_experiment_config(
        campaign,
        registry,
        dataset_root="/immutable/data",
        pde="darcy",
        method="ufno_2d",
        view="random_50pct",
        epochs=500,
    )
    temporal = build_experiment_config(
        campaign,
        registry,
        dataset_root="/immutable/data",
        pde="heat",
        method="fno",
        view="clustered_50pct",
        epochs=3,
    )
    pino = build_experiment_config(
        campaign,
        registry,
        dataset_root="/immutable/data",
        pde="navier_stokes",
        method="pino",
        view="random_50pct",
        epochs=3,
    )
    gnot = build_experiment_config(
        campaign,
        registry,
        dataset_root="/immutable/data",
        pde="darcy",
        method="gnot",
        view="random_50pct",
        epochs=3,
    )
    transolver_static = build_experiment_config(
        campaign,
        registry,
        dataset_root="/immutable/data",
        pde="darcy",
        method="transolver",
        view="random_50pct",
        epochs=3,
    )
    transolver_temporal = build_experiment_config(
        campaign,
        registry,
        dataset_root="/immutable/data",
        pde="heat",
        method="transolver",
        view="random_50pct",
        epochs=3,
    )

    assert static["task"] == "recovery"
    assert static["method"]["name"] == "ufno"
    assert static["training"]["epochs"] == 500
    assert static["training"]["scheduler_step_size"] == 100
    assert temporal["task"] == "rollout"
    assert temporal["method"]["name"] == "autoregressive"
    assert temporal["data"]["training_horizons"] == [3]
    assert temporal["training"]["teacher_forcing_ratio"] == 0.0
    assert pino["method"]["base"]["kwargs"]["physics_contract"] == (
        "pino_rollout_spectral_v1"
    )
    assert pino["training"]["physics_loss"] == "pino_rollout_spectral_v1"
    assert pino["training"]["data_loss_weight"] == 5.0
    assert pino["training"]["grad_clip"] == 1.0
    assert pino["training"]["gradient_warning_fatal"] is False
    assert registry["settings"]["pino"]["objective"]["temporal_normalization"] == (
        "differentiable_scale_neutral_for_homogeneous_rollouts"
    )
    assert gnot["training"]["scheduler"] == "one_cycle"
    assert gnot["training"]["scheduler_steps_per_epoch"] == 23
    assert gnot["training"]["gradient_warning_fatal"] is False
    assert gnot["training"]["grad_clip"] == 1000.0
    assert transolver_static["method"]["kwargs"]["hidden"] == 128
    assert transolver_static["method"]["kwargs"]["layers"] == 8
    assert transolver_static["training"]["grad_clip"] == 0.1
    assert transolver_temporal["method"]["base"]["kwargs"]["hidden"] == 256
    assert transolver_temporal["training"]["batch_size"] == 2
    assert transolver_temporal["training"]["grad_clip"] is None
    assert transolver_temporal["training"]["gradient_warning_fatal"] is True


def test_exact_production_split_and_preflight_subset_are_disjoint_contracts() -> None:
    metadata = [
        {
            "sample_id": f"sample-{index:04d}",
            "regime": ("low", "medium", "high")[index % 3],
            "split": "train" if index % 10 < 7 else ("validation" if index % 10 == 7 else "test"),
        }
        for index in range(2000)
    ]
    train, test, receipt = stable_split(metadata, 20260804, "darcy|dirichlet|smooth_grf")
    preflight, preflight_receipt = preflight_subset(
        metadata, 20260804, "darcy|dirichlet|smooth_grf"
    )

    assert len(train) == 1800
    assert len(test) == 200
    assert not set(train) & set(test)
    assert receipt["validation_records"] == 0
    assert len(preflight) == 90
    assert {metadata[index]["split"] for index in preflight} == {"train"}
    assert preflight_receipt["regime_counts"] == {"low": 30, "medium": 30, "high": 30}
