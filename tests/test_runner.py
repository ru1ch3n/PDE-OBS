from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pdeobs.runner import (
    _dataset,
    _evaluation_horizons,
    _evaluation_metric_records,
    _training_horizons,
    run_benchmark,
    run_evaluate,
    run_train,
)
from pdeobs.schema import Sample
from pdeobs.storage import AtomicHDF5ShardWriter


def test_runner_materializes_official_ood_membership(tmp_path: Path) -> None:
    shard = tmp_path / "ood.h5"
    with AtomicHDF5ShardWriter(shard, expected_count=2, spec={"test": True}) as writer:
        for index, is_ood in enumerate((False, True)):
            field = np.full((8, 8, 1), index + 1, dtype=np.float32)
            writer.append(
                Sample(
                    condition=field,
                    trajectory=field[None],
                    geometry=np.zeros_like(field),
                    metadata={
                        "sample_id": f"ood-{index}",
                        "split": "test",
                        "pde": "poisson",
                        "boundary_ood": is_ood,
                    },
                )
            )
    config = {
        "task": "recovery",
        "data": {
            "root": str(tmp_path),
            "glob": "*.h5",
            "ood_view": "boundary",
            "mask": {"protocol": "random", "count": 4},
        },
    }

    iid = _dataset(config, "test", ood_membership=False)
    ood = _dataset(config, "test", ood_membership=True)

    assert iid is not None and ood is not None
    assert iid.metadata[0]["sample_id"] == "ood-0"
    assert ood.metadata[0]["sample_id"] == "ood-1"

    alias_config = {**config, "data": {**config["data"], "split": "boundary_ood"}}
    alias_config["data"].pop("ood_view")
    alias_ood = _dataset(alias_config, "test", ood_membership=True)
    assert alias_ood is not None
    assert alias_ood.metadata[0]["sample_id"] == "ood-1"


def _write_temporal_ood_fixture(root: Path) -> Path:
    shard = root / "temporal.h5"
    with AtomicHDF5ShardWriter(shard, expected_count=2, spec={"test": "rollout-ood"}) as writer:
        for index, (boundary, is_ood) in enumerate((("periodic", False), ("robin_obstacle", True))):
            frames = np.stack(
                [np.full((8, 8, 1), index + step / 10, dtype=np.float32) for step in range(9)]
            )
            writer.append(
                Sample(
                    condition=frames[0],
                    trajectory=frames,
                    geometry=np.zeros((8, 8, 1), dtype=np.float32),
                    metadata={
                        "sample_id": f"temporal-{index}",
                        "split": "test",
                        "pde": "heat",
                        "boundary": boundary,
                        "setting": "smooth_grf",
                        "regime": "low",
                        "resolution": [8, 8],
                        "boundary_ood": is_ood,
                    },
                )
            )
    config = root / "rollout.yaml"
    config.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "name: rollout-ood-test",
                "task: rollout",
                "seed: 7",
                "data:",
                f"  root: '{root.as_posix()}'",
                '  train_glob: "*.h5"',
                "  ood_view: boundary",
                "  training_horizons: [1, 2]",
                "  input_horizon: 1",
                "  mask: {protocol: random, count: 4}",
                "method: {name: persistence, kwargs: {}}",
                "training: {batch_size: 2, num_workers: 0}",
                "evaluation:",
                "  horizons: [1, 2, 4]",
                "  ood_views: [boundary, time_horizon]",
                "  mask_protocols: [random_1pct]",
                "  include_frequency_bands: false",
                "  include_stability: false",
                f"output: {{root: '{(root / 'runs').as_posix()}'}}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def test_rollout_training_and_evaluation_horizons_are_separate() -> None:
    config = {
        "task": "rollout",
        "data": {"rollout_horizons": [1, 2, 4, 8]},
    }

    assert _training_horizons(config) == (1, 2)
    assert _evaluation_horizons(config) == (1, 2, 4, 8)


def test_explicit_observation_views_do_not_insert_a_training_mask(tmp_path: Path) -> None:
    shard = tmp_path / "observation-views.h5"
    field = np.arange(8 * 8, dtype=np.float32).reshape(8, 8, 1)
    with AtomicHDF5ShardWriter(shard, expected_count=1, spec={"test": True}) as writer:
        writer.append(
            Sample(
                condition=field,
                trajectory=field[None],
                geometry=np.zeros_like(field),
                metadata={
                    "sample_id": "view-0",
                    "split": "test",
                    "pde": "poisson",
                    "boundary": "dirichlet",
                    "setting": "smooth_grf",
                    "regime": "low",
                    "resolution": [8, 8],
                },
            )
        )
    config = {
        "task": "recovery",
        "data": {
            "root": str(tmp_path),
            "glob": "*.h5",
            "mask": {
                "protocol": "mixed_partial",
                "protocols": ["random_1pct", "regular_grid"],
            },
        },
        "method": {"name": "mean", "kwargs": {}},
        "training": {"batch_size": 1, "num_workers": 0},
        "evaluation": {
            "observation_protocols": ["full", "random_1pct", "regular_grid"],
            "include_frequency_bands": False,
            "include_stability": False,
        },
    }

    result = run_evaluate(config_path=config, output=tmp_path / "views" / "metrics.json")

    assert result["mask_ood"] == {}
    assert set(result["observation_views"]) == {"full", "random_1pct", "regular_grid"}
    assert result["observation_views"]["full"]["mask_protocol"] == "full"
    records = _evaluation_metric_records(
        result,
        experiment_index=0,
        experiment_config={"name": "explicit-views"},
    )
    assert len(records) == 3
    assert all(row["split"] == "test" and row["is_ood"] is False for row in records)


def test_runner_fits_and_loads_gappy_pod_without_a_torch_checkpoint(tmp_path: Path) -> None:
    shard = tmp_path / "gappy-runner.h5"
    splits = ("train",) * 6 + ("validation",) * 3 + ("test",)
    yy, xx = np.mgrid[:8, :8]
    with AtomicHDF5ShardWriter(shard, expected_count=len(splits), spec={"test": True}) as writer:
        for index, split in enumerate(splits):
            field = (
                (0.2 + index / 20) * np.sin(2 * np.pi * xx / 8)
                + (1.0 - index / 30) * np.cos(2 * np.pi * yy / 8)
            )[..., None].astype(np.float32)
            writer.append(
                Sample(
                    condition=field,
                    trajectory=field[None],
                    geometry=np.zeros_like(field),
                    metadata={
                        "sample_id": f"gappy-{index}",
                        "split": split,
                        "pde": "poisson",
                        "boundary": "dirichlet",
                        "setting": "smooth_grf",
                        "regime": "low",
                        "resolution": [8, 8],
                    },
                )
            )
    config = {
        "task": "recovery",
        "seed": 9,
        "data": {"root": str(tmp_path), "glob": "*.h5", "mask": {"protocol": "full"}},
        "method": {
            "name": "gappy_pod",
            "kwargs": {
                "rank_candidates": [1, 2],
                "oversampling": 1,
                "fit_batch_size": 3,
                "validation_samples_per_stratum": 1,
                "validation_protocols": ["random_10pct", "regular_grid"],
                "seed": 9,
            },
        },
        "training": {"batch_size": 2, "num_workers": 0},
        "evaluation": {
            "observation_protocols": ["full", "random_10pct"],
            "include_frequency_bands": False,
            "include_stability": False,
        },
    }

    trained = run_train(config_path=config, output=tmp_path / "gappy-fit")
    evaluated = run_evaluate(
        config_path=config,
        output=tmp_path / "gappy-eval" / "metrics.json",
        checkpoint=trained["checkpoint"],
    )

    assert Path(trained["checkpoint"]).suffix == ".npz"
    assert trained["fit_receipt"]["rank_selection"]["split"] == "validation"
    assert set(evaluated["observation_views"]) == {"full", "random_10pct"}


def test_paired_ood_results_feed_benchmark_and_analysis(tmp_path: Path) -> None:
    experiment = _write_temporal_ood_fixture(tmp_path)
    evaluation = run_evaluate(
        config_path=experiment,
        output=tmp_path / "evaluation" / "metrics.json",
    )

    assert set(evaluation["ood"]) == {"boundary", "time_horizon"}
    assert evaluation["ood"]["boundary"]["iid"]["split"] == "iid"
    assert evaluation["ood"]["boundary"]["ood"]["split"] == "ood"
    time_horizon = evaluation["ood"]["time_horizon"]
    assert set(time_horizon["iid"]) == {"1", "2"}
    assert set(time_horizon["ood"]) == {"4"}
    assert "rel_l2_at_horizon" in time_horizon["ood"]["4"]["result"]["metrics"]
    assert set(evaluation["mask_ood"]) == {"iid", "random_1pct"}
    assert time_horizon["iid"]["1"]["samples"] == 1
    assert time_horizon["iid"]["1"]["boundary"] == "periodic"
    assert time_horizon["iid"]["1"]["factor_ood_view"] == "boundary"
    assert evaluation["mask_ood"]["iid"]["samples"] == 1
    assert evaluation["mask_ood"]["iid"]["factor_is_ood"] is False

    records = _evaluation_metric_records(
        evaluation,
        experiment_index=0,
        experiment_config={
            "name": "rollout-ood-test",
            "data": {"filters": {"pde": "heat"}},
        },
    )
    assert len(records) == 7
    assert any(
        row.get("ood_view") == "time_horizon"
        and row.get("rollout_horizon") == 4
        and row.get("is_ood") is True
        for row in records
    )

    benchmark_config = tmp_path / "benchmark.yaml"
    benchmark_config.write_text(
        "\n".join(
            (
                "name: paired-ood-benchmark",
                f"output: {{root: '{(tmp_path / 'benchmark-runs').as_posix()}'}}",
                "experiments:",
                f"  - config: '{experiment.as_posix()}'",
                "    mode: eval",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark = run_benchmark(
        config_path=benchmark_config,
        output=tmp_path / "benchmark",
    )
    analysis_path = Path(benchmark["analysis_records"]["json"])
    analysis_records = json.loads(analysis_path.read_text(encoding="utf-8"))

    assert benchmark["analysis_records"]["count"] == 7
    assert len(analysis_records) == 7
    assert any(
        row.get("ood_view") == "boundary" and row.get("split") == "ood" for row in analysis_records
    )
    assert any(
        row.get("ood_view") == "mask" and row.get("mask_protocol") == "random_1pct"
        for row in analysis_records
    )
    assert any(
        row.get("ood_view") == "time_horizon" and row.get("rollout_horizon") == 4
        for row in benchmark["leaderboard"]
    )
