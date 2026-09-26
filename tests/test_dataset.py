from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pdeobs.dataset import BenchmarkDataset, collate_benchmark
from pdeobs.schema import Sample
from pdeobs.storage import AtomicHDF5ShardWriter


def test_recovery_dataset_filters_and_applies_exact_mask(tmp_path: Path) -> None:
    shard = tmp_path / "fixture.h5"
    with AtomicHDF5ShardWriter(shard, expected_count=2, spec={"test": True}) as writer:
        for index, split in enumerate(("train", "validation")):
            target = np.full((1, 8, 8, 1), index + 1, dtype=np.float32)
            writer.append(
                Sample(
                    condition=target[0],
                    trajectory=target,
                    geometry=np.zeros((8, 8, 1), dtype=np.float32),
                    metadata={"sample_id": f"sample-{index}", "split": split, "pde": "heat"},
                )
            )

    dataset = BenchmarkDataset(
        shard,
        task="recovery",
        split="train",
        filters={"pde": "heat"},
        mask={"protocol": "random", "count": 5},
        seed=9,
        verify=True,
    )
    row = dataset[0]
    batch = collate_benchmark([row])

    assert len(dataset) == 1
    assert int(row["mask"].sum()) == 5
    assert row["metadata"]["mask_id"] == "random_3pct"
    assert isinstance(row["metadata"]["mask_seed"], int)
    assert row["metadata"]["observation_count"] == 5
    assert batch["observations"].shape == (1, 8, 8, 1)
    assert np.count_nonzero(batch["observations"]) == 5


def test_spatial_adapter_area_downsamples_before_masking(tmp_path: Path) -> None:
    shard = tmp_path / "spatial-adapter.h5"
    target = np.arange(64, dtype=np.float32).reshape(1, 8, 8, 1)
    with AtomicHDF5ShardWriter(shard, expected_count=1, spec={"test": True}) as writer:
        writer.append(
            Sample(
                condition=target[0],
                trajectory=target,
                geometry=np.zeros((8, 8, 1), dtype=np.float32),
                metadata={
                    "sample_id": "spatial-0",
                    "split": "train",
                    "pde": "poisson",
                    "resolution": [8, 8],
                },
            )
        )

    row = BenchmarkDataset(
        shard,
        task="recovery",
        spatial_resolution=[4, 4],
        mask={"protocol": "random", "count": 3},
        verify=True,
    )[0]
    expected = target[0].reshape(4, 2, 4, 2, 1).mean(axis=(1, 3))
    np.testing.assert_allclose(row["target"], expected)
    assert row["target"].shape == (4, 4, 1)
    assert row["mask"].shape == (4, 4, 1)
    assert int(row["mask"].sum()) == 3
    assert row["metadata"]["native_resolution"] == [8, 8]
    assert row["metadata"]["resolution"] == [4, 4]
    assert row["metadata"]["spatial_adapter"] == (
        "deterministic_area_average_integer_factor_v1"
    )


def test_medium_release_tier_reconstructs_the_canonical_nested_prefix(tmp_path: Path) -> None:
    shard = tmp_path / "full-release.h5"
    regimes = ("low", "medium", "high")
    with AtomicHDF5ShardWriter(shard, expected_count=6, spec={"test": True}) as writer:
        for regime_index in range(2):
            for regime in regimes:
                value = 10 * regime_index + regimes.index(regime)
                target = np.full((1, 8, 8, 1), value, dtype=np.float32)
                writer.append(
                    Sample(
                        condition=target[0],
                        trajectory=target,
                        geometry=np.zeros((8, 8, 1), dtype=np.float32),
                        metadata={
                            "sample_id": f"{regime}-{regime_index}",
                            "split": "train",
                            "pde": "heat",
                            "regime": regime,
                            "regime_sample_index": regime_index,
                        },
                    )
                )

    tiny_prefix = BenchmarkDataset(shard, release_tier=5)

    assert len(tiny_prefix) == 5
    assert {row["metadata"]["sample_id"] for row in tiny_prefix} == {
        "low-0",
        "medium-0",
        "high-0",
        "low-1",
        "medium-1",
    }


def test_release_tier_requires_canonical_regime_metadata(tmp_path: Path) -> None:
    shard = tmp_path / "missing-regime.h5"
    target = np.zeros((1, 8, 8, 1), dtype=np.float32)
    with AtomicHDF5ShardWriter(shard, expected_count=1, spec={"test": True}) as writer:
        writer.append(
            Sample(
                condition=target[0],
                trajectory=target,
                geometry=np.zeros((8, 8, 1), dtype=np.float32),
                metadata={"sample_id": "missing", "split": "train", "pde": "heat"},
            )
        )

    with pytest.raises(ValueError, match="canonical regime metadata"):
        BenchmarkDataset(shard, release_tier="medium")


def test_observation_protocol_never_changes_stored_ground_truth(tmp_path: Path) -> None:
    shard = tmp_path / "immutable-ground-truth.h5"
    trajectory = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8, 1)
    condition = trajectory[0].copy()
    geometry = np.zeros((8, 8, 1), dtype=bool)
    with AtomicHDF5ShardWriter(shard, expected_count=1, spec={"test": True}) as writer:
        writer.append(
            Sample(
                condition=condition,
                trajectory=trajectory,
                geometry=geometry,
                metadata={"sample_id": "immutable-0", "split": "train", "pde": "heat"},
            )
        )

    sparse = BenchmarkDataset(
        shard,
        task="recovery",
        mask={"protocol": "random", "count": 5},
        verify=True,
    )[0]
    structured = BenchmarkDataset(
        shard,
        task="recovery",
        mask={"protocol": "grid", "spacing": 2},
        verify=True,
    )[0]

    assert not np.array_equal(sparse["mask"], structured["mask"])
    np.testing.assert_array_equal(sparse["target"], trajectory[-1])
    np.testing.assert_array_equal(structured["target"], trajectory[-1])
    np.testing.assert_array_equal(sparse["target"], structured["target"])
    np.testing.assert_array_equal(sparse["geometry"], structured["geometry"])


def test_rollout_uses_requested_sparse_history_and_future_horizon(tmp_path: Path) -> None:
    shard = tmp_path / "temporal.h5"
    trajectory = np.arange(6 * 8 * 8, dtype=np.float32).reshape(6, 8, 8, 1)
    with AtomicHDF5ShardWriter(shard, expected_count=1, spec={"test": True}) as writer:
        writer.append(
            Sample(
                condition=trajectory[0],
                trajectory=trajectory,
                geometry=np.zeros((8, 8, 1), dtype=np.float32),
                metadata={
                    "sample_id": "temporal-0",
                    "split": "train",
                    "pde": "heat",
                    "resolution": [8, 8],
                },
            )
        )

    dataset = BenchmarkDataset(
        shard,
        task="rollout",
        history_steps=2,
        horizon=3,
        mask={"protocol": "random", "count": 7},
    )
    row = dataset[0]

    assert row["observations"].shape == (2, 8, 8, 1)
    assert row["mask"].shape == (2, 8, 8, 1)
    assert int(row["mask"][0].sum()) == 7
    assert row["target"].shape == (3, 8, 8, 1)
    assert row["initial_state"].shape == (2, 8, 8, 1)
    np.testing.assert_array_equal(row["initial_state"], trajectory[:2])
    np.testing.assert_array_equal(row["target"], trajectory[2:5])


def test_rollout_rejects_a_short_trajectory_instead_of_truncating(tmp_path: Path) -> None:
    shard = tmp_path / "short-temporal.h5"
    trajectory = np.zeros((4, 8, 8, 1), dtype=np.float32)
    with AtomicHDF5ShardWriter(shard, expected_count=1, spec={"test": True}) as writer:
        writer.append(
            Sample(
                condition=trajectory[0],
                trajectory=trajectory,
                geometry=np.zeros((8, 8, 1), dtype=np.float32),
                metadata={"sample_id": "short-0", "split": "train", "pde": "heat"},
            )
        )

    dataset = BenchmarkDataset(
        shard,
        task="rollout",
        history_steps=2,
        horizon=3,
        mask={"protocol": "random", "count": 7},
    )

    with pytest.raises(ValueError, match=r"2 history \+ 3 future"):
        _ = dataset[0]


def test_forward_masks_the_condition(tmp_path: Path) -> None:
    shard = tmp_path / "forward.h5"
    condition = np.ones((8, 8, 1), dtype=np.float32)
    with AtomicHDF5ShardWriter(shard, expected_count=1, spec={"test": True}) as writer:
        writer.append(
            Sample(
                condition=condition,
                trajectory=condition[None],
                geometry=np.zeros_like(condition),
                metadata={"sample_id": "forward-0", "split": "test", "pde": "poisson"},
            )
        )
    row = BenchmarkDataset(
        shard,
        task="forward",
        mask={"protocol": "random", "count": 5},
    )[0]
    assert int(row["mask"].sum()) == 5
    assert np.count_nonzero(row["observations"]) == 5


def test_inverse_targets_the_original_condition(tmp_path: Path) -> None:
    shard = tmp_path / "inverse.h5"
    condition = np.full((8, 8, 1), 3.0, dtype=np.float32)
    solution = np.full((1, 8, 8, 1), 7.0, dtype=np.float32)
    with AtomicHDF5ShardWriter(shard, expected_count=1, spec={"test": True}) as writer:
        writer.append(
            Sample(
                condition=condition,
                trajectory=solution,
                geometry=np.zeros_like(condition),
                metadata={"sample_id": "inverse-0", "split": "test", "pde": "darcy"},
            )
        )
    row = BenchmarkDataset(
        shard,
        task="inverse",
        mask={"protocol": "random", "count": 6},
    )[0]
    assert int(row["mask"].sum()) == 6
    assert np.count_nonzero(row["observations"]) == 6
    np.testing.assert_array_equal(row["target"], condition)


@pytest.mark.parametrize(
    "stored_representation", ("bounded_vorticity", "bounded_obstacle_vorticity")
)
def test_bounded_vorticity_tags_are_canonical_vorticity_without_conversion(
    tmp_path: Path, stored_representation: str
) -> None:
    shard = tmp_path / f"{stored_representation}.h5"
    trajectory = np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8, 1)
    with AtomicHDF5ShardWriter(shard, expected_count=1, spec={"test": True}) as writer:
        writer.append(
            Sample(
                condition=trajectory[0],
                trajectory=trajectory,
                geometry=np.zeros((8, 8, 1), dtype=np.float32),
                metadata={
                    "sample_id": f"{stored_representation}-0",
                    "split": "test",
                    "pde": "navier_stokes",
                    "resolution": [8, 8],
                    "state_representation": stored_representation,
                },
            )
        )

    row = BenchmarkDataset(
        shard,
        task="recovery",
        state_representation="vorticity",
        mask={"protocol": "random", "count": 5},
        verify=True,
    )[0]

    np.testing.assert_array_equal(row["target"], trajectory[-1])
    assert row["metadata"]["state_representation"] == "vorticity"
    assert row["metadata"]["native_state_representation"] == stored_representation


def test_full_observation_mask_exposes_every_cell(tmp_path: Path) -> None:
    shard = tmp_path / "full-observation.h5"
    field = np.arange(8 * 8, dtype=np.float32).reshape(8, 8, 1)
    with AtomicHDF5ShardWriter(shard, expected_count=1, spec={"test": True}) as writer:
        writer.append(
            Sample(
                condition=field,
                trajectory=field[None],
                geometry=np.zeros_like(field),
                metadata={"sample_id": "full-0", "split": "train", "pde": "poisson"},
            )
        )

    row = BenchmarkDataset(shard, task="recovery", mask={"protocol": "full"})[0]

    assert np.all(row["mask"] == 1)
    assert np.array_equal(row["observations"], row["target"])
    assert np.array_equal(row["pde_condition"], field)
    assert row["metadata"]["mask_id"] == "full"
    assert row["metadata"]["observation_ratio"] == 1.0


def test_mixed_partial_mask_is_deterministic_and_uses_declared_protocols(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "mixed-partial.h5"
    field = np.arange(8 * 8, dtype=np.float32).reshape(8, 8, 1)
    with AtomicHDF5ShardWriter(shard, expected_count=12, spec={"test": True}) as writer:
        for index in range(12):
            writer.append(
                Sample(
                    condition=field,
                    trajectory=field[None],
                    geometry=np.zeros_like(field),
                    metadata={
                        "sample_id": f"mixed-{index:02d}",
                        "split": "train",
                        "pde": "poisson",
                    },
                )
            )
    options = {
        "protocol": "mixed_partial",
        "protocols": ["random_1pct", "regular_grid", "block_missing"],
    }
    first = BenchmarkDataset(shard, task="recovery", mask=options, seed=17)
    second = BenchmarkDataset(shard, task="recovery", mask=options, seed=17)

    first_rows = [first[index] for index in range(len(first))]
    second_rows = [second[index] for index in range(len(second))]

    assert [row["metadata"]["mask_id"] for row in first_rows] == [
        row["metadata"]["mask_id"] for row in second_rows
    ]
    assert all(
        np.array_equal(left["mask"], right["mask"])
        for left, right in zip(first_rows, second_rows, strict=True)
    )
    assert {row["metadata"]["mask_id"] for row in first_rows} <= {
        "random_1pct",
        "regular_grid",
        "block_missing",
    }
    counts = {
        protocol: sum(row["metadata"]["mask_id"] == protocol for row in first_rows)
        for protocol in ("random_1pct", "regular_grid", "block_missing")
    }
    assert counts == {"random_1pct": 4, "regular_grid": 4, "block_missing": 4}
    assert all(row["metadata"]["mask_policy"] == "mixed_partial" for row in first_rows)
