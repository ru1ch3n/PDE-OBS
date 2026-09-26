"""Mask specs: protocol + parameter shorthand, deterministic mixtures, and the pinned mask seed.

Every case is synthetic and CPU-only; nothing here touches a paper view, a registered protocol or an
existing seed derivation (the legacy-equivalence test proves that bit-for-bit)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pdeobs.dataset import BenchmarkDataset, collate_benchmark
from pdeobs.mask_specs import (
    canonical_view_id,
    describe_mask_config,
    is_mask_spec,
    parse_mask_spec,
    realized_observed_fraction,
)
from pdeobs.schema import Sample
from pdeobs.storage import AtomicHDF5ShardWriter


def test_parse_shorthand_and_weighted_mixture() -> None:
    spec = parse_mask_spec("uniform 1 x2 + sensor 20 + block 50")
    assert [c.protocol for c in spec.components] == ["random_3pct", "clustered_sensors", "block_missing"]
    assert [c.percent for c in spec.components] == [1.0, 20.0, 50.0]
    assert [c.weight for c in spec.components] == [2, 1, 1]
    assert spec.cycle == (0, 0, 1, 2)
    assert spec.is_mixture
    assert spec.canonical_text() == "random_3pct 1 x2 + clustered_sensors 20 + block_missing 50"
    assert spec.view_id == "random_3pct-p1-x2__clustered_sensors-p20__block_missing-p50"
    # the canonical text re-parses to the same spec
    assert parse_mask_spec(spec.canonical_text()) == spec
    assert parse_mask_spec("line 30 + random 20").view_id == "line_sensors-orientation-both-p30__random_3pct-p20"
    single = parse_mask_spec("uniform 1")
    assert not single.is_mixture and single.view_id == "random_3pct-p1"
    assert single.target_fraction() == pytest.approx(0.01)
    assert parse_mask_spec("uniform 20 + block 50").target_fraction() == pytest.approx(0.35)


def test_parse_explicit_kwargs_registered_names_and_mappings() -> None:
    spec = parse_mask_spec("line_sensors(num_lines=64, orientation=horizontal) + random_1pct")
    first, second = spec.components
    assert first.protocol == "line_sensors"
    assert dict(first.kwargs) == {"num_lines": 64, "orientation": "horizontal"}
    assert first.percent is None
    assert second.protocol == "random_1pct" and second.percent is None and second.kwargs == ()
    assert second.target_fraction() == pytest.approx(0.01)
    mapping = parse_mask_spec(
        {"protocol": "mixture", "components": [{"protocol": "uniform", "percent": 1, "weight": 2}, "block 50"]}
    )
    assert mapping.view_id == "random_3pct-p1-x2__block_missing-p50"
    # a percentage may sit beside arguments that do not set the same quantity
    mixed = parse_mask_spec("line(orientation=horizontal) 30").components[0]
    assert mixed.protocol == "line_sensors" and mixed.percent == 30.0
    assert mixed.resolved_kwargs((128, 128)) == {"orientation": "horizontal", "num_lines": 38}
    # a name glued to its number is accepted too
    assert parse_mask_spec("uniform20").components[0].percent == 20.0


@pytest.mark.parametrize(
    "text",
    ["", "nonsense 3", "uniform 0", "uniform 101", "block 100", "uniform(ratio=0.2) 20", "block(missing_fraction=0.3) 50",
     "uniform 1 + uniform 1", "full 50", "uniform 1 +", "uniform 1 x0", "line(num_lines=4 30"],
)
def test_parse_rejects_invalid_specs(text: str) -> None:
    with pytest.raises(ValueError):
        parse_mask_spec(text)


def test_registered_names_are_never_treated_as_specs() -> None:
    for name in ("random_3pct", "random", "block_missing", "lines", "full", "mixed_partial", "regular-grid"):
        assert not is_mask_spec(name)
        assert canonical_view_id(name) == name
    assert is_mask_spec("uniform 1") and is_mask_spec("line 30 + random 20")
    assert canonical_view_id("line 30 + random 20") == "line_sensors-orientation-both-p30__random_3pct-p20"
    assert describe_mask_config({"protocol": "random_3pct", "ratio": 0.5}) is None
    assert describe_mask_config({"protocol": "uniform 20"})["view_id"] == "random_3pct-p20"


@pytest.mark.parametrize(
    "text,expected",
    [("uniform 1", 0.01), ("uniform 20", 0.20), ("sensor 20", 0.20), ("block 50", 0.50), ("line 30", 0.30),
     ("hline 25", 0.25), ("vline 25", 0.25), ("boundary 50", 0.50), ("grid 25", 0.25), ("random_5pct 10", 0.10),
     ("cluster 5", 0.05), ("edge 20", 0.20)],
)
def test_percent_shorthand_realizes_the_observed_fraction_at_128(text: str, expected: float) -> None:
    component = parse_mask_spec(text).components[0]
    realized = realized_observed_fraction(component, (128, 128), seed=3)
    assert abs(realized - expected) <= 0.025, (text, realized)


def test_percent_shorthand_reproduces_the_paper_geometry() -> None:
    # the frozen 128 x 128 paper views: boundary width 19 covers 50.6 %, the closest band to 50 %;
    # 75 lines (38 rows + 37 columns) cover 50.02 %, closer to 50 % than the paper's 76 (50.6 %),
    # which stays available as an explicit argument.
    assert parse_mask_spec("boundary 50").components[0].resolved_kwargs((128, 128))["width"] == 19
    assert parse_mask_spec("line 50").components[0].resolved_kwargs((128, 128))["num_lines"] == 75
    assert dict(parse_mask_spec("line_sensors(num_lines=76)").components[0].kwargs) == {"num_lines": 76}
    assert parse_mask_spec("block 50").components[0].resolved_kwargs((128, 128))["missing_fraction"] == pytest.approx(0.5)
    assert parse_mask_spec("hline 50").components[0].resolved_kwargs((128, 128))["num_lines"] == 64


def _write(shard: Path, count: int, split: str = "train") -> None:
    field = np.arange(16 * 16, dtype=np.float32).reshape(16, 16, 1)
    with AtomicHDF5ShardWriter(shard, expected_count=count, spec={"test": True}) as writer:
        for index in range(count):
            writer.append(
                Sample(
                    condition=field,
                    trajectory=field[None],
                    geometry=np.zeros_like(field),
                    metadata={
                        "sample_id": f"s-{index:02d}",
                        "split": split,
                        "pde": "poisson",
                        "boundary": "dirichlet",
                        "setting": "smooth_grf",
                        "regime": "low",
                        "resolution": [16, 16],
                    },
                )
            )


def test_mixture_dataset_rotates_components_by_weight_and_is_deterministic(tmp_path: Path) -> None:
    shard = tmp_path / "mix.h5"
    _write(shard, 12)
    options = {"protocol": "uniform 10 x2 + block 50 + line 25"}
    first = BenchmarkDataset(shard, task="recovery", mask=options, seed=17)
    second = BenchmarkDataset(shard, task="recovery", mask=options, seed=17)
    rows = [first[index] for index in range(12)]
    again = [second[index] for index in range(12)]
    assert all(np.array_equal(a["mask"], b["mask"]) for a, b in zip(rows, again, strict=True))
    ids = [row["metadata"]["mask_component_id"] for row in rows]
    assert {key: ids.count(key) for key in set(ids)} == {
        "random_3pct-p10": 6,
        "block_missing-p50": 3,
        "line_sensors-orientation-both-p25": 3,
    }
    meta = rows[0]["metadata"]
    assert meta["mask_policy"] == "mixture"
    assert meta["mask_spec"] == "random_3pct 10 x2 + block_missing 50 + line_sensors(orientation=both) 25"
    assert meta["mask_view_id"] == "random_3pct-p10-x2__block_missing-p50__line_sensors-orientation-both-p25"
    assert len(meta["mask_mixture_candidates"]) == 4
    targets = {"random_3pct-p10": 0.10, "block_missing-p50": 0.50, "line_sensors-orientation-both-p25": 0.25}
    for row in rows:
        component = row["metadata"]["mask_component_id"]
        assert abs(row["metadata"]["observation_ratio"] - targets[component]) <= 0.08  # 16 x 16 is coarse
        assert row["metadata"]["mask_id"] == component.split("-")[0]
    # metadata still collates
    collate_benchmark(rows[:4])


def test_spec_masks_match_the_legacy_parameterised_masks_bit_for_bit(tmp_path: Path) -> None:
    shard = tmp_path / "legacy.h5"
    _write(shard, 6)
    legacy = BenchmarkDataset(shard, task="recovery", mask={"protocol": "random", "ratio": 0.2}, seed=99)
    spec = BenchmarkDataset(shard, task="recovery", mask={"protocol": "uniform 20"}, seed=99)
    for index in range(6):
        assert np.array_equal(legacy[index]["mask"], spec[index]["mask"])
        assert legacy[index]["metadata"]["mask_seed"] == spec[index]["metadata"]["mask_seed"]
        assert spec[index]["metadata"]["mask_policy"] == "parameterized"
    # inside a mixture the same component reproduces the standalone mask for the same sample
    mixture = BenchmarkDataset(shard, task="recovery", mask={"protocol": "line 30 + uniform 20"}, seed=99)
    hits = 0
    for index in range(6):
        row = mixture[index]
        if row["metadata"]["mask_component_id"] == "random_3pct-p20":
            hits += 1
            assert np.array_equal(row["mask"], spec[index]["mask"])
    assert hits == 3


def test_mask_seed_pins_the_sensors_while_the_dataset_seed_moves(tmp_path: Path) -> None:
    shard = tmp_path / "pin.h5"
    _write(shard, 4)
    pinned_a = BenchmarkDataset(shard, task="recovery", mask={"protocol": "uniform 20"}, seed=1, mask_seed=99)
    pinned_b = BenchmarkDataset(shard, task="recovery", mask={"protocol": "uniform 20"}, seed=2, mask_seed=99)
    other = BenchmarkDataset(shard, task="recovery", mask={"protocol": "uniform 20"}, seed=2, mask_seed=100)
    default = BenchmarkDataset(shard, task="recovery", mask={"protocol": "uniform 20"}, seed=99)
    for index in range(4):
        assert np.array_equal(pinned_a[index]["mask"], pinned_b[index]["mask"])
        assert np.array_equal(pinned_a[index]["mask"], default[index]["mask"])  # mask_seed defaults to seed
    assert any(not np.array_equal(pinned_b[index]["mask"], other[index]["mask"]) for index in range(4))
    # the legacy mixed_partial policy keys on the same pinned seed
    mixed_a = BenchmarkDataset(shard, task="recovery", mask={"protocol": "mixed_partial", "protocols": ["random_1pct", "block_missing"]}, seed=1, mask_seed=7)
    mixed_b = BenchmarkDataset(shard, task="recovery", mask={"protocol": "mixed_partial", "protocols": ["random_1pct", "block_missing"]}, seed=2, mask_seed=7)
    assert all(np.array_equal(mixed_a[i]["mask"], mixed_b[i]["mask"]) for i in range(4))


def test_full_component_inside_a_mixture_and_rejected_extra_options(tmp_path: Path) -> None:
    shard = tmp_path / "full.h5"
    _write(shard, 4)
    dataset = BenchmarkDataset(shard, task="recovery", mask={"protocol": "full + uniform 10"}, seed=3)
    rows = [dataset[index] for index in range(4)]
    ids = sorted(row["metadata"]["mask_component_id"] for row in rows)
    assert ids == ["full", "full", "random_3pct-p10", "random_3pct-p10"]
    full_rows = [row for row in rows if row["metadata"]["mask_component_id"] == "full"]
    assert all(row["metadata"]["observation_ratio"] == 1.0 and row["metadata"]["mask_seed"] is None for row in full_rows)
    bad = BenchmarkDataset(shard, task="recovery", mask={"protocol": "uniform 10", "ratio": 0.5}, seed=3)
    with pytest.raises(ValueError, match="no extra options"):
        bad[0]


def test_runner_evaluates_parameterised_and_mixture_views(tmp_path: Path) -> None:
    from pdeobs.runner import run_evaluate

    shard = tmp_path / "views.h5"
    _write(shard, 6, split="test")
    config = {
        "task": "recovery",
        "seed": 11,
        "data": {"root": str(tmp_path), "glob": "*.h5", "mask": {"protocol": "uniform 10 + block 50"}, "mask_seed": 5},
        "method": {"name": "mean", "kwargs": {}},
        "training": {"batch_size": 2, "num_workers": 0, "seed": 4},
        "evaluation": {
            "observation_protocols": ["full", "uniform 20", "line 30 + random 20"],
            "include_frequency_bands": False,
            "include_stability": False,
        },
    }
    result = run_evaluate(config_path=config, output=tmp_path / "out" / "metrics.json")
    assert set(result["observation_views"]) == {
        "full",
        "random_3pct-p20",
        "line_sensors-orientation-both-p30__random_3pct-p20",
    }
    view = result["observation_views"]["line_sensors-orientation-both-p30__random_3pct-p20"]
    assert view["mask_spec"] == "line_sensors(orientation=both) 30 + random_3pct 20"
    assert view["mask_mixture"] is True
    assert view["observation_ratio"] == pytest.approx(0.25)
    assert [c["id"] for c in view["mask_components"]] == ["line_sensors-orientation-both-p30", "random_3pct-p20"]
    single = result["observation_views"]["random_3pct-p20"]
    assert single["mask_mixture"] is False and single["observation_ratio"] == pytest.approx(0.20)
    duplicate = dict(config, evaluation=dict(config["evaluation"], observation_protocols=["uniform 20", "random 20"]))
    with pytest.raises(ValueError, match="duplicate views"):
        run_evaluate(config_path=duplicate, output=tmp_path / "dup" / "metrics.json")


def test_easy_api_accepts_specs_in_the_general_namespace(tmp_path: Path) -> None:
    from pdeobs.api import observation as O

    spec = O.make_observation("uniform 1 x2 + sensor 20 + block 50")
    assert spec.namespace == "general"
    assert spec.name == "random_3pct-p1-x2__clustered_sensors-p20__block_missing-p50"
    assert spec.mask_config == {"protocol": "random_3pct 1 x2 + clustered_sensors 20 + block_missing 50"}
    assert spec.observation_id.startswith("general:random_3pct-p1-x2__")
    # the pipeline mapping form and the CLI string form resolve identically
    assert O.make_observation({"protocol": "line 30 + random 20"}).name == "line_sensors-orientation-both-p30__random_3pct-p20"
    with pytest.raises(ValueError, match="own parameters"):
        O.make_observation("uniform 20", ratio=0.5)
    # the mask_config is what BenchmarkDataset receives, verbatim
    shard = tmp_path / "api.h5"
    _write(shard, 4)
    row = BenchmarkDataset(shard, task="recovery", mask=spec.mask_config, seed=1)[0]
    assert row["metadata"]["mask_policy"] == "mixture"
    # the legacy general namespace and the paper-view guard are untouched
    assert O.make_observation("random", ratio=0.5).mask_config == {"protocol": "random_3pct", "ratio": 0.5}
    with pytest.raises(ValueError):
        O.make_observation("random_65pct")


def test_package_version_is_0_2_1() -> None:
    import pdeobs

    assert pdeobs.__version__ == "0.2.1"
