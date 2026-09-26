"""Single-row control path tests, not production benchmark accuracy tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from pdeobs.cli import main
from pdeobs.one_setting import PUBLIC_LEARNED_METHODS, stable_split
from pdeobs.paper_row import create_paper_row_demo_data, resolve_row, row_split, run_paper_row, validate_c500_frames

ROOT = Path(__file__).resolve().parents[1]


def config_copy(tmp_path, *, family="poisson", cohort="demo", **changes):
    path = ROOT / "configs/demo" / ("paper_row_rollout.yaml" if family == "heat" else "paper_row_static.yaml")
    value = yaml.safe_load(path.read_text())
    value["campaign"] = str((path.parent / value["campaign"]).resolve())
    value["registry"] = str((path.parent / value["registry"]).resolve())
    value["cohort"] = cohort
    if cohort != "demo":
        value.pop("demo_epochs")
        value.pop("demo_test_views")
    value.update(changes)
    out = tmp_path / "row.yaml"
    out.write_text(yaml.safe_dump(value), encoding="utf-8")
    return out


def canonical_metadata():
    return [{"sample_id": f"actual-identity-{i:04d}", "regime": ("low", "medium", "high")[i % 3],
             "split": "wrong-source-split"} for i in range(2000)]


def test_c500_calls_real_stable_split_and_ignores_shard_split_labels(monkeypatch):
    from pdeobs import paper_row
    rows = canonical_metadata()
    calls = []
    def observed(*args):
        calls.append(args)
        return stable_split(*args)
    monkeypatch.setattr(paper_row, "stable_split", observed)
    train, test, receipt = row_split(rows, 20260804, "poisson|dirichlet|smooth_grf", "original500")
    reference = stable_split(rows, 20260804, "poisson|dirichlet|smooth_grf")
    assert len(calls) == 1
    assert (train, test) == reference[:2]
    assert receipt["macrodomain_identity"] == "poisson|dirichlet|smooth_grf"
    assert (len(train), len(test), receipt["validation_records"]) == (1800, 200, 0)
    assert not set(train) & set(test)
    assert all(value["train"] == 600 for value in receipt["regime_counts"].values())


@pytest.mark.parametrize("n", [1999, 2001, 9])
def test_production_count_mismatch_cannot_be_relabeled_demo(n):
    rows = canonical_metadata()
    rows = rows[:n] if n < 2000 else rows + [{"sample_id": "extra", "regime": "high"}]
    with pytest.raises(ValueError, match="exactly 2000"):
        row_split(rows, 1, "poisson|dirichlet|smooth_grf", "original500")


@pytest.mark.parametrize("method", PUBLIC_LEARNED_METHODS)
@pytest.mark.parametrize("family", ["poisson", "heat"])
def test_c500_materialization_preserves_registry_and_budget(tmp_path, method, family):
    path = config_copy(tmp_path, family=family, cohort="original500", method=method)
    row, config, _ = resolve_row(path, tmp_path / "explicit-data")
    assert config["training"]["epochs"] == 500
    assert config["training"]["early_stopping_patience"] is None
    assert config["training"]["teacher_forcing_ratio"] == 0
    assert len(row["test_views"]) == 9
    assert row["registry_sha256"]
    assert config["method"]["name"] == ("autoregressive" if family == "heat" else {"ufno_2d": "ufno"}.get(method, method))


@pytest.mark.parametrize("cohort", ["fixed200", "budget200_min120", "interrupted_or_recovered"])
def test_unpublished_cohort_adapters_are_explicitly_rejected(tmp_path, cohort):
    config = config_copy(tmp_path, cohort=cohort)
    with pytest.raises(ValueError, match="unsupported cohort"):
        resolve_row(config, tmp_path)


def test_c500_rejects_budget_override(tmp_path):
    config = config_copy(tmp_path, cohort="original500", demo_epochs=1)
    with pytest.raises(ValueError, match="demo overrides"):
        resolve_row(config, tmp_path)


@pytest.mark.parametrize("family", ["poisson", "heat"])
def test_real_demo_row_trains_then_loads_last_then_strict_scores_without_test_leakage(tmp_path, monkeypatch, family):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from pdeobs.dataset import BenchmarkDataset
    from pdeobs.storage import LazyHDF5Dataset
    from pdeobs.training import Trainer

    data = tmp_path / "data"
    create_paper_row_demo_data(data, family)
    config = config_copy(tmp_path, family=family)
    output = tmp_path / "run"
    touched = []
    stage = {"training": False, "training_finished": False}
    original_item, original_fit = LazyHDF5Dataset.__getitem__, Trainer.fit
    def read_item(self, index):
        sample = original_item(self, index)
        if stage["training"]:
            touched.append(sample.metadata["sample_id"])
        else:
            # Setup reads metadata and shape headers, never numerical targets.
            assert stage["training_finished"], "target read before training"
            assert (output / "training_completion.json").is_file()
            assert (output / "checkpoints/last.pt").is_file()
        return sample
    def fit(self, train_loader, val_loader=None):
        assert val_loader is None
        stage["training"] = True
        result = original_fit(self, train_loader, val_loader)
        stage.update(training=False, training_finished=True)
        return result
    monkeypatch.setattr(LazyHDF5Dataset, "__getitem__", read_item)
    monkeypatch.setattr(Trainer, "fit", fit)
    result = run_paper_row(config, data_root=data, manifest_path=data / "identities.json", output=output)
    assert result["status"] == "complete", result
    split = json.loads((output / "split.json").read_text())
    assert split["macrodomain_identity"] == f"{family}|{'periodic' if family == 'heat' else 'dirichlet'}|smooth_grf"
    assert set(touched) == set(split["train_ids"])
    assert not set(touched) & set(split["test_ids"])
    assert (result["training_records"], result["test_records"], result["actual_epochs"]) == (6, 3, 1)
    assert result["final_checkpoint_reloaded"] is True
    assert result["validation_records"] == 0
    assert result["historical_results_replayed"] is False
    assert len(result["evaluation_blocks"]) == 2
    assert all(item["status"] == "valid" and item["scored_identity_count"] == 3 for item in result["evaluation_blocks"])
    for item in result["evaluation_blocks"]:
        score = json.loads((output / "evaluation" / item["test_view"] / "score.json").read_text())
        assert score["config"]["checkpoint_id"] == result["checkpoint_id"]
        assert score["config"]["training_seed"] == 20260804
        assert score["config"]["training_config_sha256"] == result["training_config_sha256"]
        assert score["artifact_contract_binding"] == "pdeobs-strict-inference-v1_sha256_bound"


def test_missing_data_is_failure_receipt_cli_nonzero_and_never_generates(tmp_path, monkeypatch):
    import pdeobs.generation
    def forbidden(*args, **kwargs):
        raise AssertionError("row runner must never generate missing paper data")
    monkeypatch.setattr(pdeobs.generation, "generate_job", forbidden)
    config = config_copy(tmp_path, cohort="original500")
    out = tmp_path / "run"
    assert main(["paper-row", "--config", str(config), "--data-root", str(tmp_path / "missing"),
                 "--manifest", str(tmp_path / "missing.json"), "--output", str(out)]) == 2
    result = json.loads((out / "receipt.json").read_text())
    assert result["status"] == "failed"
    assert result["stage"] == "manifest_and_metadata"
    assert not (out / "checkpoints").exists()


def test_config_typo_cannot_silently_change_scientific_settings(tmp_path):
    config = config_copy(tmp_path, epochs=1)
    with pytest.raises(ValueError, match="unknown row config keys"):
        resolve_row(config, tmp_path)


@pytest.mark.parametrize("key,value", [("training_seed", -1), ("training_seed", 2**32),
                                      ("training_seed", True), ("run_id", "../bad"),
                                      ("attempt_id", "space forbidden")])
def test_invalid_seed_and_run_identity_rejected_before_training(tmp_path, key, value):
    config = config_copy(tmp_path, **{key: value})
    with pytest.raises(ValueError, match="training_seed|portable"):
        resolve_row(config, tmp_path)


def test_c500_does_not_silently_relabel_thinned_time_coordinates():
    validate_c500_frames([{"stored_frame_indices": [0, 1, 2, 3]}],
                         [{"target_source_frame_indices": [1, 2, 3]}], "rollout")
    with pytest.raises(ValueError, match="thinned"):
        validate_c500_frames([{"stored_frame_indices": [0, 2, 4, 6]}],
                             [{"target_source_frame_indices": [2, 4, 6]}], "rollout")
