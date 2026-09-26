"""Real canonical generation -> ordinary inference -> bound strict file tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from pdeobs.cli import main
from pdeobs.dataset import BenchmarkDataset, collate_benchmark
from pdeobs.evaluation import EvaluationConfig, evaluate_model
from pdeobs.generation import GenerationJob, generate_job
from pdeobs.methods import create_method
from pdeobs.strict_inference import (
    ARTIFACT_VERSION, batches, create_identity_manifest, evaluate_model_strict,
    load_identity_dataset, metadata_target_shape, run_strict_infer, target_frame_metadata,
)
from pdeobs.strict_score import CONTRACT_VERSION, _canonical_hash, score_prediction_file


@pytest.fixture(params=["poisson", "heat"])
def source(request, tmp_path):
    family = request.param
    root = tmp_path / "data"
    root.mkdir()
    path = root / "physical.h5"
    generate_job(GenerationJob(pde=family, boundary="periodic", setting="smooth_grf", regime="low",
                 sample_start=0, sample_count=3, shard_index=0, output_path=str(path),
                 resolution=16, seed=37, macro_size=30, tier="custom",
                 time_steps=7 if family == "heat" else 1,
                 stored_time_steps=4 if family == "heat" else None,
                 quality={"profile": "report"}), resume=False, overwrite=False)
    task = "rollout" if family == "heat" else "recovery"
    method_name = "persistence" if task == "rollout" else "nearest"
    config = {"task": task, "seed": 37, "method": {"name": method_name},
              "data": {"mask": {"protocol": "random_3pct", "ratio": .5},
                       "input_horizon": 1, "horizon": 3},
              "evaluation": {"horizons": [1, 2, 3]}, "training": {"batch_size": 2}}
    dataset = BenchmarkDataset([path], task=task, mask=config["data"]["mask"], horizon=3, seed=37)
    ec = EvaluationConfig(task=task, device="cpu", horizon=3, data_layout="channels_last",
                          history_steps=1, rollout_target_offset=0)
    manifest = create_identity_manifest(root, [path.name], root / "manifest.json")
    contract = {"schema_version": CONTRACT_VERSION, "task": task, "split": "demo",
                "expected_ids": [row["sample_id"] for row in dataset.metadata],
                "expected_shape": metadata_target_shape(dataset, ec),
                "target_time_indices": target_frame_metadata(dataset.metadata[0], ec)["target_source_frame_indices"],
                "observation_id": "random50", "mask_config": config["data"]["mask"],
                "checkpoint_id": "unfitted:" + _canonical_hash(config["method"]),
                "dataset_manifest_sha256": _canonical_hash(manifest)}
    try:
        yield root, dataset, ec, config, contract
    finally:
        dataset.close()


def test_normal_static_and_rollout_inference_exports_strict_file(source, tmp_path):
    root, dataset, ec, config, contract = source
    path = tmp_path / "predictions.h5"
    result = evaluate_model_strict(create_method(config["method"]["name"]), batches(dataset, 2),
                      config=ec, contract=contract, predictions_path=path)
    assert result["status"] == "valid", result
    assert result["scored_identity_count"] == 3
    assert result["artifact_contract_binding"] == "pdeobs-strict-inference-v1_sha256_bound"
    replay = score_prediction_file(path, contract)
    assert replay == result
    with h5py.File(path) as handle:
        assert handle.attrs["artifact_version"] == ARTIFACT_VERSION
        assert handle["prediction_ids"].asstr()[:].tolist() == contract["expected_ids"]
        assert handle["target_ids"].asstr()[:].tolist() == contract["expected_ids"]
        metadata = [json.loads(value) for value in handle["metadata_json"].asstr()[:]]
        assert metadata[0]["target_source_frame_indices"] == contract["target_time_indices"]
        if ec.task == "rollout":
            assert contract["target_time_indices"] == [2, 4, 6]
            assert len(metadata[0]["target_physical_time_values"]) == 3
            assert len(result["summary"]["rel_l2_by_horizon_mean"]) == 3


def test_cli_inference_needs_no_manual_prediction_repackaging(source, tmp_path, capsys):
    root, dataset, ec, config, contract = source
    cfg, ctr = tmp_path / "infer.yaml", tmp_path / "contract.json"
    cfg.write_text(yaml.safe_dump(config), encoding="utf-8")
    ctr.write_text(json.dumps(contract), encoding="utf-8")
    out = tmp_path / "run"
    assert main(["strict-infer", "--config", str(cfg), "--contract", str(ctr),
                 "--data-root", str(root), "--manifest", str(root / "manifest.json"),
                 "--output", str(out)]) == 0
    assert main(["strict-score", "--predictions", str(out / "predictions.h5"),
                 "--contract", str(ctr), "--output", str(tmp_path / "replay.json")]) == 0
    report = json.loads((out / "score.json").read_text())
    assert report["status"] == "valid"
    capsys.readouterr()


@pytest.mark.parametrize("change", ["view", "identity", "time", "shape", "checkpoint", "manifest"])
def test_exporter_rejects_misbound_setup(source, tmp_path, change):
    root, dataset, ec, config, contract = source
    bad = copy.deepcopy(contract)
    if change == "view":
        bad["mask_config"]["ratio"] = .8
    elif change == "identity":
        bad["expected_ids"][0] = "not-present"
    elif change == "time":
        bad["target_time_indices"] = [value + 1 for value in bad["target_time_indices"]]
    elif change == "shape":
        bad["expected_shape"][-1] = 2
    elif change == "checkpoint":
        bad["checkpoint_id"] = "wrong"
    else:
        bad["dataset_manifest_sha256"] = "0" * 64
    cfg, ctr = tmp_path / "config.yaml", tmp_path / "contract.json"
    cfg.write_text(yaml.safe_dump(config), encoding="utf-8")
    ctr.write_text(json.dumps(bad), encoding="utf-8")
    report = run_strict_infer(cfg, ctr, data_root=root, manifest_path=root / "manifest.json", output=tmp_path / "run")
    assert report["status"] == "invalid"
    assert report["summary"] is None
    assert report["scored_identity_count"] == 0
    assert json.loads((tmp_path / "run/score.json").read_text())["status"] == "invalid"


def test_contract_binding_blocks_posthoc_relabeling(source, tmp_path):
    _, dataset, ec, config, contract = source
    path = tmp_path / "predictions.h5"
    assert evaluate_model_strict(create_method(config["method"]["name"]), batches(dataset, 3),
                  config=ec, contract=contract, predictions_path=path)["status"] == "valid"
    bad = {**contract, "observation_id": "pretend-other-view"}
    result = score_prediction_file(path, bad)
    assert result["status"] == "invalid"
    assert "different scoring contract" in result["errors"][0]


def test_duplicate_metadata_and_missing_time_are_fail_closed(source, tmp_path):
    _, dataset, ec, config, contract = source
    rows = [dataset[0], dataset[0], dataset[2]]
    result = evaluate_model_strict(create_method(config["method"]["name"]), [collate_benchmark(rows)],
                   config=ec, contract=contract, predictions_path=tmp_path / "duplicate.h5")
    assert result["status"] == "invalid"
    assert not (tmp_path / "duplicate.h5").exists()
    rows = [dataset[i] for i in range(3)]
    rows[0]["metadata"].pop("stored_frame_indices")
    result = evaluate_model_strict(create_method(config["method"]["name"]), [collate_benchmark(rows)],
                   config=ec, contract=contract, predictions_path=tmp_path / "time.h5")
    assert result["status"] == "invalid"


def test_legacy_writer_api_unchanged_and_explicitly_not_strict(source, tmp_path):
    _, dataset, ec, config, contract = source
    from dataclasses import replace
    path = tmp_path / "legacy.h5"
    result = evaluate_model(create_method(config["method"]["name"]), batches(dataset, 2),
                            config=replace(ec, predictions_path=str(path)))
    assert result["samples"] == 3
    with h5py.File(path) as handle:
        assert "sample_id" in handle
        assert "prediction_ids" not in handle
    assert score_prediction_file(path, contract)["status"] == "invalid"


def test_nonfinite_raw_prediction_is_preserved_but_never_scored(source, tmp_path):
    _, dataset, ec, config, contract = source
    base = create_method(config["method"]["name"])
    class Nonfinite:
        def predict(self, observed, mask, **kwargs):
            values = base.predict(observed, mask, **kwargs)
            values.reshape(-1)[0] = np.nan
            return values
    result = evaluate_model_strict(Nonfinite(), batches(dataset, 2), config=ec,
                contract=contract, predictions_path=tmp_path / "bad.h5", report_path=tmp_path / "score.json")
    assert result["status"] == "invalid"
    assert result["summary"] is None
    assert result["scored_identity_count"] == 0
    assert "NaN" not in (tmp_path / "score.json").read_text()


def test_changed_shard_rejected_before_any_target_loading(source, monkeypatch):
    root, dataset, _, _, _ = source
    def forbidden(*args, **kwargs):
        raise AssertionError("target field accessed before manifest verification")
    monkeypatch.setattr(BenchmarkDataset, "__getitem__", forbidden)
    with (root / "physical.h5").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        load_identity_dataset(root, root / "manifest.json")


def test_manifest_escape_rejected_without_fallback(tmp_path):
    outside = tmp_path / "outside.h5"
    outside.write_bytes(b"not-a-shard")
    root = tmp_path / "data"
    root.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        create_identity_manifest(root, ["../outside.h5"], root / "manifest.json")


def test_malformed_contract_gets_standard_invalid_json_cli_nonzero(tmp_path):
    config = tmp_path / "infer.yaml"
    config.write_text("task: recovery\nmethod: {name: nearest}\n", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text("[]", encoding="utf-8")
    out = tmp_path / "run"
    assert main(["strict-infer", "--config", str(config), "--contract", str(contract),
                 "--manifest", str(tmp_path / "missing.json"), "--data-root", str(tmp_path),
                 "--output", str(out)]) == 2
    report = json.loads((out / "score.json").read_text())
    assert report["status"] == "invalid"
    assert report["summary"] is None
