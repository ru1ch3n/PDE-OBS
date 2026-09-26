"""Positive and negative release acceptance for strict-v1 (no training loss edits)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pdeobs.strict_score import CONTRACT_VERSION, score_arrays


def case(n=3, task="recovery"):
    shape = (3, 4, 4, 1) if task == "rollout" else (4, 4, 1)
    target = np.arange(n * np.prod(shape), dtype=np.float64).reshape(n, *shape) / 11 + 1
    times = [1, 2, 3] if task == "rollout" else [0]
    ids = [f"demo-{j}" for j in range(n)]
    result = dict(prediction=target * 1.1, target=target, prediction_ids=ids[:], target_ids=ids[:],
                  prediction_time_indices=times, target_time_indices=times,
                  contract=dict(schema_version=CONTRACT_VERSION, expected_ids=ids, task=task,
                                expected_shape=list(shape), target_time_indices=times,
                                split="demo", observation_id="fixed-mask", checkpoint_id="demo-model",
                                projection=False))
    if task == "recovery":
        mask = np.zeros((n, *shape), dtype=bool)
        mask[:, ::2] = True
        result.update(mask=mask, observation=np.where(mask, target, 0))
    return result


def test_valid_mean_is_per_identity_not_global_norm():
    c = case(2)
    c["prediction"][0] = c["target"][0] * 2
    c["prediction"][1] = c["target"][1]
    report = score_arrays(**c)
    assert report["status"] == "valid"
    assert report["summary"]["rel_l2_joint_mean"] == pytest.approx(0.5)
    assert report["scored_identity_count"] == 2
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("task", ["recovery", "rollout"])
def test_finite_strict_scores_preserve_legacy_relative_l2_mean(task):
    from pdeobs.metrics import relative_l2
    c = case(task=task)
    result = score_arrays(**c)
    assert result["status"] == "valid"
    assert result["summary"]["rel_l2_joint_mean"] == pytest.approx(relative_l2(c["prediction"], c["target"]), rel=1e-14)


@pytest.mark.parametrize("location", ["prediction", "target", "observation"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected_before_projection(location, bad):
    c = case()
    c[location].flat[0] = bad
    c["contract"]["projection"] = True
    result = score_arrays(**c)
    assert result["status"] == "invalid"
    assert result["summary"] is None
    assert result["scored_identity_count"] == 0
    assert not result["per_identity"]
    json.dumps(result, allow_nan=False)


def test_all_nan_rejected():
    c = case()
    c["prediction"][:] = np.nan
    assert score_arrays(**c)["status"] == "invalid"


@pytest.mark.parametrize("key", ["prediction_ids", "target_ids"])
@pytest.mark.parametrize("violation", ["duplicate", "missing", "extra", "blank"])
def test_identity_violations(key, violation):
    c = case()
    if violation == "duplicate":
        c[key][1] = c[key][0]
    elif violation == "missing":
        c[key] = c[key][:-1]
    elif violation == "extra":
        c[key].append("unrequested")
    else:
        c[key][0] = ""
    result = score_arrays(**c)
    assert result["status"] == "invalid"
    assert result["scored_identity_count"] == 0


def test_identity_order_is_explicitly_aligned():
    c = case()
    c["prediction_ids"].reverse()
    c["prediction"] = c["prediction"][::-1]
    result = score_arrays(**c)
    assert result["status"] == "valid"
    assert result["summary"]["rel_l2_joint_mean"] == pytest.approx(0.1)


@pytest.mark.parametrize("shape", [(3, 16, 1), (3, 4, 4, 2), (2, 4, 4, 1)])
def test_shape_violations(shape):
    c = case()
    c["prediction"] = np.ones(shape)
    assert score_arrays(**c)["status"] == "invalid"


@pytest.mark.parametrize("times", [[0, 1, 2], [1, 1, 3], [1, 3, 2], [1.0, 2.0, 3.0], [1, 2]])
def test_time_index_violations(times):
    c = case(task="rollout")
    c["prediction_time_indices"] = times
    assert score_arrays(**c)["status"] == "invalid"


def test_joint_and_horizon_are_distinct_names_and_reductions():
    c = case(1, "rollout")
    c["prediction"] = c["target"].copy()
    c["prediction"][:, 0] *= 2
    result = score_arrays(**c)
    assert result["status"] == "valid"
    horizons = result["summary"]["rel_l2_by_horizon_mean"]
    assert [r["rel_l2_mean"] for r in horizons] == pytest.approx([1, 0, 0])
    assert result["summary"]["rel_l2_joint_mean"] != pytest.approx(1 / 3)


def test_all_observed_hidden_is_not_applicable():
    c = case()
    c["mask"][:] = True
    c["observation"] = c["target"].copy()
    result = score_arrays(**c)
    assert result["status"] == "valid"
    for row in result["per_identity"]:
        assert row["static_diagnostics"]["hidden_only_status"] == "not_applicable"
        assert row["static_diagnostics"]["hidden_only_rel_l2"] is None
    assert result["summary"]["static_diagnostics"]["hidden_only_rel_l2_mean"] is None


def test_one_observed_common_denominators_and_projection():
    c = case()
    c["mask"][:] = False
    c["mask"][:, 0, 0] = True
    c["observation"] = np.where(c["mask"], c["target"], 0)
    c["contract"]["projection"] = True
    result = score_arrays(**c)
    assert result["status"] == "valid"
    for row in result["per_identity"]:
        d = row["static_diagnostics"]
        assert d["full_common_denominator"] ** 2 == pytest.approx(d["observed_common_denominator"] ** 2 + d["hidden_common_denominator"] ** 2)
        assert row["projected_rel_l2_joint"] == pytest.approx(d["hidden_common_denominator"])
    assert result["projection_applied"]


def test_constant_bad_finite_prediction_is_scored_not_quarantined():
    c = case()
    c["prediction"][:] = 999
    result = score_arrays(**c)
    assert result["status"] == "valid"
    assert result["summary"]["rel_l2_joint_mean"] > 100


def test_near_zero_target_epsilon_is_finite_and_not_false_zero():
    c = case()
    c["target"][:] = 1e-30
    c["observation"] = np.where(c["mask"], c["target"], 0)
    c["prediction"][:] = 1e-10
    result = score_arrays(**c)
    assert result["status"] == "valid"
    assert result["summary"]["rel_l2_joint_mean"] == pytest.approx(400)


def test_paper_split_requires_explicit_200():
    c = case()
    c["contract"]["split"] = "paper-test"
    assert score_arrays(**c)["status"] == "invalid"


def test_future_projection_rejected():
    c = case(task="rollout")
    c["contract"]["projection"] = True
    assert score_arrays(**c)["status"] == "invalid"


def test_tail_batch_scored_without_losing_identity():
    from pdeobs.methods import create_method
    from pdeobs.release_demo import _predict

    c = case()
    rows = [{"observations": c["observation"][j], "mask": c["mask"][j], "target": c["target"][j],
             "metadata": {"sample_id": f"demo-{j}"}} for j in range(3)]
    prediction, target, sizes = _predict(create_method("zero"), rows, "recovery")
    assert sizes == [2, 1]
    c.update(prediction=prediction, target=target)
    result = score_arrays(**c)
    assert result["status"] == "valid"
    assert result["summary"]["identity_count"] == 3


@pytest.mark.parametrize("invalid", [False, True])
def test_cli_nonzero_and_standard_json_on_invalid(tmp_path, invalid):
    c = case()
    contract = c.pop("contract")
    if invalid:
        c["prediction"].flat[0] = np.nan
    artifact = tmp_path / "arrays.npz"
    np.savez_compressed(artifact, **c)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    output = tmp_path / "result.json"
    child = subprocess.run([sys.executable, "-B", "-m", "pdeobs", "strict-score", "--predictions", str(artifact),
                            "--contract", str(contract_path), "--output", str(output)], capture_output=True, text=True)
    assert child.returncode == (2 if invalid else 0), child.stderr
    result = json.loads(output.read_text(encoding="utf-8"), parse_constant=lambda x: pytest.fail(x))
    assert result["status"] == ("invalid" if invalid else "valid")


def test_missing_artifact_and_malformed_contract_emit_invalid(tmp_path):
    from pdeobs.runner import run_strict_score
    invalid_contract = tmp_path / "bad.json"
    invalid_contract.write_text("{", encoding="utf-8")
    result = run_strict_score(tmp_path / "absent.npz", invalid_contract, tmp_path / "bad-result.json")
    assert result["status"] == "invalid"
    result = run_strict_score(tmp_path / "absent.npz", case()["contract"], tmp_path / "missing-result.json")
    assert result["status"] == "invalid"


def test_no_overwrite_of_old_score(tmp_path):
    from pdeobs.strict_score import write_report
    path = tmp_path / "score.json"
    write_report(path, score_arrays(**case()))
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_report(path, {"status": "replacement"})
    assert path.read_bytes() == original


@pytest.mark.parametrize("key", ["checkpoint_id", "task", "extra_unvalidated_metadata"])
def test_nonstandard_nan_contract_still_emits_standard_invalid_json(key, tmp_path):
    from pdeobs.strict_score import write_report
    c = case()
    c["contract"][key] = float("nan")
    report = score_arrays(**c)
    assert report["status"] == "invalid"
    write_report(tmp_path / "invalid.json", report)
    json.loads((tmp_path / "invalid.json").read_text(encoding="utf-8"), parse_constant=lambda x: pytest.fail(x))
