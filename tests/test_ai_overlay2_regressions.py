"""Overlay 2 regressions: a prediction bundle is validated when it is built.

Overlay 1 made ``PredictionBundle.load`` reject fractional or boolean frame labels, but a bundle
built in memory with ``time_indices=[0.9]`` could still be saved (``save`` casts to int64, storing 0)
and then reloaded and scored as valid.  Construction now applies the strict scorer's own identity and
frame rules, so no save/load path can launder an invalid label.  Facade only; no kernel, mask, split,
loss or metric change.
"""
from __future__ import annotations

import numpy as np
import pytest

from pdeobs import api
from pdeobs.api.infer import PredictionBundle


def _arrays():
    targets = np.stack([np.ones((4, 4, 1)), np.full((4, 4, 1), 2)]).astype(np.float32)
    inputs = api.InferenceInput(targets.copy(), np.ones_like(targets), sample_ids=["a", "b"])
    return targets, inputs


@pytest.mark.parametrize("bad", [[0.9], [0.0], [True], [2, 1], [0, 0], [-1], []])
def test_bundle_rejects_non_integer_or_non_increasing_frames(bad):
    targets, inputs = _arrays()
    with pytest.raises(ValueError, match="time_indices"):
        PredictionBundle(targets, ["a", "b"], bad, "recovery", inputs=inputs)


@pytest.mark.parametrize("ids", [["a", "a"], ["a"], ["a", "b", "c"], [], ["a", ""], ["a", 1]])
def test_bundle_rejects_duplicate_missing_extra_or_blank_ids(ids):
    targets, _ = _arrays()
    with pytest.raises(ValueError, match="sample_ids"):
        PredictionBundle(targets, ids, [0], "recovery")


def test_bundle_frame_count_must_match_prediction_frames():
    targets, _ = _arrays()
    with pytest.raises(ValueError, match="1 frame"):
        PredictionBundle(targets, ["a", "b"], [0, 1], "recovery")
    rollout = np.repeat(targets[:, None], 3, axis=1)  # (2, 3, 4, 4, 1)
    with pytest.raises(ValueError, match="3 frame"):
        PredictionBundle(rollout, ["a", "b"], [1, 2], "rollout")
    bundle = PredictionBundle(rollout, ["a", "b"], [1, 2, 3], "rollout")
    assert bundle.time_indices == [1, 2, 3]


def test_bundle_rejects_inputs_whose_ids_disagree():
    targets, inputs = _arrays()
    with pytest.raises(ValueError, match="inputs.sample_ids"):
        PredictionBundle(targets, ["b", "a"], [0], "recovery", inputs=inputs)


def test_bundle_rejects_non_channels_last_rank():
    targets, _ = _arrays()
    with pytest.raises(ValueError, match="channels-last"):
        PredictionBundle(targets[..., 0], ["a", "b"], [0], "recovery")


def test_valid_bundle_round_trips_and_scores_unchanged(tmp_path):
    targets, inputs = _arrays()
    bundle = PredictionBundle(targets.copy(), ["a", "b"], [0], "recovery", inputs=inputs)
    path = tmp_path / "bundle.npz"
    bundle.save(path)
    loaded = PredictionBundle.load(path)
    assert loaded.sample_ids == ["a", "b"] and loaded.time_indices == [0]
    report = api.evaluate(loaded, targets, target_ids=["a", "b"], target_time_indices=[0])
    assert report["status"] == "valid" and report["summary"]["rel_l2_joint_mean"] == 0.0


def test_laundering_path_is_closed(tmp_path):
    """The exact overlay-1 gap: a fractional frame label can no longer reach save()."""
    targets, inputs = _arrays()
    with pytest.raises(ValueError):
        PredictionBundle(targets.copy(), ["a", "b"], [0.9], "recovery", inputs=inputs).save(tmp_path / "laundered.npz")
    assert not (tmp_path / "laundered.npz").exists()
