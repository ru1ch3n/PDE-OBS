"""L0/L1 CPU tests: the six one-line workflows end to end on tiny generated records.

Torch-gated (importskip inside each test), consistent with the repo's convention.
"""
from __future__ import annotations

import numpy as np
import pytest

from pdeobs import api


def _static_dataset(tmp_path):
    return api.create_dataset(pde="poisson", boundary="periodic", setting="smooth_grf", regime="low",
                              out=tmp_path / "poisson", num_samples=6, resolution=16, seed=3, time_steps=None)


def _rollout_dataset(tmp_path):
    return api.create_dataset(pde="heat", boundary="periodic", setting="smooth_grf", regime="low",
                              out=tmp_path / "heat", num_samples=6, resolution=16, seed=5, time_steps=4)


def test_create_and_load_dataset(tmp_path):
    pytest.importorskip("h5py")
    handle = _static_dataset(tmp_path)
    assert handle.sample_count == 6
    assert handle.resolution == [16, 16]
    assert "recovery" in handle.supported_tasks and "rollout" not in handle.supported_tasks
    reloaded = api.load_dataset(handle.path)
    assert reloaded.sample_ids_sha256 == handle.sample_ids_sha256


def test_dataset_from_user_arrays(tmp_path):
    pytest.importorskip("h5py")
    n = 16
    axis = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(axis, axis)
    source = np.stack([np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * (k + 1) for k in range(3)])
    handle = api.create_dataset_from_arrays(pde="poisson", out=tmp_path / "arrays", boundary="periodic",
                                            arrays={"source": source}, physical={})
    assert handle.sample_count == 3 and handle.source == "arrays"


def test_recovery_train_predict_evaluate(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    handle = _static_dataset(tmp_path)
    obs = api.make_observation("random", ratio=0.5)
    result = api.train(handle, task="recovery", model="fno", preset="smoke", observation=obs,
                       out=tmp_path / "run", budget={"max_steps": 3}, seed=1, batch_size=2)
    assert result.optimizer_steps >= 1
    assert result.resolved["parameters_changed_after_training"] is True
    predictor = api.load_predictor(result.artifact.path)
    # target-free inference from an observation-only package
    package, targets = api.inference_input_from_dataset(handle, obs, task="recovery")
    bundle = api.predict(predictor, package, out=tmp_path / "preds.npz")
    assert bundle.predictions.shape == targets.shape
    assert bundle.provenance["targets_used"] is False
    # strict scoring only with the explicit target
    report = api.evaluate(bundle, targets, out=tmp_path / "score")
    assert report["status"] == "valid"
    assert report["summary"]["identity_count"] == handle.sample_count


def test_rollout_train_predict_reload(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    handle = _rollout_dataset(tmp_path)
    obs = api.make_observation("random", ratio=0.5)
    result = api.train(handle, task="rollout", model="fno", preset="smoke", observation=obs,
                       out=tmp_path / "run", budget={"max_steps": 3}, seed=1, batch_size=2, history_steps=1, horizon=3)
    # reload in a fresh predictor and check the structure comes from the manifest
    predictor = api.load_predictor(result.artifact.path)
    assert predictor.config.task == "rollout"
    package, targets = api.inference_input_from_dataset(handle, obs, task="rollout", history_steps=1)
    bundle = api.predict(predictor, package, horizon=3)
    assert bundle.predictions.shape[1] == 3  # horizon
    report = api.evaluate(bundle, targets)
    assert report["status"] == "valid"
    assert "rel_l2_by_horizon_mean" in report["summary"]


def test_predict_requires_trained_weights(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    handle = _static_dataset(tmp_path)
    obs = api.make_observation("random", ratio=0.5)
    result = api.train(handle, task="recovery", model="fno", preset="smoke", observation=obs,
                       out=tmp_path / "run", budget={"max_steps": 2}, batch_size=2)
    untrained = api.load_predictor(result.artifact.path, allow_untrained=True)
    package, _ = api.inference_input_from_dataset(handle, obs, task="recovery")
    with pytest.raises(ValueError, match="no trained weights"):
        api.predict(untrained, package)


def test_evaluate_needs_target(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    handle = _static_dataset(tmp_path)
    obs = api.make_observation("random", ratio=0.5)
    result = api.train(handle, task="recovery", model="fno", preset="smoke", observation=obs,
                       out=tmp_path / "run", budget={"max_steps": 2}, batch_size=2)
    predictor = api.load_predictor(result.artifact.path)
    package = api.InferenceInput.load(api.inference_input_from_dataset(handle, obs, task="recovery")[0].save(tmp_path / "in.npz"))
    bundle = api.predict(predictor, package)
    with pytest.raises(ValueError, match="target"):
        api.evaluate(bundle)  # no targets available -> refuse to fabricate a score


def test_training_requires_explicit_budget(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    handle = _static_dataset(tmp_path)
    obs = api.make_observation("random", ratio=0.5)
    with pytest.raises(ValueError, match="budget"):
        api.train(handle, task="recovery", model="fno", preset="smoke", observation=obs, out=tmp_path / "run", budget=None)


def test_runs_not_overwritten(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    handle = _static_dataset(tmp_path)
    obs = api.make_observation("random", ratio=0.5)
    api.train(handle, task="recovery", model="fno", preset="smoke", observation=obs, out=tmp_path / "run", budget={"max_steps": 2}, batch_size=2)
    with pytest.raises(FileExistsError):
        api.train(handle, task="recovery", model="fno", preset="smoke", observation=obs, out=tmp_path / "run", budget={"max_steps": 2}, batch_size=2)


def test_old_and_new_entrypoint_equivalence(tmp_path):
    """The facade predictions equal the underlying runner/evaluate path for the same weights."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("h5py")
    handle = _static_dataset(tmp_path)
    obs = api.make_observation("random", ratio=0.5)
    result = api.train(handle, task="recovery", model="fno", preset="smoke", observation=obs,
                       out=tmp_path / "run", budget={"max_steps": 3}, batch_size=2, seed=7)
    predictor = api.load_predictor(result.artifact.path)
    package, targets = api.inference_input_from_dataset(handle, obs, task="recovery")
    facade = api.predict(predictor, package).predictions

    # direct path: rebuild the same module, load weights, run predict_batch
    from pdeobs.dataset import BenchmarkDataset, collate_benchmark
    from pdeobs.evaluation import EvaluationConfig, predict_batch
    module = api.build_model(predictor.config)
    from pdeobs.training import load_checkpoint_payload
    module.load_state_dict(load_checkpoint_payload(result.artifact.weights))
    module.eval()
    ds = BenchmarkDataset(handle.shards, task="recovery", mask=dict(obs.mask_config), seed=0)
    try:
        rows = [ds[i] for i in range(len(ds))]
    finally:
        ds.close()
    cfg = EvaluationConfig(task="recovery", device="cpu", data_layout="channels_last")
    direct, _ = predict_batch(module, collate_benchmark(rows), cfg)
    assert np.allclose(facade, direct, atol=1e-5)


def test_paper_view_evaluation(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    # paper views are 128x128; a 16x16 dataset must be rejected for that namespace during training
    handle = _static_dataset(tmp_path)
    with pytest.raises(ValueError, match="128"):
        api.train(handle, task="recovery", model="fno", preset="smoke", observation=api.make_observation("paper:R50"),
                  out=tmp_path / "run", budget={"max_steps": 2})


# ----------------------------------------------------------------------------- regressions from the v0.2.0 full-coverage sweep
def test_parameter_free_models_round_trip(tmp_path):
    """T3 regression: a parameter-free artifact stores ``weights: null`` and must load and predict."""
    pytest.importorskip("h5py")
    handle = _static_dataset(tmp_path)
    obs = api.make_observation("random", ratio=0.5)
    for name in ("zero", "mean", "nearest", "bilinear", "rbf"):
        run = api.train(handle, task="recovery", model=name, observation=obs, out=tmp_path / name, budget={"max_steps": 1}, device="cpu")
        assert run.artifact.manifest["weights"] is None and run.artifact.manifest["kind"] == "parameter_free_method"
        predictor = api.load_predictor(run.artifact.path)
        assert predictor.trained and predictor.provenance["weights_sha256"] is None
        package, targets = api.inference_input_from_dataset(handle, obs, task="recovery")
        report = api.evaluate(api.predict(predictor, package), targets)
        assert report["status"] == "valid", name


def test_evaluate_views_accepts_same_protocol_at_two_settings(tmp_path):
    """T5 regression: two general views with the same protocol name must get distinct output directories."""
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    handle = _static_dataset(tmp_path)
    run = api.train(handle, task="recovery", model="fno", preset="smoke", observation=api.make_observation("random", ratio=0.5),
                    out=tmp_path / "run", budget={"max_steps": 1}, batch_size=2, device="cpu")
    predictor = api.load_predictor(run.artifact.path)
    views = [api.make_observation("random", ratio=0.5), api.make_observation("random", ratio=0.2), api.make_observation("block", missing_fraction=0.3)]
    summary = api.evaluate_views(predictor, handle, views, out=tmp_path / "views", batch_size=2, device="cpu", namespace="general")
    assert summary["count"] == 3 and summary["all_valid"]
    keys = list(summary["views"])
    assert "block" in keys and len({v["observation_id"] for v in summary["views"].values()}) == 3
    for key, entry in summary["views"].items():
        assert (tmp_path / "views" / key / "score.json").is_file() and entry["directory"] == key
    with pytest.raises(ValueError, match="more than once"):
        api.evaluate_views(predictor, handle, [views[0], views[0]], out=tmp_path / "dup", batch_size=2, device="cpu", namespace="general")


def test_block_shape_side_length_is_a_public_int(tmp_path):
    """T2 regression: ``block_shape`` is documented as one side length; the mask factory takes (height, width)."""
    spec = api.make_observation("block", block_shape=6)
    assert spec.mask_config["block_shape"] == (6, 6) and spec.params["block_shape"] == 6
    from pdeobs.api.observation import realized_count
    assert realized_count(spec, (16, 16)) == 16 * 16 - 36
    with pytest.raises(ValueError, match="conflict"):
        api.make_observation("block", block_shape=6, missing_fraction=0.3)
    with pytest.raises(ValueError, match="expects int"):
        api.make_observation("block", block_shape="six")


def test_navier_stokes_served_representation_sets_channels(tmp_path):
    """T4 regression: the model structure follows the served representation (velocity=2, vorticity=1), not the stored one."""
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    handle = api.create_dataset(pde="navier_stokes", boundary="periodic", setting="smooth_grf", regime="low",
                                out=tmp_path / "ns", num_samples=4, resolution=16, seed=7, time_steps=3)
    assert handle.pde == "navier_stokes"
    obs = api.make_observation("random", ratio=0.5)
    for rep, channels in (("vorticity", 1), ("velocity", 2)):
        run = api.train(handle, task="rollout", model="fno", preset="smoke", observation=obs, out=tmp_path / rep,
                        budget={"max_steps": 1}, batch_size=2, device="cpu", history_steps=1, horizon=2, state_representation=rep)
        assert run.artifact.manifest["io_schema"]["in_channels"] == channels
        assert run.artifact.manifest["io_schema"]["state_representation"] == rep
        predictor = api.load_predictor(run.artifact.path)
        package, targets = api.inference_input_from_dataset(handle, obs, task="rollout", history_steps=1, state_representation=rep)
        assert package.observations.shape[-1] == channels
        batch = api.predict(predictor, package, horizon=2)
        assert batch.predictions.shape[-1] == channels
        assert api.evaluate(batch, targets)["status"] == "valid"
