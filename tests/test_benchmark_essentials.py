"""Minimal must-have test set for a PDE benchmark (one fast test per category).

Categories (see docs/testing_checklist.md):
  A solver correctness      B data determinism        C protocol freeze (split / views)
  D scorer known answers    E baseline sanity         F leakage
  G interface contracts     H model coverage          I reproducibility
  J provenance / tamper     K docs-as-tests           L version separation
Each test is deliberately tiny (16-32 grids, 1-2 optimizer steps); they document the invariant, not
performance.  Torch/h5py-dependent tests skip when the extra is not installed.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from pdeobs import __solver_version__, __version__, api
from pdeobs.api import specs as S


# ----------------------------------------------------------------------------- A solver correctness
def test_A_poisson_manufactured_solution_periodic():
    from pdeobs.physical_inputs import solve_custom_elliptic
    n = 32
    ax = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(ax, ax)
    exact = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    source = 8 * np.pi**2 * exact  # -laplace(exact)
    u, info = solve_custom_elliptic(source, boundary="periodic", dx=1 / n, dy=1 / n)
    assert bool(info.converged)
    u = u - u.mean(); exact = exact - exact.mean()
    rel = np.linalg.norm(u - exact) / np.linalg.norm(exact)
    assert rel < 0.02, f"second-order FD should reproduce the manufactured solution, rel err {rel:.4f}"


def test_A_heat_single_mode_decay_periodic():
    from pdeobs.physical_inputs import advance_custom_heat
    n = 16
    ax = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(ax, ax)
    initial = np.cos(2 * np.pi * x) * np.sin(4 * np.pi * y)
    advanced, info = advance_custom_heat(initial, diffusivity=0.02, dt=0.125, boundary="periodic", dx=1 / n, dy=1 / n)
    reference = initial * np.exp(-0.02 * 0.125 * (2 * np.pi) ** 2 * (1 + 4))
    assert bool(info.converged)
    assert np.abs(advanced - reference).max() < 1e-6


# ----------------------------------------------------------------------------- B data determinism
def test_B_generation_is_seed_deterministic(tmp_path):
    pytest.importorskip("h5py")
    from pdeobs.storage import LazyHDF5Dataset
    kw = dict(pde="poisson", boundary="periodic", setting="smooth_grf", regime="low", num_samples=2, resolution=16, seed=5)
    h1 = api.create_dataset(out=tmp_path / "a", **kw)
    h2 = api.create_dataset(out=tmp_path / "b", **kw)
    r1, r2 = LazyHDF5Dataset(h1.shards), LazyHDF5Dataset(h2.shards)
    try:
        assert np.array_equal(r1[0].trajectory, r2[0].trajectory) and np.array_equal(r1[1].condition, r2[1].condition)
    finally:
        r1.close(); r2.close()
    assert h1.sample_ids_sha256 == h2.sample_ids_sha256
    h3 = api.create_dataset(out=tmp_path / "c", **{**kw, "seed": 6})
    assert h3.sample_ids_sha256 != h1.sample_ids_sha256


# ----------------------------------------------------------------------------- C protocol freeze
@pytest.mark.parametrize("view", list(S.PAPER_VIEWS))
def test_C_paper_view_counts_are_frozen(view):
    spec = api.make_observation(f"paper:{view}")
    assert api.realized_count(spec, (128, 128), seed=0) == S.PAPER_VIEWS[view]["expected_count_128"]


def test_C_stable_split_is_exact_and_frozen():
    from pdeobs.one_setting import stable_split
    meta = [{"sample_id": f"s/{i:04d}", "regime": ["low", "medium", "high"][i % 3]} for i in range(2000)]
    tr, te, receipt = stable_split(meta, 20260804, "poisson|dirichlet|smooth_grf")
    assert (len(tr), len(te)) == (1800, 200) and not set(tr) & set(te)
    tr2, te2, receipt2 = stable_split(meta, 20260804, "poisson|dirichlet|smooth_grf")
    assert te == te2 and receipt["test_sample_ids_sha256"] == receipt2["test_sample_ids_sha256"]
    _, te3, _ = stable_split(meta, 1, "poisson|dirichlet|smooth_grf")
    assert te3 != te, "a different seed must change the held-out set"


# ----------------------------------------------------------------------------- D scorer known answers
def _contract(ids, shape):
    from pdeobs.strict_score import CONTRACT_VERSION
    return {"schema_version": CONTRACT_VERSION, "task": "recovery", "split": "demo", "expected_ids": ids, "expected_shape": list(shape),
            "target_time_indices": [0], "observation_id": "test", "checkpoint_id": "test", "projection": False}


def _recovery_case(n=3, h=8):
    rng = np.random.default_rng(0)
    target = rng.standard_normal((n, h, h, 1))
    mask = rng.random((n, h, h, 1)) < 0.5
    obs = np.where(mask, target, 0.0)
    ids = [f"id{i}" for i in range(n)]
    return target, mask, obs, ids


def test_D_scorer_perfect_zero_and_invalid_cases():
    from pdeobs.strict_score import score_arrays
    target, mask, obs, ids = _recovery_case()
    c = _contract(ids, target.shape[1:])
    kw = dict(prediction_ids=ids, target_ids=ids, prediction_time_indices=[0], target_time_indices=[0], contract=c, mask=mask, observation=obs)
    perfect = score_arrays(target.copy(), target, **kw)
    assert perfect["status"] == "valid" and abs(perfect["summary"]["rel_l2_joint_mean"]) < 1e-12
    zero = score_arrays(np.zeros_like(target), target, **kw)
    assert zero["status"] == "valid" and abs(zero["summary"]["rel_l2_joint_mean"] - 1.0) < 1e-12
    nan = target.copy(); nan[0, 0, 0, 0] = np.nan
    assert score_arrays(nan, target, **kw)["status"] == "invalid"
    missing = score_arrays(target[:2], target[:2], **{**kw, "prediction_ids": ids[:2], "target_ids": ids[:2], "mask": mask[:2], "observation": obs[:2]})
    assert missing["status"] == "invalid", "a missing identity must invalidate, never shrink the denominator"


def test_D_scorer_is_batch_invariant():
    from pdeobs.strict_score import score_arrays
    target, mask, obs, ids = _recovery_case(n=4)
    pred = target + 0.1
    full = score_arrays(pred, target, prediction_ids=ids, target_ids=ids, prediction_time_indices=[0], target_time_indices=[0],
                        contract=_contract(ids, target.shape[1:]), mask=mask, observation=obs)
    per = {row["identity"]: row["rel_l2_joint"] for row in full["per_identity"]}
    for i in range(4):
        one = score_arrays(pred[i:i + 1], target[i:i + 1], prediction_ids=ids[i:i + 1], target_ids=ids[i:i + 1], prediction_time_indices=[0],
                           target_time_indices=[0], contract=_contract(ids[i:i + 1], target.shape[1:]), mask=mask[i:i + 1], observation=obs[i:i + 1])
        assert abs(one["per_identity"][0]["rel_l2_joint"] - per[ids[i]]) < 1e-12


# ----------------------------------------------------------------------------- E baseline sanity
def test_E_trivial_baselines_have_expected_order(tmp_path):
    pytest.importorskip("h5py")
    h = api.create_dataset(pde="poisson", out=tmp_path / "d", boundary="periodic", setting="smooth_grf", regime="low", num_samples=3, resolution=16, seed=2)
    obs = api.make_observation("random", ratio=0.5)
    scores = {}
    for m in ("zero", "mean", "nearest"):
        r = api.train(h, task="recovery", model=m, observation=obs, out=tmp_path / m, budget={"max_steps": 1}, device="cpu")
        assert r.artifact.weights is None and r.artifact.fitted_state is None  # parameter-free: manifest weights are null
        pkg, tg = api.inference_input_from_dataset(h, obs, task="recovery")
        rep = api.evaluate(api.predict(api.load_predictor(r.artifact.path), pkg), tg)
        assert rep["status"] == "valid"
        scores[m] = rep["summary"]["rel_l2_joint_mean"]
    # zero fill keeps the observed cells, so its joint error is exactly the unobserved share of the target energy
    missing = 1.0 - np.asarray(pkg.mask, dtype=np.float64)
    expected = [float(np.sqrt(((t * missing[i]) ** 2).sum() / (t ** 2).sum())) for i, t in enumerate(np.asarray(tg, dtype=np.float64))]
    assert abs(scores["zero"] - float(np.mean(expected))) < 1e-6, "zero-fill scores the unobserved energy fraction exactly"
    assert 0.5 < scores["zero"] < 0.9  # 50% random view on a smooth field: neither trivial nor total loss
    # nearest-neighbour interpolation beats any constant fill; on a zero-mean GRF the observed-mean constant is
    # close to zero, so mean fill and zero fill agree closely (which one is marginally better is data dependent)
    assert scores["nearest"] < min(scores["zero"], scores["mean"])
    assert abs(scores["mean"] - scores["zero"]) < 0.05 * scores["zero"]


# ----------------------------------------------------------------------------- F leakage
def test_F_train_and_test_identities_never_overlap():
    meta = [{"sample_id": f"s/{i:04d}", "regime": ["low", "medium", "high"][i % 3], "split": "train" if i % 5 else "test"} for i in range(2000)]
    for split in ("paper", "holdout:0.1", "stored"):
        p = api.plan_split(meta, split, seed=3, identity="x")
        assert p.train_ids and not set(p.train_ids) & set(p.test_ids)
    with pytest.raises(ValueError):
        api.plan_split(meta, {"train_ids": ["s/0000"], "test_ids": ["s/0000"]}, seed=3, identity="x")


# ----------------------------------------------------------------------------- G interface contracts
def test_G_every_public_model_resolves_for_its_declared_tasks():
    for name, spec in S.MODEL_SPECS.items():
        if spec.upstream_wrapper:
            with pytest.raises(ValueError):
                api.resolve_model(name, task=spec.tasks[0])
            continue
        for task in spec.tasks:
            cfg = api.resolve_model(name, task=task, preset="smoke" if "smoke" in spec.presets else None)
            assert cfg.name == name and cfg.task == task
        undeclared = [t for t in S.TASKS if t not in spec.tasks]
        for task in undeclared:
            with pytest.raises(ValueError):
                api.resolve_model(name, task=task)


def test_G_every_observation_protocol_resolves_and_rejects_conflicts():
    for name, spec in S.OBSERVATION_PROTOCOLS.items():
        if name == "custom":
            continue
        s = api.make_observation(name)
        assert s.mask_config.get("protocol")
        for group in spec.exclusive:
            with pytest.raises(ValueError):
                api.make_observation(name, **{g: 1 for g in group})


# ----------------------------------------------------------------------------- H model coverage
@pytest.mark.parametrize("name", [n for n, s in S.MODEL_SPECS.items() if s.family == "neural"])
def test_H_neural_model_forward_backward_is_finite(name):
    torch = pytest.importorskip("torch")
    spec = S.MODEL_SPECS[name]
    task = "rollout" if "recovery" not in spec.tasks else "recovery"
    cfg = api.resolve_model(name, task=task, preset="smoke" if "smoke" in spec.presets else None)
    model = api.build_model(cfg)
    x = torch.randn(2, cfg.kwargs.get("in_channels", 1), 16, 16)
    if task == "rollout":
        x = x[:, None]
    mask = (torch.rand(2, 1, 16, 16) > 0.5).float()
    geom = torch.zeros(2, 1, 16, 16)
    from pdeobs.training import TrainingConfig, _forward
    tc = TrainingConfig(task=task, epochs=1, amp=False, data_layout="channels_first", horizon=2, rollout_target_offset=0)
    y = _forward(model, x, mask, None, tc, True, geom, None, horizon=2)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# ----------------------------------------------------------------------------- I reproducibility
def test_I_same_seed_same_weights_cpu(tmp_path):
    pytest.importorskip("torch"); pytest.importorskip("h5py")
    h = api.create_dataset(pde="poisson", out=tmp_path / "d", boundary="periodic", setting="smooth_grf", regime="low", num_samples=4, resolution=16, seed=2)
    obs = api.make_observation("random", ratio=0.5)
    shas = []
    for i in range(2):
        r = api.train(h, task="recovery", model="fno", preset="smoke", observation=obs, out=tmp_path / f"r{i}", budget={"max_steps": 2}, batch_size=2, device="cpu", seed=9)
        shas.append(hashlib.sha256(r.artifact.weights.read_bytes()).hexdigest())
    assert shas[0] == shas[1]


# ----------------------------------------------------------------------------- J provenance / tamper
def test_J_artifact_records_weight_hash_and_detects_tampering(tmp_path):
    pytest.importorskip("torch"); pytest.importorskip("h5py")
    h = api.create_dataset(pde="poisson", out=tmp_path / "d", boundary="periodic", setting="smooth_grf", regime="low", num_samples=4, resolution=16, seed=2)
    r = api.train(h, task="recovery", model="fno", preset="smoke", observation=api.make_observation("random", ratio=0.5), out=tmp_path / "r",
                  budget={"max_steps": 1}, batch_size=2, device="cpu")
    manifest = json.loads((r.artifact.path / "model.json").read_text())
    assert manifest["weights"]["sha256"] == hashlib.sha256(r.artifact.weights.read_bytes()).hexdigest()
    assert manifest["numerical_kernel_version"] == __solver_version__
    with open(r.artifact.weights, "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(ValueError, match="sha256"):
        api.load_predictor(r.artifact.path)


# ----------------------------------------------------------------------------- K docs-as-tests
def test_K_readme_one_liners_parse():
    from pdeobs.cli import build_parser
    p = build_parser()
    for line in ("easy data --source generate --pde heat --resolution 32 --samples 12 --frames 4 --out demo/heat",
                 "easy train --data demo/heat --task rollout --model fno --model-param width=32 --model-param modes=8 --model-param layers=4 "
                 "--obs random --obs-param ratio=0.5 --max-steps 5 --out demo/fno",
                 "easy infer --model-from demo/fno/model --input observations.npz --out predictions.h5",
                 "easy eval --predictions predictions.h5 --targets targets.npz --out demo/score",
                 "easy run --config pipeline.yaml", "easy list models", "easy describe transolver", "easy validate --config pipeline.yaml",
                 "easy plan --config pipeline.yaml"):
        assert hasattr(p.parse_args(line.split()), "handler"), line


# ----------------------------------------------------------------------------- L version separation
def test_L_kernel_version_is_independent_of_package_version():
    assert __version__ == "0.2.1" and __solver_version__ == "0.1.0"
    assert S.API_VERSION == "pdeobs-easy-api/v1"
