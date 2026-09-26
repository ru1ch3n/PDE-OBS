"""Release checks: physical reference, paper masks, fixed-mask input permissions."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from pdeobs.masks import apply_mask, generate_mask
from pdeobs.physical_inputs import advance_custom_heat, solve_custom_elliptic


@pytest.mark.parametrize("n", [16, 32])
def test_heat_periodic_continuous_fourier_reference(n):
    coordinates = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(coordinates, coordinates)
    initial = np.cos(2 * np.pi * x) * np.sin(4 * np.pi * y)
    alpha, dt = 0.02, 0.125
    actual, _ = advance_custom_heat(initial, diffusivity=alpha, dt=dt,
                                     boundary="periodic", dx=1/n, dy=1/n)
    exact = initial * np.exp(-alpha * dt * (2 * np.pi)**2 * 5)
    assert np.max(np.abs(actual - exact)) < 1e-12


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_custom_physics_rejects_nonfinite_before_existing_solver(bad):
    source = np.ones((8, 8))
    source[0, 0] = bad
    with pytest.raises(ValueError):
        solve_custom_elliptic(source, boundary="periodic", dx=1/8, dy=1/8)


def test_custom_physical_demo_receipt_serializes_numpy_solver_flags(tmp_path):
    import json
    from pdeobs.release_demo import run_demo
    report = run_demo("E4", data_root=tmp_path / "data", output_root=tmp_path / "runs")
    assert report["status"] == "passed"
    assert report["details"]["heat_analytic_max_absolute_error"] < 1e-12
    written = json.loads((tmp_path / "runs/E4/receipt.json").read_text(encoding="utf-8"))
    assert written["details"]["elliptic_solver"]["converged"] is True


@pytest.mark.parametrize("protocol,kwargs,count", [
    ("random_3pct", {"ratio": .5}, 8192),
    ("random_3pct", {"ratio": .65}, 10650),
    ("random_3pct", {"ratio": .8}, 13107),
    ("block_missing", {"missing_fraction": .49}, 8284),
    ("line_sensors", {"num_lines": 76, "orientation": "both"}, 8284),
    ("line_sensors", {"num_lines": 64, "orientation": "horizontal"}, 8192),
    ("line_sensors", {"num_lines": 64, "orientation": "vertical"}, 8192),
    ("boundary_sensors", {"width": 19}, 8284),
    ("clustered_sensors", {"ratio": .5}, 8192),
])
def test_paper_mask_exact_counts(protocol, kwargs, count):
    mask = generate_mask(protocol, (128, 128), seed=33, **kwargs)
    assert mask.dtype == bool
    assert mask.sum() == count


def test_random_density_masks_are_not_claimed_nested():
    low = generate_mask("random_3pct", (128, 128), seed=33, ratio=.5)
    high = generate_mask("random_3pct", (128, 128), seed=33, ratio=.65)
    # The contract promises reproducibility and count, not (non-)nesting.
    # Particular NumPy versions/seeds can yield nested samples accidentally.
    np.testing.assert_array_equal(low, generate_mask("random_3pct", (128, 128), seed=33, ratio=.5))
    np.testing.assert_array_equal(high, generate_mask("random_3pct", (128, 128), seed=33, ratio=.65))
    assert (int(low.sum()), int(high.sum())) == (8192, 10650)


def _rows(task):
    rng = np.random.default_rng(29)
    field = rng.standard_normal((16, 16, 1)).astype(np.float32)
    # Instantiate once. Do not change sample identity or regenerate its seed.
    mask = generate_mask("random_3pct", (16, 16), seed=3, ratio=.5)
    observed = apply_mask(field, mask)
    if task == "rollout":
        row = {"observations": observed[None], "mask": mask[None, ..., None].astype(np.float32),
               "target": rng.standard_normal((3, 16, 16, 1)).astype(np.float32), "initial_state": field[None]}
    else:
        row = {"observations": observed, "mask": mask[..., None].astype(np.float32),
               "target": field.copy(), "pde_condition": field.copy()}
    row.update(geometry=np.zeros((16, 16, 1), np.float32), metadata={"sample_id": "fixed-demo-id"})
    changed = copy.deepcopy(row)
    if task == "rollout":
        changed["target"][:] = 12345
        changed["initial_state"][:, ~mask] = -1000
    else:
        changed["target"][~mask] = 12345
        changed["pde_condition"][:] = -1000
    assert np.array_equal(row["mask"], changed["mask"])
    assert np.array_equal(row["observations"], changed["observations"])
    return row, changed


@pytest.mark.parametrize("task", ["recovery", "rollout"])
@pytest.mark.parametrize("neural", [False, True])
def test_builtin_prediction_ignores_hidden_target_and_future_values(task, neural):
    from pdeobs.dataset import collate_benchmark
    from pdeobs.evaluation import EvaluationConfig, predict_batch
    from pdeobs.methods import create_method
    if neural:
        torch = pytest.importorskip("torch")
        torch.set_num_threads(1)
        torch.manual_seed(12)
        base = create_method("unet", width=4)
        if task == "rollout":
            from pdeobs.methods.neural import AutoregressiveModel
            method = AutoregressiveModel(one_step_model=base)
        else:
            method = base
    else:
        method = create_method("persistence" if task == "rollout" else "nearest")
    first, second = _rows(task)
    configuration = EvaluationConfig(task=task, device="cpu", data_layout="channels_last", horizon=3,
                                      history_steps=1, rollout_target_offset=0)
    a, _ = predict_batch(method, collate_benchmark([first]), configuration)
    b, _ = predict_batch(method, collate_benchmark([second]), configuration)
    np.testing.assert_array_equal(a, b)
