import hashlib
import subprocess

import numpy as np
import pytest

from pdeobs.methods import (
    MethodCapabilities,
    available_methods,
    create_method,
    create_model,
    register_method,
)
from pdeobs.methods import base as method_base
from pdeobs.methods.paper_official import _git_blob_sha256
from pdeobs.registry import METHOD_REGISTRY as PROJECT_METHOD_REGISTRY


def sparse_plane():
    y, x = np.mgrid[:9, :9]
    field = (2 * y + x)[..., None].astype(float)
    mask = np.zeros_like(field, dtype=bool)
    mask[::4, ::4] = True
    return np.where(mask, field, 0.0), mask, field


def test_fill_baselines_keep_observations():
    observed, mask, _ = sparse_plane()
    for name in ("zero", "mean", "nearest", "bilinear", "rbf"):
        predicted = create_method(name).predict(observed, mask)
        assert predicted.shape == observed.shape
        np.testing.assert_allclose(predicted[mask], observed[mask], atol=2e-3)
        assert np.isfinite(predicted).all()


def test_gappy_pod_fits_reconstructs_and_round_trips_checkpoint(tmp_path):
    height = width = 8
    yy, xx = np.mgrid[:height, :width]
    first_mode = np.sin(2 * np.pi * xx / width)
    second_mode = np.cos(2 * np.pi * yy / height)
    fields = np.stack(
        [
            (0.25 + index / 10) * first_mode + (1.0 - index / 20) * second_mode
            for index in range(12)
        ],
        axis=0,
    )[..., None].astype(np.float32)
    method = create_method(
        "gappy_pod",
        rank=2,
        rank_candidates=(1, 2),
        oversampling=1,
        fit_batch_size=4,
        seed=7,
    ).fit_arrays(fields)
    mask = np.zeros((1, height, width, 1), dtype=bool)
    mask[:, ::2, ::2] = True
    observed = np.where(mask, fields[[3]], 0.0)

    prediction = method.predict(observed, mask)

    np.testing.assert_allclose(prediction, fields[[3]], atol=2.0e-4)
    checkpoint = method.save(tmp_path / "gappy.npz")
    restored = create_method("gappy_pod", state_path=checkpoint)
    np.testing.assert_allclose(restored.predict(observed, mask), prediction, atol=1.0e-7)
    assert restored.selected_rank == 2
    assert restored.capabilities.trainable


def test_rbf_rollout_reconstructs_only_t0_then_persists():
    observed, mask, _field = sparse_plane()
    observations = observed[None, None]
    visible = mask[None, None]
    method = create_method(
        "rbf_persistence",
        epsilon=2.0,
        max_centers=81,
        normalized_coordinates=True,
    )

    prediction = method.predict(observations, visible, horizon=3)

    assert prediction.shape == (1, 3, 9, 9, 1)
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(prediction[:, 0], prediction[:, 1])
    np.testing.assert_array_equal(prediction[:, 1], prediction[:, 2])
    np.testing.assert_allclose(prediction[:, 0][visible[:, 0]], observations[:, 0][visible[:, 0]], atol=2e-3)


def test_gappy_pod_dmd_uses_sparse_t0_and_free_rollout_and_round_trips(tmp_path):
    height = width = 8
    yy, xx = np.mgrid[:height, :width]
    modes = np.stack(
        (
            np.sin(2 * np.pi * xx / width),
            np.cos(2 * np.pi * yy / height),
        )
    )[..., None].astype(np.float32)
    transition = np.asarray((0.8, 0.6), dtype=np.float32)
    rows = []
    for amplitude in (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0):
        coefficients = np.asarray((amplitude, -0.4 * amplitude), dtype=np.float32)
        states = []
        for _step in range(4):
            states.append(np.tensordot(coefficients, modes, axes=(0, 0)))
            coefficients *= transition
        sequence = np.stack(states)
        rows.append({"initial_state": sequence[:1], "target": sequence[1:]})
    method = create_method(
        "gappy_pod_dmd",
        rank=2,
        ridge=0.0,
        oversampling=1,
        fit_batch_size=4,
        seed=7,
    )
    receipt = method.fit_dataset(rows)
    mask = np.zeros((1, 1, height, width, 1), dtype=bool)
    mask[:, :, ::2, ::2] = True
    observed = np.where(mask, rows[-1]["initial_state"][None], 0.0)

    prediction = method.predict(observed, mask, horizon=3)

    np.testing.assert_allclose(prediction[0], rows[-1]["target"], atol=3.0e-4)
    assert receipt["future_truth_at_inference"] is False
    assert receipt["consecutive_training_pairs"] == len(rows) * 3
    checkpoint = method.save(tmp_path / "gappy-dmd.npz")
    restored = create_method("gappy_pod_dmd", state_path=checkpoint)
    np.testing.assert_allclose(restored.predict(observed, mask, horizon=3), prediction, atol=1e-7)


def test_mean_fill_is_per_channel():
    values = np.zeros((4, 4, 2))
    mask = np.zeros((4, 4, 1), dtype=bool)
    mask[0, 0] = mask[-1, -1] = True
    values[0, 0] = (2, 10)
    values[-1, -1] = (4, 20)
    result = create_method("mean_fill").predict(values, mask)
    np.testing.assert_allclose(result[1, 1], (3, 15))


def test_persistence_rollout_shape():
    final_state = np.arange(16).reshape(4, 4, 1)
    result = create_method("persistence").predict(final_state, horizon=4)
    assert result.shape == (4, 4, 4, 1)
    np.testing.assert_array_equal(result[3], final_state)


def test_external_method_decorator(monkeypatch):
    # Keep this plugin-registration test isolated from later tests that compare
    # generated website options with the built-in method registry.
    monkeypatch.setattr(method_base, "METHOD_REGISTRY", dict(method_base.METHOD_REGISTRY))
    monkeypatch.setattr(method_base, "_PRIMARY_NAMES", set(method_base._PRIMARY_NAMES))

    @register_method("test_external_method", replace=True)
    class External:
        name = "test_external_method"
        capabilities = MethodCapabilities(tasks=frozenset({"recovery"}))

        def predict(self, observations, mask=None, **kwargs):
            return observations

    assert "test_external_method" in available_methods()
    assert isinstance(create_method("test-external-method"), External)


def test_formal_paper_adapters_are_explicit_but_not_publicly_enumerated():
    formal = {"paper_unet", "paper_fno", "paper_cno"}
    assert formal.issubset(method_base.METHOD_REGISTRY)
    assert formal.isdisjoint(available_methods())


def test_official_source_hash_uses_committed_blob(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(upstream), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init")
    git("config", "user.name", "PDE-OBS test")
    git("config", "user.email", "pdeobs-test@example.invalid")
    source = upstream / "source.py"
    canonical = b"alpha\nbeta\n"
    source.write_bytes(canonical)
    git("add", "source.py")
    git("commit", "-m", "source")
    revision = git("rev-parse", "HEAD")

    # A checkout filter may produce CRLF working-tree bytes while the committed
    # source is unchanged. Provenance must attest the exact Git blob.
    git("config", "core.autocrlf", "true")
    source.unlink()
    git("checkout", "HEAD", "--", "source.py")
    assert not git("status", "--porcelain")
    assert _git_blob_sha256(upstream, revision, "source.py") == hashlib.sha256(
        canonical
    ).hexdigest()


def test_compact_neural_shapes_and_reference_label():
    torch = pytest.importorskip("torch")
    inputs = torch.randn(2, 1, 17, 19)
    mask = torch.ones(2, 1, 17, 19)
    for name in ("unet", "fno", "ufno", "cno"):
        kwargs = {"width": 8}
        if name == "fno":
            kwargs.update(modes=3, layers=2)
        elif name == "ufno":
            kwargs.update(modes=3, padding=7)
        model = create_model(name, **kwargs)
        assert model(inputs, mask=mask).shape == inputs.shape
        assert model.capabilities.reference_only
        assert "not an exact" in model.capabilities.notes.lower()


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("unet_paper_guided", {"width": 4, "levels": 3}),
        (
            "deeponet",
            {
                "hidden": 16,
                "latent": 12,
                "branch_layers": 1,
                "trunk_layers": 1,
                "branch_mode": "spatial_cnn",
                "branch_grid": 2,
                "branch_width": 8,
            },
        ),
        ("transolver", {"hidden": 16, "layers": 2, "heads": 2, "slices": 4}),
        ("gnot", {"hidden": 16, "layers": 2, "heads": 1, "experts": 2, "mlp_layers": 1}),
    ],
)
def test_paper_guided_operator_adapters_shape_backward_and_context(name, kwargs):
    torch = pytest.importorskip("torch")
    observations = torch.randn(2, 1, 9, 7, requires_grad=True)
    mask = torch.ones(2, 1, 9, 7)
    mask[:, :, ::2, ::2] = 0
    geometry = torch.zeros(2, 1, 9, 7)
    geometry[:, :, 2:5, 3:6] = 1
    model = create_method(name, geometry_channels=1, **kwargs)

    prediction = model(observations, mask=mask, geometry=geometry)
    assert prediction.shape == observations.shape
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert model.capabilities.reference_only
    assert "not an exact" in model.capabilities.notes.lower()

    with torch.no_grad():
        changed_mask = model(observations, mask=torch.ones_like(mask), geometry=geometry)
        changed_geometry = model(observations, mask=mask, geometry=1.0 - geometry)
    assert not torch.allclose(prediction, changed_mask)
    assert not torch.allclose(prediction, changed_geometry)


@pytest.mark.parametrize("name", ["unet", "fno", "ufno", "cno"])
def test_compact_recovery_models_can_require_geometry(name):
    torch = pytest.importorskip("torch")
    kwargs = {"width": 8, "geometry_channels": 1}
    if name == "fno":
        kwargs.update(modes=3, layers=2)
    elif name == "ufno":
        kwargs.update(modes=3, padding=7)
    model = create_model(name, **kwargs)
    observations = torch.randn(2, 1, 9, 7)
    mask = torch.ones_like(observations)
    geometry = torch.zeros_like(observations)

    with pytest.raises(ValueError, match="geometry is required"):
        model(observations, mask=mask)
    prediction = model(observations, mask=mask, geometry=geometry)
    assert prediction.shape == observations.shape


def test_compact_fno_keeps_fft_outside_mixed_precision():
    torch = pytest.importorskip("torch")
    inputs = torch.randn(2, 1, 17, 19)
    mask = torch.ones_like(inputs)
    model = create_model("fno", width=8, modes=3, layers=2)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = model(inputs, mask=mask)

    assert result.shape == inputs.shape
    assert torch.isfinite(result).all()


def test_rollout_neural_shapes():
    torch = pytest.importorskip("torch")
    model = create_model("convlstm", hidden_channels=8)
    result = model(torch.randn(2, 2, 1, 8, 8), horizon=3)
    assert result.shape == (2, 3, 1, 8, 8)


def test_autoregressive_wrapper_never_uses_future_truth():
    torch = pytest.importorskip("torch")

    class Increment(torch.nn.Module):
        def forward(self, values, mask=None, geometry=None):
            del mask, geometry
            return values + 1.0

    model = create_model("autoregressive", one_step_model=Increment())
    initial = torch.zeros(2, 1, 1, 5, 5)
    deliberately_wrong_truth = torch.full((2, 3, 1, 5, 5), 100.0)
    prediction = model(
        initial,
        mask=torch.ones_like(initial),
        horizon=3,
        teacher_forcing=None,
        geometry=torch.zeros(2, 1, 5, 5),
    )
    torch.testing.assert_close(prediction[:, 0], torch.ones_like(prediction[:, 0]))
    torch.testing.assert_close(prediction[:, 1], 2.0 * torch.ones_like(prediction[:, 1]))
    torch.testing.assert_close(prediction[:, 2], 3.0 * torch.ones_like(prediction[:, 2]))
    assert not torch.allclose(prediction, deliberately_wrong_truth)


@pytest.mark.parametrize("base_name", ("deeponet", "gnot", "transolver"))
def test_runner_wraps_registered_operator_adapters_for_free_rollout(base_name):
    torch = pytest.importorskip("torch")
    from pdeobs.runner import _method

    kwargs = {
        "deeponet": {
            "hidden": 8,
            "latent": 8,
            "branch_layers": 1,
            "trunk_layers": 1,
        },
        "gnot": {
            "hidden": 8,
            "layers": 1,
            "heads": 1,
            "experts": 1,
            "mlp_layers": 1,
        },
        "transolver": {"hidden": 8, "layers": 1, "heads": 1, "slices": 2},
    }[base_name]
    model = _method(
        {
            "method": {
                "name": "autoregressive",
                "base": {
                    "name": base_name,
                    "kwargs": {
                        "in_channels": 1,
                        "out_channels": 1,
                        "geometry_channels": 1,
                        **kwargs,
                    },
                },
            }
        }
    )
    initial = torch.randn(1, 1, 1, 8, 8)
    mask = torch.ones_like(initial)
    geometry = torch.zeros(1, 1, 8, 8)
    output = model(initial, mask=mask, geometry=geometry, horizon=3)

    assert output.shape == (1, 3, 1, 8, 8)
    assert torch.isfinite(output).all()


def test_residual_encoder_multitask_shapes_and_registry():
    torch = pytest.importorskip("torch")
    inputs = torch.randn(3, 1, 17, 19)
    mask = torch.ones(3, 1, 17, 19)
    model = create_model(
        "resnet_encoder",
        width=4,
        embedding_dim=12,
        class_counts={"family": 7, "boundary": 4, "setting": 10, "regime": 3},
    )

    embedding = model(inputs, mask=mask)
    assert embedding.shape == (3, 12)
    torch.testing.assert_close(torch.linalg.vector_norm(embedding, dim=1), torch.ones(3))
    logits = model(inputs, mask=mask, output="logits")
    assert {name: tuple(value.shape) for name, value in logits.items()} == {
        "family": (3, 7),
        "boundary": (3, 4),
        "setting": (3, 10),
        "regime": (3, 3),
    }
    targets = {name: torch.zeros(3, dtype=torch.long) for name in logits}
    loss = model.supervised_loss(logits, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()

    discovered = create_method("supervised-multitask-small", width=4, embedding_dim=8)
    assert discovered.capabilities.supports("retrieval")
    assert discovered.capabilities.supports("supervised_multitask")
    assert discovered.capabilities.reference_only
    assert "not an exact" in discovered.capabilities.notes.lower()
    assert "residual_cnn" in available_methods()
    assert "residual_cnn" in PROJECT_METHOD_REGISTRY.names()


def test_mae_small_masks_reconstructs_and_preserves_visible_values():
    torch = pytest.importorskip("torch")
    inputs = torch.randn(2, 1, 17, 19)
    mask = torch.ones(2, 1, 17, 19)
    mask[:, :, :2] = 0
    model = create_method(
        "masked-autoencoder-small",
        width=4,
        latent_channels=8,
        patch_size=4,
        mask_ratio=0.5,
    )

    model.eval()
    reconstruction = model(inputs, mask=mask)
    assert reconstruction.shape == inputs.shape
    torch.testing.assert_close(reconstruction[mask.bool()], inputs[mask.bool()])
    assert model.embedding(inputs, mask=mask).shape == (2, 8)

    torch.manual_seed(7)
    model.train()
    masked, visible = model.mask_inputs(inputs, mask=mask, force_random=True)
    assert masked.shape == inputs.shape
    assert visible.shape == (2, 1, 17, 19)
    assert torch.all(visible <= mask)
    assert torch.count_nonzero(visible) < torch.count_nonzero(mask)
    assert torch.count_nonzero(masked * (1 - visible)) == 0
    training_reconstruction = model(inputs, mask=mask)
    assert training_reconstruction.shape == inputs.shape

    hidden_loss = model.reconstruction_loss(training_reconstruction, inputs, visible)
    assert hidden_loss.ndim == 0 and torch.isfinite(hidden_loss)
    assert model.capabilities.supports("pretraining")
    assert model.capabilities.supports("inverse")
    assert model.capabilities.reference_only
    assert "not a vit-mae" in model.capabilities.notes.lower()
    assert "mae_small" in available_methods()
    assert "mae_small" in PROJECT_METHOD_REGISTRY.names()
