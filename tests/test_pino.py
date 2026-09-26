from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FROZEN_CANDIDATE = (
    ROOT
    / "configs/campaign/full_to_partial_7x4x10_one_seed_v1_mixed_unet_lr3e4.candidate.jsonl"
)
FROZEN_CANDIDATE_SHA256 = (
    "397a35a1b52e086854fc1a294aea6ce510783f6182bae30a840b470344b7eb17"
)


def _periodic_poisson(batch_size: int = 1, size: int = 12):
    torch = pytest.importorskip("torch")
    coordinate = (torch.arange(size, dtype=torch.float32) + 0.5) / size
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    solution = torch.sin(2.0 * torch.pi * xx) * torch.sin(2.0 * torch.pi * yy)
    eigenvalue = 8.0 * torch.sin(torch.tensor(torch.pi / size)).square() * size**2
    source = eigenvalue * solution
    solution = solution[None, None].repeat(batch_size, 1, 1, 1)
    source = source[None, None].repeat(batch_size, 1, 1, 1)
    geometry = torch.zeros_like(solution)
    metadata = [
        {
            "pde": "poisson",
            "boundary": "periodic",
            "parameters": {"domain_id": "unit_square_cell_centered_v1"},
        }
        for _ in range(batch_size)
    ]
    return solution, source, geometry, metadata


def test_static_pino_residual_is_differentiable_and_equation_exact() -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.pino import static_pino_residual_loss

    solution, source, geometry, metadata = _periodic_poisson()
    prediction = solution.clone().requires_grad_(True)
    exact_loss = static_pino_residual_loss(prediction, source, geometry, metadata)
    exact_loss.backward()

    assert torch.isfinite(exact_loss)
    assert exact_loss.item() < 1.0e-8
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()

    perturbed = (solution + 0.05 * torch.randn_like(solution)).requires_grad_(True)
    perturbed_loss = static_pino_residual_loss(perturbed, source, geometry, metadata)
    assert perturbed_loss > exact_loss + 1.0e-4


@pytest.mark.parametrize("family", ("darcy", "poisson", "helmholtz"))
@pytest.mark.parametrize("boundary", ("dirichlet", "neumann", "periodic", "robin"))
def test_static_pino_residual_matches_registered_dataset_equations(
    family: str, boundary: str
) -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.pdes import generate_sample
    from pdeobs.pino import static_pino_residual_loss

    output = generate_sample(
        family,
        boundary=boundary,
        setting="smooth_grf",
        regime="medium",
        seed=47,
        resolution=12,
    )

    def bchw(array: np.ndarray) -> object:
        return torch.from_numpy(np.asarray(array)).permute(2, 0, 1)[None]

    prediction = bchw(output.trajectory[0]).requires_grad_(True)
    condition = bchw(output.condition)
    geometry = bchw(output.geometry)
    metadata = [
        {
            "pde": output.family,
            "boundary": output.boundary,
            "parameters": dict(output.parameters),
        }
    ]
    loss = static_pino_residual_loss(prediction, condition, geometry, metadata)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() < 1.0e-4
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_temporal_final_field_fails_closed() -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.pino import static_pino_residual_loss

    prediction = torch.zeros((1, 1, 8, 8))
    metadata = [
        {
            "pde": "heat",
            "boundary": "periodic",
            "parameters": {"domain_id": "unit_square_cell_centered_v1"},
        }
    ]
    with pytest.raises(ValueError, match="requires a predicted trajectory"):
        static_pino_residual_loss(prediction, prediction, prediction, metadata)


@pytest.mark.parametrize(
    "family", ("heat", "reaction_diffusion", "burgers", "navier_stokes")
)
def test_temporal_pino_residual_is_finite_and_differentiable(family: str) -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.pdes import generate_sample
    from pdeobs.pino import temporal_pino_residual_loss

    output = generate_sample(
        family,
        boundary="periodic",
        setting="smooth_grf",
        regime="medium",
        seed=47,
        resolution=12,
    )
    trajectory = torch.from_numpy(np.asarray(output.trajectory[:4])).permute(0, 3, 1, 2)
    prediction = trajectory[1:][None].clone().requires_grad_(True)
    geometry = torch.from_numpy(np.asarray(output.geometry)).permute(2, 0, 1)[None]
    stored_times = np.linspace(
        0.0,
        float(output.parameters["final_time"]),
        int(output.parameters["time_steps"]),
    ).tolist()
    metadata = [
        {
            "pde": output.family,
            "boundary": output.boundary,
            "parameters": dict(output.parameters),
            "stored_time_values": stored_times,
        }
    ]

    loss = temporal_pino_residual_loss(prediction, geometry, metadata)
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_heat_temporal_normalization_has_no_artificial_amplitude_gradient() -> None:
    """The homogeneous Heat residual must not push every prediction toward zero."""

    torch = pytest.importorskip("torch")
    from pdeobs.pino import temporal_pino_residual_loss

    generator = torch.Generator().manual_seed(47)
    prediction = (0.1 * torch.randn((1, 3, 1, 12, 12), generator=generator)).requires_grad_(
        True
    )
    geometry = torch.zeros((1, 1, 12, 12))
    metadata = [
        {
            "pde": "heat",
            "boundary": "periodic",
            "parameters": {"diffusivity": 0.05},
            "stored_time_values": [0.0, 1.0, 2.0, 3.0],
        }
    ]

    loss = temporal_pino_residual_loss(prediction, geometry, metadata)
    gradient = torch.autograd.grad(loss, prediction)[0]
    radial_derivative = torch.sum(gradient * prediction)

    scaled_loss = temporal_pino_residual_loss(
        2.5 * prediction.detach(), geometry, metadata
    )
    assert scaled_loss.item() == pytest.approx(loss.item(), rel=2.0e-5, abs=2.0e-7)
    assert abs(float(radial_derivative)) <= 2.0e-5 * max(abs(float(loss)), 1.0)


def test_temporal_pino_trainer_uses_free_three_step_rollout(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.methods import create_model
    from pdeobs.training import Trainer, TrainingConfig

    output = __import__("pdeobs.pdes", fromlist=["generate_sample"]).generate_sample(
        "heat",
        boundary="periodic",
        setting="smooth_grf",
        regime="medium",
        seed=47,
        resolution=8,
    )
    trajectory = torch.from_numpy(np.asarray(output.trajectory[:4])).permute(0, 3, 1, 2)
    geometry = torch.from_numpy(np.asarray(output.geometry)).permute(2, 0, 1)
    stored_times = np.linspace(
        0.0,
        float(output.parameters["final_time"]),
        int(output.parameters["time_steps"]),
    ).tolist()
    metadata = [
        {
            "pde": output.family,
            "boundary": output.boundary,
            "parameters": dict(output.parameters),
            "stored_time_values": stored_times,
        }
    ]
    loader = [
        {
            "observations": trajectory[:1][None],
            "mask": torch.ones_like(trajectory[:1][None]),
            "target": trajectory[1:4][None],
            "geometry": geometry[None],
            "metadata": metadata,
        }
    ]
    one_step = create_model(
        "pino",
        in_channels=1,
        out_channels=1,
        width=4,
        modes=2,
        layers=1,
        geometry_channels=1,
        physics_contract="pino_rollout_spectral_v1",
    )
    model = create_model("autoregressive", one_step_model=one_step)
    trainer = Trainer(
        model,
        TrainingConfig(
            task="rollout",
            epochs=1,
            horizon=3,
            history_steps=1,
            rollout_target_offset=0,
            teacher_forcing_ratio=0.0,
            loss="relative_l2",
            physics_loss="pino_rollout_spectral_v1",
            physics_loss_weight=1.0,
            amp=False,
            device="cpu",
            health_policy="record",
            checkpoint_dir=str(tmp_path),
        ),
    )
    history = trainer.fit(loader)

    assert np.isfinite(history[0]["train_data_loss"])
    assert np.isfinite(history[0]["train_physics_loss"])
    assert history[0]["train_physics_loss"] >= 0.0


def test_pino_model_and_trainer_contract_is_fail_closed(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.methods import create_model
    from pdeobs.training import Trainer, TrainingConfig

    model = create_model(
        "pino",
        in_channels=1,
        out_channels=1,
        width=4,
        modes=2,
        layers=1,
        geometry_channels=1,
    )
    with pytest.raises(ValueError, match="physics contract mismatch"):
        Trainer(model, TrainingConfig(amp=False, device="cpu"))

    solution, source, geometry, metadata = _periodic_poisson(batch_size=2, size=8)
    loader = [
        {
            "observations": solution.clone(),
            "mask": torch.ones_like(solution),
            "target": solution,
            "pde_condition": source,
            "geometry": geometry,
            "metadata": metadata,
        }
    ]
    trainer = Trainer(
        create_model(
            "pino",
            in_channels=1,
            out_channels=1,
            width=4,
            modes=2,
            layers=1,
            geometry_channels=1,
        ),
        TrainingConfig(
            epochs=1,
            loss="mse",
            physics_loss="pino_static_fd_v1",
            physics_loss_weight=1.0,
            amp=False,
            device="cpu",
            health_policy="record",
            checkpoint_dir=str(tmp_path),
        ),
    )
    history = trainer.fit(loader)

    assert np.isfinite(history[0]["train_data_loss"])
    assert np.isfinite(history[0]["train_physics_loss"])
    assert history[0]["train_physics_loss"] > 0.0
    assert history[0]["val_physics_loss"] == history[0]["train_physics_loss"]


