"""Differentiable physics residuals for the preregistered PINO adapter.

The executable contract is deliberately narrow: it supports the three
elliptic PDE families whose residual is identifiable from one predicted field
and the stored PDE condition.  It also implements a separate periodic,
two-dimensional free-rollout contract for the four temporal families.  The
two contracts are intentionally distinct so a final-field prediction can never
be mistaken for a trajectory-level PINO objective.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

try:  # PyTorch is optional for data-only installations.
    import torch
    import torch.nn.functional as F
    from torch import Tensor
except ImportError:  # pragma: no cover - exercised by minimal installations
    torch = None
    F = None
    Tensor = Any


PINO_STATIC_FAMILIES = ("darcy", "poisson", "helmholtz")
PINO_TEMPORAL_FAMILIES = ("heat", "reaction_diffusion", "burgers", "navier_stokes")
PINO_STATIC_RESIDUAL_CONTRACT = "pino_static_fd_v1"
PINO_TEMPORAL_RESIDUAL_CONTRACT = "pino_rollout_spectral_v1"


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PINO residual training requires PyTorch")


def _canonical_boundary(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "robin_obstacle": "robin",
        "mixed": "robin",
        "mixed_robin": "robin",
        "no_slip": "dirichlet",
        "free_slip": "neumann",
    }
    result = aliases.get(token, token)
    if result not in {"dirichlet", "neumann", "periodic", "robin"}:
        raise ValueError(f"PINO residual received unsupported boundary {value!r}")
    return result


def _metadata_rows(metadata: Any, batch_size: int) -> list[Mapping[str, Any]]:
    if isinstance(metadata, Mapping):
        if batch_size != 1:
            raise ValueError(
                "PINO requires one metadata mapping per batch item; "
                "use the PDE-OBS benchmark collator"
            )
        return [metadata]
    if isinstance(metadata, Sequence) and not isinstance(metadata, (str, bytes)):
        rows = list(metadata)
        if len(rows) != batch_size or not all(isinstance(row, Mapping) for row in rows):
            raise ValueError("PINO metadata must align one-to-one with the batch")
        return rows
    raise ValueError("PINO requires per-sample PDE metadata")


def _finite_parameter(
    parameters: Mapping[str, Any],
    name: str,
    *,
    positive: bool = False,
) -> float:
    try:
        value = float(parameters[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"PINO requires numeric metadata parameter {name!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"PINO metadata parameter {name!r} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"PINO metadata parameter {name!r} must be positive")
    return value


def _neighbors(values: Tensor, boundary: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if boundary == "periodic":
        return (
            torch.roll(values, -1, dims=-1),
            torch.roll(values, 1, dims=-1),
            torch.roll(values, -1, dims=-2),
            torch.roll(values, 1, dims=-2),
        )
    padded = F.pad(values[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    return (
        padded[1:-1, 2:],
        padded[1:-1, :-2],
        padded[2:, 1:-1],
        padded[:-2, 1:-1],
    )


def _laplacian(values: Tensor, boundary: str, dx: float, dy: float) -> Tensor:
    east, west, north, south = _neighbors(values, boundary)
    return (east - 2.0 * values + west) / dx**2 + (
        north - 2.0 * values + south
    ) / dy**2


def _variable_diffusion(
    values: Tensor,
    coefficient: Tensor,
    boundary: str,
    dx: float,
    dy: float,
) -> Tensor:
    east, west, north, south = _neighbors(values, boundary)
    ae, aw, an, ass = _neighbors(coefficient, boundary)
    ae = 0.5 * (coefficient + ae)
    aw = 0.5 * (coefficient + aw)
    an = 0.5 * (coefficient + an)
    ass = 0.5 * (coefficient + ass)
    return (ae * (east - values) - aw * (values - west)) / dx**2 + (
        an * (north - values) - ass * (values - south)
    ) / dy**2


def _active_stencil_mask(geometry: Tensor, boundary: str) -> Tensor:
    if geometry.ndim != 3 or geometry.shape[0] != 1:
        raise ValueError("PINO geometry must have shape 1HW")
    fluid = geometry[0] <= 0.5
    active = fluid.clone()
    if boundary != "periodic":
        active[[0, -1], :] = False
        active[:, [0, -1]] = False
    solid = ~fluid
    if bool(torch.any(solid)):
        halo = solid.clone()
        for dimension in (-2, -1):
            halo |= torch.roll(solid, 1, dims=dimension)
            halo |= torch.roll(solid, -1, dims=dimension)
        active &= ~halo
    if not bool(torch.any(active)):
        raise ValueError("PINO residual has no active stencil cells")
    return active


def _darcy_forcing(
    height: int,
    width: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    x = (torch.arange(width, dtype=dtype, device=device) + 0.5) / width
    y = (torch.arange(height, dtype=dtype, device=device) + 0.5) / height
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    forcing = torch.sin(2.0 * torch.pi * xx) * torch.sin(2.0 * torch.pi * yy)
    forcing = forcing + 0.2 * torch.sin(4.0 * torch.pi * xx + 0.3) * torch.sin(
        2.0 * torch.pi * yy
    )
    return forcing - forcing.mean()


def _normalized_residual_mse(
    residual: Tensor,
    components: Sequence[Tensor],
    active: Tensor,
    epsilon: float,
    *,
    differentiate_scale: bool = False,
) -> Tensor:
    selected = residual[active]
    scales = [torch.sqrt(torch.mean(component[active].square())) for component in components]
    scale = torch.stack(scales).sum()
    if not differentiate_scale:
        # Static residuals include a fixed source term. Detaching their scale
        # prevents a model from reducing the normalized objective by merely
        # inflating the learned operator term.
        scale = scale.detach()
    # A homogeneous temporal residual is different: all terms scale with the
    # predicted trajectory. Detaching that scale introduces a spurious radial
    # gradient (proportional to 1 / prediction amplitude) even though the
    # normalized residual value is scale invariant. Keeping the temporal scale
    # in the graph removes that artificial pressure toward the zero solution.
    scale = scale.clamp_min(float(epsilon))
    return torch.mean((selected / scale).square())


def static_pino_residual_loss(
    prediction: Tensor,
    condition: Tensor,
    geometry: Tensor,
    metadata: Any,
    *,
    epsilon: float = 1.0e-12,
) -> Tensor:
    """Return a dimensionless differentiable residual loss for elliptic rows.

    ``prediction``, ``condition``, and ``geometry`` use BCHW layout. The
    condition is used only by the declared training objective; it is not
    appended to the recovery model's test input.
    """

    _require_torch()
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("static PINO predictions must have shape B1HW")
    if condition.shape != prediction.shape:
        raise ValueError("static PINO condition and prediction shapes must match")
    if geometry.ndim != 4 or geometry.shape[0] != prediction.shape[0]:
        raise ValueError("static PINO geometry must have shape B1HW")
    if geometry.shape[1] != 1 or geometry.shape[-2:] != prediction.shape[-2:]:
        raise ValueError("static PINO geometry must have one aligned channel")
    if (
        not torch.isfinite(prediction).all()
        or not torch.isfinite(condition).all()
        or not torch.isfinite(geometry).all()
    ):
        raise FloatingPointError("PINO residual inputs contain non-finite values")
    if not math.isfinite(float(epsilon)) or epsilon <= 0.0:
        raise ValueError("PINO residual epsilon must be finite and positive")

    rows = _metadata_rows(metadata, prediction.shape[0])
    losses: list[Tensor] = []
    for index, row in enumerate(rows):
        family = str(row.get("pde", row.get("family", ""))).strip().lower().replace("-", "_")
        if family in PINO_TEMPORAL_FAMILIES:
            raise ValueError(
                f"{family} PINO requires a predicted trajectory and reviewed temporal residual; "
                "a recovered final field is scientifically insufficient"
            )
        if family not in PINO_STATIC_FAMILIES:
            raise ValueError(f"no PINO residual contract is registered for {family!r}")
        boundary = _canonical_boundary(row.get("boundary"))
        parameters = row.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("PINO metadata parameters must be a mapping")
        domain_id = parameters.get("domain_id")
        if domain_id not in {
            "unit_square_cell_centered_v1",
            "unit_square_node_centered_v1",
        }:
            raise ValueError("PINO requires a registered unit-square domain_id")

        solution = prediction[index, 0].float()
        pde_condition = condition[index, 0].float()
        sample_geometry = geometry[index].float()
        height, width = solution.shape
        if domain_id == "unit_square_node_centered_v1":
            dx, dy = 1.0 / (width - 1), 1.0 / (height - 1)
        else:
            dx, dy = 1.0 / width, 1.0 / height
        active = _active_stencil_mask(sample_geometry, boundary)

        if family == "poisson":
            operator = -_laplacian(solution, boundary, dx, dy)
            source = pde_condition
        elif family == "helmholtz":
            wavenumber = _finite_parameter(parameters, "wavenumber", positive=True)
            operator = -_laplacian(solution, boundary, dx, dy) - wavenumber**2 * solution
            source = pde_condition
        else:
            if parameters.get("forcing_id") != "unit_square_sine_mix_v1":
                raise ValueError("Darcy PINO requires forcing_id=unit_square_sine_mix_v1")
            if not bool(torch.all(pde_condition > 0.0)):
                raise ValueError("Darcy PINO coefficient must be strictly positive")
            amplitude = _finite_parameter(parameters, "forcing_amplitude")
            source = amplitude * _darcy_forcing(
                height,
                width,
                dtype=solution.dtype,
                device=solution.device,
            )
            operator = -_variable_diffusion(
                solution,
                pde_condition,
                boundary,
                dx,
                dy,
            )
        losses.append(
            _normalized_residual_mse(operator - source, (operator, source), active, epsilon)
        )
    return torch.stack(losses).mean()


def _periodic_spectral_terms(values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return x/y derivatives and Laplacian of one periodic 2-D field."""

    height, width = values.shape
    ky = 2.0 * torch.pi * torch.fft.fftfreq(
        height, d=1.0 / height, device=values.device
    )
    kx = 2.0 * torch.pi * torch.fft.fftfreq(
        width, d=1.0 / width, device=values.device
    )
    wave_x = kx[None, :]
    wave_y = ky[:, None]
    transformed = torch.fft.fft2(values.float())
    derivative_x = torch.fft.ifft2(1j * wave_x * transformed).real
    derivative_y = torch.fft.ifft2(1j * wave_y * transformed).real
    laplacian = torch.fft.ifft2(
        -(wave_x.square() + wave_y.square()) * transformed
    ).real
    return derivative_x, derivative_y, laplacian


def _stored_future_times(row: Mapping[str, Any], horizon: int) -> list[float]:
    raw = row.get("stored_time_values")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("temporal PINO requires stored_time_values metadata")
    values = [float(value) for value in raw]
    if len(values) < horizon + 1 or not all(math.isfinite(value) for value in values):
        raise ValueError("temporal PINO stored_time_values are missing or non-finite")
    future = values[1 : horizon + 1]
    if any(right <= left for left, right in zip(future, future[1:], strict=False)):
        raise ValueError("temporal PINO time values must be strictly increasing")
    return future


def _navier_stokes_velocity(vorticity: Tensor) -> tuple[Tensor, Tensor]:
    height, width = vorticity.shape
    ky = 2.0 * torch.pi * torch.fft.fftfreq(
        height, d=1.0 / height, device=vorticity.device
    )
    kx = 2.0 * torch.pi * torch.fft.fftfreq(
        width, d=1.0 / width, device=vorticity.device
    )
    wave_x = kx[None, :]
    wave_y = ky[:, None]
    wave2 = wave_x.square() + wave_y.square()
    safe_wave2 = wave2.clone()
    safe_wave2[0, 0] = 1.0
    omega_hat = torch.fft.fft2(vorticity.float())
    psi_hat = omega_hat / safe_wave2
    psi_hat = psi_hat.clone()
    psi_hat[0, 0] = 0.0
    velocity_x = torch.fft.ifft2(1j * wave_y * psi_hat).real
    velocity_y = torch.fft.ifft2(-1j * wave_x * psi_hat).real
    return velocity_x, velocity_y


def _periodic_ns_forcing(values: Tensor, amplitude: float) -> Tensor:
    height, width = values.shape
    x = torch.arange(width, dtype=values.dtype, device=values.device) / width
    y = torch.arange(height, dtype=values.dtype, device=values.device) / height
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    phase = 2.0 * torch.pi * (xx + yy)
    return float(amplitude) * (torch.sin(phase) + torch.cos(phase))


def temporal_pino_residual_loss(
    prediction: Tensor,
    geometry: Tensor,
    metadata: Any,
    *,
    epsilon: float = 1.0e-12,
) -> Tensor:
    """PINO residual for a free autoregressive 2-D periodic rollout.

    ``prediction`` has shape ``BT1HW`` and contains t1, t2, ... generated
    autoregressively from the sparsely observed t0.  Residuals are evaluated
    only between predicted states (t1->t2, t2->t3, ...), so hidden ground-truth
    states are never fed back to the model or to the PDE residual.
    """

    _require_torch()
    if prediction.ndim != 5 or prediction.shape[2] != 1:
        raise ValueError("temporal PINO predictions must have shape BT1HW")
    if prediction.shape[1] < 2:
        raise ValueError("temporal PINO requires at least two predicted future states")
    if geometry.ndim != 4 or geometry.shape[0] != prediction.shape[0]:
        raise ValueError("temporal PINO geometry must have shape B1HW")
    if geometry.shape[1] != 1 or geometry.shape[-2:] != prediction.shape[-2:]:
        raise ValueError("temporal PINO geometry must have one aligned channel")
    if not torch.isfinite(prediction).all() or not torch.isfinite(geometry).all():
        raise FloatingPointError("temporal PINO inputs contain non-finite values")
    if not math.isfinite(float(epsilon)) or epsilon <= 0.0:
        raise ValueError("PINO residual epsilon must be finite and positive")

    rows = _metadata_rows(metadata, prediction.shape[0])
    losses: list[Tensor] = []
    for batch_index, row in enumerate(rows):
        family = str(row.get("pde", row.get("family", ""))).strip().lower().replace("-", "_")
        if family not in PINO_TEMPORAL_FAMILIES:
            raise ValueError(f"temporal PINO does not support family {family!r}")
        if _canonical_boundary(row.get("boundary")) != "periodic":
            raise ValueError("pino_rollout_spectral_v1 is periodic-only")
        parameters = row.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("PINO metadata parameters must be a mapping")
        times = _stored_future_times(row, prediction.shape[1])
        active = _active_stencil_mask(geometry[batch_index].float(), "periodic")
        states = prediction[batch_index, :, 0].float()

        for step in range(states.shape[0] - 1):
            previous = states[step]
            following = states[step + 1]
            dt = times[step + 1] - times[step]
            time_derivative = (following - previous) / dt
            midpoint = 0.5 * (previous + following)
            derivative_x, derivative_y, laplacian = _periodic_spectral_terms(midpoint)

            if family == "heat":
                diffusion = _finite_parameter(parameters, "diffusivity", positive=True) * laplacian
                residual = time_derivative - diffusion
                components = (time_derivative, diffusion)
            elif family == "reaction_diffusion":
                diffusion = _finite_parameter(parameters, "diffusivity", positive=True) * laplacian
                reaction_rate = _finite_parameter(parameters, "reaction_rate", positive=True)
                reaction = reaction_rate * (midpoint - midpoint.pow(3))
                residual = time_derivative - diffusion - reaction
                components = (time_derivative, diffusion, reaction)
            elif family == "burgers":
                viscosity = _finite_parameter(parameters, "viscosity", positive=True)
                advection = midpoint * (derivative_x + derivative_y)
                diffusion = viscosity * laplacian
                residual = time_derivative + advection - diffusion
                components = (time_derivative, advection, diffusion)
            else:
                viscosity = _finite_parameter(parameters, "viscosity", positive=True)
                velocity_x, velocity_y = _navier_stokes_velocity(midpoint)
                advection = velocity_x * derivative_x + velocity_y * derivative_y
                diffusion = viscosity * laplacian
                forcing = _periodic_ns_forcing(
                    midpoint,
                    _finite_parameter(parameters, "forcing_amplitude"),
                )
                residual = time_derivative + advection - diffusion - forcing
                components = (time_derivative, advection, diffusion, forcing)
            losses.append(
                _normalized_residual_mse(
                    residual,
                    components,
                    active,
                    epsilon,
                    differentiate_scale=True,
                )
            )
    if not losses:
        raise ValueError("temporal PINO produced no inter-prediction residuals")
    return torch.stack(losses).mean()


__all__ = [
    "PINO_STATIC_FAMILIES",
    "PINO_STATIC_RESIDUAL_CONTRACT",
    "PINO_TEMPORAL_FAMILIES",
    "PINO_TEMPORAL_RESIDUAL_CONTRACT",
    "static_pino_residual_loss",
    "temporal_pino_residual_loss",
]
