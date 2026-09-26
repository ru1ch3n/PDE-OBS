"""Reproducible recovery and rollout training loops.

The adapter accepts ordinary PyTorch ``DataLoader`` batches, including batches
backed by lazily opened HDF5 datasets. No h5py objects are retained here, which
keeps multi-worker loading and batch jobs safe when the dataset opens files in
each worker process.
"""

from __future__ import annotations

import inspect
import json
import os
import random
import time
from collections import deque
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from .pino import static_pino_residual_loss, temporal_pino_residual_loss

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - optional for non-learning baselines
    torch = None
    Tensor = Any
    nn = None


@dataclass
class TrainingConfig:
    task: str = "recovery"
    epochs: int = 50
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "relative_l2"
    data_loss_weight: float = 1.0
    physics_loss: str = "none"
    physics_loss_weight: float = 0.0
    physics_loss_epsilon: float = 1.0e-12
    scheduler: str = "none"
    scheduler_step_size: int = 1
    scheduler_gamma: float = 1.0
    scheduler_steps_per_epoch: int | None = None
    scheduler_pct_start: float = 0.3
    scheduler_div_factor: float = 25.0
    scheduler_final_div_factor: float = 1.0e4
    scheduler_anneal_strategy: str = "cos"
    horizon: int = 1
    history_steps: int = 1
    rollout_target_offset: int = 1
    target_step: int = -1
    data_layout: str = "auto"
    device: str = "auto"
    amp: bool = True
    amp_initial_scale: float = 1.0
    grad_clip: float | None = 1.0
    seed: int = 0
    deterministic: bool = True
    checkpoint_dir: str = "runs/checkpoints"
    checkpoint_every: int = 1
    resume_from: str | None = None
    strict_resume: bool = True
    monitor: str = "val_loss"
    early_stopping_patience: int | None = None
    teacher_forcing_ratio: float = 0.0
    log_every: int = 20
    health_policy: str = "record"
    loss_spike_ratio: float = 10.0
    loss_spike_consecutive_windows: int = 3
    loss_spike_scope: str = "batch"
    gradient_norm_warning: float = 1000.0
    gradient_warning_consecutive_windows: int = 3
    gradient_warning_fatal: bool = True
    output_std_ratio_collapse_max: float = 1.0e-4
    collapse_consecutive_windows: int = 100
    validation_improvement_min_fraction: float | None = None

    def __post_init__(self) -> None:
        self.task = self.task.lower().replace("-", "_")
        if self.task not in {"recovery", "forward", "inverse", "rollout"}:
            raise ValueError("task must be recovery, forward, inverse, or rollout")
        if self.epochs < 1 or self.horizon < 1 or self.history_steps < 1:
            raise ValueError("epochs, horizon, and history_steps must be positive")
        self.optimizer = self.optimizer.lower().replace("-", "_")
        if self.optimizer not in {"adam", "adamw"}:
            raise ValueError("optimizer must be adam or adamw")
        for name, value in (
            ("data_loss_weight", self.data_loss_weight),
            ("physics_loss_weight", self.physics_loss_weight),
            ("physics_loss_epsilon", self.physics_loss_epsilon),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.data_loss_weight <= 0.0:
            raise ValueError("data_loss_weight must be positive")
        if self.physics_loss_weight < 0.0:
            raise ValueError("physics_loss_weight must be non-negative")
        if self.physics_loss_epsilon <= 0.0:
            raise ValueError("physics_loss_epsilon must be positive")
        self.physics_loss = str(self.physics_loss).strip().lower().replace("-", "_")
        if self.physics_loss not in {
            "none",
            "pino_static_fd_v1",
            "pino_rollout_spectral_v1",
        }:
            raise ValueError(
                "physics_loss must be none, pino_static_fd_v1, or "
                "pino_rollout_spectral_v1"
            )
        if self.physics_loss == "none" and self.physics_loss_weight != 0.0:
            raise ValueError("physics_loss_weight must be zero when physics_loss is none")
        if self.physics_loss != "none":
            if self.physics_loss_weight <= 0.0:
                raise ValueError("PINO physics_loss_weight must be positive")
            if self.physics_loss == "pino_static_fd_v1" and self.task not in {
                "recovery",
                "forward",
            }:
                raise ValueError(
                    "pino_static_fd_v1 requires an elliptic recovery/forward task"
                )
            if self.physics_loss == "pino_rollout_spectral_v1" and self.task != "rollout":
                raise ValueError("pino_rollout_spectral_v1 requires task=rollout")
            if self.amp:
                raise ValueError("PINO residual training requires amp=false")
        self.scheduler = self.scheduler.lower().replace("-", "_")
        if self.scheduler not in {"none", "step", "one_cycle"}:
            raise ValueError("scheduler must be none, step, or one_cycle")
        if self.scheduler_step_size < 1:
            raise ValueError("scheduler_step_size must be positive")
        if not 0.0 < self.scheduler_gamma <= 1.0:
            raise ValueError("scheduler_gamma must be in (0, 1]")
        if self.scheduler == "one_cycle" and (
            self.scheduler_steps_per_epoch is None or self.scheduler_steps_per_epoch < 1
        ):
            raise ValueError(
                "scheduler_steps_per_epoch must be positive for one_cycle"
            )
        if not 0.0 < self.scheduler_pct_start < 1.0:
            raise ValueError("scheduler_pct_start must lie in (0, 1)")
        if self.scheduler_div_factor <= 0.0:
            raise ValueError("scheduler_div_factor must be positive")
        if self.scheduler_final_div_factor <= 0.0:
            raise ValueError("scheduler_final_div_factor must be positive")
        self.scheduler_anneal_strategy = (
            str(self.scheduler_anneal_strategy).strip().lower()
        )
        if self.scheduler_anneal_strategy not in {"cos", "linear"}:
            raise ValueError("scheduler_anneal_strategy must be cos or linear")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        if self.rollout_target_offset < 0:
            raise ValueError("rollout_target_offset must be non-negative")
        if self.data_layout not in {"auto", "channels_first", "channels_last"}:
            raise ValueError("data_layout must be auto, channels_first, or channels_last")
        if not np.isfinite(self.amp_initial_scale) or self.amp_initial_scale <= 0.0:
            raise ValueError("amp_initial_scale must be finite and positive")
        if not 0.0 <= self.teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be between zero and one")
        self.health_policy = str(self.health_policy).strip().lower().replace("-", "_")
        if self.health_policy not in {"off", "record", "fail"}:
            raise ValueError("health_policy must be off, record, or fail")
        if self.loss_spike_ratio <= 1.0:
            raise ValueError("loss_spike_ratio must be greater than one")
        if self.loss_spike_consecutive_windows < 1:
            raise ValueError("loss_spike_consecutive_windows must be positive")
        self.loss_spike_scope = str(self.loss_spike_scope).strip().lower().replace("-", "_")
        if self.loss_spike_scope not in {"batch", "epoch"}:
            raise ValueError("loss_spike_scope must be batch or epoch")
        if self.gradient_norm_warning <= 0.0:
            raise ValueError("gradient_norm_warning must be positive")
        if self.gradient_warning_consecutive_windows < 1:
            raise ValueError("gradient_warning_consecutive_windows must be positive")
        if not 0.0 <= self.output_std_ratio_collapse_max < 1.0:
            raise ValueError("output_std_ratio_collapse_max must lie in [0, 1)")
        if self.collapse_consecutive_windows < 1:
            raise ValueError("collapse_consecutive_windows must be positive")
        if (
            self.validation_improvement_min_fraction is not None
            and not 0.0 <= self.validation_improvement_min_fraction <= 1.0
        ):
            raise ValueError("validation_improvement_min_fraction must lie in [0, 1]")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> TrainingConfig:
        """Create a config while rejecting misspelled keys early."""

        known = {item.name for item in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown training configuration keys: {', '.join(sorted(unknown))}")
        return cls(**dict(values))


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "Training neural baselines requires PyTorch. Install pdeobs[torch] or torch."
        )


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:  # older PyTorch
                torch.use_deterministic_algorithms(True)


def _numpy_rng_payload(state: tuple[Any, ...]) -> dict[str, Any]:
    """Convert NumPy's RNG tuple to weights-only-safe primitive values."""

    return {
        "bit_generator": str(state[0]),
        "keys": np.asarray(state[1], dtype=np.uint32).tolist(),
        "position": int(state[2]),
        "has_gaussian": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _restore_numpy_rng(payload: Mapping[str, Any]) -> None:
    np.random.set_state(
        (
            str(payload["bit_generator"]),
            np.asarray(payload["keys"], dtype=np.uint32),
            int(payload["position"]),
            int(payload["has_gaussian"]),
            float(payload["cached_gaussian"]),
        )
    )


def load_checkpoint_payload(path: str | Path, *, map_location: Any = "cpu") -> Mapping[str, Any]:
    """Load tensor/state dictionaries without enabling arbitrary pickle execution."""

    _require_torch()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Checkpoint {source} is not compatible with safe weights-only loading. "
            "PDE-OBS refuses arbitrary-pickle checkpoints; recreate it with this version "
            "or export a plain state_dict."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint {source} must contain a state mapping")
    return payload


def resolve_device(requested: str = "auto") -> torch.device:
    _require_torch()
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and device.index is None:
            local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
            return torch.device(f"cuda:{local_rank % max(torch.cuda.device_count(), 1)}")
        return device
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        return torch.device(f"cuda:{local_rank % max(torch.cuda.device_count(), 1)}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_tensor(
    value: Any, device: torch.device, layout: str, *, is_mask: bool = False
) -> Tensor | None:
    if value is None:
        return None
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
    if tensor.ndim == 2:  # an unbatched scalar field
        tensor = tensor[None, None]
    elif tensor.ndim == 3:  # B,H,W
        tensor = tensor[:, None]
    elif tensor.ndim == 4:
        channel_last = layout == "channels_last" or (
            layout == "auto" and tensor.shape[-1] <= 8 and tensor.shape[-1] < tensor.shape[1]
        )
        if channel_last:  # B,H,W,C -> B,C,H,W
            tensor = tensor.permute(0, 3, 1, 2)
    elif tensor.ndim == 5:
        channel_last = layout == "channels_last" or (
            layout == "auto" and tensor.shape[-1] <= 8 and tensor.shape[-1] < tensor.shape[2]
        )
        if channel_last:  # B,T,H,W,C -> B,T,C,H,W
            tensor = tensor.permute(0, 1, 4, 2, 3)
    if is_mask:
        tensor = (tensor > 0).to(dtype=torch.float32)
    return tensor.contiguous()


_INPUT_KEYS = ("observations", "observation", "observed", "y", "input", "inputs", "condition")
_MASK_KEYS = ("mask", "observation_mask", "sensor_mask")
_TARGET_KEYS = ("target", "targets", "trajectory", "solution", "state", "field")
_PDE_CONDITION_KEYS = ("pde_condition", "physics_condition")


def _first(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any | None:
    return next((mapping[key] for key in keys if key in mapping), None)


def unpack_batch_context(
    batch: Any,
) -> tuple[Any | None, Any | None, Any, Any | None, Any | None]:
    """Return input, mask, target, geometry, and metadata without dropping context."""

    metadata = None
    if isinstance(batch, Mapping):
        inputs = _first(batch, _INPUT_KEYS)
        mask = _first(batch, _MASK_KEYS)
        target = _first(batch, _TARGET_KEYS)
        metadata = batch.get("metadata")
        geometry = batch.get("geometry")
        if target is None:
            raise KeyError(f"Batch needs one target key from {_TARGET_KEYS}")
        return inputs, mask, target, geometry, metadata
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            return batch[0], None, batch[1], None, None
        if len(batch) == 3:
            return batch[0], batch[1], batch[2], None, None
        if len(batch) == 4:
            return batch[0], batch[1], batch[2], None, batch[3]
        if len(batch) == 5:
            return batch[0], batch[1], batch[2], batch[3], batch[4]
    # Canonical LazyHDF5Dataset items are Sample dataclasses. This branch avoids
    # importing schema.py and keeps the training layer usable by external data
    # packages exposing the same attributes.
    if hasattr(batch, "condition") and hasattr(batch, "trajectory"):
        return (
            batch.condition,
            None,
            batch.trajectory,
            getattr(batch, "geometry", None),
            getattr(batch, "metadata", None),
        )
    raise TypeError("Batch must be a mapping or a 2-5 item tuple/list")


def unpack_batch(batch: Any) -> tuple[Any | None, Any | None, Any, Any | None]:
    """Backward-compatible input, mask, target, metadata batch adapter."""

    inputs, mask, target, _, metadata = unpack_batch_context(batch)
    return inputs, mask, target, metadata


def collate_samples(samples: Iterable[Any]) -> dict[str, Any]:
    """Collate canonical HDF5 ``Sample`` objects for a PyTorch DataLoader.

    Use ``DataLoader(dataset, collate_fn=collate_samples)``. Metadata remains a
    list of dictionaries; arrays are stacked without opening or retaining HDF5
    handles in worker processes.
    """

    rows = list(samples)
    if not rows:
        raise ValueError("cannot collate an empty sample list")
    if not all(hasattr(row, "condition") and hasattr(row, "trajectory") for row in rows):
        raise TypeError("collate_samples expects canonical Sample-like objects")
    return {
        "condition": np.stack([np.asarray(row.condition) for row in rows]),
        "trajectory": np.stack([np.asarray(row.trajectory) for row in rows]),
        "geometry": np.stack([np.asarray(row.geometry) for row in rows]),
        "metadata": [getattr(row, "metadata", {}) for row in rows],
    }


def prepare_batch_with_context(
    batch: Any, config: TrainingConfig, device: torch.device
) -> tuple[Tensor, Tensor | None, Tensor, Tensor | None, Any | None]:
    raw_input, raw_mask, raw_target, raw_geometry, metadata = unpack_batch_context(batch)
    target = _to_tensor(raw_target, device, config.data_layout)
    mask = _to_tensor(raw_mask, device, config.data_layout, is_mask=True)
    inputs = _to_tensor(raw_input, device, config.data_layout)
    geometry = _to_tensor(raw_geometry, device, config.data_layout)
    assert target is not None

    if config.task == "rollout":
        if target.ndim != 5:
            raise ValueError("rollout targets must have shape BTCHW (or BTHWC before adaptation)")
        if inputs is None:
            required_steps = config.history_steps + config.horizon
            if target.shape[1] < required_steps:
                raise ValueError(
                    f"rollout trajectory has {target.shape[1]} steps, but requires "
                    f"{config.history_steps} history + {config.horizon} future steps"
                )
            history = config.history_steps
            horizon = config.horizon
            inputs = target[:, :history]
            target = target[:, history : history + horizon]
            if mask is not None and mask.ndim == 5:
                mask = mask[:, :history]
        else:
            # Canonical HDF5 trajectories include t0, while condition stores the
            # initial state. Set rollout_target_offset=0 for loaders whose target
            # tensor already begins at the first future state.
            start = config.rollout_target_offset
            available = target.shape[1] - start
            if available < config.horizon:
                raise ValueError(
                    f"rollout target has {max(0, available)} steps after offset {start}, "
                    f"but horizon {config.horizon} was requested"
                )
            horizon = config.horizon
            target = target[:, start : start + horizon]
    else:
        if target.ndim == 5:
            target = target[:, config.target_step]
        if inputs is None:
            inputs = target
            if mask is not None:
                inputs = inputs * mask
    assert inputs is not None
    return inputs, mask, target, geometry, metadata


def prepare_batch(
    batch: Any, config: TrainingConfig, device: torch.device
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Prepare the three legacy tensors; use prepare_batch_with_context for plugins."""

    inputs, mask, target, _, _ = prepare_batch_with_context(batch, config, device)
    return inputs, mask, target


def prepare_pde_condition(
    batch: Any, config: TrainingConfig, device: torch.device
) -> Tensor | None:
    """Return the declared PDE condition without exposing it to the model."""

    if config.physics_loss != "pino_static_fd_v1":
        return None
    if not isinstance(batch, Mapping):
        raise ValueError("PINO training batches must include a pde_condition mapping field")
    raw_condition = _first(batch, _PDE_CONDITION_KEYS)
    if raw_condition is None:
        raise ValueError("PINO training requires the stored pde_condition")
    condition = _to_tensor(raw_condition, device, config.data_layout)
    if condition is None:
        raise ValueError("PINO training requires the stored pde_condition")
    return condition


def _loss_function(name: str):
    key = name.lower().replace("-", "_")
    if key == "mse":
        return lambda prediction, target: torch.mean((prediction - target) ** 2)
    if key == "mae":
        return lambda prediction, target: torch.mean(torch.abs(prediction - target))
    if key in {"relative_l1", "rel_l1"}:

        def relative_l1(prediction: Tensor, target: Tensor) -> Tensor:
            # Keep global reductions outside float16.  With 128x128 fields the
            # default AMP loss scale can otherwise overflow otherwise-finite
            # first-step gradients before GradScaler has calibrated itself.
            prediction_float = prediction.float()
            target_float = target.float()
            difference = torch.abs(prediction_float - target_float).reshape(
                prediction.shape[0], -1
            )
            reference = torch.abs(target_float).reshape(target.shape[0], -1)
            return torch.mean(
                torch.sum(difference, dim=1)
                / torch.clamp(torch.sum(reference, dim=1), min=1e-12)
            )

        return relative_l1
    if key in {"relative_l2", "rel_l2"}:

        def relative(prediction: Tensor, target: Tensor) -> Tensor:
            prediction_float = prediction.float()
            target_float = target.float()
            difference = (prediction_float - target_float).reshape(
                prediction.shape[0], -1
            )
            reference = target_float.reshape(target.shape[0], -1)
            return torch.mean(
                torch.linalg.vector_norm(difference, dim=1)
                / torch.clamp(torch.linalg.vector_norm(reference, dim=1), min=1e-12)
            )

        return relative
    raise ValueError("loss must be mse, mae, relative_l1, or relative_l2")


def _forward(
    model: nn.Module,
    inputs: Tensor,
    mask: Tensor | None,
    target: Tensor,
    config: TrainingConfig,
    training: bool,
    geometry: Tensor | None = None,
    metadata: Any | None = None,
    *,
    horizon: int | None = None,
) -> Tensor:
    """Invoke the model; ``target`` may be None for target-free inference (then ``horizon`` sets rollout length)."""

    def invoke(**kwargs: Any) -> Tensor:
        callable_target = model.forward if hasattr(model, "forward") else model
        try:
            signature = inspect.signature(callable_target)
        except (TypeError, ValueError):
            return model(inputs, **kwargs)
        accepts_extra = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        context = {"geometry": geometry, "metadata": metadata}
        for name, value in context.items():
            if value is not None and (accepts_extra or name in signature.parameters):
                kwargs[name] = value
        filtered = (
            kwargs
            if accepts_extra
            else {name: value for name, value in kwargs.items() if name in signature.parameters}
        )
        return model(inputs, **filtered)

    if config.task == "rollout":
        if target is None:
            if horizon is None or int(horizon) < 1:
                raise ValueError("target-free rollout inference requires an explicit positive horizon")
            steps = int(horizon)
        else:
            steps = int(target.shape[1])
        teacher = (
            target if training and target is not None and random.random() < config.teacher_forcing_ratio else None
        )
        return invoke(mask=mask, horizon=steps, teacher_forcing=teacher)
    return invoke(mask=mask)


class Trainer:
    """Stateful trainer with portable checkpoints and resume support."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | Mapping[str, Any],
        optimizer: Any | None = None,
    ) -> None:
        _require_torch()
        self.config = (
            config if isinstance(config, TrainingConfig) else TrainingConfig.from_mapping(config)
        )
        seed_everything(self.config.seed, self.config.deterministic)
        self.device = resolve_device(self.config.device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.model = model.to(self.device)
        model_physics_loss = str(
            getattr(self.model, "required_physics_loss", "none")
        ).strip().lower()
        if model_physics_loss != self.config.physics_loss:
            raise ValueError(
                "model/trainer physics contract mismatch: "
                f"model requires {model_physics_loss!r}, but training config declares "
                f"{self.config.physics_loss!r}"
            )
        optimizer_types = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }
        self.optimizer = optimizer or optimizer_types[self.config.optimizer](
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        if self.config.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
        elif self.config.scheduler == "one_cycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=int(self.config.scheduler_steps_per_epoch or 0),
                pct_start=self.config.scheduler_pct_start,
                div_factor=self.config.scheduler_div_factor,
                final_div_factor=self.config.scheduler_final_div_factor,
                anneal_strategy=self.config.scheduler_anneal_strategy,
            )
        else:
            self.scheduler = None
        self._scheduler_steps_per_batch = self.config.scheduler == "one_cycle"
        self.loss_fn = _loss_function(self.config.loss)
        scaler_enabled = self.config.amp and self.device.type == "cuda"
        self.scaler = (
            torch.amp.GradScaler(
                "cuda",
                enabled=scaler_enabled,
                init_scale=self.config.amp_initial_scale,
            )
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
            else torch.cuda.amp.GradScaler(
                enabled=scaler_enabled,
                init_scale=self.config.amp_initial_scale,
            )
        )
        self.start_epoch = 0
        self.best_metric = float("inf")
        self.stale_epochs = 0
        self.history: list[dict[str, float | int]] = []
        self.health_events: list[dict[str, Any]] = []
        self.health_warnings: list[dict[str, Any]] = []
        self.last_epoch_health: dict[str, float | int] = {}
        self._loss_window: deque[float] = deque(maxlen=20)
        self._loss_spike_windows = 0
        self._gradient_warning_windows = 0
        self._collapse_windows = 0
        self._optimizer_steps = 0
        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)

    @property
    def is_primary(self) -> bool:
        return int(os.environ.get("RANK", "0")) == 0

    def _autocast(self):
        if self.device.type == "cuda":
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                return torch.amp.autocast("cuda", enabled=self.config.amp)
            return torch.cuda.amp.autocast(enabled=self.config.amp)
        return nullcontext()

    def _record_health_event(self, kind: str, **values: Any) -> None:
        if self.config.health_policy == "off":
            return
        event = {"kind": kind, "optimizer_step": int(self._optimizer_steps), **values}
        self.health_events.append(event)
        if self.config.health_policy == "fail":
            raise RuntimeError(f"training pathology: {kind}: {values}")

    def _record_health_warning(self, kind: str, **values: Any) -> None:
        if self.config.health_policy == "off":
            return
        self.health_warnings.append(
            {"kind": kind, "optimizer_step": int(self._optimizer_steps), **values}
        )

    def _check_loss_spike(self, loss_value: float, **context: Any) -> None:
        if self._loss_window:
            trailing_median = float(np.median(tuple(self._loss_window)))
            is_spike = (
                trailing_median > 0.0
                and loss_value > self.config.loss_spike_ratio * trailing_median
            )
            self._loss_spike_windows = self._loss_spike_windows + 1 if is_spike else 0
            if self._loss_spike_windows >= self.config.loss_spike_consecutive_windows:
                self._loss_spike_windows = 0
                self._record_health_event(
                    "sustained_loss_spike",
                    loss=loss_value,
                    trailing_median=trailing_median,
                    ratio=loss_value / trailing_median,
                    **context,
                )
        self._loss_window.append(loss_value)

    @staticmethod
    def _tensor_norm_and_finiteness(tensors: Iterable[Tensor]) -> tuple[float, int]:
        squared = 0.0
        nonfinite = 0
        for tensor in tensors:
            detached = tensor.detach()
            invalid = int(torch.count_nonzero(~torch.isfinite(detached)).item())
            nonfinite += invalid
            if invalid == 0:
                magnitude = torch.abs(detached) if torch.is_complex(detached) else detached
                squared += float(torch.sum(magnitude.float().square()).item())
        return float(np.sqrt(squared)), nonfinite

    def run_epoch(self, loader: Iterable[Any], *, training: bool) -> float:
        self.model.train(training)
        total, total_data, total_physics, count = 0.0, 0.0, 0.0, 0
        started = time.perf_counter()
        preclip_norms: list[float] = []
        postclip_norms: list[float] = []
        amp_scales: list[float] = []
        output_std_ratios: list[float] = []
        gradient_warning_steps = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        for batch in loader:
            inputs, mask, target, geometry, metadata = prepare_batch_with_context(
                batch, self.config, self.device
            )
            pde_condition = prepare_pde_condition(batch, self.config, self.device)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training), self._autocast():
                prediction = _forward(
                    self.model,
                    inputs,
                    mask,
                    target,
                    self.config,
                    training,
                    geometry,
                    metadata,
                )
                if prediction.shape != target.shape:
                    raise ValueError(
                        f"Model output {tuple(prediction.shape)} does not match target {tuple(target.shape)}"
                    )
                data_loss = self.loss_fn(prediction, target)
            physics_loss = prediction.new_zeros(())
            if self.config.physics_loss == "pino_static_fd_v1":
                if pde_condition is None or geometry is None:
                    raise ValueError(
                        "PINO training requires aligned pde_condition and geometry tensors"
                    )
                physics_loss = static_pino_residual_loss(
                    prediction,
                    pde_condition,
                    geometry,
                    metadata,
                    epsilon=self.config.physics_loss_epsilon,
                )
            elif self.config.physics_loss == "pino_rollout_spectral_v1":
                if geometry is None:
                    raise ValueError("temporal PINO training requires aligned geometry")
                physics_loss = temporal_pino_residual_loss(
                    prediction,
                    geometry,
                    metadata,
                    epsilon=self.config.physics_loss_epsilon,
                )
            if not torch.isfinite(data_loss):
                raise FloatingPointError(
                    f"Non-finite data loss encountered: {data_loss.item()}"
                )
            if not torch.isfinite(physics_loss):
                raise FloatingPointError(
                    f"Non-finite physics loss encountered: {physics_loss.item()}"
                )
            loss = (
                self.config.data_loss_weight * data_loss
                + self.config.physics_loss_weight * physics_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss encountered: {loss.item()}")
            if training:
                loss_value = float(loss.detach())
                if self.config.loss_spike_scope == "batch":
                    self._check_loss_spike(loss_value)

                target_std = float(target.detach().float().std(unbiased=False).item())
                prediction_std = float(
                    prediction.detach().float().std(unbiased=False).item()
                )
                if target_std > 1.0e-12:
                    std_ratio = prediction_std / target_std
                    output_std_ratios.append(std_ratio)
                    collapsed = std_ratio <= self.config.output_std_ratio_collapse_max
                    self._collapse_windows = self._collapse_windows + 1 if collapsed else 0
                    if self._collapse_windows >= self.config.collapse_consecutive_windows:
                        self._collapse_windows = 0
                        self._record_health_event(
                            "constant_output_collapse",
                            output_std_ratio=std_ratio,
                        )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                gradients = [
                    parameter.grad
                    for parameter in self.model.parameters()
                    if parameter.grad is not None
                ]
                preclip_norm, nonfinite_gradients = self._tensor_norm_and_finiteness(gradients)
                if nonfinite_gradients:
                    raise FloatingPointError(
                        f"Non-finite gradients encountered: {nonfinite_gradients} values"
                    )
                preclip_norms.append(preclip_norm)
                above_warning = preclip_norm > self.config.gradient_norm_warning
                gradient_warning_steps += int(above_warning)
                self._gradient_warning_windows = (
                    self._gradient_warning_windows + 1 if above_warning else 0
                )
                if (
                    self._gradient_warning_windows
                    >= self.config.gradient_warning_consecutive_windows
                ):
                    self._gradient_warning_windows = 0
                    recorder = (
                        self._record_health_event
                        if self.config.gradient_warning_fatal
                        else self._record_health_warning
                    )
                    recorder(
                        "sustained_large_gradient",
                        preclip_gradient_norm=preclip_norm,
                        warning_threshold=self.config.gradient_norm_warning,
                        postclip_guard=(
                            None
                            if self.config.grad_clip is None
                            else float(self.config.grad_clip)
                        ),
                    )
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                postclip_norm, nonfinite_postclip = self._tensor_norm_and_finiteness(gradients)
                if nonfinite_postclip:
                    raise FloatingPointError(
                        f"Non-finite post-clip gradients encountered: {nonfinite_postclip} values"
                    )
                postclip_norms.append(postclip_norm)
                amp_scales.append(float(self.scaler.get_scale()))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self._optimizer_steps += 1
                if self.scheduler is not None and self._scheduler_steps_per_batch:
                    self.scheduler.step()
                _, nonfinite_parameters = self._tensor_norm_and_finiteness(
                    parameter.data for parameter in self.model.parameters()
                )
                if nonfinite_parameters:
                    raise FloatingPointError(
                        f"Non-finite parameters encountered: {nonfinite_parameters} values"
                    )
            batch_size = target.shape[0]
            total += float(loss.detach()) * batch_size
            total_data += float(data_loss.detach()) * batch_size
            total_physics += float(physics_loss.detach()) * batch_size
            count += batch_size
        if not count:
            raise ValueError("Data loader produced no batches")
        elapsed = max(time.perf_counter() - started, np.finfo(float).eps)
        peak_memory = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )
        self.last_epoch_health = {
            "samples": int(count),
            "data_loss": total_data / count,
            "physics_loss": total_physics / count,
            "seconds": float(elapsed),
            "throughput_samples_per_second": float(count / elapsed),
            "peak_gpu_memory_bytes": peak_memory,
            "preclip_gradient_norm_max": max(preclip_norms, default=0.0),
            "preclip_gradient_norm_mean": (
                float(np.mean(preclip_norms)) if preclip_norms else 0.0
            ),
            "postclip_gradient_norm_max": max(postclip_norms, default=0.0),
            "amp_scale_min": min(amp_scales, default=1.0),
            "amp_scale_max": max(amp_scales, default=1.0),
            "output_std_ratio_min": min(output_std_ratios, default=1.0),
            "gradient_warning_steps": int(gradient_warning_steps),
        }
        return total / count

    def fit(
        self, train_loader: Iterable[Any], val_loader: Iterable[Any] | None = None
    ) -> list[dict[str, float | int]]:
        for epoch in range(self.start_epoch, self.config.epochs):
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            train_loss = self.run_epoch(train_loader, training=True)
            if self.config.loss_spike_scope == "epoch":
                self._check_loss_spike(train_loss, epoch=epoch + 1)
            train_health = dict(self.last_epoch_health)
            val_loss = (
                self.run_epoch(val_loader, training=False) if val_loader is not None else train_loss
            )
            validation_health = dict(self.last_epoch_health) if val_loader is not None else {}
            row: dict[str, float | int] = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_data_loss": train_health["data_loss"],
                "train_physics_loss": train_health["physics_loss"],
                "val_data_loss": validation_health.get(
                    "data_loss", train_health["data_loss"]
                ),
                "val_physics_loss": validation_health.get(
                    "physics_loss", train_health["physics_loss"]
                ),
                "learning_rate": learning_rate,
                "learning_rate_end": float(self.optimizer.param_groups[0]["lr"]),
                "throughput_samples_per_second": train_health[
                    "throughput_samples_per_second"
                ],
                "peak_gpu_memory_bytes": train_health["peak_gpu_memory_bytes"],
                "preclip_gradient_norm_max": train_health["preclip_gradient_norm_max"],
                "preclip_gradient_norm_mean": train_health["preclip_gradient_norm_mean"],
                "postclip_gradient_norm_max": train_health["postclip_gradient_norm_max"],
                "amp_scale_min": train_health["amp_scale_min"],
                "amp_scale_max": train_health["amp_scale_max"],
                "output_std_ratio_min": train_health["output_std_ratio_min"],
                "gradient_warning_steps": train_health["gradient_warning_steps"],
                "validation_throughput_samples_per_second": validation_health.get(
                    "throughput_samples_per_second", 0.0
                ),
                "pathology_event_count": len(self.health_events),
            }
            self.history.append(row)
            improved = val_loss < self.best_metric
            if improved:
                self.best_metric, self.stale_epochs = val_loss, 0
            else:
                self.stale_epochs += 1
            if self.scheduler is not None and not self._scheduler_steps_per_batch:
                self.scheduler.step()
            if improved and self.is_primary:
                self.save_checkpoint("best.pt", epoch + 1)
            if self.is_primary and (
                (epoch + 1) % self.config.checkpoint_every == 0 or epoch + 1 == self.config.epochs
            ):
                self.save_checkpoint("last.pt", epoch + 1)
            if (
                self.config.early_stopping_patience is not None
                and self.stale_epochs >= self.config.early_stopping_patience
            ):
                break
        if self.history and self.config.validation_improvement_min_fraction is not None:
            initial = float(self.history[0]["val_loss"])
            improvement = (initial - float(self.best_metric)) / max(abs(initial), 1.0e-12)
            if improvement < self.config.validation_improvement_min_fraction:
                self._record_health_event(
                    "insufficient_validation_improvement",
                    improvement_fraction=improvement,
                    required_fraction=self.config.validation_improvement_min_fraction,
                )
        return self.history

    def _model_state(self) -> Mapping[str, Tensor]:
        return (
            self.model.module.state_dict()
            if hasattr(self.model, "module")
            else self.model.state_dict()
        )

    def _scheduler_checkpoint_state(self) -> tuple[Mapping[str, Any] | None, list[str]]:
        """Return a weights-only-safe scheduler state.

        PyTorch 2.2 stores ``OneCycleLR.anneal_func`` as a Python function in
        ``state_dict``.  Newer releases store a string instead.  The function is
        derivable from the freshly constructed scheduler configuration and must
        not be pickled into an otherwise safe checkpoint, so only callable
        top-level entries are omitted and recorded.
        """

        if self.scheduler is None:
            return None, []
        state = dict(self.scheduler.state_dict())
        omitted = sorted(key for key, value in state.items() if callable(value))
        return {key: value for key, value in state.items() if key not in omitted}, omitted

    def save_checkpoint(self, filename: str, epoch: int) -> Path:
        directory = Path(self.config.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        scheduler_state, scheduler_omitted = self._scheduler_checkpoint_state()
        payload = {
            "format_version": 2,
            "epoch": epoch,
            "model_state": self._model_state(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": scheduler_state,
            "scheduler_state_omitted_callable_keys": scheduler_omitted,
            "scaler_state": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "stale_epochs": self.stale_epochs,
            "history": self.history,
            "health_events": self.health_events,
            "health_warnings": self.health_warnings,
            "optimizer_steps": self._optimizer_steps,
            "config": asdict(self.config),
            "rng": {
                "python": random.getstate(),
                "numpy": _numpy_rng_payload(np.random.get_state()),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        (directory / "training_config.json").write_text(
            json.dumps(asdict(self.config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination

    def load_checkpoint(self, path: str | Path) -> Mapping[str, Any]:
        checkpoint = load_checkpoint_payload(path, map_location=self.device)
        target = self.model.module if hasattr(self.model, "module") else self.model
        target.load_state_dict(checkpoint["model_state"], strict=self.config.strict_resume)
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler_state = checkpoint.get("scheduler_state")
        if self.scheduler is not None:
            if scheduler_state is None and int(checkpoint.get("epoch", 0)) > 0:
                if self.config.strict_resume:
                    raise ValueError(
                        "Checkpoint lacks scheduler_state required for an exact scheduled resume"
                    )
            elif scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)
        if "scaler_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.best_metric = float(checkpoint.get("best_metric", float("inf")))
        self.stale_epochs = int(checkpoint.get("stale_epochs", 0))
        self.history = list(checkpoint.get("history", []))
        self.health_events = list(checkpoint.get("health_events", []))
        self.health_warnings = list(checkpoint.get("health_warnings", []))
        self._optimizer_steps = int(checkpoint.get("optimizer_steps", 0))
        rng = checkpoint.get("rng", {})
        if rng:
            random.setstate(rng["python"])
            _restore_numpy_rng(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        return checkpoint


def train_model(
    model: nn.Module,
    train_loader: Iterable[Any],
    val_loader: Iterable[Any] | None = None,
    *,
    config: TrainingConfig | Mapping[str, Any] | None = None,
) -> Trainer:
    """Convenience API used by the CLI and Python examples."""

    trainer = Trainer(model, config or TrainingConfig())
    trainer.fit(train_loader, val_loader)
    return trainer
