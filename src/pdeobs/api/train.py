"""One-line training with an explicit budget, producing a reusable model artifact.

The facade only assembles: existing ``BenchmarkDataset`` records + observation
config -> existing ``Trainer`` with a ``TrainingConfig`` -> artifact.  No loss,
mask or numerical logic lives here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .. import __solver_version__, __version__
from . import specs as S
from .artifacts import ModelArtifact, save_artifact
from .data import DatasetHandle, load_dataset
from .models import ModelConfig, build, check_grid_compatibility, count_parameters, resolve_model
from .observation import ObservationSpec, make_observation


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass
class Budget:
    """Explicit training budget.  Exactly one of epochs / max_steps."""

    epochs: int | None = None
    max_steps: int | None = None

    @classmethod
    def parse(cls, value: Any) -> Budget:
        if isinstance(value, Budget):
            return value
        if value is None:
            raise ValueError("training needs an explicit budget: {'epochs': N} or {'max_steps': N}; no default long run is started")
        if isinstance(value, int):
            return cls(epochs=int(value))
        if isinstance(value, str):
            key, _, num = value.partition("=")
            return cls.parse({key.strip(): int(num)})
        if isinstance(value, Mapping):
            known = {"epochs", "max_steps"}
            unknown = set(value) - known
            if unknown:
                raise ValueError(f"budget accepts only {sorted(known)}, got {sorted(unknown)}")
            epochs = value.get("epochs")
            steps = value.get("max_steps")
            if (epochs is None) == (steps is None):
                raise ValueError("give exactly one of budget.epochs or budget.max_steps")
            b = cls(epochs=None if epochs is None else int(epochs), max_steps=None if steps is None else int(steps))
            if (b.epochs is not None and b.epochs < 1) or (b.max_steps is not None and b.max_steps < 1):
                raise ValueError("budget must be positive")
            return b
        raise ValueError(f"unsupported budget {value!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"epochs": self.epochs, "max_steps": self.max_steps,
                "kind": "software_step_budget" if self.max_steps is not None else "epoch_budget"}


class _StepBudgetLoader:
    """Yield exactly ``max_steps`` batches in one epoch, cycling the base loader as needed.

    Used only for the ``max_steps`` (software step budget) mode with ``epochs=1``, so each acceptance
    flow performs a fixed, small number of optimizer steps regardless of the dataset size.
    """

    def __init__(self, loader: Any, max_steps: int) -> None:
        self.loader = loader
        self.max_steps = int(max_steps)

    def __iter__(self):
        produced = 0
        while produced < self.max_steps:
            emitted_this_pass = 0
            for batch in self.loader:
                yield batch
                produced += 1
                emitted_this_pass += 1
                if produced >= self.max_steps:
                    return
            if emitted_this_pass == 0:  # the base loader is empty; avoid an infinite loop
                return

    def __len__(self) -> int:
        return self.max_steps


@dataclass
class SplitPlan:
    name: str
    train_ids: list[str]
    test_ids: list[str]
    receipt: dict[str, Any] = field(default_factory=dict)


def plan_split(metadata: Sequence[Mapping[str, Any]], split: Any, *, seed: int, identity: str) -> SplitPlan:
    """Decide which record identities train.  Never relabels stored split fields.

    ``split`` may be ``"all"`` (every record trains; no held-out identities), ``"paper"``
    (the exact 2000-record stable split, 1800/200), ``"holdout:<fraction>"`` (deterministic
    sha256-ranked fraction held out), ``"stored"`` (use the records' own ``split`` metadata:
    train vs test), or an explicit mapping ``{"train_ids": [...], "test_ids": [...]}``.
    """
    ids = [str(r["sample_id"]) for r in metadata]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample identities in the selected records")
    if isinstance(split, Mapping):
        train = [str(s) for s in split.get("train_ids", [])]
        test = [str(s) for s in split.get("test_ids", [])]
        known = set(ids)
        missing = [s for s in train + test if s not in known]
        if missing:
            raise ValueError(f"explicit split names unknown identities: {missing[:5]}")
        if set(train) & set(test):
            raise ValueError("explicit split has overlapping train/test identities")
        if not train:
            raise ValueError("explicit split has no training identities")
        return SplitPlan("explicit", train, test, {"algorithm": "explicit_identity_lists"})
    text = str(split).strip().lower()
    if text == "all":
        return SplitPlan("all", ids, [], {"algorithm": "all_records_train", "held_out": 0})
    if text in {"paper", "paper", "paper"}:
        from ..one_setting import stable_split
        train_idx, test_idx, receipt = stable_split(list(metadata), seed, identity)
        return SplitPlan("paper", [ids[i] for i in train_idx], [ids[i] for i in test_idx], dict(receipt))
    if text == "stored":
        train = [s for s, r in zip(ids, metadata) if str(r.get("split")) == "train"]
        test = [s for s, r in zip(ids, metadata) if str(r.get("split")) == "test"]
        if not train:
            raise ValueError("no records carry split='train' metadata; use 'all' or an explicit split")
        return SplitPlan("stored", train, test, {"algorithm": "stored_split_metadata"})
    if text.startswith("holdout:"):
        fraction = float(text.split(":", 1)[1])
        if not 0.0 < fraction < 1.0:
            raise ValueError("holdout fraction must be in (0, 1)")
        ranked = sorted(ids, key=lambda s: hashlib.sha256(f"{seed}|{identity}|{s}".encode()).hexdigest())
        n_test = max(1, int(round(fraction * len(ids))))
        if n_test >= len(ids):
            raise ValueError("holdout leaves no training records")
        test = sorted(ranked[:n_test])
        test_set = set(test)
        train = [s for s in ids if s not in test_set]
        return SplitPlan(f"holdout:{fraction}", train, test, {"algorithm": "sha256_rank_holdout", "fraction": fraction, "seed": seed})
    raise ValueError(f"unknown split {split!r}; use all, paper, stored, holdout:<fraction> or explicit id lists")


@dataclass
class TrainResult:
    artifact: ModelArtifact
    out: Path
    history: list[dict[str, Any]]
    optimizer_steps: int
    resolved: dict[str, Any]
    split: SplitPlan
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"artifact": str(self.artifact.path), "out": str(self.out), "epochs": len(self.history),
                "optimizer_steps": self.optimizer_steps, "split": {"name": self.split.name, "train": len(self.split.train_ids),
                                                                   "test": len(self.split.test_ids)},
                "summary": self.summary}


def _training_values(config: ModelConfig, *, task: str, budget: Budget, n_train: int, batch_size: int, seed: int,
                     device: str, history_steps: int, horizon: int, overrides: Mapping[str, Any], preset: str | None,
                     temporal: bool) -> dict[str, Any]:
    values: dict[str, Any] = {"task": task, "loss": "relative_l2", "optimizer": "adamw", "learning_rate": 1e-3,
                              "weight_decay": 1e-4, "scheduler": "none", "amp": False, "grad_clip": 1.0,
                              "deterministic": True, "seed": int(seed), "device": device, "data_layout": "channels_last",
                              "history_steps": int(history_steps), "horizon": int(horizon), "rollout_target_offset": 0,
                              "target_step": -1, "checkpoint_every": 1, "teacher_forcing_ratio": 0.0,
                              "health_policy": "record", "log_every": 20}
    if preset == "paper" and config.name in S.PAPER_OPTIMIZER:
        paper = dict(S.PAPER_OPTIMIZER[config.name])
        if temporal:
            paper.update(S.PAPER_TEMPORAL_OPTIMIZER.get(config.name, {}))
        paper.pop("batch_size", None)
        values.update(paper)
    if config.name == "pino":
        values["physics_loss"] = config.physics_contract
        values.setdefault("physics_loss_weight", 1.0)
        values.setdefault("data_loss_weight", 1.0 if preset != "paper" else 5.0)
        values["amp"] = False
    if budget.max_steps is not None:
        # one epoch that cycles to exactly max_steps optimizer steps
        values["epochs"] = 1
        values["scheduler_steps_per_epoch"] = budget.max_steps
    else:
        values["epochs"] = budget.epochs
        values["scheduler_steps_per_epoch"] = max(1, math.ceil(n_train / batch_size))
    if values.get("scheduler") not in {"one_cycle"}:
        values.pop("scheduler_steps_per_epoch", None)
    clean = {k: v for k, v in overrides.items() if v is not None}
    if "amp" in clean and config.name == "pino" and clean["amp"]:
        raise ValueError("PINO training requires amp=False")
    if "physics_loss" in clean and config.name != "pino":
        raise ValueError("physics_loss is only meaningful for PINO")
    values.update(clean)
    return values


def effective_channels(handle: DatasetHandle, state_representation: str) -> tuple[int, int]:
    """(state, condition) channel counts after ``BenchmarkDataset`` applies ``state_representation``.

    Navier-Stokes records can be served as vorticity (1 channel) or velocity (2 channels) regardless
    of how they were stored; the model structure must match the served representation, not the file.
    """
    rep = str(state_representation).lower()
    if handle.pde == "navier_stokes" and rep in {"velocity", "vorticity"}:
        served = 2 if rep == "velocity" else 1
        return served, served
    return handle.state_channels, handle.condition_channels


def train(data: DatasetHandle | str | Path, *, task: str, model: str | Mapping[str, Any] | ModelConfig,
          observation: str | Mapping[str, Any] | ObservationSpec, out: str | Path, budget: Any, preset: str | None = None,
          model_params: Mapping[str, Any] | None = None, seed: int = 0, device: str = "auto", batch_size: int | None = None,
          split: Any = "all", history_steps: int = 1, horizon: int | None = None, training: Mapping[str, Any] | None = None,
          num_workers: int = 0, label: str | None = None, max_samples: int | None = None,
          state_representation: str | None = None, overwrite: bool = False) -> TrainResult:
    """Train a model on canonical records under a stated budget and save a reusable artifact."""
    handle = data if isinstance(data, DatasetHandle) else load_dataset(data)
    task = str(task).lower().replace("-", "_")
    if task not in S.TASKS:
        raise ValueError(f"unknown task {task!r}")
    if task not in handle.supported_tasks:
        raise ValueError(f"records at {handle.path} support {handle.supported_tasks}; task {task!r} is not available (T={handle.time_steps})")
    budget_obj = Budget.parse(budget)
    obs = make_observation(observation) if not isinstance(observation, ObservationSpec) else observation
    if obs.namespace == "stored":
        raise ValueError("training needs a synthetic observation over complete records; 'stored' is only for inference inputs")
    if obs.namespace == "paper" and tuple(handle.resolution) != S.PAPER_RESOLUTION:
        raise ValueError(f"paper views are defined at 128x128; the records are {handle.resolution}. Use the general namespace for other grids")
    temporal = task == "rollout"
    rep = state_representation or ("vorticity" if handle.state_representation == "vorticity" else "native")
    state_channels, condition_channels = effective_channels(handle, rep)
    config = resolve_model(model, task=task, pde=handle.pde, preset=preset, params=model_params, in_channels=state_channels)
    if task == "inverse" and "out_channels" in config.kwargs:
        config.kwargs["out_channels"] = condition_channels
    problems = check_grid_compatibility(config, handle.shape_hw)
    if problems:
        raise ValueError("; ".join(problems))
    out_dir = Path(out)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory {out_dir} already holds a run; choose a new directory (runs are never overwritten by default)")
    out_dir.mkdir(parents=True, exist_ok=True)
    if temporal:
        if horizon is None:
            horizon = handle.time_steps - history_steps
        if history_steps + horizon > handle.time_steps:
            raise ValueError(f"history {history_steps} + horizon {horizon} exceeds the stored {handle.time_steps} frames")
    else:
        horizon = 1
    from ..dataset import BenchmarkDataset
    from ..one_setting import subset_dataset

    dataset = BenchmarkDataset(handle.shards, task=task, mask=dict(obs.mask_config), history_steps=history_steps,
                               horizon=horizon if temporal else 8, seed=seed, state_representation=rep, max_samples=max_samples)
    spec = S.MODEL_SPECS[config.name]
    result: TrainResult
    try:
        identity = "|".join(str(x) for x in (handle.pde, handle.boundary, handle.setting))
        plan = plan_split(dataset.metadata, split, seed=seed, identity=identity)
        positions = {row["sample_id"]: i for i, row in enumerate(dataset.metadata)}
        train_set = subset_dataset(dataset, [positions[s] for s in plan.train_ids])
        n_train = len(train_set)
        default_bs = S.PAPER_OPTIMIZER.get(config.name, {}).get("batch_size", 4) if preset == "paper" else 2
        if temporal and preset == "paper":
            default_bs = S.PAPER_TEMPORAL_OPTIMIZER.get(config.name, {}).get("batch_size", default_bs)
        bs = int(batch_size or min(default_bs, n_train))
        if bs < 1:
            raise ValueError("batch_size must be positive")
        if config.name == "ufno" and bs < 2 and spec.family == "neural":
            raise ValueError("U-FNO's U-branches use BatchNorm2d: training needs batch_size >= 2")
        if spec.family not in {"classical", "fitted"}:
            # seed before the structure is built: weight initialisation draws from the global RNGs,
            # so seeding only before the optimiser loop would make repeated calls in one process diverge
            from ..training import seed_everything
            seed_everything(int(seed), True)
        model_obj = build(config)
        resolved: dict[str, Any] = {
            "schema_version": S.PIPELINE_CONFIG_VERSION, "stage": "train", "api_version": S.API_VERSION,
            "pdeobs_version": __version__, "numerical_kernel_version": __solver_version__, "task": task, "seed": int(seed),
            "data": {"path": handle.path, "shards": handle.shards, "sample_ids_sha256": handle.sample_ids_sha256, "pde": handle.pde,
                     "boundary": handle.boundary, "setting": handle.setting, "regime": handle.regime, "resolution": handle.resolution,
                     "time_steps": handle.time_steps, "state_representation": rep, "history_steps": history_steps, "horizon": horizon,
                     "stored_frame_indices": handle.stored_frame_indices, "stored_time_values": handle.stored_time_values},
            "observation": obs.to_dict(), "model": config.to_dict(), "split": {"name": plan.name, "receipt": plan.receipt,
                                                                               "train_records": n_train, "test_records": len(plan.test_ids)},
            "budget": budget_obj.to_dict(), "batch_size": bs, "num_workers": int(num_workers), "device_requested": device,
            "label": label or ("paper" if preset == "paper" else "custom"),
        }
        if spec.family in {"classical"}:
            history: list[dict[str, Any]] = []
            steps = 0
            training_values: dict[str, Any] = {"note": "parameter-free method; nothing was trained"}
            checkpoint = None
            device_used = "cpu"
        elif spec.family == "fitted":
            history = []
            steps = 0
            training_values = {"note": "fitted through fit_dataset"}
            model_obj.fit_dataset(train_set, None)
            checkpoint = None
            device_used = "cpu"
        else:
            from ..runner import _loader
            from ..training import Trainer, TrainingConfig, seed_everything
            training_values = _training_values(config, task=task, budget=budget_obj, n_train=n_train, batch_size=bs, seed=seed,
                                               device=device, history_steps=history_steps, horizon=horizon, overrides=training or {},
                                               preset=preset, temporal=temporal)
            training_values["checkpoint_dir"] = str(out_dir / "checkpoints")
            tcfg = TrainingConfig.from_mapping(training_values)
            seed_everything(int(seed), True)  # again: the data loader shuffle and dropout draw from the same RNGs
            loader = _loader(train_set, {"training": {"batch_size": bs, "num_workers": int(num_workers), "pin_memory": False},
                                         "seed": seed}, shuffle=True)
            if budget_obj.max_steps is not None:
                loader = _StepBudgetLoader(loader, budget_obj.max_steps)
            started = time.perf_counter()
            trainer = Trainer(model_obj, tcfg)
            before = {k: v.detach().cpu().clone() for k, v in list(model_obj.state_dict().items())[:4]}
            history = trainer.fit(loader, val_loader=None)
            steps = int(trainer._optimizer_steps)
            after = {k: v.detach().cpu() for k, v in model_obj.state_dict().items() if k in before}
            changed = any(not np.array_equal(before[k].numpy(), after[k].numpy()) for k in before)
            resolved["parameters_changed_after_training"] = bool(changed)
            resolved["training_seconds"] = round(time.perf_counter() - started, 3)
            resolved["peak_gpu_memory_bytes"] = max((int(h.get("peak_gpu_memory_bytes", 0)) for h in history), default=0)
            resolved["gradient_norm_max"] = max((float(h.get("preclip_gradient_norm_max", 0.0)) for h in history), default=0.0)
            resolved["loss_finite"] = all(np.isfinite(float(h.get("train_loss", np.nan))) for h in history)
            checkpoint = out_dir / "checkpoints" / "last.pt"
            device_used = str(trainer.device)
            (out_dir / "history.json").write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
            (out_dir / "health.json").write_text(json.dumps({"events": trainer.health_events, "warnings": getattr(trainer, "health_warnings", []),
                                                            "last_epoch_health": trainer.last_epoch_health},
                                                           indent=2, default=str), encoding="utf-8")
        resolved["training"] = training_values
        resolved["device_used"] = device_used
        resolved["optimizer_steps"] = steps
        resolved["training_config_sha256"] = _canonical_hash(training_values)
        recipe = copy.deepcopy(resolved)
        for key in ("data", "device_requested", "device_used", "training_seconds", "peak_gpu_memory_bytes", "seed"):
            recipe.pop(key, None)
        resolved["training_recipe_sha256"] = _canonical_hash(recipe)
        io_schema = {"layout": "channels_last", "input": "observations (N,H,W,C)" if not temporal else "history (N,T_h,H,W,C)",
                     "mask": "(N,H,W,1) 1=observed", "geometry": "(N,H,W,1) 1=solid" if config.kwargs.get("geometry_channels", 0) else "not used",
                     "output": "(N,H,W,C_out)" if not temporal else "(N,horizon,H,W,C_out)", "in_channels": config.kwargs.get("in_channels", 1),
                     "out_channels": config.kwargs.get("out_channels", 1), "resolution_trained": handle.resolution,
                     "state_representation": rep, "history_steps": history_steps, "horizon_trained": horizon,
                     "time_stride": (handle.stored_time_values[1] - handle.stored_time_values[0]) if handle.stored_time_values and len(handle.stored_time_values) > 1 else None,
                     "normalization": "none (raw physical fields; relative-L2 objective)",
                     "coordinates": "unit-square grid implied by the model (FNO/UFNO [0,1], CNO [-1,1]); not stored"}
        contract = {"schema_version": "pdeobs-training-contract/v1", "task": task, "observation": obs.to_dict(),
                    "dataset": resolved["data"], "split": resolved["split"], "budget": budget_obj.to_dict(), "loss": training_values.get("loss"),
                    "physics_loss": training_values.get("physics_loss", "none"), "physics_loss_weight": training_values.get("physics_loss_weight", 0.0),
                    "data_loss_weight": training_values.get("data_loss_weight", 1.0), "optimizer_steps": steps, "epochs_completed": len(history),
                    "training_config_sha256": resolved["training_config_sha256"], "training_recipe_sha256": resolved["training_recipe_sha256"],
                    "seed": int(seed), "kind": resolved["label"], "not_a_benchmark_result": budget_obj.max_steps is not None or (budget_obj.epochs or 0) < 500}
        artifact = save_artifact(out_dir / "model", model=model_obj, config=config, training_contract=contract, resolved_config=resolved,
                                 io_schema=io_schema, trainer_checkpoint=checkpoint,
                                 extra={"kind": "smoke_or_example_checkpoint" if contract["not_a_benchmark_result"] else "trained_checkpoint",
                                        "label": resolved["label"]})
        (out_dir / "split.json").write_text(json.dumps({"name": plan.name, "train_ids": plan.train_ids, "test_ids": plan.test_ids,
                                                        "receipt": plan.receipt}, indent=2), encoding="utf-8")
        (out_dir / "resolved.json").write_text(json.dumps(resolved, indent=2, sort_keys=True, default=str), encoding="utf-8")
        summary = {"model": config.name, "task": task, "parameters": count_parameters(model_obj), "epochs": len(history),
                   "optimizer_steps": steps, "final_train_loss": (history[-1].get("train_loss") if history else None),
                   "device": device_used, "artifact": str(artifact.path)}
        result = TrainResult(artifact=artifact, out=out_dir, history=history, optimizer_steps=steps, resolved=resolved, split=plan, summary=summary)
    finally:
        dataset.close()
    return result
