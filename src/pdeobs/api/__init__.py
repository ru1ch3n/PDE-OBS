"""High-level PDE-OBS API (``from pdeobs import api``).

One-line operations over the existing generators, datasets, masks, models,
Trainer and strict scorer::

    from pdeobs import api
    data = api.create_dataset(pde="heat", out="demo/heat", num_samples=12, resolution=32, time_steps=4)
    obs = api.make_observation("random", ratio=0.5)
    run = api.train(data, task="rollout", model="fno", model_params={"width": 16, "modes": 6, "layers": 2},
                    observation=obs, budget={"max_steps": 5}, out="demo/fno")
    predictor = api.load_predictor(run.artifact.path)
    package, targets = api.inference_input_from_dataset(data, obs, task="rollout")
    bundle = api.predict(predictor, package, horizon=3)          # no target needed
    score = api.evaluate(bundle, targets)                         # strict score when a target exists

Every operation prints/saves its resolved configuration.  Loading data or a
model never trains; training always requires an explicit budget.
"""
from __future__ import annotations

from .artifacts import ModelArtifact, Predictor, load_artifact, load_legacy_checkpoint, load_predictor, save_artifact
from .data import (DatasetHandle, InferenceInput, create_dataset, create_dataset_from_arrays, inference_input_from_dataset,
                   load_dataset)
from .evaluate import build_contract, evaluate, evaluate_records, evaluate_views
from .infer import PredictionBundle, predict
from .models import (ModelConfig, build as build_model, check_grid_compatibility, count_parameters, describe_model,
                     list_models, resolve_model, summarize)
from .observation import ObservationSpec, describe_observations, make_observation, paper_views, realized_count
from .pipeline import load_pipeline_config, plan, run, validate_pipeline
from .specs import (API_VERSION, MAIN_PAPER_MODELS, MODEL_SPECS, OBSERVATION_PROTOCOLS, PAPER_PROBLEMS, PAPER_VIEWS,
                    STATIC_PDES, TEMPORAL_PDES)
from .train import Budget, SplitPlan, TrainResult, plan_split, train


def create_model(model, *, task: str = "recovery", preset: str | None = None, params=None, pde: str | None = None,
                 in_channels: int | None = None):
    """Resolve + construct a model in one call; returns ``(module, ModelConfig)``."""
    config = resolve_model(model, task=task, preset=preset, params=params, pde=pde, in_channels=in_channels)
    return build_model(config), config


def list_pdes() -> list[dict]:
    from ..generation import BOUNDARIES
    from ..settings import SETTING_NAMES
    return [{"pde": p, "temporal": p in TEMPORAL_PDES, "boundaries": list(BOUNDARIES), "settings": list(SETTING_NAMES),
             "regimes": ["low", "medium", "high"], "paper": PAPER_PROBLEMS.get(p)} for p in STATIC_PDES + TEMPORAL_PDES]


def list_presets() -> dict:
    return {name: sorted(spec.presets) for name, spec in MODEL_SPECS.items() if spec.presets}


__all__ = [name for name in dir() if not name.startswith("_")]
