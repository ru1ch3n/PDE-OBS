"""L0 CPU tests: the v0.2.0 facade resolvers, validation and parameter schema (no torch needed)."""
from __future__ import annotations

import pytest

from pdeobs.api import models as M
from pdeobs.api import observation as O
from pdeobs.api import specs as S


def test_main_paper_models_present():
    assert set(S.MAIN_PAPER_MODELS) == {"fno", "pino", "ufno", "cno", "deeponet", "gnot", "transolver"}


def test_model_alias_resolution():
    assert S.normalize_model_name("ufno_2d") == "ufno"
    assert S.normalize_model_name("Compact-FNO") == "fno"
    with pytest.raises(ValueError):
        S.normalize_model_name("does_not_exist")


@pytest.mark.parametrize("model", S.MAIN_PAPER_MODELS)
def test_paper_preset_resolves(model):
    cfg = M.resolve_model(model, task="recovery" if model != "pino" else "recovery", preset="paper")
    assert cfg.origin == "paper"
    assert cfg.kwargs["geometry_channels"] == 1


def test_fno_groupnorm_constraint_rejected():
    with pytest.raises(ValueError, match="divisible"):
        M.resolve_model("fno", task="recovery", params={"width": 36})
    M.resolve_model("fno", task="recovery", params={"width": 40})  # divisible by 8 -> ok


def test_unknown_and_alias_confused_params_rejected():
    with pytest.raises(ValueError, match="does not accept"):
        M.resolve_model("fno", task="recovery", params={"widht": 3})
    with pytest.raises(ValueError, match="does not accept"):
        M.resolve_model("cno", task="recovery", params={"depth": 3})
    with pytest.raises(ValueError):
        M.resolve_model("gnot", task="recovery", params={"hidden": 30, "heads": 4})


def test_pino_contract_follows_task():
    assert M.resolve_model("pino", task="recovery").physics_contract == "pino_static_fd_v1"
    assert M.resolve_model("pino", task="rollout").physics_contract == "pino_rollout_spectral_v1"
    with pytest.raises(ValueError):
        M.resolve_model("pino", task="inverse")
    with pytest.raises(ValueError, match="does not match task"):
        M.resolve_model("pino", task="recovery", params={"physics_contract": "pino_rollout_spectral_v1"})


def test_task_support_enforced():
    with pytest.raises(ValueError, match="does not support task"):
        M.resolve_model("convlstm", task="recovery")
    with pytest.raises(ValueError, match="does not support task"):
        M.resolve_model("mae_small", task="rollout")


def test_static_pde_rollout_rejected():
    with pytest.raises(ValueError, match="static"):
        M.resolve_model("fno", task="rollout", pde="poisson")


def test_upstream_wrapper_not_served():
    with pytest.raises(ValueError, match="upstream"):
        M.resolve_model("paper_fno", task="recovery")


def test_rollout_wraps_autoregressive():
    method = M.resolve_model("fno", task="rollout", preset="smoke").method_config()
    assert method["name"] == "autoregressive"
    assert method["base"]["name"] == "fno"


def test_grid_compatibility_flags_modes():
    cfg = M.resolve_model("fno", task="recovery", params={"width": 16, "modes": 40, "layers": 2})
    problems = M.check_grid_compatibility(cfg, (16, 16))
    assert problems and "modes" in problems[0]


def test_observation_general_and_paper_split():
    assert O.make_observation("random", ratio=0.5).mask_config == {"protocol": "random_3pct", "ratio": 0.5}
    assert O.make_observation("paper:R65").mask_config == {"protocol": "random_3pct", "ratio": 0.65}
    with pytest.raises(ValueError, match="paper view"):
        O.make_observation("random_65pct")  # must be requested explicitly in the paper namespace


def test_observation_conflicts_and_unknown_params():
    with pytest.raises(ValueError, match="conflict"):
        O.make_observation("random", ratio=0.5, count=100)
    with pytest.raises(ValueError, match="does not accept"):
        O.make_observation("random", clusters=4)
    with pytest.raises(ValueError):
        O.make_observation("full", ratio=0.5)


@pytest.mark.parametrize("view,expected", [("random_50pct", 8192), ("random_65pct", 10650), ("random_80pct", 13107),
                                           ("boundary_band_50pct", 8284), ("clustered_50pct", 8192),
                                           ("horizontal_lines_50pct", 8192), ("vertical_lines_50pct", 8192)])
def test_paper_realized_counts_match(view, expected):
    spec = O.make_observation(f"paper:{view}")
    assert O.realized_count(spec, (128, 128)) == expected


def test_stored_observation_namespace():
    spec = O.make_observation("stored")
    assert spec.namespace == "stored" and spec.mask_config == {"protocol": "stored"}


def test_pipeline_validation_rejects_download_and_missing_time_steps():
    from pdeobs.api.pipeline import validate_pipeline
    with pytest.raises(ValueError, match="time_steps"):
        validate_pipeline({"task": "rollout", "data": {"source": "generate", "pde": "heat", "out": "x", "num_samples": 2, "resolution": 16}})
    with pytest.raises(ValueError, match="download"):
        validate_pipeline({"task": "recovery", "data": {"source": "download", "path": "x"}})
