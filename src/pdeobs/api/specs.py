"""Declarative specifications behind the high-level API.

Nothing here constructs a model or a mask.  The tables describe, for every
public model and observation protocol, which parameters the *actual*
constructors accept (checked against the v0.1.1 implementations), their types
and constraints, the frozen ``paper`` structures and the small ``smoke``
presets.  Unknown parameters are rejected before any expensive work starts.

Versions are managed separately: the numerical kernel keeps
``pdeobs.__solver_version__``; this facade carries ``API_VERSION`` and the
observation/score protocol versions it delegates to.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

API_VERSION = "pdeobs-easy-api/v1"
MODEL_ARTIFACT_VERSION = "pdeobs-model-artifact/v1"
INFERENCE_INPUT_VERSION = "pdeobs-inference-input/v1"
PREDICTION_BUNDLE_VERSION = "pdeobs-prediction-bundle/v1"
PIPELINE_CONFIG_VERSION = "pdeobs-pipeline/v1"
OBSERVATION_GENERAL_VERSION = "pdeobs-observation-general/v1"
OBSERVATION_PAPER_VERSION = "pdeobs.paper-observations/v1"

STATIC_PDES = ("darcy", "poisson", "helmholtz")
TEMPORAL_PDES = ("heat", "reaction_diffusion", "burgers", "navier_stokes")
ALL_PDES = STATIC_PDES + TEMPORAL_PDES
TASKS = ("recovery", "forward", "inverse", "rollout")


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str  # int | float | str | bool | int?  (trailing ? = optional)
    default: Any
    doc: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] | None = None
    effect: str = "structure"  # structure | regularization | io | physics | fixed

    def validate(self, value: Any, model: str) -> Any:
        optional = self.type.endswith("?")
        base = self.type.rstrip("?")
        if value is None:
            if optional:
                return None
            raise ValueError(f"{model}: parameter {self.name!r} must not be null")
        if isinstance(value, bool) and base != "bool":
            raise ValueError(f"{model}: parameter {self.name!r} expects {base}, got bool")
        if base == "int":
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if not isinstance(value, int):
                raise ValueError(f"{model}: parameter {self.name!r} expects int, got {value!r}")
        elif base == "float":
            if not isinstance(value, (int, float)):
                raise ValueError(f"{model}: parameter {self.name!r} expects float, got {value!r}")
            value = float(value)
        elif base == "str":
            if not isinstance(value, str):
                raise ValueError(f"{model}: parameter {self.name!r} expects str, got {value!r}")
        elif base == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{model}: parameter {self.name!r} expects bool, got {value!r}")
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{model}: parameter {self.name!r}={value!r} is below the minimum {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{model}: parameter {self.name!r}={value!r} is above the maximum {self.maximum}")
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"{model}: parameter {self.name!r}={value!r}; valid choices: {list(self.choices)}")
        return value


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    family: str  # neural | classical | fitted | reference | official
    factory: str  # "model" -> methods.neural.create_model, "method" -> methods.create_method
    tasks: tuple[str, ...]
    params: tuple[ParamSpec, ...]
    aliases: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    presets: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    temporal_presets: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    notes: str = ""
    requires_torch: bool = True
    main_paper_model: bool = False
    temporal_wrapper: str | None = "autoregressive"  # how rollout is served
    upstream_wrapper: bool = False
    physics: bool = False

    def param(self, name: str) -> ParamSpec | None:
        return next((p for p in self.params if p.name == name), None)

    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)


_IO = (
    ParamSpec("in_channels", "int", 1, "input state channels (1 scalar field; 2 for NS velocity)", minimum=1, effect="io"),
    ParamSpec("out_channels", "int", 1, "output state channels", minimum=1, effect="io"),
    ParamSpec("geometry_channels", "int", 1, "obstacle/geometry channels appended to the input (0 disables; 1 is the paper setting)", minimum=0, maximum=1, effect="io"),
)


def _spec(**kwargs: Any) -> ModelSpec:
    return ModelSpec(**kwargs)


MODEL_SPECS: dict[str, ModelSpec] = {}


def _register(spec: ModelSpec) -> None:
    MODEL_SPECS[spec.name] = spec


_register(_spec(
    name="fno", label="Fourier Neural Operator (compact PDE-OBS adaptation)", family="neural", factory="model",
    tasks=("recovery", "forward", "inverse", "rollout"), main_paper_model=True,
    aliases=("fno2d", "compact_fno"),
    params=_IO + (
        ParamSpec("width", "int", 32, "channel width of every spectral layer; must be divisible by min(8, width) (GroupNorm)", minimum=1),
        ParamSpec("modes", "int", 12, "retained Fourier modes per axis; modes > H//2 are clipped at run time", minimum=1),
        ParamSpec("layers", "int", 4, "number of spectral+local layers", minimum=1),
    ),
    constraints=("width % min(8, width) == 0",),
    presets={"smoke": {"width": 16, "modes": 6, "layers": 2}, "paper": {"width": 32, "modes": 12, "layers": 4}},
))

_register(_spec(
    name="pino", label="Physics-Informed Neural Operator (PDE-OBS adaptation, FNO backbone + residual loss)",
    family="neural", factory="model", tasks=("recovery", "forward", "rollout"), main_paper_model=True, physics=True,
    aliases=("pino_static", "physics_informed_fno"),
    params=_IO + (
        ParamSpec("width", "int", 32, "FNO backbone width; must be divisible by min(8, width)", minimum=1),
        ParamSpec("modes", "int", 12, "retained Fourier modes per axis", minimum=1),
        ParamSpec("layers", "int", 4, "number of spectral+local layers", minimum=1),
        ParamSpec("physics_contract", "str", "pino_static_fd_v1", "residual contract; static tasks use pino_static_fd_v1, periodic rollout uses pino_rollout_spectral_v1",
                  choices=("pino_static_fd_v1", "pino_rollout_spectral_v1"), effect="physics"),
    ),
    constraints=("width % min(8, width) == 0", "physics_contract must match task: static->pino_static_fd_v1, rollout->pino_rollout_spectral_v1",
                 "training requires physics_loss == physics_contract, physics_loss_weight > 0 and amp == False",
                 "inverse task is not supported by any PINO residual"),
    presets={"smoke": {"width": 16, "modes": 6, "layers": 2}, "paper": {"width": 64, "modes": 20, "layers": 5}},
))

_register(_spec(
    name="ufno", label="U-FNO-2D (PDE-OBS adaptation)", family="neural", factory="model",
    tasks=("recovery", "forward", "inverse", "rollout"), main_paper_model=True,
    aliases=("u_fno", "ufno2d", "u_fno_2d", "ufno_2d"),
    params=_IO + (
        ParamSpec("width", "int", 36, "channel width (no GroupNorm; any positive width)", minimum=1),
        ParamSpec("modes", "int", 12, "retained Fourier modes per axis", minimum=1),
        ParamSpec("padding", "int", 8, "zero padding applied before the spectral stack", minimum=0),
        ParamSpec("dropout", "float", 0.0, "dropout inside the three U-branches", minimum=0.0, maximum=1.0, effect="regularization"),
    ),
    constraints=("block count is fixed: 6 spectral layers, U-branches on layers 4-6 (no `layers` parameter)",
                 "U-branches use BatchNorm2d: training batch size must be >= 2"),
    presets={"smoke": {"width": 12, "modes": 6, "padding": 2}, "paper": {"width": 36, "modes": 12, "padding": 8, "dropout": 0.0}},
))

_register(_spec(
    name="cno", label="CNO-inspired anti-aliased operator (compact PDE-OBS adaptation)", family="neural", factory="model",
    tasks=("recovery", "forward", "inverse", "rollout"), main_paper_model=True,
    aliases=("cno2d", "compact_cno", "cno_inspired"),
    params=_IO + (
        ParamSpec("width", "int", 32, "encoder width; the decoder uses 2x width", minimum=1),
    ),
    constraints=("depth is fixed (encoder / mid / decoder); there is no `layers` or `depth` parameter",),
    presets={"smoke": {"width": 8}, "paper": {"width": 32}},
))

_register(_spec(
    name="deeponet", label="Variable-sensor DeepONet (PDE-OBS adaptation)", family="neural", factory="method",
    tasks=("recovery", "forward", "inverse", "rollout"), main_paper_model=True,
    aliases=("variable_sensor_deeponet",),
    params=_IO + (
        ParamSpec("hidden", "int", 128, "hidden width of branch/trunk MLPs", minimum=1),
        ParamSpec("latent", "int", 128, "branch/trunk latent dimension", minimum=1),
        ParamSpec("branch_layers", "int", 2, "branch depth", minimum=1),
        ParamSpec("trunk_layers", "int", 2, "trunk depth", minimum=1),
        ParamSpec("branch_mode", "str", "sensor_set", "sensor_set (permutation-invariant) or spatial_cnn (ordered mask-aware)", choices=("sensor_set", "spatial_cnn")),
        ParamSpec("branch_grid", "int", 4, "pooling grid of the spatial_cnn branch", minimum=1),
        ParamSpec("branch_width", "int?", None, "conv width of the spatial_cnn branch (default min(hidden, 64))", minimum=1),
    ),
    presets={"smoke": {"hidden": 16, "latent": 16, "branch_layers": 1, "trunk_layers": 1, "branch_mode": "spatial_cnn", "branch_grid": 2, "branch_width": 8},
             "paper": {"hidden": 128, "latent": 128, "branch_layers": 2, "trunk_layers": 2, "branch_mode": "spatial_cnn", "branch_grid": 4, "branch_width": 64}},
))

_register(_spec(
    name="gnot", label="GNOT (mask-conditioned PDE-OBS adaptation)", family="neural", factory="method",
    tasks=("recovery", "forward", "inverse", "rollout"), main_paper_model=True,
    aliases=("mask_conditioned_gnot",),
    params=_IO + (
        ParamSpec("hidden", "int", 128, "token width; must be divisible by heads", minimum=1),
        ParamSpec("layers", "int", 3, "number of attention blocks", minimum=1),
        ParamSpec("heads", "int", 1, "attention heads", minimum=1),
        ParamSpec("experts", "int", 2, "gated MLP experts per block", minimum=1),
        ParamSpec("inner_ratio", "int", 4, "expert MLP expansion ratio", minimum=1),
        ParamSpec("mlp_layers", "int", 3, "depth of the source/query encoders", minimum=1),
        ParamSpec("dropout", "float", 0.0, "attention/MLP dropout", minimum=0.0, maximum=1.0, effect="regularization"),
    ),
    constraints=("hidden % heads == 0",),
    presets={"smoke": {"hidden": 16, "layers": 1, "heads": 1, "experts": 1, "inner_ratio": 2, "mlp_layers": 1},
             "paper": {"hidden": 128, "layers": 3, "heads": 1, "experts": 2}},
))

_register(_spec(
    name="transolver", label="Transolver (mask-conditioned PDE-OBS adaptation)", family="neural", factory="method",
    tasks=("recovery", "forward", "inverse", "rollout"), main_paper_model=True,
    aliases=("transolver_2d",),
    params=_IO + (
        ParamSpec("hidden", "int", 256, "token width; must be divisible by heads", minimum=1),
        ParamSpec("layers", "int", 5, "number of physics-attention blocks", minimum=1),
        ParamSpec("heads", "int", 8, "attention heads", minimum=1),
        ParamSpec("slices", "int", 32, "physics slices per head", minimum=1),
        ParamSpec("mlp_ratio", "int", 1, "MLP expansion ratio", minimum=1),
        ParamSpec("dropout", "float", 0.0, "dropout", minimum=0.0, maximum=1.0, effect="regularization"),
    ),
    constraints=("hidden % heads == 0",),
    presets={"smoke": {"hidden": 16, "layers": 1, "heads": 2, "slices": 4, "mlp_ratio": 1},
             "paper": {"hidden": 128, "layers": 8, "heads": 8, "slices": 64, "mlp_ratio": 1}},
    temporal_presets={"paper": {"hidden": 256, "layers": 8, "heads": 8, "slices": 32, "mlp_ratio": 1}},
))

# ---- other public reference models -------------------------------------------------
_register(_spec(
    name="unet", label="Mask-channel U-Net (compact reference)", family="neural", factory="model",
    tasks=("recovery", "forward", "inverse", "rollout"), aliases=("mask_unet", "mask_channel_unet"),
    params=_IO + (ParamSpec("width", "int", 32, "base width; doubled per level", minimum=1),),
    presets={"smoke": {"width": 4}, "paper": {"width": 32}},
    notes="not part of the paper's seven learned models",
))
_register(_spec(
    name="unet_paper_guided", label="Deep paper-guided mask U-Net", family="neural", factory="method",
    tasks=("recovery", "forward", "inverse", "rollout"), aliases=("deep_mask_unet",),
    params=_IO + (ParamSpec("width", "int", 64, "base width", minimum=1), ParamSpec("levels", "int", 4, "encoder levels", minimum=1)),
    presets={"smoke": {"width": 4, "levels": 2}},
))
_register(_spec(
    name="convlstm", label="ConvLSTM rollout baseline", family="neural", factory="model", tasks=("rollout",),
    aliases=("conv_lstm",), temporal_wrapper=None,
    params=(ParamSpec("in_channels", "int", 1, "input channels", minimum=1, effect="io"),
            ParamSpec("out_channels", "int", 1, "output channels", minimum=1, effect="io"),
            ParamSpec("hidden_channels", "int", 32, "recurrent hidden channels", minimum=1)),
    constraints=("no geometry input",),
    presets={"smoke": {"hidden_channels": 4}},
))
_register(_spec(
    name="mae_small", label="Masked autoencoder (small reference)", family="neural", factory="model",
    tasks=("recovery", "forward", "inverse"), aliases=("masked_autoencoder", "masked_autoencoder_small", "mae_style_small"),
    temporal_wrapper=None,
    params=(ParamSpec("in_channels", "int", 1, "input channels", minimum=1, effect="io"),
            ParamSpec("out_channels", "int", 1, "output channels", minimum=1, effect="io"),
            ParamSpec("width", "int", 32, "width", minimum=1), ParamSpec("latent_channels", "int", 128, "latent channels", minimum=1),
            ParamSpec("patch_size", "int", 8, "patch size", minimum=1),
            ParamSpec("mask_ratio", "float", 0.75, "pretraining mask ratio", minimum=0.0, maximum=1.0, effect="regularization"),
            ParamSpec("preserve_visible", "bool", True, "copy visible values through", effect="io")),
    constraints=("no geometry input",),
    presets={"smoke": {"width": 4, "latent_channels": 8, "patch_size": 4}},
))

# ---- classical / fitted baselines ----------------------------------------------------
for _name, _label, _aliases, _tasks, _params in (
    ("zero", "Zero fill", (), ("recovery", "forward", "inverse"), ()),
    ("mean", "Mean of observed values", (), ("recovery", "forward", "inverse"), ()),
    ("nearest", "Nearest-neighbour interpolation", (), ("recovery", "forward", "inverse"), ()),
    ("bilinear", "Bilinear interpolation", (), ("recovery", "forward", "inverse"), ()),
    ("rbf", "Gaussian RBF interpolation", ("gaussian_rbf",), ("recovery", "forward", "inverse"),
     (ParamSpec("smoothing", "float", 1e-6, "RBF smoothing", minimum=0.0), ParamSpec("epsilon", "float?", None, "kernel shape parameter"),
      ParamSpec("max_centers", "int", 512, "maximum sensors used as centers", minimum=1),
      ParamSpec("normalized_coordinates", "bool", False, "use [0,1] coordinates"))),
    ("persistence", "Persistence rollout", (), ("rollout",), ()),
):
    _register(_spec(name=_name, label=_label, family="classical", factory="method", tasks=_tasks, aliases=_aliases,
                    params=tuple(_params), requires_torch=False, temporal_wrapper=None,
                    presets={"smoke": {}, "paper": {"smoothing": 1e-6, "epsilon": 2.0, "max_centers": 512, "normalized_coordinates": True} if _name == "rbf" else {}}))

_register(_spec(
    name="gappy_pod", label="Gappy POD (fitted linear reduced-order method)", family="fitted", factory="method",
    tasks=("recovery", "forward", "inverse"), requires_torch=False, temporal_wrapper=None,
    params=(ParamSpec("rank", "int?", None, "POD rank (None selects among rank_candidates)", minimum=1),
            ParamSpec("ridge", "float", 1e-6, "ridge regularization", minimum=0.0),
            ParamSpec("oversampling", "int", 16, "oversampling", minimum=0), ParamSpec("seed", "int", 0, "seed")),
    presets={"smoke": {"rank": 4}, "paper": {"rank": 64, "ridge": 1e-6}},
    notes="trained through fit_dataset, not the Trainer; served through the legacy `train` runner path",
))

# ---- optional exact upstream wrappers (dependency gated) ---------------------------
for _name in ("paper_unet", "paper_fno", "paper_cno"):
    _register(_spec(name=_name, label=f"Official upstream wrapper {_name} (requires pinned external checkout)", family="official",
                    factory="method", tasks=("recovery",), params=(), upstream_wrapper=True, temporal_wrapper=None,
                    notes="fail-closed: needs upstream_root/upstream_revision/source_file_sha256; never substituted by a compact model"))


MAIN_PAPER_MODELS = tuple(name for name, spec in MODEL_SPECS.items() if spec.main_paper_model)
_ALIAS_INDEX: dict[str, str] = {}
for _name, _s in MODEL_SPECS.items():
    _ALIAS_INDEX[_name] = _name
    for _a in _s.aliases:
        _ALIAS_INDEX[_a] = _name


def normalize_model_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _ALIAS_INDEX:
        raise ValueError(f"unknown model {name!r}; public models: {', '.join(sorted(MODEL_SPECS))}")
    return _ALIAS_INDEX[key]


def model_spec(name: str) -> ModelSpec:
    return MODEL_SPECS[normalize_model_name(name)]


# ---- observation protocols -----------------------------------------------------------

@dataclass(frozen=True)
class ObservationProtocolSpec:
    name: str
    doc: str
    params: tuple[ParamSpec, ...]
    aliases: tuple[str, ...] = ()
    exclusive: tuple[tuple[str, ...], ...] = ()  # parameter groups that must not be combined


OBSERVATION_PROTOCOLS: dict[str, ObservationProtocolSpec] = {}


def _obs(spec: ObservationProtocolSpec) -> None:
    OBSERVATION_PROTOCOLS[spec.name] = spec


_RATIO = ParamSpec("ratio", "float?", None, "observed fraction of grid cells (actual count = round(ratio*H*W))", minimum=0.0, maximum=1.0)
_COUNT = ParamSpec("count", "int?", None, "exact number of observed cells", minimum=1)
_obs(ObservationProtocolSpec("random", "uniformly random sensors (exact discrete count); default 3% (128x128 default: 500 sensors)",
                             (_RATIO, _COUNT), aliases=("random_3pct", "random_sensors"), exclusive=(("ratio", "count"),)))
_obs(ObservationProtocolSpec("grid", "regular grid of sensors approximating a target ratio (spacing is discrete; exact ratio not guaranteed)",
                             (ParamSpec("ratio", "float", 0.03, "target observed fraction", minimum=0.0, maximum=1.0),
                              ParamSpec("spacing", "int?", None, "explicit grid spacing (overrides ratio)", minimum=1),
                              ParamSpec("random_offset", "bool", False, "random phase of the grid")),
                             aliases=("regular_grid", "regular-grid"), exclusive=(("ratio", "spacing"),)))
_obs(ObservationProtocolSpec("block", "everything observed except one missing square block (missing_fraction of the area)",
                             (ParamSpec("missing_fraction", "float", 0.25, "area fraction of the missing block", minimum=0.0, maximum=1.0),
                              ParamSpec("block_shape", "int?", None, "explicit side length of the missing block", minimum=1)),
                             aliases=("block_missing", "missing_block"), exclusive=(("missing_fraction", "block_shape"),)))
_obs(ObservationProtocolSpec("line", "full rows and/or columns of sensors",
                             (ParamSpec("ratio", "float", 0.03, "target observed fraction", minimum=0.0, maximum=1.0),
                              ParamSpec("num_lines", "int?", None, "explicit number of lines", minimum=1),
                              ParamSpec("orientation", "str", "both", "horizontal, vertical or both", choices=("horizontal", "vertical", "both"))),
                             aliases=("line_sensors", "lines"), exclusive=(("ratio", "num_lines"),)))
_obs(ObservationProtocolSpec("horizontal", "full rows of sensors (line protocol, horizontal)",
                             (ParamSpec("ratio", "float", 0.03, "target observed fraction", minimum=0.0, maximum=1.0),
                              ParamSpec("num_lines", "int?", None, "explicit number of rows", minimum=1)),
                             aliases=("horizontal_lines",), exclusive=(("ratio", "num_lines"),)))
_obs(ObservationProtocolSpec("vertical", "full columns of sensors (line protocol, vertical)",
                             (ParamSpec("ratio", "float", 0.03, "target observed fraction", minimum=0.0, maximum=1.0),
                              ParamSpec("num_lines", "int?", None, "explicit number of columns", minimum=1)),
                             aliases=("vertical_lines",), exclusive=(("ratio", "num_lines"),)))
_obs(ObservationProtocolSpec("boundary", "a band of `width` cells along the array perimeter (deterministic; not a physical boundary condition)",
                             (ParamSpec("width", "int", 1, "band width in cells; requires 1 <= width <= min(H,W)//2", minimum=1),
                              ParamSpec("count", "int?", None, "sub-sample the band to an exact count", minimum=1)),
                             aliases=("boundary_sensors", "edge", "boundary_band")))
_obs(ObservationProtocolSpec("clustered", "sensors grouped in Gaussian clusters",
                             (ParamSpec("ratio", "float", 0.03, "target observed fraction", minimum=0.0, maximum=1.0),
                              ParamSpec("count", "int?", None, "exact count", minimum=1),
                              ParamSpec("clusters", "int", 4, "number of clusters", minimum=1),
                              ParamSpec("spread", "float", 0.08, "cluster spread (fraction of the domain)", minimum=0.0)),
                             aliases=("clustered_sensors", "clusters"), exclusive=(("ratio", "count"),)))
_obs(ObservationProtocolSpec("full", "every cell observed (dataset-level pseudo protocol; accepts no parameters)", (), aliases=("full_observation", "all_visible")))
_obs(ObservationProtocolSpec("custom", "a mask factory registered in pdeobs.registry.MASK_REGISTRY, or an explicit mask array",
                             (ParamSpec("protocol", "str", "", "registered factory name", effect="io"),), aliases=("registered",)))

# canonical protocol -> registry protocol + fixed kwargs
_GENERAL_TO_REGISTRY = {
    "random": ("random_3pct", {}),
    "grid": ("regular_grid", {}),
    "block": ("block_missing", {}),
    "line": ("line_sensors", {}),
    "horizontal": ("line_sensors", {"orientation": "horizontal"}),
    "vertical": ("line_sensors", {"orientation": "vertical"}),
    "boundary": ("boundary_sensors", {}),
    "clustered": ("clustered_sensors", {}),
    "full": ("full", {}),
}
_OBS_ALIAS_INDEX: dict[str, str] = {}
for _n, _o in OBSERVATION_PROTOCOLS.items():
    _OBS_ALIAS_INDEX[_n] = _n
    for _a in _o.aliases:
        _OBS_ALIAS_INDEX[_a] = _n
# registry names are also accepted directly
for _reg in ("random_1pct", "random_5pct", "random_10pct"):
    _OBS_ALIAS_INDEX[_reg] = "random"


def normalize_observation_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key in _OBS_ALIAS_INDEX:
        return _OBS_ALIAS_INDEX[key]
    raise ValueError(f"unknown observation protocol {name!r}; general protocols: {', '.join(OBSERVATION_PROTOCOLS)}; "
                     f"paper views: {', '.join(PAPER_VIEWS)}")


# The paper's nine frozen views (configs/paper/observations.yaml) at 128x128.  The mask
# configuration is passed verbatim to the existing dataset mask construction so the
# arrays and seeds match the original experiments.
PAPER_VIEWS: dict[str, dict[str, Any]] = {
    "random_50pct": {"label": "R50", "mask": {"protocol": "random_3pct", "ratio": 0.50}, "expected_count_128": 8192},
    "random_65pct": {"label": "R65", "mask": {"protocol": "random_3pct", "ratio": 0.65}, "expected_count_128": 10650},
    "random_80pct": {"label": "R80", "mask": {"protocol": "random_3pct", "ratio": 0.80}, "expected_count_128": 13107},
    "block_observed_50pct": {"label": "BL", "mask": {"protocol": "block_missing", "missing_fraction": 0.49}, "expected_count_128": 8284},
    "line_sensors_50pct": {"label": "LI", "mask": {"protocol": "line_sensors", "num_lines": 76, "orientation": "both"}, "expected_count_128": 8284},
    "horizontal_lines_50pct": {"label": "H", "mask": {"protocol": "line_sensors", "num_lines": 64, "orientation": "horizontal"}, "expected_count_128": 8192},
    "vertical_lines_50pct": {"label": "V", "mask": {"protocol": "line_sensors", "num_lines": 64, "orientation": "vertical"}, "expected_count_128": 8192},
    "boundary_band_50pct": {"label": "BD", "mask": {"protocol": "boundary_sensors", "width": 19}, "expected_count_128": 8284},
    "clustered_50pct": {"label": "CL", "mask": {"protocol": "clustered_sensors", "ratio": 0.50}, "expected_count_128": 8192},
}
PAPER_VIEW_LABELS = {v["label"]: k for k, v in PAPER_VIEWS.items()}
PAPER_RESOLUTION = (128, 128)

# paper problem settings (configs/campaign/all_pde_one_setting_10method_9x9.yaml)
PAPER_PROBLEMS: dict[str, dict[str, str]] = {
    "darcy": {"task": "recovery", "boundary": "dirichlet", "setting": "smooth_grf", "state_representation": "native"},
    "poisson": {"task": "recovery", "boundary": "dirichlet", "setting": "smooth_grf", "state_representation": "native"},
    "helmholtz": {"task": "recovery", "boundary": "dirichlet", "setting": "smooth_grf", "state_representation": "native"},
    "heat": {"task": "rollout", "boundary": "periodic", "setting": "smooth_grf", "state_representation": "native"},
    "reaction_diffusion": {"task": "rollout", "boundary": "periodic", "setting": "smooth_grf", "state_representation": "native"},
    "burgers": {"task": "rollout", "boundary": "periodic", "setting": "smooth_grf", "state_representation": "native"},
    "navier_stokes": {"task": "rollout", "boundary": "periodic", "setting": "smooth_grf", "state_representation": "vorticity"},
}

# paper optimizer settings per main model (configs/method/all_pde_source_faithful_registry.yaml)
PAPER_OPTIMIZER: dict[str, dict[str, Any]] = {
    "ufno": {"optimizer": "adam", "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 4},
    "fno": {"optimizer": "adam", "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 8},
    "cno": {"optimizer": "adamw", "learning_rate": 1e-3, "weight_decay": 1e-6, "batch_size": 8},
    "deeponet": {"optimizer": "adam", "learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 4},
    "gnot": {"optimizer": "adamw", "learning_rate": 1e-3, "weight_decay": 5e-5, "batch_size": 4,
             "scheduler": "one_cycle", "scheduler_pct_start": 0.2, "scheduler_div_factor": 1e4, "scheduler_final_div_factor": 1e4,
             "grad_clip": 1000.0, "gradient_warning_fatal": False},
    "transolver": {"optimizer": "adamw", "learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 4,
                   "scheduler": "one_cycle", "scheduler_pct_start": 0.3, "scheduler_div_factor": 25.0, "scheduler_final_div_factor": 1e4,
                   "grad_clip": 0.1, "gradient_warning_fatal": False},
    "pino": {"optimizer": "adam", "learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 4,
             "scheduler": "step", "scheduler_step_size": 100, "scheduler_gamma": 0.5,
             "data_loss_weight": 5.0, "physics_loss_weight": 1.0, "grad_clip": 1.0, "gradient_warning_fatal": False},
}
PAPER_TEMPORAL_OPTIMIZER: dict[str, dict[str, Any]] = {
    "transolver": {"batch_size": 2, "grad_clip": None, "gradient_warning_fatal": True},
}
