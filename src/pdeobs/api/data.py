"""Data sources: numerical generation, local canonical records, manifests, and
observation-only inference input packages.

Three different objects are kept apart on purpose:

* a **numerical solver** (``create_dataset(source="generate")`` or ``source="arrays"``)
  computes complete physical records with the existing kernels;
* a **dataset source** (``load_dataset``) reads records that already exist;
* a **trained model artifact** (see :mod:`pdeobs.api.artifacts`) predicts.

None of them is ever substituted for another.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .. import __solver_version__, __version__
from . import specs as S

CANONICAL_FORMATS = ("hdf5",)
INFERENCE_FORMATS = ("npz",)


@dataclass
class DatasetHandle:
    """What a user gets back from create/load: where the records are and what they contain."""

    path: str
    shards: list[str]
    sample_count: int
    pde: str | None
    boundary: str | None
    setting: str | None
    regime: str | None
    resolution: list[int]
    time_steps: int
    condition_channels: int
    state_channels: int
    state_representation: str
    stored_frame_indices: list[int] | None
    stored_time_values: list[float] | None
    supported_tasks: list[str]
    solver_version: str | None
    pdeobs_version: str | None
    source: str  # generate | local | manifest | arrays
    identity: dict[str, Any] = field(default_factory=dict)
    sample_ids_sha256: str | None = None
    generation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def shape_hw(self) -> tuple[int, int]:
        return int(self.resolution[0]), int(self.resolution[1])

    @property
    def is_temporal(self) -> bool:
        return self.time_steps > 1

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps({"schema_version": "pdeobs-dataset-handle/v1", **self.to_dict()}, indent=2, sort_keys=True),
                          encoding="utf-8")
        return target


def _sha_ids(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _inspect(shards: Sequence[Path], *, source: str, root: Path) -> DatasetHandle:
    from ..storage import LazyHDF5Dataset

    reader = LazyHDF5Dataset(list(shards), verify=False)
    try:
        rows = list(reader.iter_metadata())
        if not rows:
            raise ValueError("dataset contains no records")
        first = reader[0]
        cond, traj = first.condition, first.trajectory
        keys = ("pde", "boundary", "setting", "regime")
        distinct = {k: sorted({str(r.get(k)) for r in rows}) for k in keys}
        ident = {k: (v[0] if len(v) == 1 else None) for k, v in distinct.items()}
        reps = sorted({str(r.get("state_representation", "scalar")) for r in rows})
        t = int(traj.shape[0])
        frames = rows[0].get("stored_frame_indices")
        times = rows[0].get("stored_time_values")
        tasks = ["recovery", "forward", "inverse"] + (["rollout"] if t > 1 else [])
        return DatasetHandle(
            path=str(root), shards=[str(p) for p in shards], sample_count=len(rows),
            pde=ident["pde"], boundary=ident["boundary"], setting=ident["setting"], regime=ident["regime"],
            resolution=[int(traj.shape[1]), int(traj.shape[2])], time_steps=t,
            condition_channels=int(cond.shape[-1]), state_channels=int(traj.shape[-1]),
            state_representation=reps[0] if len(reps) == 1 else "mixed",
            stored_frame_indices=list(frames) if frames is not None else None,
            stored_time_values=[float(x) for x in times] if times is not None else None,
            supported_tasks=tasks, solver_version=rows[0].get("solver_version"), pdeobs_version=rows[0].get("pdeobs_version"),
            source=source, identity={"distinct": distinct, "seeds": sorted({int(r.get("seed", 0)) for r in rows})[:8]},
            sample_ids_sha256=_sha_ids([str(r.get("sample_id")) for r in rows]))
    finally:
        reader.close()


def load_dataset(path: str | Path, *, source: str = "local", verify: bool = False,
                 manifest: str | Path | None = None) -> DatasetHandle:
    """Read existing canonical HDF5 records (a shard, a directory of shards, or a manifest).

    ``source="manifest"`` reads an identity manifest (``pdeobs identity-manifest``) and verifies the
    listed shard checksums.  Remote/published datasets are **not** downloaded here: when the release
    manifest declares no verified public endpoint this call reports that explicitly rather than
    regenerating data under a download label.
    """
    root = Path(path)
    if source == "manifest" or manifest is not None:
        from ..strict_inference import load_identity_dataset
        manifest_path = Path(manifest) if manifest is not None else root
        data_root = root if manifest is not None else root.parent
        dataset, _ = load_identity_dataset(data_root, manifest_path)
        try:
            shards = [Path(p) for p in dataset.base.paths]
        finally:
            dataset.close()
        handle = _inspect(shards, source="manifest", root=data_root)
        handle.identity["manifest"] = str(manifest_path)
        return handle
    if source == "download":
        raise ValueError("no verified public download endpoint is registered for this release candidate; "
                         "use source='generate' to compute records locally or source='local' for records you already hold")
    if source != "local":
        raise ValueError("load_dataset source must be local or manifest")
    if not root.exists():
        raise FileNotFoundError(f"dataset path does not exist: {root}")
    if root.is_dir():
        shards = sorted(root.rglob("*.h5")) + sorted(root.rglob("*.hdf5"))
    else:
        if root.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"unsupported dataset format {root.suffix!r}; canonical records are HDF5 (.h5)")
        shards = [root]
    if not shards:
        raise FileNotFoundError(f"no canonical HDF5 shards under {root}")
    if verify:
        from ..storage import is_shard_complete
        for shard in shards:
            if not is_shard_complete(shard):
                raise ValueError(f"shard is not a complete verified record: {shard}")
    return _inspect(shards, source="local", root=root)


def _normalize_pde(pde: str) -> str:
    from ..generation import _canonical_pde
    return _canonical_pde(pde)


def create_dataset(*, pde: str, out: str | Path, boundary: str = "periodic", setting: str = "smooth_grf",
                   regime: str = "low", num_samples: int = 4, resolution: int | Sequence[int] = 32, seed: int = 0,
                   time_steps: int | None = None, stored_time_steps: int | None = None, format: str = "hdf5",
                   shard_size: int | None = None, dtype: str = "float32", numerics: Mapping[str, Any] | None = None,
                   source: str = "generate", arrays: Mapping[str, Any] | None = None, physical: Mapping[str, Any] | None = None,
                   overwrite: bool = False) -> DatasetHandle:
    """Compute complete physical records with the built-in numerical routes and store them canonically.

    ``source="generate"`` uses the deterministic condition generators; ``source="arrays"`` takes the
    user's own source/coefficient/initial arrays (see :func:`create_dataset_from_arrays`).
    Only ``format="hdf5"`` is a canonical complete record; nothing else is silently accepted.
    """
    if format.lower() not in CANONICAL_FORMATS:
        raise ValueError(f"unsupported dataset format {format!r}; canonical complete records use {CANONICAL_FORMATS}")
    if source == "arrays":
        if arrays is None:
            raise ValueError("source='arrays' needs the arrays mapping")
        return create_dataset_from_arrays(pde=pde, out=out, boundary=boundary, arrays=arrays, physical=physical or {},
                                          resolution=resolution, seed=seed, dtype=dtype, overwrite=overwrite)
    if source != "generate":
        raise ValueError("create_dataset source must be generate or arrays")
    from ..generation import GenerationJob, generate_job

    family = _normalize_pde(pde)
    if boundary == "robin":
        boundary = "robin_obstacle"
    res = normalize_res(resolution)
    static = family in S.STATIC_PDES
    if static and time_steps not in (None, 1):
        raise ValueError(f"{family} is a static family: time_steps must be 1 (or omitted)")
    if not static and time_steps is None:
        raise ValueError(f"{family} is temporal: give time_steps (number of stored frames, >= 2; the paper uses 15)")
    if not static and int(time_steps) < 2:
        raise ValueError("temporal records need time_steps >= 2")
    out_dir = Path(out)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory {out_dir} is not empty; choose a new directory or overwrite=True")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(num_samples)
    if n < 1:
        raise ValueError("num_samples must be positive")
    per_shard = int(shard_size or n)
    if per_shard < 1:
        raise ValueError("shard_size must be positive")
    options = dict(numerics or {})
    jobs = []
    start = 0
    index = 0
    while start < n:
        count = min(per_shard, n - start)
        jobs.append(GenerationJob(pde=family, boundary=boundary, setting=setting, regime=regime, sample_start=start,
                                  sample_count=count, shard_index=index, output_path=str(out_dir / f"shard_{index:05d}.h5"),
                                  resolution=res, seed=int(seed), time_steps=None if static else int(time_steps),
                                  stored_time_steps=stored_time_steps, dtype=dtype, tier="custom",
                                  macro_size=max(2000, n * 3), options=options or None, quality={"profile": "report"},
                                  provenance={"created_by": "pdeobs.api.create_dataset", "api_version": S.API_VERSION}))
        start += count
        index += 1
    results = [generate_job(job, resume=not overwrite, overwrite=overwrite) for job in jobs]
    produced = sum(r.sample_count for r in results)
    if produced != n:
        raise RuntimeError(f"generation produced {produced} of {n} requested samples; no silent re-sampling is performed")
    handle = _inspect([Path(r.output_path) for r in results], source="generate", root=out_dir)
    handle.generation = {"jobs": [j.to_dict() for j in jobs], "results": [asdict(r) for r in results],
                         "solver_version": __solver_version__, "pdeobs_version": __version__,
                         "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    handle.save(out_dir / "dataset.json")
    return handle


def normalize_res(resolution: int | Sequence[int]) -> tuple[int, int]:
    from ..schema import normalize_resolution
    return tuple(int(v) for v in normalize_resolution(resolution))  # type: ignore[return-value]


def create_dataset_from_arrays(*, pde: str, out: str | Path, boundary: str, arrays: Mapping[str, Any],
                               physical: Mapping[str, Any], resolution: int | Sequence[int] | None = None, seed: int = 0,
                               dtype: str = "float32", overwrite: bool = False) -> DatasetHandle:
    """Solve the user's own physical inputs with the existing low-level kernels and store canonical records.

    Supported routes (complete inputs only; nothing is imputed):
      poisson   : arrays["source"] (N,H,W)                          -> -div(grad u) = source
      darcy     : arrays["source"], arrays["coefficient"] (N,H,W)   -> -div(a grad u) = source
      helmholtz : arrays["source"] (N,H,W), physical["reaction"]    -> -div(grad u) + reaction*u = source
      heat      : arrays["initial_state"] (N,H,W), physical{diffusivity, dt, steps}
    Other families need the full generator and are rejected explicitly.
    """
    from ..physical_inputs import advance_custom_heat, solve_custom_elliptic
    from ..schema import Sample
    from ..storage import AtomicHDF5ShardWriter

    family = _normalize_pde(pde)
    out_dir = Path(out)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory {out_dir} is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    solver_boundary = "robin" if boundary == "robin_obstacle" else boundary
    if family in {"poisson", "darcy", "helmholtz"}:
        source = np.asarray(arrays.get("source"), dtype=np.float64)
        if source.ndim != 3:
            raise ValueError("arrays['source'] must have shape (N, H, W)")
        n, h, w = source.shape
        coefficient = None
        if family == "darcy":
            if "coefficient" not in arrays:
                raise ValueError("darcy needs arrays['coefficient'] (N,H,W); the coefficient is not inferred")
            coefficient = np.asarray(arrays["coefficient"], dtype=np.float64)
            if coefficient.shape != source.shape:
                raise ValueError("coefficient and source shapes differ")
        reaction = float(physical.get("reaction", 0.0)) if family == "helmholtz" else 0.0
        dx, dy = float(physical.get("dx", 1.0 / w)), float(physical.get("dy", 1.0 / h))
        fields = []
        for i in range(n):
            solution, info = solve_custom_elliptic(source[i], boundary=solver_boundary, dx=dx, dy=dy,
                                                   coefficient=None if coefficient is None else coefficient[i], reaction=reaction)
            if not bool(info.converged):
                raise RuntimeError(f"elliptic solve did not converge for sample {i}; the record is not stored")
            cond = source[i] if coefficient is None else coefficient[i]
            fields.append((cond, solution[None], {"solver_iterations": int(info.iterations),
                                                  "solver_relative_residual": float(info.relative_residual)}))
        t = 1
        frames = [0]
        times = None
    elif family == "heat":
        initial = np.asarray(arrays.get("initial_state"), dtype=np.float64)
        if initial.ndim != 3:
            raise ValueError("arrays['initial_state'] must have shape (N, H, W)")
        n, h, w = initial.shape
        diffusivity = float(physical["diffusivity"]) if "diffusivity" in physical else None
        dt = float(physical["dt"]) if "dt" in physical else None
        steps = int(physical.get("steps", 1))
        if diffusivity is None or dt is None or steps < 1:
            raise ValueError("heat needs physical={'diffusivity': ..., 'dt': ..., 'steps': >=1}")
        dx, dy = float(physical.get("dx", 1.0 / w)), float(physical.get("dy", 1.0 / h))
        fields = []
        for i in range(n):
            state = initial[i]
            traj = [state]
            for _ in range(steps):
                state, info = advance_custom_heat(state, diffusivity=diffusivity, dt=dt, boundary=solver_boundary, dx=dx, dy=dy)
                if not bool(info.converged):
                    raise RuntimeError(f"heat step did not converge for sample {i}")
                traj.append(state)
            fields.append((initial[i], np.stack(traj), {"diffusivity": diffusivity, "dt": dt}))
        t = steps + 1
        frames = list(range(t))
        times = [k * dt for k in range(t)]
    else:
        raise ValueError(f"create_dataset_from_arrays supports poisson, darcy, helmholtz and heat; {family} requires "
                         "the full generator route (source='generate')")
    shard = out_dir / "shard_00000.h5"
    spec = {"pde": family, "boundary": boundary, "source": "user_arrays", "api_version": S.API_VERSION}
    with AtomicHDF5ShardWriter(shard, expected_count=len(fields), spec=spec, resume=False, overwrite=overwrite) as writer:
        for i, (cond, traj, diag) in enumerate(fields):
            metadata = {"sample_id": f"user-{seed}/{family}/{boundary}/arrays/{i:06d}", "pde": family, "boundary": boundary,
                        "setting": "user_arrays", "regime": "user", "split": "user", "seed": int(seed), "T": t,
                        "resolution": [int(traj.shape[1]), int(traj.shape[2])], "state_representation": "scalar",
                        "solver_version": __solver_version__, "pdeobs_version": __version__,
                        "solver_fidelity": "compact_reference", "stored_frame_indices": frames,
                        "parameters": {"dx": dx, "dy": dy, **({"final_time": times[-1]} if times else {})}, "diagnostics": diag}
            if times is not None:
                metadata["stored_time_values"] = times
            writer.append(Sample(condition=np.asarray(cond, dtype=dtype)[..., None], trajectory=np.asarray(traj, dtype=dtype)[..., None],
                                 geometry=np.zeros((*traj.shape[1:], 1), dtype=np.float32), metadata=metadata))
    handle = _inspect([shard], source="arrays", root=out_dir)
    handle.save(out_dir / "dataset.json")
    return handle


# ----------------------------------------------------------------------------------------------
# Observation-only inference input packages (NPZ with an explicit schema)
# ----------------------------------------------------------------------------------------------

INFERENCE_KEYS = ("observations", "mask")
OPTIONAL_INFERENCE_KEYS = ("geometry", "sample_ids", "time_indices", "physical_times", "condition")


@dataclass
class InferenceInput:
    """Observations + mask (+ geometry/coords) without a target field."""

    observations: np.ndarray  # (N,H,W,C) static or (N,T_h,H,W,C) rollout history, channels-last
    mask: np.ndarray  # (N,H,W,1) bool/float, 1 = observed; may be (N,T_h,H,W,1)
    geometry: np.ndarray | None = None  # (N,H,W,1), 1 = solid
    sample_ids: list[str] = field(default_factory=list)
    time_indices: list[int] | None = None  # source frame indices of the history frames
    physical_times: list[float] | None = None
    condition: np.ndarray | None = None  # (N,H,W,C) PDE condition, needed only by PINO physics loss (not inference)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        obs = np.asarray(self.observations, dtype=np.float32)
        mask = np.asarray(self.mask)
        if obs.ndim not in (4, 5):
            raise ValueError("observations must be channels-last (N,H,W,C) or (N,T,H,W,C)")
        if obs.shape[-1] > 8:
            raise ValueError("observations must be channels-last with <= 8 channels; a layout must not be guessed from sizes")
        if mask.ndim == obs.ndim - 1:
            mask = mask[..., None]
        if mask.ndim != obs.ndim or mask.shape[:-1] != obs.shape[:-1] or mask.shape[-1] != 1:
            raise ValueError(f"mask shape {mask.shape} must match observations {obs.shape} with a single trailing channel")
        if not np.isfinite(obs).all():
            raise ValueError("observations contain non-finite values")
        # Validate BEFORE conversion: NaN/Inf or soft masks must not become valid
        # binary observations merely by passing through the convenience API.
        if mask.dtype.kind not in "biuf" or not np.isfinite(mask).all():
            raise ValueError("mask must contain finite real binary values")
        if not np.isin(mask, (0, 1)).all():
            raise ValueError("mask must contain only 0 and 1")
        self.observations = obs
        self.mask = mask.astype(np.float32)
        if self.geometry is not None:
            geom = np.asarray(self.geometry, dtype=np.float32)
            if geom.ndim == 3:
                geom = geom[..., None]
            if geom.shape != (obs.shape[0], *obs.shape[-3:-1], 1):
                raise ValueError(f"geometry shape {geom.shape} must be (N,H,W,1)")
            if not np.isfinite(geom).all():
                raise ValueError("geometry contains non-finite values")
            self.geometry = geom
        if self.time_indices is not None:
            from ..strict_score import _times
            self.time_indices = _times(self.time_indices, "input time_indices")
            if len(self.time_indices) != (obs.shape[1] if obs.ndim == 5 else 1):
                raise ValueError("input time_indices length differs from observation history")
        n = obs.shape[0]
        if not self.sample_ids:
            self.sample_ids = [f"input-{i:06d}" for i in range(n)]
        if len(self.sample_ids) != n or len(set(self.sample_ids)) != n:
            raise ValueError("sample_ids must be unique and match the batch size")
        if self.condition is not None:
            cond = np.asarray(self.condition, dtype=np.float32)
            if cond.ndim == 3:
                cond = cond[..., None]
            if cond.shape[:3] != (n, *obs.shape[-3:-1]):
                raise ValueError("condition must be (N,H,W,C)")
            self.condition = cond

    @property
    def batch_size(self) -> int:
        return int(self.observations.shape[0])

    @property
    def shape_hw(self) -> tuple[int, int]:
        return int(self.observations.shape[-3]), int(self.observations.shape[-2])

    @property
    def is_temporal(self) -> bool:
        return self.observations.ndim == 5

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        if target.suffix.lower() != ".npz":
            raise ValueError("inference inputs are stored as .npz")
        payload: dict[str, Any] = {"observations": self.observations, "mask": self.mask,
                                   "sample_ids": np.asarray(self.sample_ids, dtype=str),
                                   "schema": np.asarray(json.dumps({"schema_version": S.INFERENCE_INPUT_VERSION,
                                                                    "layout": "channels_last", "mask_semantics": "1=observed",
                                                                    "geometry_semantics": "1=solid", **self.metadata}))}
        if self.geometry is not None:
            payload["geometry"] = self.geometry
        if self.time_indices is not None:
            payload["time_indices"] = np.asarray(self.time_indices, dtype=np.int64)
        if self.physical_times is not None:
            payload["physical_times"] = np.asarray(self.physical_times, dtype=np.float64)
        if self.condition is not None:
            payload["condition"] = self.condition
        np.savez_compressed(target, **payload)
        return target

    @classmethod
    def load(cls, path: str | Path) -> InferenceInput:
        target = Path(path)
        if target.suffix.lower() != ".npz":
            raise ValueError(f"unsupported inference input format {target.suffix!r}; expected an .npz package with "
                             f"keys {INFERENCE_KEYS} (+ optional {OPTIONAL_INFERENCE_KEYS}) and a schema entry")
        with np.load(target, allow_pickle=False) as data:
            keys = set(data.files)
            missing = [k for k in INFERENCE_KEYS if k not in keys]
            if missing:
                raise ValueError(f"inference input {target} lacks required keys {missing}; keys present: {sorted(keys)}")
            if "schema" not in keys:
                raise ValueError("inference input has no 'schema' entry; bare arrays without a declared layout are not accepted")
            schema = json.loads(str(data["schema"]))
            if schema.get("schema_version") != S.INFERENCE_INPUT_VERSION:
                raise ValueError(f"inference input schema {schema.get('schema_version')!r} is not {S.INFERENCE_INPUT_VERSION!r}")
            if schema.get("layout") != "channels_last":
                raise ValueError("inference inputs must declare layout=channels_last")
            unknown = keys - set(INFERENCE_KEYS) - set(OPTIONAL_INFERENCE_KEYS) - {"schema"}
            if unknown:
                raise ValueError(f"inference input has undeclared keys {sorted(unknown)}")
            meta = {k: v for k, v in schema.items() if k not in {"schema_version", "layout", "mask_semantics", "geometry_semantics"}}
            return cls(observations=data["observations"], mask=data["mask"],
                       geometry=data["geometry"] if "geometry" in keys else None,
                       sample_ids=[str(s) for s in data["sample_ids"]] if "sample_ids" in keys else [],
                       time_indices=data["time_indices"].tolist() if "time_indices" in keys else None,
                       physical_times=[float(v) for v in data["physical_times"]] if "physical_times" in keys else None,
                       condition=data["condition"] if "condition" in keys else None, metadata=meta)


def inference_input_from_dataset(handle: DatasetHandle | str | Path, observation: Any, *, task: str = "recovery",
                                 history_steps: int = 1, sample_ids: Sequence[str] | None = None, seed: int = 0,
                                 max_samples: int | None = None, state_representation: str | None = None) -> tuple[InferenceInput, np.ndarray]:
    """Build an observation-only package from complete records (benchmark mode) and return the hidden targets separately.

    The targets are returned to the *caller* only so that a later ``evaluate`` can score; ``predict`` never sees them.
    """
    from ..dataset import BenchmarkDataset
    from .observation import make_observation

    handle = handle if isinstance(handle, DatasetHandle) else load_dataset(handle)
    obs_spec = make_observation(observation) if not hasattr(observation, "mask_config") else observation
    dataset = BenchmarkDataset(handle.shards, task=task, mask=dict(obs_spec.mask_config), history_steps=history_steps,
                               horizon=max(1, handle.time_steps - history_steps) if task == "rollout" else 8, seed=seed,
                               state_representation=state_representation or ("vorticity" if handle.state_representation == "vorticity" else "native"),
                               max_samples=max_samples)
    try:
        positions = list(range(len(dataset)))
        if sample_ids is not None:
            index = {row["sample_id"]: i for i, row in enumerate(dataset.metadata)}
            positions = [index[s] for s in sample_ids]
        rows = [dataset[i] for i in positions]
    finally:
        dataset.close()
    obs = np.stack([r["observations"] for r in rows]).astype(np.float32)
    mask = np.stack([r["mask"] for r in rows]).astype(np.float32)
    geom = np.stack([r["geometry"] for r in rows]).astype(np.float32)
    ids = [str(r["metadata"]["sample_id"]) for r in rows]
    targets = np.stack([r["target"] for r in rows]).astype(np.float32)
    frames = rows[0]["metadata"].get("stored_frame_indices")
    times = rows[0]["metadata"].get("stored_time_values")
    time_indices = list(frames[:history_steps]) if (task == "rollout" and frames is not None) else ([frames[0]] if frames else None)
    phys = list(times[:history_steps]) if (task == "rollout" and times is not None) else None
    cond = np.stack([r["pde_condition"] for r in rows]).astype(np.float32) if "pde_condition" in rows[0] else None
    package = InferenceInput(observations=obs, mask=mask, geometry=geom, sample_ids=ids, time_indices=time_indices,
                             physical_times=phys, condition=cond,
                             metadata={"observation": obs_spec.to_dict(), "task": task, "dataset_sha256": handle.sample_ids_sha256,
                                       "pde": handle.pde, "mask_id": str(rows[0]["metadata"].get("mask_id")),
                                       "stored_frame_indices": frames, "stored_time_values": times})
    return package, targets
