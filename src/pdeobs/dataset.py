"""Task views over immutable canonical PDE-OBS shards."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .masks import MASK_PROTOCOL_NAMES, apply_mask, generate_mask
from .mask_specs import is_mask_spec, parse_mask_spec
from .registry import normalize_name as _normalize_mask_name
from .registry import MASK_REGISTRY
from .schema import derive_seed, normalize_resolution
from .splits import REGIMES, resolve_tier
from .storage import LazyHDF5Dataset

_BUILTIN_VORTICITY_REPRESENTATIONS = frozenset(
    {"vorticity", "bounded_vorticity", "bounded_obstacle_vorticity"}
)


def _canonical_state_representation(value: Any) -> str:
    """Collapse built-in solver-specific vorticity tags to their field family."""

    token = str(value).strip().lower()
    if token in _BUILTIN_VORTICITY_REPRESENTATIONS:
        return "vorticity"
    return token


def find_shards(root: str | Path, pattern: str = "**/*.h5") -> list[Path]:
    directory = Path(root).expanduser()
    if directory.is_file():
        return [directory]
    return sorted(path for path in directory.glob(pattern) if path.is_file())


def _area_downsample(values: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Area-average a channel-last HWC/THWC field by exact integer factors."""

    array = np.asarray(values)
    if array.ndim not in {3, 4}:
        raise ValueError("spatial adaptation expects HWC or THWC arrays")
    source = tuple(int(value) for value in array.shape[-3:-1])
    if source == target:
        return array
    if source[0] % target[0] or source[1] % target[1]:
        raise ValueError(
            f"area downsampling requires integer factors: source={source}, target={target}"
        )
    factors = source[0] // target[0], source[1] // target[1]
    if factors[0] < 1 or factors[1] < 1:
        raise ValueError(f"spatial adaptation may not upsample: source={source}, target={target}")
    prefix = array.shape[:-3]
    reshaped = array.reshape(
        *prefix,
        target[0],
        factors[0],
        target[1],
        factors[1],
        array.shape[-1],
    )
    return np.asarray(reshaped.mean(axis=(-4, -2)), dtype=np.float32)


class BenchmarkDataset(Sequence[dict[str, Any]]):
    """Expose recovery, forward, inverse, or rollout examples from canonical shards.

    Filtering reads only the small metadata column. Arrays remain lazy and each
    DataLoader worker opens its own HDF5 handle.
    """

    def __init__(
        self,
        shards: str | Path | Sequence[str | Path],
        *,
        task: str = "recovery",
        split: str | None = None,
        filters: Mapping[str, Any] | None = None,
        mask: Mapping[str, Any] | None = None,
        target_step: int = -1,
        horizon: int = 8,
        history_steps: int = 1,
        state_representation: str = "native",
        spatial_resolution: int | tuple[int, int] | list[int] | None = None,
        seed: int = 0,
        mask_seed: int | None = None,
        max_samples: int | None = None,
        release_tier: str | int | None = None,
        verify: bool = False,
    ) -> None:
        self.base = LazyHDF5Dataset(shards, verify=verify)
        self.task = task.lower().replace("-", "_")
        if self.task not in {"recovery", "forward", "inverse", "rollout"}:
            raise ValueError("task must be recovery, forward, inverse, or rollout")
        self.target_step = int(target_step)
        self.horizon = int(horizon)
        self.history_steps = int(history_steps)
        if self.horizon < 1 or self.history_steps < 1:
            raise ValueError("horizon and history_steps must be positive")
        self.state_representation = str(state_representation).lower()
        if self.state_representation not in {"native", "velocity", "vorticity"}:
            raise ValueError("state_representation must be native, velocity, or vorticity")
        self.spatial_resolution = (
            normalize_resolution(spatial_resolution) if spatial_resolution is not None else None
        )
        self.seed = int(seed)
        # Masks are drawn from ``mask_seed`` so a controlled multi-seed study can vary the
        # training seed while every sample keeps the same sensors; it defaults to ``seed``.
        self.mask_seed = int(seed if mask_seed is None else mask_seed)
        self.mask_config = dict(mask or {"protocol": "random_3pct"})
        self.filters = dict(filters or {})
        self.release_tier = release_tier
        self.release_tier_size = (
            resolve_tier(release_tier, full_size=2000) if release_tier is not None else None
        )
        if split is not None:
            self.filters["split"] = split

        self.indices: list[int] = []
        self.metadata: list[dict[str, Any]] = []
        sample_ids: set[str] = set()
        signatures: set[tuple[Any, ...]] = set()
        for index, metadata in enumerate(self.base.iter_metadata()):
            if self.release_tier_size is not None and not _in_release_tier(
                metadata, self.release_tier_size
            ):
                continue
            if all(_matches(metadata.get(key), expected) for key, expected in self.filters.items()):
                sample_id = str(metadata.get("sample_id", ""))
                if sample_id and sample_id in sample_ids:
                    raise ValueError(f"duplicate sample_id across selected shards: {sample_id}")
                if sample_id:
                    sample_ids.add(sample_id)
                signatures.add(
                    (
                        tuple(metadata.get("resolution", ())),
                        _canonical_state_representation(
                            metadata.get("state_representation", "scalar")
                        ),
                    )
                )
                self.indices.append(index)
                self.metadata.append(metadata)
                if max_samples is not None and len(self.indices) >= int(max_samples):
                    break
        if not self.indices:
            raise ValueError(f"no samples match filters {self.filters}")
        resolutions = {signature[0] for signature in signatures if signature[0]}
        if len(resolutions) > 1:
            raise ValueError(f"selected shards mix spatial resolutions: {sorted(resolutions)}")
        representations = {
            signature[1] for signature in signatures if signature[1] in {"velocity", "vorticity"}
        }
        if self.state_representation == "native" and len(representations) > 1:
            raise ValueError(
                "selected Navier-Stokes shards mix velocity and vorticity; set "
                "data.state_representation to one canonical representation"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def _mask(
        self,
        target: np.ndarray,
        metadata: MutableMapping[str, Any],
        *,
        dataset_index: int,
    ) -> np.ndarray:
        options = dict(self.mask_config)
        requested = str(options.pop("protocol", "random_3pct")).strip().lower()
        if requested in {"full", "full_observation", "all_visible"}:
            if options:
                raise ValueError("the full-observation mask does not accept extra options")
            spatial = target.shape[:2] if target.ndim <= 3 else target.shape[-3:-1]
            mask = np.ones(spatial, dtype=bool)
            metadata["mask_id"] = "full"
            metadata["mask_seed"] = None
            metadata["observation_count"] = int(mask.size)
            metadata["observation_ratio"] = 1.0
            return mask
        if requested in {"mixed_partial", "balanced_partial", "all_partial"}:
            candidates = tuple(options.pop("protocols", MASK_PROTOCOL_NAMES))
            if options:
                raise ValueError(
                    "the mixed-partial mask accepts only an optional protocols list"
                )
            if not candidates:
                raise ValueError("the mixed-partial mask needs at least one protocol")
            protocols = tuple(MASK_REGISTRY.resolve_name(str(item)) for item in candidates)
            if len(set(protocols)) != len(protocols):
                raise ValueError("the mixed-partial protocol list contains duplicates")
            factor_offset_seed = derive_seed(
                self.mask_seed,
                "mixed_partial_protocol",
                metadata.get("pde", ""),
                metadata.get("boundary", ""),
                metadata.get("setting", ""),
                metadata.get("regime", ""),
            )
            try:
                stratum_position = int(metadata.get("regime_sample_index", dataset_index))
            except (TypeError, ValueError):
                stratum_position = int(dataset_index)
            protocol_position = (stratum_position + int(factor_offset_seed)) % len(protocols)
            protocol = protocols[protocol_position]
            metadata["mask_policy"] = "mixed_partial"
            metadata["mask_protocol_selection_seed"] = int(factor_offset_seed)
            metadata["mask_protocol_stratum_position"] = int(stratum_position)
            metadata["mask_protocol_selection_index"] = int(protocol_position)
            metadata["mask_protocol_candidates"] = list(protocols)
        elif _normalize_mask_name(requested) in {"mixture", "mix", "mixed"} or is_mask_spec(requested):
            return self._spec_mask(target, metadata, dataset_index=dataset_index, requested=requested, options=options)
        else:
            protocol = MASK_REGISTRY.resolve_name(requested)
        mask_seed = derive_seed(
            self.mask_seed,
            "mask",
            metadata.get("sample_id", metadata.get("seed", 0)),
            protocol,
        )
        spatial = target.shape[:2] if target.ndim <= 3 else target.shape[-3:-1]
        mask = generate_mask(protocol, spatial, seed=mask_seed, **options)
        metadata["mask_id"] = protocol
        metadata["mask_seed"] = int(mask_seed)
        metadata["observation_count"] = int(np.count_nonzero(mask))
        metadata["observation_ratio"] = float(np.mean(mask))
        return mask

    def _spec_mask(
        self,
        target: np.ndarray,
        metadata: MutableMapping[str, Any],
        *,
        dataset_index: int,
        requested: str,
        options: dict[str, Any],
    ) -> np.ndarray:
        """Mask from a parameterised protocol or a deterministic mixture (see ``mask_specs``)."""

        if _normalize_mask_name(requested) in {"mixture", "mix", "mixed"}:
            spec = parse_mask_spec({"protocol": "mixture", **options})
        else:
            if options:
                raise ValueError(
                    "a mask spec string takes no extra options; put arguments inside the spec, "
                    "e.g. 'line_sensors(num_lines=64, orientation=horizontal) + uniform 20'"
                )
            spec = parse_mask_spec(requested)
        spatial = target.shape[:2] if target.ndim <= 3 else target.shape[-3:-1]
        try:
            stratum_position = int(metadata.get("regime_sample_index", dataset_index))
        except (TypeError, ValueError):
            stratum_position = int(dataset_index)
        stratum = (
            metadata.get("pde", ""),
            metadata.get("boundary", ""),
            metadata.get("setting", ""),
            metadata.get("regime", ""),
        )
        index, selection_seed, cycle_position = spec.select(
            mask_seed=self.mask_seed, stratum=stratum, position=stratum_position
        )
        component = spec.components[index]
        metadata["mask_policy"] = "mixture" if spec.is_mixture else "parameterized"
        metadata["mask_spec"] = spec.canonical_text()
        metadata["mask_view_id"] = spec.view_id
        metadata["mask_component_id"] = component.base_id
        metadata["mask_component"] = component.canonical_text()
        if spec.is_mixture:
            metadata["mask_mixture_selection_seed"] = int(selection_seed)
            metadata["mask_mixture_stratum_position"] = int(stratum_position)
            metadata["mask_mixture_selection_index"] = int(cycle_position)
            metadata["mask_mixture_candidates"] = [spec.components[i].base_id for i in spec.cycle]
        if component.protocol == "full":
            mask = np.ones(spatial, dtype=bool)
            metadata["mask_id"] = "full"
            metadata["mask_seed"] = None
        else:
            # Same derivation as a fixed view: the resolved protocol name, never its arguments,
            # so "uniform 20" reproduces the legacy {protocol: random, ratio: 0.2} mask bit-for-bit.
            mask_seed = derive_seed(
                self.mask_seed,
                "mask",
                metadata.get("sample_id", metadata.get("seed", 0)),
                component.protocol,
            )
            mask = component.generate(spatial, mask_seed)
            metadata["mask_id"] = component.protocol
            metadata["mask_seed"] = int(mask_seed)
        metadata["observation_count"] = int(np.count_nonzero(mask))
        metadata["observation_ratio"] = float(np.mean(mask))
        return mask

    def _state_view(
        self,
        condition: np.ndarray,
        trajectory: np.ndarray,
        metadata: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        if metadata.get("pde") != "navier_stokes" or self.state_representation == "native":
            return condition, trajectory
        native_tag = str(metadata.get("state_representation", "native")).strip().lower()
        native = _canonical_state_representation(native_tag)
        if native == self.state_representation:
            if native_tag != self.state_representation:
                metadata["native_state_representation"] = native_tag
                metadata["state_representation"] = self.state_representation
            return condition, trajectory

        height, width = trajectory.shape[1:3]
        dy, dx = 1.0 / height, 1.0 / width

        def to_vorticity(values: np.ndarray) -> np.ndarray:
            u, v = values[..., 0], values[..., 1]
            return (np.gradient(v, dx, axis=-1) - np.gradient(u, dy, axis=-2))[..., None]

        def to_velocity(values: np.ndarray) -> np.ndarray:
            from .pdes.common import stream_velocity

            frames = values if values.ndim == 4 else values[None]
            converted = []
            for frame in frames:
                velocity_x, velocity_y = stream_velocity(frame[..., 0], dx, dy)
                converted.append(np.stack((velocity_x, velocity_y), axis=-1))
            result = np.stack(converted)
            return result if values.ndim == 4 else result[0]

        if native == "velocity" and self.state_representation == "vorticity":
            converted_condition = to_vorticity(condition)
            converted_trajectory = to_vorticity(trajectory)
        elif (
            native == "vorticity"
            and native_tag == "vorticity"
            and self.state_representation == "velocity"
        ):
            converted_condition = to_velocity(condition)
            converted_trajectory = to_velocity(trajectory)
        else:
            raise ValueError(
                f"cannot convert Navier-Stokes representation {native_tag!r} to "
                f"{self.state_representation!r}"
            )
        metadata["native_state_representation"] = native_tag
        metadata["state_representation"] = self.state_representation
        return converted_condition.astype(np.float32), converted_trajectory.astype(np.float32)

    def _spatial_view(
        self,
        condition: np.ndarray,
        trajectory: np.ndarray,
        geometry: np.ndarray,
        metadata: MutableMapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.spatial_resolution is None:
            return condition, trajectory, geometry
        source = tuple(int(value) for value in trajectory.shape[1:3])
        target = self.spatial_resolution
        if source != target:
            condition = _area_downsample(condition, target)
            trajectory = _area_downsample(trajectory, target)
            geometry = _area_downsample(geometry, target)
        metadata["native_resolution"] = list(source)
        metadata["resolution"] = list(target)
        metadata["spatial_adapter"] = "deterministic_area_average_integer_factor_v1"
        return condition, trajectory, geometry

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sample = self.base[self.indices[index]]
        metadata = dict(sample.metadata)
        condition, trajectory = self._state_view(
            np.asarray(sample.condition), np.asarray(sample.trajectory), metadata
        )
        condition, trajectory, geometry = self._spatial_view(
            condition,
            trajectory,
            np.asarray(sample.geometry),
            metadata,
        )

        if self.task == "recovery":
            target = np.asarray(trajectory[self.target_step])
            mask = self._mask(target, metadata, dataset_index=index)
            return {
                "observations": apply_mask(target, mask).astype(np.float32, copy=False),
                "mask": mask[..., None].astype(np.float32),
                "target": target.astype(np.float32, copy=False),
                # Physics-informed objectives may use the stored coefficient or
                # source as declared training context. Ordinary recovery models
                # and every test-time forward call continue to receive only the
                # observation, mask, and geometry.
                "pde_condition": np.asarray(condition, dtype=np.float32),
                "geometry": np.asarray(geometry, dtype=np.float32),
                "metadata": metadata,
            }

        if self.task == "forward":
            target = np.asarray(trajectory[self.target_step], dtype=np.float32)
            condition = np.asarray(condition, dtype=np.float32)
            mask = self._mask(condition, metadata, dataset_index=index)
            return {
                "observations": apply_mask(condition, mask).astype(np.float32, copy=False),
                "mask": mask[..., None].astype(np.float32),
                "target": target,
                # PINO may use the complete PDE coefficient/source only inside
                # its declared training loss. It is never passed to the model.
                "pde_condition": condition,
                "geometry": np.asarray(geometry, dtype=np.float32),
                "metadata": metadata,
            }

        if self.task == "inverse":
            observed_state = np.asarray(trajectory[self.target_step], dtype=np.float32)
            target = np.asarray(condition, dtype=np.float32)
            mask = self._mask(observed_state, metadata, dataset_index=index)
            return {
                "observations": apply_mask(observed_state, mask).astype(np.float32, copy=False),
                "mask": mask[..., None].astype(np.float32),
                "target": target,
                "geometry": np.asarray(geometry, dtype=np.float32),
                "metadata": metadata,
            }

        required_steps = self.history_steps + self.horizon
        if trajectory.shape[0] < required_steps:
            raise ValueError(
                f"sample {metadata.get('sample_id')} has {trajectory.shape[0]} trajectory "
                f"steps, but rollout requires {self.history_steps} history + "
                f"{self.horizon} future steps"
            )
        history = self.history_steps
        horizon = self.horizon
        history_values = np.asarray(trajectory[:history], dtype=np.float32)
        target = np.asarray(trajectory[history : history + horizon], dtype=np.float32)
        mask = self._mask(history_values, metadata, dataset_index=index)
        return {
            "observations": apply_mask(history_values, mask).astype(np.float32, copy=False),
            "mask": np.broadcast_to(mask[None, ..., None], (history, *mask.shape, 1)).astype(
                np.float32
            ),
            "target": target,
            # Training-only access for classical reduced-order fitting and
            # audit receipts. Learned rollout models receive `observations`
            # only, so the complete initial state cannot become teacher forcing.
            "initial_state": history_values,
            "geometry": np.asarray(geometry, dtype=np.float32),
            "metadata": metadata,
        }

    def close(self) -> None:
        self.base.close()

    def __getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, (list, tuple, set, frozenset)):
        return value in expected
    return value == expected


def _in_release_tier(metadata: Mapping[str, Any], tier_size: int) -> bool:
    """Reconstruct the canonical nested-tier rank stored implicitly by old shards.

    Full-tier production records predate a dedicated ``tier_rank`` metadata
    column. Generation nevertheless used the authoritative round-robin order
    from :func:`build_split_plan`: low[0], medium[0], high[0], low[1], ... .
    Deriving the rank from the persisted regime pair lets medium experiments
    reuse the immutable full release without copying or regenerating data.
    """

    regime = str(metadata.get("regime", ""))
    if regime not in REGIMES:
        raise ValueError(
            "release_tier selection requires canonical regime metadata; "
            f"found {regime!r}"
        )
    try:
        regime_index = int(metadata["regime_sample_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "release_tier selection requires integer regime_sample_index metadata"
        ) from exc
    if regime_index < 0:
        raise ValueError("regime_sample_index must be non-negative")
    tier_rank = len(REGIMES) * regime_index + REGIMES.index(regime)
    return tier_rank < tier_size


def collate_benchmark(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot collate an empty batch")
    keys = {key for row in rows for key in row if key != "metadata"}
    batch: dict[str, Any] = {
        key: np.stack([np.asarray(row[key]) for row in rows]) for key in sorted(keys)
    }
    batch["metadata"] = [dict(row.get("metadata", {})) for row in rows]
    return batch
