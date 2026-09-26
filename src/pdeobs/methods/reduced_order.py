"""Leakage-free reduced-order baselines for partial-field recovery."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..masks import MASK_PROTOCOL_NAMES, apply_mask, generate_mask
from ..schema import derive_seed
from .base import MethodCapabilities, register_method
from .interpolation import _as_bhwc, _restore

_GAPPY_CAPABILITIES = MethodCapabilities(
    tasks=frozenset({"recovery"}),
    trainable=True,
    requires_mask=True,
    reference_only=True,
    notes=(
        "Paper-guided Gappy POD with a declared streaming randomized-SVD adapter; "
        "not an exact reproduction of one application-specific POD experiment."
    ),
)

_GAPPY_DMD_CAPABILITIES = MethodCapabilities(
    tasks=frozenset({"rollout"}),
    trainable=True,
    temporal=True,
    requires_mask=True,
    reference_only=True,
    notes=(
        "Training-split-only Gappy POD reconstruction at t0 followed by a "
        "linear DMD coefficient transition and prediction-only rollout."
    ),
)


def _field_batches(dataset: Sequence[Mapping[str, Any]], batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(dataset), batch_size):
        rows = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        yield np.stack([np.asarray(row["target"], dtype=np.float32) for row in rows])


@register_method("gappy_pod", aliases=("gappy-pod", "pod"))
class GappyPOD:
    """Gappy POD reconstruction with a basis fitted only on complete train fields.

    Large datasets use a three-pass randomized range finder: one pass computes
    the mean, one builds the sample-space sketch, and one forms the small
    projected matrix. Memory is ``O(N r + r D)`` rather than ``O(N D)`` for
    ``N`` fields, ``D`` cells, and requested rank ``r``.
    """

    name = "gappy_pod"
    capabilities = _GAPPY_CAPABILITIES

    def __init__(
        self,
        *,
        rank: int | None = None,
        rank_candidates: Sequence[int] = (8, 16, 32, 64, 128),
        ridge: float = 1.0e-6,
        oversampling: int = 16,
        fit_batch_size: int = 64,
        validation_samples_per_stratum: int = 3,
        validation_protocols: Sequence[str] = MASK_PROTOCOL_NAMES,
        seed: int = 0,
        state_path: str | Path | None = None,
    ) -> None:
        candidates = tuple(sorted({int(value) for value in rank_candidates}))
        if not candidates or candidates[0] < 1:
            raise ValueError("rank_candidates must contain positive integers")
        if rank is not None and int(rank) < 1:
            raise ValueError("rank must be positive")
        if ridge < 0.0 or not np.isfinite(ridge):
            raise ValueError("ridge must be finite and non-negative")
        if oversampling < 0 or fit_batch_size < 1 or validation_samples_per_stratum < 1:
            raise ValueError("oversampling must be non-negative and sample counts positive")
        self.rank = int(rank) if rank is not None else None
        self.rank_candidates = candidates
        self.ridge = float(ridge)
        self.oversampling = int(oversampling)
        self.fit_batch_size = int(fit_batch_size)
        self.validation_samples_per_stratum = int(validation_samples_per_stratum)
        self.validation_protocols = tuple(str(value) for value in validation_protocols)
        if not self.validation_protocols:
            raise ValueError("validation_protocols must not be empty")
        self.seed = int(seed)
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.field_shape_: tuple[int, ...] | None = None
        self.fit_receipt_: dict[str, Any] = {}
        if state_path is not None:
            self.load(state_path)

    @property
    def selected_rank(self) -> int:
        if self.components_ is None:
            raise RuntimeError("GappyPOD has not been fitted")
        return min(self.rank or self.components_.shape[0], self.components_.shape[0])

    def _fit_factory(self, factory: Callable[[], Iterable[np.ndarray]]) -> None:
        count = 0
        total: np.ndarray | None = None
        field_shape: tuple[int, ...] | None = None
        for batch in factory():
            values = np.asarray(batch, dtype=np.float32)
            if values.ndim < 3:
                raise ValueError("POD fields need batch plus spatial dimensions")
            shape = tuple(int(value) for value in values.shape[1:])
            if field_shape is None:
                field_shape = shape
                total = np.zeros(int(np.prod(shape)), dtype=np.float64)
            elif shape != field_shape:
                raise ValueError(f"POD fields mix shapes: {field_shape} and {shape}")
            flattened = values.reshape(len(values), -1)
            assert total is not None
            total += flattened.sum(axis=0, dtype=np.float64)
            count += len(values)
        if count < 2 or total is None or field_shape is None:
            raise ValueError("GappyPOD needs at least two training fields")
        mean = total / count
        dimension = len(mean)
        requested = max(self.rank_candidates + ((self.rank,) if self.rank is not None else ()))
        sketch_rank = min(requested + self.oversampling, count, dimension)
        rng = np.random.default_rng(self.seed)
        projection = rng.standard_normal((dimension, sketch_rank), dtype=np.float32)

        sketches: list[np.ndarray] = []
        for batch in factory():
            centered = np.asarray(batch, dtype=np.float32).reshape(len(batch), -1)
            centered = centered - mean.astype(np.float32, copy=False)
            sketches.append(centered @ projection)
        sample_sketch = np.concatenate(sketches, axis=0)
        q, _ = np.linalg.qr(sample_sketch, mode="reduced")

        projected = np.zeros((q.shape[1], dimension), dtype=np.float64)
        offset = 0
        for batch in factory():
            centered = np.asarray(batch, dtype=np.float32).reshape(len(batch), -1)
            centered = centered - mean.astype(np.float32, copy=False)
            stop = offset + len(centered)
            projected += q[offset:stop].T.astype(np.float64) @ centered.astype(np.float64)
            offset = stop
        if offset != count:
            raise RuntimeError("POD streaming passes produced different sample counts")
        _, singular_values, components = np.linalg.svd(projected, full_matrices=False)
        kept = min(requested, len(singular_values))
        self.mean_ = mean.astype(np.float32)
        self.components_ = components[:kept].astype(np.float32)
        self.singular_values_ = singular_values[:kept].astype(np.float64)
        self.field_shape_ = field_shape
        self.fit_receipt_ = {
            "algorithm": "three_pass_randomized_svd_v1",
            "training_samples": count,
            "field_shape": list(field_shape),
            "sketch_rank": int(sketch_rank),
            "available_rank": int(kept),
            "seed": self.seed,
        }

    def fit_arrays(self, fields: np.ndarray) -> GappyPOD:
        values = np.asarray(fields, dtype=np.float32)
        if values.ndim < 3:
            raise ValueError("fields need batch plus spatial dimensions")

        def factory() -> Iterable[np.ndarray]:
            for start in range(0, len(values), self.fit_batch_size):
                yield values[start : start + self.fit_batch_size]

        self._fit_factory(factory)
        if self.rank is None:
            self.rank = min(self.rank_candidates[0], self.components_.shape[0])
        return self

    def _predict_bhwc(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        geometry: np.ndarray | None,
        *,
        rank: int,
    ) -> np.ndarray:
        if self.mean_ is None or self.components_ is None or self.field_shape_ is None:
            raise RuntimeError("GappyPOD has not been fitted or loaded")
        if tuple(values.shape[1:]) != self.field_shape_:
            raise ValueError(
                f"GappyPOD expected field shape {self.field_shape_}, got {values.shape[1:]}"
            )
        basis = self.components_[:rank].T.astype(np.float64)
        mean = self.mean_.astype(np.float64)
        result = np.empty_like(values, dtype=np.float64)
        geometry_values = None if geometry is None else np.asarray(geometry)
        for batch_index in range(len(values)):
            observed = mask[batch_index].reshape(-1).astype(bool)
            if geometry_values is not None:
                valid_geometry = geometry_values[batch_index]
                if valid_geometry.shape[-1] == 1 and values.shape[-1] != 1:
                    valid_geometry = np.broadcast_to(valid_geometry, values.shape[1:])
                observed &= valid_geometry.reshape(-1) < 0.5
            if not np.any(observed):
                reconstructed = mean.copy()
            else:
                design = basis[observed]
                rhs = values[batch_index].reshape(-1)[observed].astype(np.float64) - mean[observed]
                gram = design.T @ design
                if self.ridge:
                    scale = max(float(np.trace(gram)) / max(rank, 1), 1.0)
                    gram.flat[:: rank + 1] += self.ridge * scale
                try:
                    coefficients = np.linalg.solve(gram, design.T @ rhs)
                except np.linalg.LinAlgError:
                    coefficients = np.linalg.lstsq(design, rhs, rcond=1.0e-8)[0]
                reconstructed = mean + basis @ coefficients
            result[batch_index] = reconstructed.reshape(self.field_shape_)
        return result.astype(np.float32)

    def predict(
        self,
        observations: Any,
        mask: Any | None = None,
        *,
        geometry: Any | None = None,
        **_: Any,
    ) -> np.ndarray:
        values, visible, original = _as_bhwc(observations, mask)
        geometry_bhwc = None
        if geometry is not None:
            geometry_bhwc, _, _ = _as_bhwc(geometry, None)
        reconstructed = self._predict_bhwc(
            values,
            visible,
            geometry_bhwc,
            rank=self.selected_rank,
        )
        return _restore(reconstructed, original)

    def _validation_indices(self, dataset: Sequence[Mapping[str, Any]]) -> list[int]:
        counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
        selected: list[int] = []
        for index in range(len(dataset)):
            metadata = getattr(dataset, "metadata", [{}] * len(dataset))[index]
            key = tuple(str(metadata.get(field, "")) for field in ("boundary", "setting", "regime"))
            if counts[key] < self.validation_samples_per_stratum:
                selected.append(index)
                counts[key] += 1
        return selected

    def select_rank(self, validation_dataset: Sequence[Mapping[str, Any]]) -> dict[int, float]:
        if self.components_ is None:
            raise RuntimeError("fit the POD basis before selecting a rank")
        candidates = tuple(rank for rank in self.rank_candidates if rank <= len(self.components_))
        if not candidates:
            candidates = (len(self.components_),)
        sums = {rank: 0.0 for rank in candidates}
        counts = {rank: 0 for rank in candidates}
        indices = self._validation_indices(validation_dataset)
        for index in indices:
            row = validation_dataset[index]
            target = np.asarray(row["target"], dtype=np.float32)
            metadata = dict(row.get("metadata", {}))
            geometry = np.asarray(row.get("geometry", np.zeros(target.shape[:-1] + (1,))))
            valid = geometry[..., 0] < 0.5
            for protocol in self.validation_protocols:
                mask_seed = derive_seed(
                    self.seed,
                    "mask",
                    metadata.get("sample_id", metadata.get("seed", 0)),
                    protocol,
                )
                mask = generate_mask(protocol, target.shape[:2], seed=mask_seed)
                observed = apply_mask(target, mask)[None]
                visible = mask[None, ..., None]
                for rank in candidates:
                    prediction = self._predict_bhwc(
                        observed,
                        visible,
                        geometry[None],
                        rank=rank,
                    )[0]
                    difference = (prediction - target)[valid]
                    reference = target[valid]
                    relative = float(
                        np.linalg.norm(difference.reshape(-1))
                        / max(np.linalg.norm(reference.reshape(-1)), np.finfo(float).eps)
                    )
                    sums[rank] += relative
                    counts[rank] += 1
        scores = {rank: sums[rank] / counts[rank] for rank in candidates if counts[rank]}
        if not scores:
            raise ValueError("validation rank selection produced no scored examples")
        self.rank = min(scores, key=lambda value: (scores[value], value))
        self.fit_receipt_["rank_selection"] = {
            "split": "validation",
            "samples": len(indices),
            "protocols": list(self.validation_protocols),
            "mean_relative_l2_by_rank": {str(key): value for key, value in scores.items()},
            "selected_rank": self.rank,
        }
        return scores

    def fit_dataset(
        self,
        training_dataset: Sequence[Mapping[str, Any]],
        validation_dataset: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self._fit_factory(lambda: _field_batches(training_dataset, self.fit_batch_size))
        if self.rank is None:
            if validation_dataset is None:
                raise ValueError("validation data is required for preregistered POD rank selection")
            self.select_rank(validation_dataset)
        self.fit_receipt_["selected_rank"] = self.selected_rank
        return dict(self.fit_receipt_)

    def save(self, path: str | Path) -> Path:
        if self.mean_ is None or self.components_ is None or self.singular_values_ is None:
            raise RuntimeError("cannot save an unfitted GappyPOD")
        destination = Path(path)
        if destination.suffix.lower() != ".npz":
            raise ValueError("GappyPOD checkpoints must use the .npz extension")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".npz.partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                mean=self.mean_,
                components=self.components_,
                singular_values=self.singular_values_,
                field_shape=np.asarray(self.field_shape_, dtype=np.int64),
                selected_rank=np.asarray(self.selected_rank, dtype=np.int64),
                ridge=np.asarray(self.ridge, dtype=np.float64),
                receipt=np.asarray(json.dumps(self.fit_receipt_, sort_keys=True)),
            )
        temporary.replace(destination)
        return destination

    def load(self, path: str | Path) -> GappyPOD:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"GappyPOD checkpoint does not exist: {source}")
        with np.load(source, allow_pickle=False) as payload:
            self.mean_ = np.asarray(payload["mean"], dtype=np.float32)
            self.components_ = np.asarray(payload["components"], dtype=np.float32)
            self.singular_values_ = np.asarray(payload["singular_values"], dtype=np.float64)
            self.field_shape_ = tuple(int(value) for value in payload["field_shape"])
            self.rank = int(payload["selected_rank"])
            if "ridge" in payload:
                self.ridge = float(payload["ridge"])
            self.fit_receipt_ = json.loads(str(payload["receipt"]))
        if self.mean_.size != int(np.prod(self.field_shape_)):
            raise ValueError("GappyPOD checkpoint mean and field shape disagree")
        if self.components_.shape[1] != self.mean_.size:
            raise ValueError("GappyPOD checkpoint components and mean disagree")
        return self


@register_method("gappy_pod_dmd", aliases=("gappy-dmd", "pod_dmd"), hidden=True)
class GappyPODDMD(GappyPOD):
    """Gappy POD at sparse ``t0`` plus a training-only linear DMD rollout.

    The POD basis and transition are fitted exclusively from complete states in
    the declared training split.  At inference, coefficients are inferred only
    from sparse ``t0`` observations.  The transition then advances those
    coefficients freely, so no future target frame can enter the recurrence.
    """

    name = "gappy_pod_dmd"
    capabilities = _GAPPY_DMD_CAPABILITIES

    def __init__(
        self,
        *,
        rank: int = 64,
        ridge: float = 1.0e-6,
        oversampling: int = 16,
        fit_batch_size: int = 64,
        seed: int = 0,
        state_path: str | Path | None = None,
    ) -> None:
        self.transition_: np.ndarray | None = None
        super().__init__(
            rank=rank,
            rank_candidates=(rank,),
            ridge=ridge,
            oversampling=oversampling,
            fit_batch_size=fit_batch_size,
            seed=seed,
        )
        if state_path is not None:
            self.load(state_path)

    @staticmethod
    def _sequence(row: Mapping[str, Any]) -> np.ndarray:
        if "initial_state" not in row:
            raise ValueError("GappyPODDMD training rows require complete initial_state")
        initial = np.asarray(row["initial_state"], dtype=np.float32)
        future = np.asarray(row["target"], dtype=np.float32)
        if initial.ndim != 4 or future.ndim != 4:
            raise ValueError("GappyPODDMD states must have THWC layout")
        if initial.shape[1:] != future.shape[1:] or len(initial) != 1 or len(future) < 1:
            raise ValueError("GappyPODDMD requires one t0 frame and aligned future frames")
        sequence = np.concatenate((initial, future), axis=0)
        if not np.isfinite(sequence).all():
            raise FloatingPointError("GappyPODDMD training states contain non-finite values")
        return sequence

    def _state_factory(
        self, dataset: Sequence[Mapping[str, Any]]
    ) -> Callable[[], Iterable[np.ndarray]]:
        def factory() -> Iterable[np.ndarray]:
            buffered: list[np.ndarray] = []
            buffered_states = 0
            for index in range(len(dataset)):
                sequence = self._sequence(dataset[index])
                buffered.append(sequence)
                buffered_states += len(sequence)
                if buffered_states >= self.fit_batch_size:
                    yield np.concatenate(buffered, axis=0)
                    buffered.clear()
                    buffered_states = 0
            if buffered:
                yield np.concatenate(buffered, axis=0)

        return factory

    def fit_dataset(
        self,
        training_dataset: Sequence[Mapping[str, Any]],
        validation_dataset: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if validation_dataset is not None:
            raise ValueError("GappyPODDMD uses no validation data")
        self._fit_factory(self._state_factory(training_dataset))
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("GappyPODDMD basis fit did not produce a state")
        rank = self.selected_rank
        basis = self.components_[:rank].astype(np.float64)
        mean = self.mean_.astype(np.float64)
        gram = np.zeros((rank, rank), dtype=np.float64)
        cross = np.zeros((rank, rank), dtype=np.float64)
        pairs = 0
        for index in range(len(training_dataset)):
            sequence = self._sequence(training_dataset[index]).reshape(-1, mean.size)
            coefficients = (sequence.astype(np.float64) - mean) @ basis.T
            previous, following = coefficients[:-1], coefficients[1:]
            gram += previous.T @ previous
            cross += previous.T @ following
            pairs += len(previous)
        if pairs < 1:
            raise ValueError("GappyPODDMD fit produced no consecutive training pairs")
        if self.ridge:
            scale = max(float(np.trace(gram)) / max(rank, 1), 1.0)
            gram.flat[:: rank + 1] += self.ridge * scale
        try:
            transition = np.linalg.solve(gram, cross)
        except np.linalg.LinAlgError:
            transition = np.linalg.lstsq(gram, cross, rcond=1.0e-10)[0]
        if not np.isfinite(transition).all():
            raise FloatingPointError("GappyPODDMD transition is non-finite")
        self.transition_ = transition.astype(np.float64)
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
        self.fit_receipt_.update(
            {
                "temporal_adapter": "gappy_t0_plus_linear_dmd_free_rollout_v1",
                "selected_rank": rank,
                "training_trajectories": len(training_dataset),
                "consecutive_training_pairs": pairs,
                "transition_spectral_radius": spectral_radius,
                "validation_samples": 0,
                "future_truth_at_inference": False,
            }
        )
        return dict(self.fit_receipt_)

    def predict(
        self,
        observations: Any,
        mask: Any | None = None,
        *,
        geometry: Any | None = None,
        horizon: int = 1,
        **_: Any,
    ) -> np.ndarray:
        if self.transition_ is None or self.mean_ is None or self.components_ is None:
            raise RuntimeError("GappyPODDMD has not been fitted or loaded")
        values = np.asarray(observations)
        if values.ndim != 5 or values.shape[1] < 1:
            raise ValueError("GappyPODDMD observations must have shape BTHWC")
        if horizon < 1:
            raise ValueError("GappyPODDMD horizon must be positive")
        visible = np.isfinite(values) if mask is None else np.asarray(mask, dtype=bool)
        try:
            visible = np.broadcast_to(visible, values.shape)
        except ValueError as exc:
            raise ValueError("GappyPODDMD mask is not broadcastable to BTHWC") from exc
        initial = np.asarray(values[:, -1], dtype=np.float32)
        initial_mask = np.asarray(visible[:, -1], dtype=bool)
        geometry_bhwc = None
        if geometry is not None:
            geometry_bhwc, _, _ = _as_bhwc(geometry, None)
        reconstructed = self._predict_bhwc(
            initial,
            initial_mask,
            geometry_bhwc,
            rank=self.selected_rank,
        )
        mean = self.mean_.astype(np.float64)
        basis = self.components_[: self.selected_rank].astype(np.float64)
        coefficients = (reconstructed.reshape(len(reconstructed), -1) - mean) @ basis.T
        predictions: list[np.ndarray] = []
        for _step in range(int(horizon)):
            coefficients = coefficients @ self.transition_
            state = mean + coefficients @ basis
            predictions.append(state.reshape((len(reconstructed), *self.field_shape_)))
        result = np.stack(predictions, axis=1).astype(np.float32)
        if not np.isfinite(result).all():
            raise FloatingPointError("GappyPODDMD rollout is non-finite")
        return result

    def save(self, path: str | Path) -> Path:
        if (
            self.mean_ is None
            or self.components_ is None
            or self.singular_values_ is None
            or self.transition_ is None
        ):
            raise RuntimeError("cannot save an unfitted GappyPODDMD")
        destination = Path(path)
        if destination.suffix.lower() != ".npz":
            raise ValueError("GappyPODDMD checkpoints must use the .npz extension")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".npz.partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                mean=self.mean_,
                components=self.components_,
                singular_values=self.singular_values_,
                transition=self.transition_,
                field_shape=np.asarray(self.field_shape_, dtype=np.int64),
                selected_rank=np.asarray(self.selected_rank, dtype=np.int64),
                ridge=np.asarray(self.ridge, dtype=np.float64),
                receipt=np.asarray(json.dumps(self.fit_receipt_, sort_keys=True)),
            )
        temporary.replace(destination)
        return destination

    def load(self, path: str | Path) -> GappyPODDMD:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"GappyPODDMD checkpoint does not exist: {source}")
        with np.load(source, allow_pickle=False) as payload:
            self.mean_ = np.asarray(payload["mean"], dtype=np.float32)
            self.components_ = np.asarray(payload["components"], dtype=np.float32)
            self.singular_values_ = np.asarray(payload["singular_values"], dtype=np.float64)
            self.transition_ = np.asarray(payload["transition"], dtype=np.float64)
            self.field_shape_ = tuple(int(value) for value in payload["field_shape"])
            self.rank = int(payload["selected_rank"])
            if "ridge" in payload:
                self.ridge = float(payload["ridge"])
            self.fit_receipt_ = json.loads(str(payload["receipt"]))
        dimension = int(np.prod(self.field_shape_))
        if self.mean_.size != dimension or self.components_.shape[1] != dimension:
            raise ValueError("GappyPODDMD checkpoint basis shape is invalid")
        if self.transition_.shape != (self.selected_rank, self.selected_rank):
            raise ValueError("GappyPODDMD checkpoint transition shape is invalid")
        return self


__all__ = ["GappyPOD", "GappyPODDMD"]
