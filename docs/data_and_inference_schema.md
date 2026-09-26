# Data and inference schema (v0.2.0)

The easy API keeps three resource kinds strictly apart. None is ever substituted for another.

| resource | what it is | how it is produced/read |
|---|---|---|
| **numerical solver** | complete physical record from physical conditions | `api.create_dataset(source="generate")` (deterministic condition generators) or `create_dataset(source="arrays")` (your own source/coefficient/initial arrays through the existing kernels) |
| **dataset source** | records that already exist | `api.load_dataset(source="local"|"manifest")` — no re-solving |
| **trained model artifact** | learned weights + structure | `api.load_predictor(<artifact>)` — no re-training |

There is no `solver=pretrained`. A registered download is only reported as available when a verified
public endpoint exists; otherwise `load_dataset(source="download")` fails explicitly rather than
regenerating data under a download label. Random weights are never presented as a pretrained predictor.

## DatasetHandle

`create_dataset` / `load_dataset` return a `DatasetHandle` (also written to `dataset.json`) with:
`path`, `shards`, `sample_count`, `pde/boundary/setting/regime`, `resolution`, `time_steps`,
`condition_channels`, `state_channels`, `state_representation`, `stored_frame_indices`,
`stored_time_values`, `supported_tasks`, `solver_version`, `pdeobs_version`, `source`,
`sample_ids_sha256`. Static families are `time_steps=1`; temporal families require an explicit
`time_steps >= 2`. Generation never silently re-samples: a short count is an error.

Canonical records are HDF5 (channels-last `condition (N,H,W,Cc)`, `trajectory (N,T,H,W,Cs)`,
`geometry (N,H,W,1)`, per-sample JSON metadata). `create_dataset(format=...)` accepts only `hdf5`;
an undeclared format is rejected, not coerced.

## InferenceInput (observation-only package)

Target-free inference reads an `.npz` package with a declared schema
(`pdeobs-inference-input/v1`, `layout=channels_last`, `mask 1=observed`, `geometry 1=solid`):

- required: `observations` `(N,H,W,C)` static or `(N,T_h,H,W,C)` rollout history; `mask` matching with a
  single trailing channel; `schema` (declared version + layout).
- optional: `geometry` `(N,H,W,1)`, `sample_ids`, `time_indices` (source frame indices of the history),
  `physical_times`, `condition` (only for PINO's physics loss; never used at inference).

A bare array without the schema entry, an undeclared key, or a layout that would have to be guessed
from sizes is rejected. `observation="stored"` uses the mask inside the package and never generates a
new one (which would hide the user's real sensor positions).

Two ways to obtain inputs:
- **benchmark mode**: `api.inference_input_from_dataset(handle, observation, task=...)` builds the
  package from complete records and returns the hidden targets *to the caller* (for a later
  `evaluate`); the predictor never sees them.
- **real mode**: build `api.InferenceInput(observations=..., mask=..., geometry=...)` from your own
  sensors, save it, and predict with `observation="stored"`.

## PredictionBundle

`api.predict` returns a `PredictionBundle` (`pdeobs-prediction-bundle/v1`): channels-last
`predictions`, `sample_ids`, `time_indices` (source frame indices of the predicted frames), `task`,
and a `provenance` record (`targets_used=false`, model config, observation, input hash, executable vs
evaluated range, warnings). Saving to `.npz`/`.h5` never overwrites an existing file.

`evaluate` scores a bundle against an **explicit** independent target with the frozen
`pdeobs-strict-v1` scorer: identities, shapes and time indices must align exactly; missing, extra,
repeated or non-finite entries invalidate the score instead of shrinking the denominator. Recovery
scoring additionally requires the observation and mask so `observation[mask] == target[mask]` holds.
No target ⇒ no supervised metric; `predict` still works.

## Time coordinates

`source-frame index`, `stored-array index` and `physical time` are kept distinct. The handle carries
`stored_frame_indices`/`stored_time_values`; prediction time indices are the source frame indices of
the predicted frames, never `[1,2,3]` forced onto every record. If a record stores thinned frames,
the real coordinates are preserved.
