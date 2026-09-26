# Checkpoint and compatibility reference (v0.2.0)

The v0.2.0 API is a facade over the frozen v0.1.1 science. It adds no numerical algorithm, changes no
loss, mask, split or scoring definition, and does not rescore historical results.

## Model artifacts (`pdeobs-model-artifact/v1`)

`api.train` writes an artifact directory:

```
model/
  model.json              manifest: architecture (name, kwargs, preset, method config, parameter count),
                          task, io_schema, versions (pdeobs, numerical kernel), weight sha256, kind
  weights.pt              plain state_dict (torch.load(weights_only=True) safe); or
  fitted_state.npz        for fitted classical methods (e.g. Gappy POD)
  training_contract.json  data identity, observation, loss, budget, physics contract, hashes, seed
  resolved_config.json    the full resolved experiment configuration
  checkpoints/last.pt     the Trainer checkpoint (format_version 2, carries the training config)
```

`api.load_predictor` restores the exact structure from the manifest. If you also pass a structure and
it disagrees with the manifest, the load is refused (no silent `strict=False` drop of parameters).
Missing weights are an error; `allow_untrained=True` builds the structure without weights and is
labelled a development mode — such a predictor cannot run inference.

Weights are verified against the recorded sha256 before loading, and only safe weights-only
checkpoints are accepted (arbitrary-pickle checkpoints are refused, as in v0.1.1).

`kind` is `smoke_or_example_checkpoint` for short/`max_steps`/`<500`-epoch runs and
`trained_checkpoint` otherwise; these software artifacts are never labelled benchmark pretrained
models.

## Legacy checkpoints

`api.load_legacy_checkpoint(checkpoint, model=..., task=..., params=...)` adapts a v0.1.x
`last.pt`/`best.pt` that has no `model.json`. The structure is supplied by the caller, never guessed
from the file name; the checkpoint's own stored training config (format_version 2) is cross-checked
for task and, for PINO, physics contract; missing fields (observation, data identity, normalization,
budget) are recorded as `unknown_fields` in the returned provenance rather than invented.

## What is preserved (hard constraints)

- numerical kernel version (`__solver_version__`) and generation identity — a packaging/API bump never
  relabels solver provenance;
- canonical HDF5 records, mask arrays and seeds, `stable_split` (1800/200, no validation);
- model structure, input layout (mask channel; FNO/UFNO `[0,1]`, CNO `[-1,1]` coords), normalization;
- relative-L2 loss, teacher forcing = 0, PINO data/physics weighting and fail-closed physics contract;
- the strict scorer (`pdeobs-strict-v1`) and the run/seed/attempt result system;
- the frozen 521 portable tests and 37 updater tests (new tests add coverage; none are removed).

## Equivalence guarantee

New API predictions equal the underlying `runner`/`evaluate_model` path for the same weights and
device (test `test_old_and_new_entrypoint_equivalence`). Across CPU/CUDA a small declared numerical
tolerance applies rather than bit-exactness.

## Overrides create new identities

When you override the paper structure, observation or budget, the run is labelled `custom` (or
`smoke`) and gets its own identity/hashes; it never reuses a paper row identifier. `paper` presets
and the `paper` split reproduce the frozen structure exactly.

## Explicitly not implemented this round

- `fixed200` / `min120` / `recovered` historical cohorts — the paper-row adapter still only executes
  `original500` and the reduced demo; custom short training is available but is not those cohorts;
- exact historical continued training without sufficient stored state;
- a full optional-backend / minimum-version / platform matrix, or GPU acceptance as a paper result;
- the initial C500 paper-table adapter still requires unthinned first source frames.
