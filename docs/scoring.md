# Strict scoring: `pdeobs-strict-v1`

This is a new, versioned path for raw-array validation and scoring. It is wired
to `pdeobs strict-score`, `runner.run_strict_score`, the versioned `strict-infer`
exporter, `paper-row`, and the release demos. It
does not alter the training loss or overwrite legacy score files. Historical
results require their own raw-prediction audit before they can be labeled with
this version.

```bash
pdeobs strict-score --predictions ./predictions.npz --contract ./contract.json --output ./strict-score.json
```

An accepted result exits zero. An invalid bundle/contract writes an explicit
invalid result where possible and returns a nonzero CLI exit status. Output is
created exclusively: an existing score is not silently overwritten. If the
output path itself is unwritable or already exists, the invocation fails rather
than claim a new receipt exists.

## Normal inference without manual repacking

`strict-infer` is the explicit versioned alternative to the unchanged legacy
`infer`/`evaluate_model` output. It runs a registered method on identities from
real canonical shards and writes a directly scoreable `predictions.h5`,
`contract.json`, and `score.json` in a new output directory:

```bash
pdeobs identity-manifest --data-root ./data --shards physical-000.h5 physical-001.h5 \
  --output ./identities.json
pdeobs strict-infer --config ./inference.yaml --contract ./test-contract.json \
  --data-root ./data --manifest ./identities.json --checkpoint ./last.pt \
  --output ./strict-run --device cpu
pdeobs strict-score --predictions ./strict-run/predictions.h5 \
  --contract ./test-contract.json --output ./strict-run/replay.json
```

The shard names above are illustrative existing files, not bundled production
data. The manifest schema is `pdeobs-identity-manifest-v1`:

```json
{
  "schema_version": "pdeobs-identity-manifest-v1",
  "expected_ids": ["actual-sample-id-0", "actual-sample-id-1"],
  "shards": [{"path": "physical-000.h5", "sha256": "actual-64-character-shard-hash"}]
}
```

`identity-manifest` reads the real IDs and hashes explicitly listed shards; it
does not generate data. `strict-infer` verifies those hashes and the full source
identity set, then selects the contract's exact expected identities. It rejects
missing files and never chooses a different root or a fallback split.

The inference YAML uses the existing `task`, `method`, `data.mask`, and
`evaluation` fields. For rollout use `data.input_horizon: 1` and
`evaluation.horizons: [1,2,3]`; targets in `BenchmarkDataset` are already future
frames, so its offset is zero. Besides the ordinary strict contract below,
the exporter requires `mask_config` equal to the actual `data.mask`,
`dataset_manifest_sha256` equal to the canonical JSON manifest hash, and
`checkpoint_id` equal to the supplied checkpoint's SHA-256. For an unfitted
baseline omit `--checkpoint` and use
`"unfitted:" + canonical_json_sha256(config["method"])` as its checkpoint ID.
The helper is `pdeobs.strict_inference.canonical_json_sha256`; it uses sorted-key,
compact JSON with no NaN. Trainable methods without a checkpoint are rejected.

IDs come from each record's `sample_id`. Time indices come from its actual
`stored_frame_indices` after the task's history/target selection. Dynamic
`stored_time_values` must be complete, finite and strictly increasing. Exported
`metadata_json` retains selected stored positions, source indices and physical
times per identity. A source stored at indices `[0,2,4,6]` therefore exports
future indices `[2,4,6]`; it is not relabeled `[1,2,3]`. HDF5 headers carry
`artifact_version=pdeobs-strict-inference-v1`, the contract JSON and its SHA-256.
`strict-score` checks the binding before reading and scoring arrays, and rejects
using a different contract to relabel the artifact. The result identifies the
binding as `pdeobs-strict-inference-v1_sha256_bound`; an old/external unbound
strict-compatible bundle is explicitly labeled
`unbound_legacy_or_external_artifact` and remains supported.

The Python entry point `pdeobs.strict_inference.evaluate_model_strict` accepts
`method`, `loader`, and the keyword arguments `config`, `contract`,
`predictions_path`, and optional `report_path`. It uses the same ordinary
`predict_batch` implementation and accepts per-identity metadata from
`collate_benchmark`. Registered and custom in-memory models share this route.
The versioned exporter currently supports recovery, forward and rollout.
Inverse export needs an explicit condition-coordinate adapter; inverse strict
array scoring remains implemented. See [paper reproduction](reproducing_the_paper.md)
for the executable single-row protocol and its narrower C500 time-coordinate
requirements.

Neither a caller-declared contract nor its hash authenticates external source
provenance. The bridge verifies consistency and raw arrays; it does not certify
a fabricated manifest or retroactively validate an old aggregate.

## Contract

For a static two-identity demo, a minimal contract is:

```json
{
  "schema_version": "pdeobs-strict-contract-v1",
  "task": "recovery",
  "split": "demo",
  "expected_ids": ["example-0", "example-1"],
  "expected_shape": [16, 16, 1],
  "target_time_indices": [0],
  "observation_id": "custom-demo-mask-v1",
  "checkpoint_id": "untrained:nearest",
  "projection": false
}
```

The strings above are illustrations, not identities that may be substituted for
a real dataset. Use the actual sample IDs and the actual task/checkpoint/mask
association of the artifact. For a learned method, identify the exact checkpoint
and keep its source/configuration receipt. The scorer verifies consistency with
the caller-declared contract; it does not authenticate a fabricated contract.

Supported tasks are `recovery`, `forward`, `inverse`, and `rollout`.
`expected_shape` excludes the leading identity axis: HWC for static tasks and
THWC for rollout. `target_time_indices` is a nonnegative, strictly increasing
integer list, with one index for a static task and T for rollout. These are
selected frame indices, not floating-point physical time values; preserve the
corresponding physical time metadata separately.

`split` is `demo` or `paper-test`. The latter requires **exactly 200 explicit
expected identities**. It does not accept an anonymous count of 200 or silently
use whichever records are present. Tiny demos use their own declared identity
set, not a hard-coded paper denominator.

## Prediction bundle

The CLI accepts `.npz`, `.h5`, or `.hdf5`. NPZ is loaded without pickle. Required
array keys are:

| Key | Meaning |
|---|---|
| `prediction` | Real numeric NHWC or NTHWC raw predictions |
| `target` | Corresponding complete targets in the same declared shape |
| `prediction_ids` | One nonempty string per prediction |
| `target_ids` | One nonempty string per target |
| `prediction_time_indices` | Integer vector matching the contract's target frames |
| `target_time_indices` | Integer vector matching the same contract |

Prediction and target order may differ; the scorer aligns both to the contract
order. Their sets must be exactly equal to the expected set. It rejects repeated,
missing, extra, empty, or non-string identities, and a mismatch between array
length and identity count.

Static recovery additionally requires:

- `mask`: NHWC with either one channel or the target channel count, containing
  only zero and one;
- `observation`: NHWC with the same shape as the target.

These two arrays are in **target identity order**. Observed values must match the
target exactly at mask-true entries: strict v1's static projection/diagnostics
contract is noiseless. A noise-aware extension would require a different,
explicitly versioned observation contract. Optional mask/observation data for
other tasks are checked for finiteness, but do not enable future-target projection.

## Validation precedes scoring

The scorer checks the complete raw prediction and target arrays for real numeric
type, nonempty finite values, shape, and frame/identity agreement. It then aligns
identity order and validates the recovery mask/observations. Only after these
checks does it compute metrics or apply optional data consistency.

Partial NaN, all NaN, positive/negative infinity, malformed shapes, or an invalid
identity/time set invalidate the **whole requested block**. There is no
`nanmean`, `nan_to_num`, filtered identity set, or reduced-denominator success.
Finite but constant, high-error, or otherwise scientifically poor predictions
remain valid mathematical inputs and remain in the result. Validity is not a
claim of model quality.

Near-zero target norms use a fixed `epsilon=1e-12`. This can yield a large finite
relative error; it is not capped. Float64 arithmetic overflow is reported invalid
rather than replaced with zero. Output is standard JSON: invalid metrics are
`null`, not JSON NaN, infinity, or a fabricated numerical score.

## Metric definitions

For each identity i, let `e_i = prediction_i - target_i` and
`d_i = max(||target_i||_2, 1e-12)`. Strict v1 computes

```text
relative_l2_i = ||e_i||_2 / d_i
reported_mean = sum_i relative_l2_i / N_expected
```

All norms and means are float64. For finite legitimate inputs this preserves the
legacy per-identity relative-L2 definition; stricter validity is the difference.
It is not the norm of all errors divided by a global target norm and is not a
mean over batches with unequal weighting. Tail batches retain their identities.

`rel_l2_joint` uses all non-identity axes. For rollout this includes time, space,
and channels. `rel_l2_by_horizon` separately uses each target frame, and summaries
retain both the one-based forecast horizon and target-frame index. Joint error
does not in general equal the mean of the horizon errors.

## Static observed/hidden diagnostics

The recovery result reports three norms with the **same full-target denominator**:

```text
full_common_denominator     = ||e||_2 / d
observed_common_denominator = ||M*e||_2 / d
hidden_common_denominator   = ||(1-M)*e||_2 / d
```

Their squared numerators partition the full squared error. The observed and
hidden values do not add linearly to the full norm.

`hidden_only_rel_l2` is a different diagnostic,
`||e_hidden||_2 / max(||target_hidden||_2, epsilon)`, and is separately named.
Counts refer to scalar entries after channel broadcasting. When an identity is
fully observed, hidden-only error is `null` with `not_applicable`. If not every
identity has a hidden-only value, the block does not report a misleading
reduced-denominator hidden-only mean; it states the applicability count and
`not_applicable_for_all_identities`.

## Optional data-consistency projection

`projection: true` is allowed only for static recovery. Raw prediction checks
still happen first. Projection replaces predicted observed entries with the
provided noiseless observations and leaves hidden entries unchanged. The raw
relative error stays in the result; `projected_rel_l2_joint` and its mean are
additional, explicitly named values. No future truth is projected into rollout.

## Result structure and provenance boundary

The result includes `status`, error messages, scoring version, task,
observation/checkpoint associations, expected/actual/scored identity counts,
per-identity values and aggregate metrics. A valid result also records the full
contract, its SHA-256, the identity-set hash, expected ordering, reduction,
epsilon, raw-check flag and whether projection was applied.

Invalid inputs have `status: invalid`, `summary: null`, an empty per-identity
score list, and a scored count of zero. Available actual counts and the reason
are preserved rather than treated as a successful smaller experiment. A failure
to read a bundle is still a failure, not a successful import-only test.

Contract agreement is not external provenance authentication. Keep dataset,
configuration and checkpoint receipts with the bundle and verify their mapping
before paper aggregation. This release does not silently rescore the historical
campaign or upgrade legacy aggregate-only results to strict status.
