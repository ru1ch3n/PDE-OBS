# Reproducing the paper protocol

The software implements a broader resource than the paper's primary experiment.
This page fixes the intended interpretation of the selected experiment and its
historical cohorts. It is not a launch instruction for an unbounded sweep, and
it does not assert that every planned result is complete.

## Three distinct scopes

| Scope | Physical records and purpose | Evidence to retain |
|---|---|---|
| Full resource design | Seven PDE families x four boundary protocols x ten condition settings x 2,000 records per macrodomain, or 560,000 records | Generation configuration, source/version, shard manifests, historical QC versus new independent checks |
| Selected paper experiment | One fixed boundary/setting macrodomain per PDE, 2,000 records each, or 14,000 underlying records | Exact identity lists, splits, trained view, test view, model/attempt/budget, checkpoint, score version |
| Release demos | Small new CPU-generated records and explicit demo identity sets | Demo command, source/package identity, environment, outputs and validity status |

Three regimes divide a 2,000-record macrodomain; they do not multiply its record
count by three. Nine masks are views of the same underlying records, not nine
independent numerical datasets.

## Selected physical cases and tasks

All seven selected cases use `smooth_grf`. Darcy, Poisson, and Helmholtz use
`dirichlet` and the recovery task. Heat, reaction-diffusion, Burgers, and
Navier-Stokes use `periodic` and rollout. The NS state is vorticity; other
selected cases use their native scalar state. The resolution is 128 x 128.

For recovery the predictor reconstructs a complete selected state from partial
observations of that state. For rollout it receives the first stored state
through a mask (`history=1`) and predicts the next three (`horizon=3`), with
prediction feedback and no future-ground-truth forcing. See
[Tasks and permissions](tasks_and_permissions.md). Forward and inverse are
implemented extensions, not additional completed primary task experiments.

The saved trajectory's physical time coordinates differ by family. The first
four monotone stored frames define `t0,t1,t2,t3`; these labels are not a claim
that all families share a common physical time step.

## Identity split, supervision, and pairing

Each selected 2,000-record macrodomain is split deterministically into exactly
1,800 training and 200 held-out test identities, with **no validation set**.
`one_setting.stable_split` ranks sample IDs within each regime using
`SHA256(seed|macrodomain_identity|sample_id)`, then allocates the 10% test set by
largest remainder. Preserve its split receipt and the actual lists. Balanced
source regime counts lead to 600 training identities per regime; the test counts
are 67/67/66 according to the recorded allocation.

The original split seed is `20260804`. The identity sets are shared by all
methods and observation views for a physical case. Training uses complete
targets; masks restrict predictor inputs, not the supervision target. Test
identities must not be substituted for missing training data or used to choose
the checkpoint. A generic loader's `split: iid` field does not reconstruct this
paper split by itself.

The exact macrodomain string passed to `stable_split` is
`pde|boundary|setting`, for example `poisson|dirichlet|smooth_grf`.
The pipe delimiters are part of the hashed input; replacing them with slashes
changes the split even if all physical records and the integer seed are equal.

## Training view v and test view w

Each learned row is a `(PDE, method, training view v)` identity. Nine training
views are trained independently. A fixed checkpoint is evaluated on nine test
views `w`, always using the same 200 test identities. The matched-view diagonal
`v=w` and the cross-view matrix serve different comparisons.

At 128 x 128, the exact observed counts are R50/H/V/CL = 8192, R65 = 10650,
R80 = 13107, and BL/LI/BD = 8284. Configuration and geometric meanings are in
[Observations](observations.md) and `configs/paper/observations.yaml`. The random
densities are not guaranteed nested. A generic 3% mask or horizon-eight preset
must not be used as an implicit replacement.

The current public learned-method set is U-FNO-2D, FNO, CNO, DeepONet, GNOT,
Transolver, and PINO: 7 x 7 x 9 = **441 planned training rows** and 441 x 9 =
**3969 possible learned cross-view blocks**. This is a denominator, not a
completion count. Deferred methods, aliases, and wrappers do not add algorithms
to it. Classical Gaussian RBF and Gappy POD have a different possible indexing,
PDE x method x test view, with no independently trained-view axis; if such
results are included, their 126 possible blocks must remain separate. This
candidate makes no claim that those classical paper blocks have been run.

## Architectures and training information

The source contains compact/local adaptations and, separately, selected exact
external-checkout wrappers. A local method name is not a claim of bitwise
reproduction of an upstream release. Model size, modes/layers, optimizer,
scheduler, batch size, input channels, geometry channels, and loss must be taken
from that row's resolved configuration and method registry, not inferred from a
generic default.

PINO additionally uses a declared training-only physics residual context. Static
and periodic dynamic residuals have separate contracts and weights. This is a
training-information difference from ordinary data-only objectives, even though
the predictor-visible observations follow the same task contract. It must remain
visible in method descriptions and result grouping.

## Historical budget cohorts are not interchangeable

| Cohort | Budget and stopping semantics | Required reporting |
|---|---|---|
| `original500` | 500 epochs, 1,800 records/epoch, no validation early stopping, final `last` checkpoint | Actual epochs, optimizer/scheduler configuration and checkpoint identity |
| `fixed200` | Declared 200-epoch attempt budget | Actual endpoint and any explicitly recorded attempt-specific settings |
| `budget200_min120` | Maximum 200 epochs; no plateau accumulation through epoch 120; after that, stop after 10 consecutive epochs without a strict improvement in the monitored training data loss | Budget version, minimum epoch, patience, `min_delta`, monitored loss and actual stop reason |
| `interrupted_or_recovered` | An interrupted checkpoint or resumed/authorized attempt, not automatically budget-complete | Recovered state, epoch, original attempt, reason and any changed parameters |

In the recorded min120 adapter, `min_delta=0`; epochs 1–120 do not accrue stale
epochs, so the earliest patience stop is epoch 130. An improvement means a new
strict minimum of the monitored training data loss, not a visual trend judgment
or a test metric. The declared final/last checkpoint remains distinct from a
best-validation checkpoint. A field named `val_loss` can mirror training loss
when no validation loader exists.

Some retries have explicit learning-rate or clipping changes. They remain new
attempts with preserved predecessors. They are not silently renamed the original
500-epoch protocol. This release documents the later adapter protocol but does
not bundle private scheduler/authentication state or a promise to reproduce a
private deployment. The frozen `one_setting.build_experiment_config` still
accepts only the original 3-epoch diagnostic and 500-epoch settings; changing its
`epochs` argument to 200 is not a faithful implementation of the later adapter.

`configs/paper/training_cohorts.yaml` is protocol metadata, not an implicit early
stopping plugin. The original diagnostic three-epoch preflight is neither a
budget-complete result nor a paper convergence experiment.

## Execute one original500 row

`configs/paper/protocol.yaml` is the concise public description of the selected
scope. Its distinct schema deliberately prevents confusion with a production
job file. The retained historical campaign/registry files remain source evidence
for the original resolver, including their broader candidate accounting. Do not
pass the new descriptive manifest into the old frozen campaign validator.

The new, portable `paper-row` command is an **executable** adapter for a single
new `original500` attempt. It calls the actual `one_setting.stable_split`, not a
generic loader's source split labels. It uses the current retained method
registry and resolver without changing optimizer, architecture, loss, or
scheduler parameters. It enforces scalar 128 x 128 source fields, exactly
2,000 distinct identities, 600 training identities per regime, 1,800 training
identities in total, 200 held-out test identities, no validation, 500 epochs,
and the final `last.pt`. All nine test views are evaluated after reloading a
new model instance from that checkpoint. The runner does not select `best.pt`,
use test loss for stopping, or reuse a historical score.

Install the optional training dependencies (`pip install '.[train]'`) and run
from the extracted release. Supply actual existing shards, with paths relative
to your explicit root. The following file names are illustrative, not bundled
production data:

```bash
pdeobs identity-manifest --data-root ./paper-data \
  --shards poisson/dirichlet/smooth_grf/low/shard-00000.h5 \
           poisson/dirichlet/smooth_grf/medium/shard-00000.h5 \
           poisson/dirichlet/smooth_grf/high/shard-00000.h5 \
  --output ./poisson-identities.json

pdeobs paper-row --config configs/paper/row_original500.yaml \
  --data-root ./paper-data --manifest ./poisson-identities.json \
  --output ./paper-runs/poisson-fno-r50-attempt-1 --device cpu
```

This is a potentially long full C500 training command, not a smoke test.
The command requests no scheduler allocation. `--device cuda` is available only
when the caller already has an authorized GPU; this release revision did not
run C500 or request a GPU. Change `pde`, `method`, and `train_view` in a new row
configuration to select another supported row. Use a new portable `run_id` and
`attempt_id`; do not reuse an output directory. Configuration-relative paths
resolve relative to the row YAML, while data and output roots are explicit.

The JSON identity manifest uses `pdeobs-identity-manifest-v1`, with an explicit
`expected_ids` list and `shards: [{path, sha256}, ...]`. Its source hash and
metadata checks run before any target-array access. Hashing verifies stored
bytes, not scientific correctness. All shards must be supplied explicitly;
the runner will not search a different root, create missing paper data, or fall
back to source `train`/`test` labels. The complete source identity set must equal
the manifest. Source records must include `T`, `stored_frame_indices`, and, for
rollout, finite, strictly increasing `stored_time_values`. Missing time metadata
fails explicitly, rather than fabricating a physical time vector.

The initial `original500` paper-table adapter additionally requires unthinned
first source frames `[0,1,2,3]` for rollout (targets `[1,2,3]`) or static target
source index `[0]`. This matches the current paper updater's coordinate contract.
A thinned corpus is rejected **before training**, even if it contains enough
stored frames; it requires a separately reviewed coordinate-mapping adapter.
Generic strict inference below supports such thinned records without relabeling
their genuine source indices.

Artifacts are `row_resolved.json`, `identity_manifest.json`, `split.json`,
`checkpoints/last.pt`, `training_completion.json`, per-view
`evaluation/<view>/{contract.json,predictions.h5,score.json}`, and `receipt.json`.
The training receipt is written before any numerical test arrays are loaded.
It records the selected IDs, hashes, actual epochs and checkpoint choice.
Failures leave their evidence and a nonzero CLI result; no retry, altered
training parameter, missing-view substitution or historical-result promotion
occurs automatically. Training health checks are inherited from the registry.
Finite poor predictions are still included by the strict scorer.

`training_seed` controls model initialization and training; the split and mask
seed remain the campaign seed for identity/view pairing. Contracts retain
`run_id`, `training_seed`, `attempt_id`, `training_config_sha256`, and the final
checkpoint hash. `training_recipe_sha256` hashes the resolved configuration
after excluding the training seed, dataset-root location, device and operational
loader fields. This seed-independent recipe tag is distinct from the full
training configuration hash and from the scoring contract's `config_sha256`.
The separately shipped paper updater still validates its required result bundle;
a row receipt alone is not a paper-table update or a history audit.

The executable row schema accepts only `original500` and the explicitly reduced
`demo` mode below. `fixed200`, `budget200_min120` and recovered/interrupted
cohorts are rejected until their separately versioned attempt adapters are
supplied. Describing those historical budgets above does not make them runnable
by changing one integer in this adapter.

## Small static and rollout integration runs

These commands explicitly generate **nine new physical records**, three per
regime, at 16 x 16. They do not substitute for absent paper data. The demo row
uses the current registry's FNO architecture, one epoch, six training and three
test identities, and two test views. Its small split has its own schema and is
not labeled the production 1,800/200 split. It exercises the same model
construction, Trainer, no-validation, final checkpoint reload, strict exporter
and scoring control path as the production adapter.

```bash
pdeobs paper-row-demo-data --pde poisson --output ./row-demo-data/poisson
pdeobs paper-row --config configs/demo/paper_row_static.yaml \
  --data-root ./row-demo-data/poisson --manifest ./row-demo-data/poisson/identities.json \
  --output ./row-demo-runs/poisson --device cpu

pdeobs paper-row-demo-data --pde heat --output ./row-demo-data/heat
pdeobs paper-row --config configs/demo/paper_row_rollout.yaml \
  --data-root ./row-demo-data/heat --manifest ./row-demo-data/heat/identities.json \
  --output ./row-demo-runs/heat --device cpu
```

Use one or two CPU threads for these demos (for example set `OMP_NUM_THREADS=1`
and `MKL_NUM_THREADS=1` in the calling process). They establish an executable
interface, not C500 completion, model convergence, or reproduction of any
reported historical score. Tests separately verify the exact production split
on a 2,000-record metadata fixture, the supported registry configurations, and
that training reads only the selected training IDs. Metadata-only tests are
not claimed as training on 2,000 generated PDE records.

## Ordinary inference directly into strict scoring

`pdeobs strict-infer` is the opt-in versioned path for registered methods; the
legacy `infer`/`evaluate_model` writer and API are unchanged. For a configured
method and a predeclared contract:

```bash
pdeobs strict-infer --config ./inference.yaml --contract ./test-contract.json \
  --data-root ./data --manifest ./identities.json --checkpoint ./last.pt \
  --output ./strict-inference --device cpu
pdeobs strict-score --predictions ./strict-inference/predictions.h5 \
  --contract ./test-contract.json --output ./strict-replay.json
```

The normal inference path writes `prediction_ids`, `target_ids`, source-frame
time indices, raw predictions/targets, observation/mask arrays, and per-identity
metadata including selected physical times. It is immediately readable by
`strict-score`, without manually editing or repacking arrays. Source solver
frame indices may differ from stored-frame positions: for example a thinned
seven-frame trajectory stored at indices `[0,2,4,6]` predicts `[2,4,6]`, not a
fabricated `[1,2,3]`. Each block preserves those distinctions.

The strict contract uses the schema described in [Scoring](scoring.md), plus
`mask_config` matching `data.mask` and `dataset_manifest_sha256` (the canonical
JSON SHA-256 of the manifest). `checkpoint_id` is the SHA-256 of the supplied
checkpoint. For an unfitted baseline with no checkpoint, it is
`"unfitted:" + canonical_json_sha256(config["method"])` (the helper in
`pdeobs.strict_inference`); trainable methods without a
checkpoint are rejected. The exporter checks identity membership, actual
target shape, source-frame metadata, checkpoint and manifest before crediting
any score. `pdeobs.strict_inference` also exposes `evaluate_model_strict` for a
custom in-memory method and normal `collate_benchmark` batches, without requiring
that method to be installed in the registry.

HDF5 artifacts carry `artifact_version=pdeobs-strict-inference-v1`, the exact
contract JSON and its SHA-256. `strict-score` rejects relabeling one bound file
with a different contract. This is consistency binding, not an attestation that
an untrusted caller supplied authentic provenance. Legacy/external unbound
strict-compatible NPZ/HDF5 inputs remain supported and are labeled
`unbound_legacy_or_external_artifact`. Versioned inference currently supports
recovery, forward and rollout. Inverse inference requires a future explicit
condition-coordinate adapter; this limitation does not remove the existing
inverse dataset/task implementation or strict array scoring support.

## Scoring and result status

The new release-default scorer is `pdeobs-strict-v1`. It verifies raw finite
prediction/target arrays, identity sets, shapes, and time indices before metrics
or projection. A paper-test contract requires exactly the specified 200 test
identities. Relative L2 is computed in float64 per identity and then averaged.
The joint dynamic norm covers time, space, and channels; per-horizon scores are
separately named. Finite poor predictions remain included. Details are in
[Scoring](scoring.md).

Historical experiments retain their recorded legacy score/version. Passing the
new strict-scorer tests does not certify their raw predictions. If a historical
aggregate lacks a raw prediction audit, preserve that status rather than
relabeling it. Main tables must identify which cohorts and scoring versions are
pooled. A missing result is `RESULT_PENDING` or an explicit absence, never zero,
an interpolated curve, or another row's value.

For a new result: verify structure and identity/version first; then generate its
table/figure; then review any affected claim; then compile the paper. Do not use
a generated plot as evidence that an underlying identity or checkpoint was valid.

## Evidence and practical limits

The candidate package includes CPU demos and tests, not the production corpus or
all trained checkpoints. A release smoke-test pass establishes an interface
worked in the recorded environment. A Heat analytic-mode check establishes one
specific reference comparison. Neither establishes accuracy for all family,
boundary, regime and generator combinations, nor reproduces all 441 training
rows. The release receipt names completed, failed, skipped and unexecuted checks
separately. Optional upstream model checkouts and dataset redistribution rights
remain explicit requirements rather than hidden fallbacks.
