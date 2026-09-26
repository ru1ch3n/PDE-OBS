# Results format

**No scored cell of the 441-row paper grid exists in this tree.** No leaderboard
is shipped and no per-method benchmark accuracy figure is claimed. The only
scored artifacts in this release are the software-acceptance blocks under
[`acceptance/`](../acceptance/), which run at toy scale (16 x 16 and 64 x 64
fields, a handful of samples, two optimizer steps) and are explicitly fenced off
in [`ACCEPTANCE_SUMMARY.md`](../ACCEPTANCE_SUMMARY.md): "Software acceptance
only; not training to convergence, not a performance number." Those blocks are
not benchmark results and must never be placed in a results table. The release
notes state the same boundary from the other direction: "No production C500 run,
441-row sweep, multi-seed benchmark or convergence study"
([`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md)).

This document fixes the **format** of a result and the **recipe** that turns
scored blocks into tables. It states one rule above all others: no number may be
reported without its contract hash and its scorer version.

Related reading: [Scoring](scoring.md) for the strict scorer itself,
[Reproducing the paper](reproducing_the_paper.md) for the protocol and the
executable row command, and [`configs/paper/protocol.yaml`](../configs/paper/protocol.yaml)
for the machine-readable scope.

---

## The unit of a result

One **strict evaluation block** is one cell:

> (PDE family, method, training view `v`, test view `w`)

scored on exactly the 200 held-out identities of that PDE's macrodomain with
the release-default scorer `pdeobs-strict-v1`
([`src/pdeobs/strict_score.py`](../src/pdeobs/strict_score.py)). A block is the
smallest thing that can be reported. There is no sub-block unit: a strict
report is valid for the complete expected identity set or it is invalid, and an
invalid block has `summary: null`, an empty `per_identity` list and
`scored_identity_count: 0`.

The identity set is fixed by the contract. A `paper-test` contract is rejected
unless `expected_ids` holds exactly 200 explicit identities; an anonymous count
of 200, a truncated set, a duplicate or an extra identity all invalidate the
block rather than shrinking the denominator.

The axes, in their frozen order
([`src/pdeobs/one_setting.py`](../src/pdeobs/one_setting.py)):

| Axis | Values |
|---|---|
| PDE (7) | `darcy`, `poisson`, `helmholtz`, `heat`, `reaction_diffusion`, `burgers`, `navier_stokes` |
| Public learned method (7) | `ufno_2d`, `fno`, `cno`, `deeponet`, `gnot`, `transolver`, `pino` |
| Training view `v` (9) | `random_50pct`, `random_65pct`, `random_80pct`, `block_observed_50pct`, `line_sensors_50pct`, `horizontal_lines_50pct`, `vertical_lines_50pct`, `boundary_band_50pct`, `clustered_50pct` |
| Test view `w` (9) | the same nine views |

7 x 7 x 9 = 441 planned training rows; 441 x 9 = 3969 possible learned
cross-view blocks. That is a **denominator, not a completion count**
([`configs/paper/protocol.yaml`](../configs/paper/protocol.yaml),
[Reproducing the paper](reproducing_the_paper.md)).

Classical Gaussian RBF and Gappy POD have a different indexing —
PDE x method x test view, with no independently trained-view axis, so
7 x 2 x 9 = 126 possible blocks — and those must stay in a separate table.

The **paper-row path** cannot produce a classical block: `resolve_row` rejects
any method outside the seven public learned methods
(`if row.get("method") not in PUBLIC_LEARNED_METHODS`,
[`src/pdeobs/paper_row.py`](../src/pdeobs/paper_row.py)). The strict-inference
path can. Both classical methods are registered — `gappy_pod` in
[`src/pdeobs/methods/reduced_order.py`](../src/pdeobs/methods/reduced_order.py)
and RBF in
[`src/pdeobs/methods/interpolation.py`](../src/pdeobs/methods/interpolation.py) —
the runner fits and saves `checkpoints/gappy_pod.npz`
([`src/pdeobs/runner.py`](../src/pdeobs/runner.py)), and `run_strict_infer` is
the CLI adapter for arbitrary registered methods with explicit test identities.
This release's own acceptance evidence contains valid strict reports for
`gappy_pod:recovery(fitted)` and `rbf:recovery`
([`acceptance/round2/T3.json`](../acceptance/round2/T3.json)). A classical block
is therefore producible, but it is not an `original500` paper cell and belongs
in its own table.

---

## On-disk layout of one row

One executed `original500` row writes one directory. The layout is produced by
[`src/pdeobs/paper_row.py`](../src/pdeobs/paper_row.py) together with the shared
trainer in [`src/pdeobs/training.py`](../src/pdeobs/training.py):

```
<row output directory>/
  row_resolved.json          # {"row": ..., "experiment": ...} the selection + resolved config
  split.json                 # split receipt: algorithm, seed, regime counts, train_ids, test_ids
  identity_manifest.json     # pdeobs-identity-manifest-v1: expected_ids + shards[{path, sha256}]
  checkpoints/last.pt        # the only checkpoint the row runner reloads and evaluates
  checkpoints/best.pt        # written by the shared Trainer; never read by the row runner
  checkpoints/training_config.json   # written beside every saved checkpoint
  training_completion.json   # cohort, actual_epochs, checkpoint_id, config/recipe/manifest/split
                             #   hashes, history, health events, test_arrays_accessed: 0
  evaluation/<test_view>/
      contract.json          # the strict contract for this (v, w) cell
      predictions.h5         # bound strict inference artifact
      score.json             # the pdeobs-strict-v1 result for this cell
  receipt.json               # row status + one entry per test view
```

**`best.pt` is present and is never used.** The row runner passes
`val_loader=None`, so `Trainer.fit` sets `val_loss = train_loss`; `improved` is
then true whenever the training loss improves, and `save_checkpoint("best.pt",
...)` fires. Every `save_checkpoint` call also writes
`checkpoints/training_config.json`. Neither file participates in evaluation: the
row runner reloads `checkpoints/last.pt` into a fresh model instance and records
`checkpoint_selection: "final_last_only"`. An auditor who finds `best.pt` in a
row directory has found a trainer artifact, not a selection.

There are nine `evaluation/<test_view>/` directories for an `original500` row,
one per frozen view, all scored against the same 200 test identities with only
the test mask swapped.

Two rules matter for provenance:

- **A row is complete only if every one of the nine view blocks is valid.**
  `receipt.json` records, per view, `{test_view, status, scored_identity_count,
  score_sha256}`, and sets the row status to `complete` only when every block
  reports `status: "valid"`; otherwise the row status is `invalid_evaluation`
  (or `failed`, with the stage and the error preserved).
- **Score files are written exclusively-create and never overwritten.**
  `write_report` opens the destination with mode `"x"`, so an existing
  `score.json` causes the write to fail rather than silently replacing a prior
  result. A failure leaves its evidence and a nonzero CLI exit status; no
  retry, no altered parameter, no missing-view substitution.

The training receipt is written **before any test array is touched**, and
records `validation_records: 0`, `test_arrays_accessed: 0`,
`checkpoint_selection: "final_last_only"` and
`scientific_acceptance: "not_inferred_from_budget_completion"`.

---

## The headline metric, declared

The primary column of any main table is:

> **`rel_l2_joint_mean`** — the strict per-identity relative L2, computed in
> float64, then **arithmetic-meaned over the complete expected identity set**.

Per identity, the value is `||prediction - target||_2 / max(||target||_2, 1e-12)`
over all non-identity axes (HWC for static recovery, THWC for rollout), computed
with a scaled norm in float64. The reduction is recorded inside every valid
report as `reduction: "float64 per-identity relative L2, then arithmetic mean"`,
with `epsilon: 1e-12`.

**Denominators are never reduced.** The strict path uses no `nanmean` and no
`nansum`; a non-finite entry invalidates the block instead of being skipped.

### Static recovery: the partial-observation columns

Recovery is what makes this a partial-observation benchmark, so the headline
column is reported beside the static diagnostics rather than alone. For each
identity, with `d = max(||target||_2, 1e-12)` and `e = prediction - target`:

| Reported value | Definition | Denominator |
|---|---|---|
| `rel_l2_joint` | norm of `e` over the whole field, divided by `d` | full target |
| `full_common_denominator` | norm of `e` over the whole field, divided by `d` | full target |
| `observed_common_denominator` | norm of `e` restricted to mask-true entries, divided by `d` | full target |
| `hidden_common_denominator` | norm of `e` restricted to mask-false entries, divided by `d` | full target |
| `hidden_only_rel_l2` | norm of `e` on hidden entries, divided by `max(norm of target on hidden entries, 1e-12)` | **hidden-entry target norm** |

All norms are Euclidean (L2) and are taken in float64.

**`rel_l2_joint` and `full_common_denominator` are the same number by
construction.** In [`src/pdeobs/strict_score.py`](../src/pdeobs/strict_score.py),
`rel_l2_joint` is `_relative(p, t)`, which computes
`_norm(p - t) / max(_norm(t), EPSILON)`, and `full_common_denominator` is
`_norm(error) / denominator` with the same `denominator = max(_norm(t),
EPSILON)`. The second name exists only so that the four common-denominator
diagnostics can be read as one family on one scale. **A table must not publish
both as separate columns**; pick one name and say which.

The first four share one full-target denominator, so they sit on a common
scale; the squared numerators of the observed and hidden parts partition the
full squared error, which means the two do **not** add linearly. The
hidden-only value is deliberately given a different name because it has a
different denominator, and it must never be presented in the same column as
the common-denominator diagnostics.

**The hidden-only mean is withheld, not rescued.** If any identity in the block
is fully observed, that identity's `hidden_only_rel_l2` is `null` with
`hidden_only_status: "not_applicable"`, and the block summary reports
`hidden_only_applicable_count`, `hidden_only_status:
"not_applicable_for_all_identities"` and `hidden_only_rel_l2_mean: null`.
The code comment states the reason plainly: mixed applicability has no
misleading mean with a reduced denominator.

An optional data-consistency `projection` is defined only for static recovery.
When requested it replaces predicted values at observed entries with the
supplied noiseless observations and reports `projected_rel_l2_joint` as an
**additional, separately named** value; the unmodified raw score is always
retained. Requesting `projection: true` on any non-recovery task invalidates
the block.

### Rollout: joint and per-horizon are different columns

For rollout, the block reports:

- `rel_l2_joint` — over time, space and channels **jointly**;
- `rel_l2_by_horizon` — one value per target frame, each carrying its 1-based
  `horizon` and the source `target_time_index`.

**The joint value is not the mean of the horizon values.** They are separate
columns and must be labelled as such.

Time indices are the record's actual stored source solver frame indices after
task selection and are never renumbered. In **generic** strict inference
([`src/pdeobs/strict_inference.py`](../src/pdeobs/strict_inference.py),
`target_frame_metadata`) a trajectory stored at `[0,2,4,6]` exports targets
`[2,4,6]`, not `[1,2,3]`. That case **cannot arise in an `original500` paper
block**: `validate_c500_frames`
([`src/pdeobs/paper_row.py`](../src/pdeobs/paper_row.py)) hard-requires the
stored prefix `[0,1,2,3]` and the target indices `[1,2,3]`, raising "C500 source
time coordinates require unthinned first frames" otherwise. The release notes
flag the same limitation: the C500 paper-table adapter still requires unthinned
first source frames. A thinned corpus needs a separately reviewed mapping
adapter that this release does not ship.

---

## Proposed long-format results schema

One row per scored value. This is the interchange format the tables will be
pivoted from; it is proposed here and is not yet emitted by any shipped tool.

| Column | Source |
|---|---|
| `pde` | contract `pde` |
| `method` | contract `method` |
| `train_view` | contract `train_view` |
| `test_view` | contract `test_view` (= contract `observation_id` for a paper row) |
| `seed` | contract `training_seed` (the split/mask seed is the campaign seed) |
| `task` | contract `task` (`recovery` or `rollout` for the primary grid) |
| `metric` | e.g. `rel_l2_joint_mean`, `rel_l2_by_horizon_mean[h=1]`, `static_diagnostics.hidden_common_denominator_mean` |
| `value` | the number, or an explicit absence marker |
| `identity_count` | summary `identity_count` (200 for a paper block) |
| `training_cohort` | contract `training_cohort` |
| `actual_epochs` | contract `actual_epochs` |
| `attempt_identity` | contract `attempt_id` |
| `resolved_config_identity` | contract `training_config_sha256` |
| `stop_reason` | contract `stop_reason` |
| `checkpoint_id` | contract `checkpoint_id` (the final checkpoint SHA-256) |
| `contract_sha256` | report `config_sha256`, the canonical SHA-256 of the contract |
| `identity_set_sha256` | report `identity_set_sha256`, over the sorted identity set |
| `scoring_version` | report `scoring_version` (`pdeobs-strict-v1`) |
| `evidence_level` | one of `implemented`, `smoke-tested`, `reference-checked`, `paper-evaluated` (`release/candidate_metadata.json`) |

**The rule, plainly: a reported number without a contract hash and a scorer
version is not a PDE-OBS result.** It may be a diagnostic, a legacy aggregate
or a demo output, but it may not enter a results table as one.

The last three provenance columns above are present so that this schema covers
the full `required_result_fields` list that the cohort description demands of
every result row — `training_cohort`, `actual_epochs`, `attempt_identity`,
`checkpoint_identity`, `resolved_config_identity`, `score_version`,
`stop_reason`
([`configs/paper/training_cohorts.yaml`](../configs/paper/training_cohorts.yaml)).
All three are carried in the contract itself, so `contract_sha256` binds them
transitively even when a table omits the columns. The same file forbids
relabelling historical results into another cohort.

---

## How the tables will be produced

The recipe:

1. Walk a run directory. For each row bundle, read `receipt.json`; for each
   `evaluation/<view>/`, read `score.json` and its sibling `contract.json`.
2. Skip nothing silently. A block whose `status` is not `valid` becomes a row
   with an explicit invalid marker, not an absent row.
3. Emit the long-format CSV above, one line per scored value.
4. Pivot:
   - **(a) matched-view diagonal**, `v = w`: one value per (PDE, method), a
     7 x 7 table;
   - **(b) cross-view matrix**, a 9 x 9 grid of (train view, test view) per
     (PDE, method);
   - **(c) per-method summary**, a 7 x 9 table of (PDE, test view) for one
     method at a fixed training view, or of (PDE, training view) at a fixed
     test view — whichever the table caption states explicitly.

**This walker does not exist yet for strict `paper-row` output, and nothing in this release
ships it.** What does ship is `tools/archived_results.py`, which walks the *archived campaign* layout
(legacy relative-L2 metrics, no prediction tensors) and emits this long format with the legacy
scoring version on every row; see [Archived results](archived_results.md).
The shipped `pdeobs aggregate` command discovers result files by four fixed
file names only — `metrics.json`, `results.json`, `metrics.csv`, `results.csv`
([`src/pdeobs/aggregate.py`](../src/pdeobs/aggregate.py), `_report_files`) — so
it does **not** pick up strict `score.json` blocks. Its `--group-by` default is
`method,task,split` and its leaderboard comes from
`reports.aggregate_records`, which emits a `runs` count per group plus
`count`/`mean`/`std`/`min`/`max` per numeric metric per group
([`src/pdeobs/reports.py`](../src/pdeobs/reports.py)), with a population
standard deviation that is `0.0` for a single run and is not a replication
protocol. Likewise `runner.run_benchmark` writes `analysis_records.json/.csv`,
`leaderboard.json/.csv` and `benchmark.json` from the **legacy** evaluation
path, not from strict blocks.

The strict aggregator is therefore a separate, still-to-be-added tool. This
document specifies it; it does not claim that it ships.

---

## Matched versus cross view, and matched versus full

Two distinct axes that must not be collapsed into one sentence.

**1. Training view `v` versus test view `w`.** Every contract records both
`train_view` and `test_view`, so the matched-view diagonal (`v = w`) is
distinguishable from the off-diagonal in the scored artifacts themselves,
without relying on directory names. The diagonal and the cross-view matrix
answer different questions and belong in different tables.

**2. Matched mask versus full observation.** `observation_mode` in
[`src/pdeobs/evaluation.py`](../src/pdeobs/evaluation.py) is a *different*
switch. `matched` is the official benchmark view, and that is enforced in code,
not only by convention: `evaluate_model_strict` refuses to export at all unless
`observation_mode == "matched"` and the layout is `channels_last`, raising
"strict export requires matched observations and explicit channels_last layout"
([`src/pdeobs/strict_inference.py`](../src/pdeobs/strict_inference.py)). **No
strict block can exist for the full-observation mode.**

`full` is a **recovery-only diagnostic control** that exposes the complete
target to an already-trained model with an all-visible mask; the dataclass
rejects it for any non-recovery task and refuses to combine it with mask-OOD or
explicit observation protocols. Its output is written on the legacy evaluation
path to its own artifact location —
`evaluation/full_observation/metrics.json`, beside the official matched report
at `evaluation/metrics.json`
([`src/pdeobs/campaign.py`](../src/pdeobs/campaign.py)) — and it must never
replace matched-mask scores.

The interpretation rule is preregistered or it is not applied. The campaign
diagnostic summary carries `role: "diagnostic_only"` and
`official_result: "matched_observation"`. If no preregistered
`full_observation_relative_l2_good_max` is supplied, the row is classified
`unclassified_no_preregistered_threshold` with the action
`report_both_metrics_without_inventing_a_cutoff`. **No shipped campaign config
in this release supplies such a threshold**, so every row would classify as
unclassified; inventing a cutoff after seeing the numbers is not an option this
format permits.

---

## Missing, invalid and pending

Three states that a table must keep distinct:

| State | What it means | How it should appear |
|---|---|---|
| **Not yet run** | The cell has no artifact. | `RESULT_PENDING`, the declared marker |
| **Run and invalid** | A strict report exists with `status: "invalid"`, `summary: null`, `per_identity: []`, `scored_identity_count: 0` and a non-empty `errors` list | an explicit invalid cell, with the failure reason available |
| **Refused** | The runner or scorer declined before producing a block (unsupported cohort, mismatched manifest, wrong resolution, non-public method, unsupported task export) | an explicit refusal with its stated reason |

**A missing result is never zero, never an interpolated curve, and never
another row's value.** That rule is already in
[Reproducing the paper](reproducing_the_paper.md); this document restates it as
a table-construction constraint.

Two honest caveats about the current artifacts:

- **States two and three are not separable today.** `run_strict_infer` catches a
  setup refusal and writes an `invalid_report` into `score.json`, so a refusal
  and a genuine scoring failure land in the same file with the same `status:
  "invalid"`, distinguished only by the text of the `errors` entry. A row
  refused before any block is produced writes `receipt.json` with `status:
  "failed"` and no `evaluation/` blocks at all. The three-state table is a
  design goal for the aggregator, not a property of the shipped artifacts.
- **`RESULT_PENDING` is declared, not implemented.** It is declared in
  configuration (`scoring.missing_result_marker` in
  [`configs/paper/protocol.yaml`](../configs/paper/protocol.yaml)) and described
  in prose in several shipped documents
  ([benchmark overview](benchmark_overview.md), [extending](extending.md),
  [reproducing the paper](reproducing_the_paper.md),
  [methods card](methods_card.md), [claims and limits](claims_and_limits.md)),
  but nothing under `src/` emits it or reads it, and no artifact currently
  distinguishes "not yet run" from "run and invalid" at the grid level.

This document proposes where the marker lives: as the `value` of a long-format
row whose `metric` is present but whose block is absent, with `contract_sha256`,
`scoring_version` and `identity_count` left explicitly null — so that a pending
cell is structurally incapable of being mistaken for a scored one. A pending
cell carries no number by construction.

---

## Strict versus legacy provenance

Two scoring paths exist in the tree and they are not interchangeable.

| Column family | Path | Semantics |
|---|---|---|
| `rel_l2_joint`, `rel_l2_by_horizon`, the static recovery diagnostics, `projected_rel_l2_joint` | **strict**, `pdeobs-strict-v1` | fail-closed: any non-finite, empty, duplicated, missing, extra, mis-shaped or time-mismatched record invalidates the whole block. Complete-set denominators only. |
| `relative_l2`, `mse`, `mae`, `spectral_centroid_error`, `high_frequency_energy_error`, `spectral_low`/`mid`/`high`, rollout stability, physical (vorticity/enstrophy/energy) errors, `<metric>_ood_degradation` | **legacy**, [`src/pdeobs/metrics.py`](../src/pdeobs/metrics.py) | tolerant: `nansum`/`nanmean` throughout, so NaNs are skipped and the effective denominator shrinks |

Concretely, legacy `relative_l2` builds its numerator and denominator with
`np.nansum` and reduces a batch with `np.nanmean`; strict v1 refuses the array
outright. **A legacy aggregate may therefore never be substituted for a strict
check, and a strict column may never be back-filled from a legacy report.**
The protocol already fixes this:
`legacy_results_automatically_strict_validated: false` and
`preserve_historical_scoring_version: true`.

Consequences for tables:

- Only the relative-L2 family is strict-validated. The spectral, stability and
  physical metrics are **never computed on the strict path**, so any table
  column built from them carries legacy semantics and must say so.
- Any table must identify **which cohorts and which scoring versions are
  pooled**. Mixing `pdeobs-strict-v1` blocks with historical legacy aggregates
  in one column, without a per-row `scoring_version`, is not a permitted
  presentation.
- Metric orientation is an open item; see below.

---

## Reproduce one cell

One cell of the grid is one `pdeobs paper-row` invocation plus the nine strict
evaluations it performs after training. The row configuration selects the axis
values; a template is
[`configs/paper/row_original500.yaml`](../configs/paper/row_original500.yaml).

```yaml
# configs/paper/row_original500.yaml
schema_version: pdeobs-paper-row-v1
cohort: original500                 # only original500 and demo are executable here
campaign: ../campaign/all_pde_one_setting_10method_9x9.yaml
registry: ../method/all_pde_source_faithful_registry.yaml
pde: poisson                        # darcy | poisson | helmholtz          -> recovery
                                    # heat | reaction_diffusion | burgers |
                                    #   navier_stokes                      -> rollout
method: fno                         # ufno_2d | fno | cno | deeponet |
                                    #   gnot | transolver | pino
train_view: random_50pct            # random_50pct  (R50) | random_65pct (R65)
                                    # random_80pct  (R80) | block_observed_50pct (BL)
                                    # line_sensors_50pct (LI)
                                    # horizontal_lines_50pct (H)
                                    # vertical_lines_50pct   (V)
                                    # boundary_band_50pct    (BD)
                                    # clustered_50pct        (CL)
training_seed: 20260804             # model init/training; split and mask use the
                                    #   campaign seed 20260804 for identity/view pairing
run_id: poisson-fno-r50-seed20260804-c500
attempt_id: attempt-1
```

**Edit a copy, and fix the two relative paths in the copy.** `resolve_row`
resolves both referenced files against the row file's own directory —
`campaign_path = (path.parent / row["campaign"]).resolve()` and
`registry_path = (path.parent / row["registry"]).resolve()`
([`src/pdeobs/paper_row.py`](../src/pdeobs/paper_row.py)). The template's
`../campaign/...` and `../method/...` are correct only for a file that sits
inside `configs/paper/`. A copy placed anywhere else must rewrite those two
values to paths valid relative to the copy, or to absolute paths; otherwise the
run fails at the configuration stage before any data is read.

```bash
# 1. Pin the identities you already hold. This hashes existing shards; it does
#    not generate data. Shard names below are illustrative, not bundled data.
#    The on-disk convention is <pde>/<boundary>/<setting>/<regime>/shard_NNNNN.h5.
pdeobs identity-manifest --data-root ./paper-data \
  --shards poisson/dirichlet/smooth_grf/low/shard_00000.h5 \
           poisson/dirichlet/smooth_grf/medium/shard_00000.h5 \
           poisson/dirichlet/smooth_grf/high/shard_00000.h5 \
  --output ./poisson-identities.json

# 2. Train the row and, after training, strict-score all nine test views.
#    This is a full 500-epoch training command, not a smoke test.
pdeobs paper-row --config ./my-row.yaml \
  --data-root ./paper-data --manifest ./poisson-identities.json \
  --output ./paper-runs/poisson-fno-r50-attempt-1 --device cpu
```

The runner enforces four preconditions that make the resulting cell reportable:

1. **Real shards plus an identity manifest.** Every listed shard's SHA-256 is
   re-verified before any target array is read; the runner will not search a
   different root, create missing data, or fall back to a loader's own
   `train`/`test` labels. The source identity set must equal the manifest, and
   `original500` additionally requires scalar 128 x 128 fields, exactly 2,000
   identities, 600 training identities per regime, 1,800 train / 200 test / 0
   validation, and 500 epochs.
2. **Training happens before any test array is touched.** `training_completion.json`
   is written first and records `test_arrays_accessed: 0`,
   `validation_records: 0` and `checkpoint_selection: "final_last_only"`; the
   fit call passes `val_loader=None`.
3. **Final-checkpoint reload into a fresh model instance.** The trained object
   is discarded, a new model is constructed, and `checkpoints/last.pt` is
   reloaded before evaluation. The runner does not select `best.pt` (which the
   shared trainer nonetheless writes; see the layout above), does not use test
   loss for stopping, and does not reuse a historical score. The saved epoch
   count and history length must equal the declared budget or the row fails.
4. **Nine strict cross-view evaluations after training.** Only the test mask is
   swapped between views; the checkpoint, the 200 identities and the contract's
   provenance hashes are identical across the nine blocks.

For a single ad-hoc block outside the paper row — for example a classical or
custom method — use `pdeobs strict-infer`. That single command already writes
`predictions.h5`, `contract.json` **and** `score.json` in its output directory;
a subsequent `pdeobs strict-score` is an independent re-score of the bound
artifact, not a required second step (see [Scoring](scoring.md)). Such a block
is not an `original500` paper cell and must not be placed in the same column as
one.

---

## What this release cannot yet state

Listed as open items rather than filled in:

- **No scored cell of the 441-row grid exists in this tree.** There is no
  results table, no benchmark leaderboard, and no per-row completion status
  file for the grid. The scored records under `acceptance/` are toy-scale
  software-acceptance evidence and are not benchmark numbers.
- **The 441-row campaign has not been run in this release** and its outputs are
  not part of this candidate. [`configs/paper/protocol.yaml`](../configs/paper/protocol.yaml)
  declares `purpose: descriptive_protocol_not_job_submission` and
  `status: anonymous_release_candidate`.
- **No multi-seed or repeat protocol, and therefore no error bars.** The
  protocol fixes a single seed (`20260804`) for the split and mask pairing; the
  population standard deviation computed by `reports.aggregate_records` across
  whatever rows land in a group is not a replication protocol and must not be
  presented as one.
- **No significance test, and no rule for ranking methods on the benchmark
  grid.** Nothing defines a confidence interval, a paired test across the 200
  identities, a normalization across families with different state magnitudes,
  a per-family weighting, or an aggregate "benchmark score" across the seven
  families and nine views. (A separate failure-analysis ranking does exist —
  `_failure_rankings` in [`src/pdeobs/difficulty.py`](../src/pdeobs/difficulty.py)
  sorts groups and records by a configured primary metric, stamps a `rank`
  field, truncates to `top_k` and writes `failure-rankings.csv` — but it ranks
  hard samples and factor combinations on legacy records, not methods on the
  strict grid.)
- **No per-block runtime or memory figure at paper scale.** Nothing records how
  long a nine-view cross-evaluation of one checkpoint on 200 identities takes,
  or what memory it needs. The acceptance flows do carry `train_s` and
  `peak_gpu_mb` per flow, but those are toy-scale software-acceptance timings on
  a single modern data-center GPU and cannot be extrapolated to a paper cell.
- **Metric orientation is not declared for the benchmark grid metrics.** No
  table in the documentation or configuration states higher- versus
  lower-is-better for the strict relative-L2 family or the legacy benchmark
  metrics, and the OOD-degradation helper
  (`ood_metric_degradation` in [`src/pdeobs/evaluation.py`](../src/pdeobs/evaluation.py))
  defaults every metric to lower-is-better unless a caller supplies an explicit
  `higher_is_better` orientation map. (The separate failure-analysis path does
  declare an orientation: `configs/analysis/difficulty.yaml` states
  "Error metrics default to lower-is-better" with
  `higher_is_better: [accuracy, ssim, psnr]`, consumed by `_metric_direction` in
  `difficulty.py`. That list does not cover the grid metrics.)
- **The strict aggregator is unwritten**, `RESULT_PENDING` is unimplemented, the
  full-observation control has no preregistered threshold, and the inverse task
  has no strict **export** adapter — `target_frame_metadata` raises "strict
  inference currently supports recovery, forward and rollout; inverse needs an
  explicit condition-coordinate adapter". Inverse arrays supplied by a caller
  *can* be strict-scored: `strict_score.score_arrays` accepts `task: "inverse"`,
  the API evaluate entry point scores caller-supplied arrays, and the API CLI
  exposes `--task inverse`. Each gap is named above at the point where it
  matters.

Two accounting numbers also need a single authoritative statement before any
table is published: the descriptive protocol declares 441 learned training rows
and 3969 blocks, while the retained historical campaign file and its validator
hard-require 504 rows, 4536 learned blocks, 126 classical blocks and 4662 in
total, because that file carries an eighth learned slot whose implementation is
deliberately not present in this repository. A results table must state which
denominator it uses and why.
