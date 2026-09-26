# AI/review overlays: not an official upstream release

Two overlays sit on top of the uploaded v0.2.0 tree. Each is identified separately, neither changes
the package version, and the original uploads are not overwritten. Overlay 2 is described first; the
overlay 1 notes are kept verbatim below it.

## Prediction-result completion (2026-09-25 UTC)

A new, separately versioned inference-only verification is complete for all 441
retained models and 3,969 full-200 blocks. Current scores are in
`results/prediction_verification_20260924/`; the historical archive is unchanged.
All prediction files and input bindings were freshly hashed during terminal audit.
Identity, shape, mask, geometry and physical-time mappings were checked against
frozen contracts; full-array finiteness receipts were authenticated against the
same bytes. No inference was repeated during final audit; no training, model,
loss, mask, split or tolerance changed.

All 793,800 per-record cell errors and each dynamic horizon are retained.
Mean/sample-SD and aggregate arithmetic were independently recomputed.
The fixed 13-pair/117-model original500 selection and complete configurations
were rechecked without selecting pairs from new scores. Counts: 3,646 old-strict
agreement, 314 additional <=1% differences, nine >1%; zero invalid blocks and
zero failed models. All valid new values are adopted.

CPU scripts and 48 metadata/regression tests are in `tools/prediction_analysis/`.
These are not new training or GPU performance measurements and do not replace
independent numerical-reference validation or establish external reviewer access.
The notes below are historical and remain unchanged.

## Overlay 2 (2026-09-23 UTC, CPU only)

Base: the overlay-1 archive `pdeobs-review-ai-ready.zip`, SHA-256
`43dbd102bf7ae703d8ae42e20fe25a4c52e364864e34ca7c8fd71576bfb05112` (304 files), which itself sits on
`pdeobs-review-main.zip` `ece1024b4d32b699e80a2918e77f3cd569ed57b4102dad2eb008035916ce0b79`.
File-level manifest: `AI_OVERLAY2_MANIFEST.json`. Logs: `review_audit/overlay2/`.

### Source change (1 facade file)

`src/pdeobs/api/infer.py`: `PredictionBundle` now validates at construction. Identities must be a
non-empty, unique string list matching the batch size; frame labels must be a non-empty,
non-negative, strictly increasing integer vector whose length matches the prediction frames; an
attached `InferenceInput` must carry the same identities in the same order. Overlay 1 had made
`load()` reject fractional or boolean frame labels, but an in-memory bundle built with
`time_indices=[0.9]` was still written as `[0]` (the writer casts to int64) and then reloaded and
scored as `valid`. That laundering path is closed at the source; `predict()` already produced valid
bundles and behaves as before. Regression tests: `tests/test_ai_overlay2_regressions.py` (18 cases).

### New module (not a protected file)

`src/pdeobs/evidence.py` (`python -m pdeobs.evidence plan|check|export`): bundle schema
`pdeobs-evidence-bundle/v1`, check report `pdeobs-evidence-check/v1`, error `pdeobs-evidence-error/v1`.
It links a paper cell to resolved configuration, identity manifest, split receipt, final checkpoint,
per-view contract, prediction artifact and strict score; hashes present files; re-scores each
`predictions.h5` with the frozen scorer; checks the split receipt against the declared cohort; and
reports each category as passed / blocked / failed on its own. `check` is read-only; `export`
refuses existing directories and demo bundles without `--allow-demo`. Tests:
`tests/test_evidence_bundle.py` (11 cases on a real demo row written by `paper_row`, plus tamper
copies). The legacy `pdeobs` command module is protected and untouched, so the tool is a separate
module rather than a subcommand. `tools/build_acceptance_index.py` generates the acceptance index
from the shipped acceptance files and writes nothing else.

`tools/archived_results.py` (`scan | pair | index`, Python 3.6+ standard library) indexes the
archived campaign runs behind the paper's grid: it walks the campaign runner's training and
evaluation outputs, hashes every record, joins evaluation to training by checkpoint SHA-256,
classifies the training cohort from the recorded protocol fields, checks the bindings, and writes
the long-format table and pivots of `docs/results_format.md` with an explicit `RESULT_PENDING`
marker for every missing cell. Those runs retained no prediction tensors, so their metrics are the
campaign evaluator's legacy relative L2 and every row says so (`docs/archived_results.md`).

### Documentation and navigation

New: `docs/paper_evidence.md`, `docs/ai/ACCEPTANCE_INDEX.md` + `docs/ai/acceptance_index.json`
(generated; hashes, superseded markers, unresolved conflicts, the U-FNO batch constraint).
Updated: `AI_START_HERE.md`, `AGENTS.md`, `llms.txt`, `docs/ai/WORKFLOWS.md`, `docs/ai/PROTOCOLS.md`,
`docs/ai/repository_index.json`, `docs/ai/evidence_status.json` (overlay-2 measurements added, the
overlay-1 block kept), `README.md` (overlay banner; a section separating provided features,
this-round acceptance, paper evidence and author-required materials; licence separation),
`docs/cli_reference.md` (pointer).

### What did not change

No solver, generator, mask, split, loss, model, metric, protected configuration, original test
assertion or archived report. `docs/ai/frozen_science_sha256.json` is the input baseline and was not
rewritten; the doctor reports 0 changed protected files. The legacy `easy` CLI error path (text on
stderr, exit 2) is unchanged because `src/pdeobs/cli.py` is protected.

### Regression evidence for overlay 2

`docs/ai/evidence_status.json` under `overlay_2_this_run`, with the raw logs under
`review_audit/overlay2/`. The totals are for the whole suite of this tree and overlap with every
earlier count; they are not additive. The tests were run on CPU only; no GPU statement is made.

### Second-audit corrections (2026-09-23, same overlay)

An independent second review of the pushed tree reproduced false-positive states in the two new
evidence tools. Corrected without touching any protected file or archived value:

- `src/pdeobs/evidence.py`: required manifest structure is checked first (`manifest_structure`); PDE, method,
  views, task, cohort, epochs, seeds and run/attempt ids are cross-checked across manifest, receipt, resolved
  configuration, completion record and contracts, including the receipt's block inventory and `score_sha256`
  (`metadata_consistency`); `training_config_sha256` and `training_recipe_sha256` are recomputed from
  `row_resolved.json` with the writer's rules and the trainer file is compared with the resolved training
  section (`configuration_identity`); finiteness is its own category, `blocked` when re-scoring is skipped and
  `failed` when an artifact cannot be read; `metric_recomputation` fails when it cannot be performed; an
  export writes a flat `manifest.json` and its `evidence_bundle.json` can be re-checked directly.
- `tools/archived_results.py`: every identity/hash disagreement fails `blocks_complete`, metrics must be finite
  real scalars, and `metrics_file_binding` and a failed `training_completion` propagate to `inconsistent`;
  index JSON is written with `allow_nan=False`. The archived index was regenerated with the corrected checker;
  no stored value changed.
- `tests/test_second_audit_regressions.py`: the reviewer's negative cases plus further ones (receipt inventory,
  seeds, score hash, missing identity hashes, test-record access, export re-check).
- Documentation: the pre-index statements ('no result numbers', 'cannot be re-run') were replaced by the
  accurate provenance statement in the current entry points; historical logs are untouched.

### Still missing (author-side)

Unchanged from overlay 1: the evaluated data or a verified identity/checksum manifest, production
checkpoints, per-cell prediction artifacts and scores, numerical-reference reports. With those files
in a `paper-row` layout, `python -m pdeobs.evidence check` runs on them without any code change.

## Overlay 1: base audit (historical notes, kept verbatim below)

Base uploaded ZIP SHA256: `ece1024b4d32b699e80a2918e77f3cd569ed57b4102dad2eb008035916ce0b79`. Base package version stays `0.2.0`;
this overlay is identified separately. Original upload is not overwritten.

### Source changes (5 facade files)

1. `src/pdeobs/api/data.py`: reject nonfinite/nonbinary masks **before** converting them;
   reject nonfinite geometry and invalid integer source-frame labels.
2. `src/pdeobs/api/evaluate.py`: preserve strict frame-type checks; support NumPy target IDs;
   align observation/mask rows to independent target ordering; do not cast raw masks to bool first.
3. `src/pdeobs/api/cli.py`: preserve target-time types until validation instead of truncating floats.
4. `src/pdeobs/api/infer.py`: check saved prediction bundle schema/version and integer frame labels.
5. `src/pdeobs/api/pipeline.py`: reject misspelled/duplicate/empty/malformed stage declarations;
   reject missing stage sections; propagate an invalid evaluation as a failed pipeline receipt.

No loss, numerical solver, mask generator, paper split algorithm, strict metric definition, original
configuration, original test assertion, or archived result is changed. Rejection of formerly
silently coerced malformed API inputs is intentional. This is not retrospective rescoring of any
paper result. The protected source hashes are in `docs/ai/frozen_science_sha256.json`.

### New AI interface

`AI_START_HERE.md`, `AGENTS.md`, `llms.txt`, a bounded context/doctor/CPU-smoke tool,
a machine-readable repository map, evidence map, and source-hash guard. Everything is local;
no AI API key, server, cloud rental, scheduler, or automatic package download is introduced.

### Regression evidence

The original source passed all597 shipped tests in this audit. The29 new facade tests plus four helper-tool tests and all original tests passed **630/630**
(0failed/0skipped) in the final run. Earlier scoped totals are recorded separately in
`docs/ai/evidence_status.json`; these are overlapping counts, not additive campaigns.
The original non-editable install ran E1–E6 outside the source cwd. The added tiny CPU smoke ran
Poisson recovery and Heat rollout with four TRAIN and two disjoint TEST demo identities.

### Existing documentation caveats, not silently rewritten

`ACCEPTANCE_SUMMARY.md` describes64^2/49 short flows and two general views. Release notes
lines53–55 describe128^2/batch1/nine paper views; round2 contains separate tiers with their own
scope. Treat the concrete receipts separately. The '521 +37' phrasing is also not a basis to add
counts: the portable runner selects521 total cases. Older release metadata concerns v0.1.1 and
is preserved as historical evidence. This overlay does not authenticate old GPU logs.

### Remaining work

Provide the evaluated data or a verified regenerable identity/checksum manifest, production model
artifacts, and actual scientific results; perform/attach numerical-reference checks and matched
controls. Those are author-side evidence tasks, not solved by adding more Markdown or unit tests.

This patch is reviewed/tested on CPU only. Review the diff in a separate working copy before
merging; do not replace code beneath already-running production jobs.

## Mask specs and a pinned mask seed (2026-09-25)

Additive interface change; no registered protocol, paper view, hash or seed derivation of an
existing configuration is altered, and the existing suite is expected to pass unchanged.

- New module `src/pdeobs/mask_specs.py`: a protocol name may carry a parameter (`uniform 1`,
  `block 50`, `line 30`, `boundary 50`), and `+` joins components into a deterministic mixture
  (`uniform 1 x2 + sensor 20 + block 50`). Accepted for training (`data.mask.protocol`) and for
  test views (`evaluation.observation_protocols`). The number is always the observed percentage and
  is converted to the protocol's own parameter from the field shape; `boundary 50`
  recovers the frozen 128 x 128 band (width 19); `line 50` solves to 75 lines (50.02 %).
- `BenchmarkDataset(mask_seed=...)` and `data.mask_seed`: sensors are drawn from the mask seed,
  which defaults to the dataset seed. Paper rows already wrote `data.mask_seed`; the runner now
  passes it through. With `training.seed` set separately, a multi-seed subset keeps split and sensors
  fixed and varies only initialisation and batch order.
- Runner: spec views are keyed by a canonical id, duplicates are rejected, and the view context
  records the spec, its components and the weight-averaged target observation ratio.
- Tests: `tests/test_mask_specs.py` (parser, realised fractions at 128 x 128, paper-geometry
  recovery, weighted rotation, bit-for-bit equivalence with the legacy parameterised mask,
  mask-seed pinning, runner views). Documentation: `docs/observations.md`.
- Follow-up the same day: the `easy` API (`api.make_observation`, `--obs`, pipeline `observation:`)
  accepts the same spec strings; package version 0.2.1 (`RELEASE_NOTES_v0.2.1.md`); reader guide
  `docs/observation_specs.md`; the protected-source manifest records the `__init__.py` revision.

## Current-release metadata relabelling (2026-09-26)

Remaining private source-commit strings, hardware UUIDs and local filesystem prefixes in shipped
records use anonymous labels or repository-relative paths. Acceptance round labels stay distinct;
their reported outcomes are unchanged. The acceptance index was rebuilt from the relabelled sources,
and live digest references were updated. The earlier relabel receipt is preserved as a historical
record; the new receipt is `results/public_deposits/relabel_receipt_20260926_followup.json`.

Scientific-contract impact: none. No numerical kernels, training configuration, arrays, weights,
scores or the frozen `results/prediction_verification_20260924/` bundle were changed. Remote historical
Git objects are outside this follow-up's scope.
