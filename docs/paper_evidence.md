# Paper evidence: linking a table cell to the files that produced it

This page describes the one route by which a number in a paper table can be traced back to the
artifacts that produced it, and the local tool that checks that route:
`python -m pdeobs.evidence`. It is part of the review overlay (see `AI_PATCH_NOTES.md`), it is a
separate module because the legacy `pdeobs` command module is a protected scientific source file,
and it never trains, generates records, downloads, or launches anything.

**Status of this tree.** No production dataset, production checkpoint or prediction artifact is
distributed with this copy; the archived campaign metrics are indexed under
`results/archived_campaign/` with cohort and scoring-provenance labels, and they are not strict-v1
blocks (README, *Datasets* and *Results*; [Archived results](archived_results.md)). The checker can
therefore be exercised here only on the two software demo rows of
[Reproducing the paper](reproducing_the_paper.md) (*Small static and rollout integration runs*)
and on synthetic fixtures in `tests/test_evidence_bundle.py`. A paper cell reaches a verified
status only when the author-supplied materials listed at the end of this page are present and
every check passes on them.

## The evidence chain

Every link below is a file that one `pdeobs paper-row` invocation writes into one output
directory (README, *Reproducing one grid cell*). The checker creates none of them.

| Link | File in the row directory | Schema |
|---|---|---|
| paper protocol (description) | `configs/paper/protocol.yaml`, `configs/paper/observations.yaml` | none |
| executable row | a copy of `configs/paper/row_original500.yaml` | none |
| resolved configuration | `row_resolved.json` | none |
| training configuration | `checkpoints/training_config.json` | none |
| identity manifest | `identity_manifest.json` | `pdeobs-identity-manifest-v1` |
| split receipt | `split.json` | `pdeobs.one-setting-split/v1` (original500) or `pdeobs-paper-row-demo-split-v1` (demo) |
| training completion | `training_completion.json` | `pdeobs-paper-row-training-completion-v1` |
| final checkpoint | `checkpoints/last.pt` | its SHA-256 is the `checkpoint_id` everywhere else |
| per-view contract | `evaluation/<view>/contract.json` | `pdeobs-strict-contract-v1` |
| per-view prediction artifact | `evaluation/<view>/predictions.h5` | `pdeobs-strict-inference-v1` (SHA-256 bound to its contract) |
| per-view score | `evaluation/<view>/score.json` | `pdeobs-strict-v1` |
| attempt receipt | `receipt.json` | `pdeobs-paper-row-v1` |

Bindings the checker verifies: the checkpoint file's SHA-256 equals `checkpoint_id` in the
receipt, the completion record and every contract; `dataset_manifest_sha256` in those files equals
the canonical hash of `identity_manifest.json`; `split_sha256` in the completion record equals the
canonical hash of `split.json`; every contract's `expected_ids` are exactly the split's test
identities, in order; `training_config_sha256` and `training_recipe_sha256` agree between the
completion record and the receipt; `checkpoint_selection` is `final_last_only`.

## The bundle manifest (`pdeobs-evidence-bundle/v1`)

`plan` derives it from `receipt.json`; it can also be written by hand for materials that were not
produced by `paper-row`, as long as every path is relative to the bundle root.

| Field | Meaning |
|---|---|
| `schema_version` | `pdeobs-evidence-bundle/v1` |
| `experiment_id` | `run_id/attempt_id` from the receipt |
| `purpose` | `paper-evidence` or `software-demo`; derived from the cohort, and a demo cohort is always checked as demo |
| `task`, `pde`, `method`, `model_adaptation` | the row axes; the method is an adaptation registered in the method registry, not an upstream reproduction |
| `training_view`, `training_protocol` | cohort, declared and actual epochs, checkpoint selection, stop reason, registry and campaign hashes |
| `test_protocol` | split label, `expected_identity_count` (200 for `paper-test`), test views |
| `seeds` | `split_seed` (campaign seed: split and masks) and `training_seed` (model only) |
| `dataset` | `version` (equal to `dataset_manifest_sha256`), `identity_manifest`, `split_manifest` |
| `artifacts` | receipt, resolved config, training config, training completion, checkpoint, and one block per test view (`contract`, `predictions`, `score`) |
| `scorer_version`, `metric` | `pdeobs-strict-v1`; per-identity relative L2 in float64, arithmetic mean over every expected identity; a dimensionless ratio, not a percentage |
| `paper_reference` | table, row and cell the bundle backs; filled by the author |
| `numerical_reference` | optional `report` path; recorded, not verified |

## Check categories

Each category reports its own status. A passed category never implies another one.

| Category | `passed` means | it never means |
|---|---|---|
| `manifest_structure` | every required manifest field, artifact declaration and block declaration is present; the declared views are unique and match `test_protocol.test_views` | anything about the files |
| `artifact_completeness` | every declared file exists and was hashed (SHA-256 recorded per file); a missing file makes the category `blocked` | that the file is the one used for the paper |
| `metadata_consistency` | PDE, method, training view, task, cohort, actual epochs, seeds and run/attempt ids agree across manifest, receipt, resolved configuration, completion record and every contract; each block's contract carries the block's view; the receipt's block inventory and `score_sha256` values match the declared blocks | that the labels are true of the paper |
| `configuration_identity` | `training_config_sha256` and `training_recipe_sha256` recomputed from `row_resolved.json` with the writer's own rules equal every record's value; the trainer file agrees with the resolved training section on every shared key, seed and task | that the configuration is the paper's |
| `identity_shape` | contract identities are unique and match `expected_identity_count`; `predictions.h5` has the contract's shape, identity count and artifact version; the stored score is `valid` with matching counts | that the identities are the paper's 200 |
| `finiteness` | the stored prediction and target arrays pass strict re-validation (no non-finite entries); `blocked` when re-scoring is skipped, `failed` when the artifact cannot be read | anything about accuracy |
| `metric_recomputation` | re-scoring `predictions.h5` against its contract with the frozen scorer reproduces `score.json` (mean and every per-identity value, relative tolerance 1e-9) and the artifact is SHA-256 bound to the contract; `failed` when re-scoring is impossible, `blocked` when skipped | that the predictions came from the claimed training |
| `split_provenance` | the split receipt has the cohort's schema and algorithm, the bundle's seed, a three-part macrodomain identity (pde, boundary, setting), disjoint train and test sets (original500: 2000 records, 1800/200, regime counts 600/600/600 and 67/67/66), and every contract lists exactly the test identities in order | that the shards themselves are authentic |
| `checkpoint_binding` | the bindings listed above hold | that the weights are the paper's |
| `purpose_guard` | the declared purpose is consistent with the artifacts; a paper-evidence claim over a demo cohort, a non-`paper-test` contract or a smoke receipt fails | tamper-proof provenance |
| `numerical_reference_evidence` | `recorded_unverified` when a report is declared and present; `not_attempted` otherwise | that the targets are reference-checked |
| `external_reproduction` | always `not_attempted`: an independent run is outside this tool | anything |

Aggregate `verification_status`: `complete` when every core category passed (exit 0),
`incomplete` when something is blocked (exit 3), `failed` when any core category failed (exit 2).
A bad manifest, an unreadable artifact, an existing output directory, or a demo bundle offered to
the paper export path is an error: one JSON line on stderr with `category`, `reason` and
`next_step`, nothing on stdout, exit 2. Successful commands print one JSON document on stdout.

## Checking one stationary block and one temporal block

The commands are identical for a demo row and a paper row; only the inputs differ. First produce
the two demo rows (CPU, nine 16 x 16 records each, one epoch; software only):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
pdeobs paper-row-demo-data --pde poisson --output ./row-demo-data/poisson
pdeobs paper-row --config configs/demo/paper_row_static.yaml \
  --data-root ./row-demo-data/poisson --manifest ./row-demo-data/poisson/identities.json \
  --output ./row-demo-runs/poisson --device cpu
pdeobs paper-row-demo-data --pde heat --output ./row-demo-data/heat
pdeobs paper-row --config configs/demo/paper_row_rollout.yaml \
  --data-root ./row-demo-data/heat --manifest ./row-demo-data/heat/identities.json \
  --output ./row-demo-runs/heat --device cpu
```

Then plan, check and (optionally) export each block into a new directory:

```bash
# stationary block: Poisson recovery, two test views
python -m pdeobs.evidence plan  --root ./row-demo-runs/poisson --out ./evidence-checks/poisson-bundle.json
python -m pdeobs.evidence check --bundle ./evidence-checks/poisson-bundle.json \
  --root ./row-demo-runs/poisson --out ./evidence-checks/poisson-check.json
python -m pdeobs.evidence export --bundle ./evidence-checks/poisson-bundle.json \
  --root ./row-demo-runs/poisson --out ./evidence-checks/poisson-export --allow-demo

# temporal block: Heat rollout, frames 1,2,3 predicted from frame 0
python -m pdeobs.evidence plan  --root ./row-demo-runs/heat --out ./evidence-checks/heat-bundle.json
python -m pdeobs.evidence check --bundle ./evidence-checks/heat-bundle.json \
  --root ./row-demo-runs/heat --out ./evidence-checks/heat-check.json
```

An export can be re-checked directly: `python -m pdeobs.evidence check --bundle <export>/evidence_bundle.json`
(the root defaults to `<export>/artifacts`; files not copied are reported as blocked, exactly as they are
absent). Expected on the demo rows: `verification_status: complete`, `purpose: software-demo`,
`split_provenance` reporting 6 train / 3 test identities, and the export labelled
`software-demo export, not paper evidence`. Without `--allow-demo` the export exits 2 with
category `demo_into_paper_export`. The scores in these reports are one-epoch numbers on nine
16 x 16 records and mean nothing scientifically.

For a paper cell, point `--root` at the author's `paper-row` output directory (cohort
`original500`). `plan` then derives `purpose: paper-evidence` and
`expected_identity_count: 200`; add the table reference with
`--paper-ref '{"table": "2", "row": "poisson/fno", "cell": "R50"}'`. Two things are deliberately
impossible: the 200 formal test identities cannot be replaced by demo identities to make a check
pass (the contract, the split receipt and the identity manifest must agree), and regenerated
records cannot stand in for the historical evaluated data (the manifest hash recorded in the
contracts must match the manifest supplied).

## Materials the authors must supply for a paper cell

Minimal list, per claimed cell:

1. The `paper-row` output directory of the attempt: `receipt.json`, `row_resolved.json`,
   `checkpoints/training_config.json`, `training_completion.json`, `checkpoints/last.pt`,
   `identity_manifest.json`, `split.json`, and `evaluation/<view>/{contract.json,
   predictions.h5, score.json}` for every claimed view. Equivalent files under other names need a
   hand-written manifest.
2. The table cell the directory backs (`paper_reference`), and the attempt that is the reported one
   when several attempts exist.
3. For the `reference-checked` claim on the evaluated targets: the numerical-reference report, so
   that it can at least be recorded next to the bundle.
4. For external reproduction (not performed by this tool): the shards named in
   `identity_manifest.json`, with the SHA-256 values it records, or a deposit that reproduces them.

What is missing from this copy for every one of the 441 planned cells is exactly items 1 to 4.
Software tests, the smoke flow and the demo rows do not shorten that list.

## Archived campaign runs

The runs behind the paper's grid were produced by a campaign runner whose outputs differ from the
`paper-row` layout above and which retained no prediction tensors. They are indexed, checked and
tabulated by `tools/archived_results.py` without re-running anything; see
[Archived results](archived_results.md) for the layout, the commands, the cohort labels and what
those checks do and do not establish.

## What the checker does not do

* It does not train, generate data, download, copy large files by default, or write outside the
  directory named by `--out`, and it refuses an existing output directory.
* It does not authenticate the origin of the data, the weights or the predictions: a passed check
  means the supplied files are mutually consistent and the stored score follows from the stored
  arrays.
* It does not turn a demo or smoke result into paper evidence, and it does not compare a score with
  any paper table: that comparison is the reader's, with the cell reference recorded in the bundle.
