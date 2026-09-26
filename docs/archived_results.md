# Archived campaign results: how the tables are generated without re-running anything

The paper's grid was executed by a campaign runner over several machines and several attempt waves.
Those runs were not rerun here; rescoring from retained arrays is unavailable because the runner kept no
arrays, and re-inference depends on access to the checkpoints and data. They were not produced by
`pdeobs paper-row`. This
page states what the archived outputs contain, how they are turned into the results tables, and what
that process does and does not establish. The tool is `tools/archived_results.py` (Python 3.6+,
standard library only, so it runs on a login node as it is).

## What one archived run consists of

| Side | Files | What they record |
|---|---|---|
| training output | `identity.json`, `completion.json`, `health.json`, `split_manifest.json`, `factor_coverage.json`, `checkpoints/training_config.json`, `checkpoints/last.pt`, `history.json`, `resolved.yaml` | PDE, method, training view, planned epochs, registry / campaign / dataset identities (SHA-256), completion status, checkpoint SHA-256, health events, the resolved training configuration, the 1800/200 split receipt |
| training output, cohort dependent | `budget-protocol.json`, `dynamic-protocol.json`, `operational-continuation*.json`, `checkpoints/budget200-state/*.json` | the budgeted protocol (max 200 epochs, no plateau accumulation through epoch 120, then patience 10 on the training data loss), or the continuation records of a resumed attempt |
| evaluation output | `completion.json`, `blocks/<test view>/metrics.json` (nine views) | the checkpoint SHA-256 that was evaluated, the frozen evaluator's identity, the acceptance policy applied, and per view: 200 test identities (their set hash), the metrics, the mask protocol and observation ratio |

Two facts about these outputs decide what can be claimed from them:

1. **No prediction tensors were retained.** The runner stored metrics, not arrays. The relative-L2
   values are therefore the campaign evaluator's output (the legacy relative-L2 family of
   [Results format](results_format.md)), not `pdeobs-strict-v1` blocks, and they cannot be recomputed
   from arrays. Every table row carries `scoring_version = legacy-campaign-evaluator:<sha>` for that reason.
2. **The training cohorts differ by attempt.** Some rows ran the original 500-epoch protocol, some the
   budgeted max-200 / min-120 / patience-10 adapter, some are recovered or salvaged attempts. The cohort
   is read from the recorded protocol fields and reported per row; no row is relabelled
   ([`configs/paper/training_cohorts.yaml`](../configs/paper/training_cohorts.yaml)).

## The generation statements

On each machine that holds archives (read-only; hashes every JSON file and, optionally, the checkpoints):

```bash
python tools/archived_results.py scan \
  --training-root   <root>/_preliminary-training/<campaign> \
  --evaluation-root <root>/_preliminary-evaluation/<campaign> \
  --label <machine label> --out ./scan-<machine label>.json          # add --hash-checkpoints to re-hash every last.pt (slow)
```

On any machine, with the scan files collected (no archives needed from here on):

```bash
# one selected attempt per identity; evaluation is joined to training by checkpoint SHA-256, or by the
# source-artifact hashes a salvage evaluation records; an explicit selection file pins the credited attempt
python tools/archived_results.py pair --scan scan-*.json --selection selection.json --out ./paired.json

# the public index, the per-identity checks, the long-format table and the pivots, into a NEW directory
python tools/archived_results.py index --paired ./paired.json --out-dir ./results
```

`results/` then holds `index.json` (one record per identity: cohort with its evidence, actual epochs,
stop reason, checkpoint identity, resolved-configuration identity, registry / campaign / dataset
identities, the split receipt, the health summary, the nine blocks with their metrics and file hashes),
`checks.json`, `results_long.csv`, `matched_diagonal.csv`, `cross_view/<pde>__<method>.csv`,
`pending.json` and `summary.json`. Absolute paths, machine names, job identifiers and device records
are not written to the index; attempts are named by the SHA-256 of their completion record.

The selection file is `{identity: {evaluation_completion_sha256 | evaluation_rel | checkpoint_sha256}}`.
Without it, the tool keeps the unique complete evaluation of an identity and, when several exist, the
latest one **flagged as an unverified choice**; with it, a selection that matches nothing is reported,
never replaced by a guess.

## What the checks establish

| Check | Passed means | Never means |
|---|---|---|
| `blocks_complete` | nine views, 200 identities each, one recorded test identity set shared by all nine blocks and equal to the split manifest's; any of these missing or disagreeing fails | that the identities are authentic |
| `checkpoint_binding` | the evaluation record, every block, the training completion record (and the checkpoint file, when re-hashed) name the same checkpoint SHA-256 | that the weights are what the paper describes |
| `metrics_file_binding` | each `metrics.json` hashes to the value the evaluation completion record stores for it; a mismatch makes the identity `inconsistent` | that the metrics are correct |
| `dataset_binding` | dataset-summary and campaign identities agree between evaluation and training | that the data are authentic |
| `split_provenance` | the split receipt has the paper algorithm, seed 20260804, 2000 records, 1800/200, regimes 600/600/600 and 67/67/66 | that the shards themselves are authentic |
| `training_completion` | the training record completed without test-record access or validation; health events are listed (`flagged`) rather than hidden | anything about quality |
| `finiteness` | `relative_l2` (and the per-horizon values where present) is a finite real scalar in every block and the recorded non-finite rate is zero | strict-v1 validation |
| `metric_recomputation` | always `blocked`: no arrays exist to recompute from | |

`verification_status` is `consistent`, `incomplete` (a training record could not be paired or a field is
unrecorded) or `inconsistent` (a binding fails). It is internal consistency of the archive, nothing more:
numerical-reference accuracy, external reproduction and any recomputation are not established by it.

## Cohort labels

| Label | Rule (from the recorded fields) |
|---|---|
| `original500` / `original500_recovered` | training completed, planned 500, completed 500; `_recovered` when continuation records exist |
| `budget200_min120` | the recorded training protocol names the max-200 / min-120 adapter; `actual_epochs` and `stop_reason` are reported with it (patience stops end between epoch 130 and 200) |
| `budget200_patience10_no_floor` | a max-200 patience-10 protocol without the epoch-120 floor, as recorded; the cohort table does not list it, so it is reported under its own label |
| `fixed200` / `fixed200_recovered` | declared 200-epoch budget completed; `_recovered` when continuation records exist |
| `interrupted_or_recovered` | anything else: a salvaged or user-terminated checkpoint evaluated before its budget was reached; `budget_complete` is false |

## What ships in this tree

`results/archived_campaign/` is the index generated on 2026-09-23 from four archives: all 441
planned identities with a complete nine-view evaluation (3969 blocks), all 441 `consistent` under
the checks above, none pending, so the grid has no missing cell. The epoch ceiling is declared per
setting rather than summarised - 500 epochs for 350 settings, 200 for 91. Fixed-budget,
training-loss plateau, continuation and interrupted/salvaged records are distinguished;
not every early endpoint is a plateau stop. The shared template and exact cohort counts are in
[Training template and records](training_template_and_records.md). Each row carries its `planned_epochs`,
`actual_epochs`, `stop_reason`, `training_cohort`, `checkpoint_id`, `resolved_config_identity` and
`scoring_version`.
Its own README lists the counts per cohort and the caveats that apply to every number in it.

A separately versioned [prediction verification](prediction_verification.md) can now retain
new prediction arrays and recompute their scores. It does not overwrite the archived index,
retroactively create missing historical arrays, or imply that all checkpoints have already
passed the new audit. The [production quality summary](../results/numerical_quality/README.md)
reports actual generation-time checks bound to the archive's dataset-summary digest.

## Missing and failed cells

A planned identity without a selected evaluation appears in `pending.json`, `results_long.csv` and
`matched_diagonal.csv` as `RESULT_PENDING` with the count and health events of its unpaired training
attempts. Whether such a cell is "still running", "terminal after bounded retries" or "never attempted" is
a campaign-ledger statement, not something the archive alone proves; the tool reports what the archive
holds and nothing more.
