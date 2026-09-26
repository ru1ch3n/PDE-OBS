# Revision report: repair-and-analysis release (2026-09-26)

Scope decided on 2026-09-26: new analyses use the **441-model main grid only**; the four completed mixed-pattern
rows keep their strict rebuilt comparisons (section 1) but no budget-matching audit is pursued; records of failed
attempts were removed from this bundle. Items that were not executed are described from the material that exists.
Nothing in this bundle was trained. Every number below is reproducible from the declared inputs with the commands in
`tools/revision/README.md`; every tool records the digests of what it read.

| item | status | evidence |
|---|---|---|
| 1. result sources and rebuilt mixed comparisons | done, five mixed rows | `mixed_summary.json`, `mixed_comparisons.csv`, `mix_strict/` |
| 2. training inventory (441) and mixture tables | done (reduced scope) | `training_inventory.csv`, `trial_inventory.json`, `mixture_assignments.csv`, `mixture_counts.csv` |
| 3. observed/hidden decomposition and projection | done, 1,701 blocks | `stationary_diagnostics_*.csv*`, `stationary_diagnostics_checks.json` |
| 4. paired test-identity bootstrap | done, 5,000 replicates | `bootstrap_intervals.csv`, `bootstrap_config.json` |
| 5. simple baselines under the formal protocol | not run | existing implementations and demo-scale strict reports only (section 5) |
| 6. independent numerical checks | not run | recorded production residual consistency only (section 6) |
| 7. anonymous access of the public deposits | re-published 2026-09-26, access and integrity verified | `access_check.json` (section 7) |

## 1. Result sources and the rebuilt mixed comparisons

`configs/revision/result_sources.yaml` declares the release inputs with digests: the prediction-verification release
`pdeobs-prediction-verification/20260925-v1` (index, per-identity errors, dataset bindings, statistics), the frozen
campaign views, the mask seed, the split algorithm and the target frames. Legacy evaluator scalars are forbidden as
numerators or denominators and appear only in an audit column.

Numerators for the four completed mixed rows come from inference-only strict rescoring of the retained study checkpoints
(`tools/revision/mix_strict_inference.py`; 48 contract-bound prediction files, zero optimizer updates); they reproduce the
study's legacy evaluator values to at most 1.6e-9. Denominators are the strict main-grid scores. The builder refuses a
block whose 200 identities, mask, target frames, checkpoint or scoring version differ from the specialist block.

| pair | population | n | mean_mix | mean_specialist | geomean ratio |
|---|---|---|---|---|---|
| helmholtz / FNO | all_nine_descriptive | 9 | 0.0531 | 0.0212 | 2.065 |
| helmholtz / DeepONet | all_nine_descriptive | 9 | 0.1214 | 0.0501 | 2.562 |
| heat / DeepONet | all_nine_descriptive | 9 | 0.1329 | 0.0983 | 1.319 |
| poisson / U-FNO | all_nine_descriptive | 9 | 0.0712 | 0.1165 | 0.460 |
| poisson / U-FNO | recipe_matched (block, boundary, clustered) | 3 | 0.1710 | 0.0550 | 3.051 |
| navier-stokes / U-FNO | all_nine_descriptive | 9 | 0.0349 | 0.0255 | 1.186 |

All 45 mixed cells lie below the median of the eight transferred specialists; 34 of 45 lie below the best transfer
(Navier-Stokes: 9 of 9 and 7 of 9; its ratios range from 0.63 on random 50 % to 2.48 on the boundary band).
The Poisson restricted population is declared from the receipts (same recipe: one-cycle schedule, learning rate 5e-4,
200-epoch ceiling with the min-120 / patience-10 stop, 200 completed epochs, 90,000 updates for the mixed model and for
each of the three specialists, all in `training_inventory.csv`); the six other Poisson specialists ran the step schedule
with patience stops (130 to 163 epochs), so the all-nine number is descriptive. The Navier-Stokes mixed row finished on
2026-09-26 (500 epochs, 225,000 updates, no health events) and was rescored the same way (12 strict blocks; legacy
evaluator reproduced to 2.8e-17).

## 2. Training inventory (441) and mixture tables

`training_inventory.csv` has one row per credited model with optimizer, schedule, batch size, planned and completed
epochs, the recorded optimizer-step counter (or `unknown`), derived examples, stopping rule, stop reason, training status
and an evidence source per field. Sources: the verification index (cohort, epochs, configuration, checkpoint digests), the
archived training receipts (step counters, repository commit; their legacy scores are never read) and the `resolved.yaml`
of every released model, digest-bound to the index (`inputs/resolved_configs_441.json`).

- cohorts: original500 257, original500_recovered 66, budget200_min120 44, fixed200_recovered 36,
  interrupted_or_recovered 28, budget200_patience10_no_floor 10;
- step counters: 412 equal completed epochs x updates per epoch, 2 exceed it (resumed segments), 27 unknown
  (`trial_inventory.json` lists them); nothing is derived from a planned budget;
- training status from the receipts: 412 completed, 2 recorded as failed (constant-output collapse, still credited with
  nine valid strict blocks), 27 unknown; all 441 have nine valid strict blocks of 200 identities.

`mixture_assignments.csv` runs the release assignment logic (`pdeobs.mask_specs`) over the frozen identity lists of the
four mixed PDEs (8,000 identities): component, mask digest and observed count per identity; `mixture_counts.csv` gives
counts by split x regime x component. The reconstructed identities reproduce the frozen split digests, the assignment is
identical under reversed identity order, and the training counts equal the assignment counts staged with each mixed row.

## 3. Observed/hidden decomposition and data-consistency projection (1,701 stationary blocks)

Every retained stationary prediction file of the verification release was reopened on the host that holds it and
rescored once with `projection: true` by the unchanged strict scorer (`tools/revision/static_diagnostics.py score`,
CPU only). For every one of the 340,200 identities the raw error equals the sealed score (1e-12), the identity
raw^2 = observed^2 + hidden^2 holds (rel 1e-9), the projected error equals the hidden contribution and never exceeds the
raw error. Block means equal the released index (1e-9).

Medians over blocks (raw and projected are block means over 200 identities; the share is
sum_i e_obs,i^2 / sum_i e_i^2, a share of squared error, not a fraction of wrong cells):

| PDE | blocks | median raw | median projected | median projected / raw | median observed squared-error share |
|---|---|---|---|---|---|
| darcy, matched | 63 | 0.0403 | 0.0276 | 0.732 | 0.431 |
| darcy, cross | 504 | 0.3628 | 0.2712 | 0.724 | 0.470 |
| poisson, matched | 63 | 0.0301 | 0.0195 | 0.737 | 0.432 |
| poisson, cross | 504 | 0.4731 | 0.3651 | 0.743 | 0.439 |
| helmholtz, matched | 63 | 0.0242 | 0.0155 | 0.735 | 0.448 |
| helmholtz, cross | 504 | 0.4736 | 0.3808 | 0.733 | 0.456 |

Across the 1,701 blocks projected / raw ranges from 0.194 to 1.000. Matched-view medians by method: U-FNO 0.713, FNO 0.747,
CNO 0.770, DeepONet 0.711, GNOT 0.717, PINO 0.721, Transolver 0.963 (its observed squared-error share is 0.043: it already
reproduces the observed cells almost exactly, so pasting observations changes little). Raw scores remain the released
results; projection is a separately labelled diagnostic.

## 4. Paired test-identity bootstrap (441 grid)

`tools/revision/bootstrap.py`: per PDE, one regime-stratified identity draw per replicate (67/67/66 with replacement)
shared by all 63 models, nine views and, for forecasting, three horizons; 5,000 replicates, seed 20260926, 95 % percentile
intervals; statistics recomputed within each replicate from the resampled cell means (C - D, C / D, E[v,w] - E[w,w],
E[v,w] / E[w,w], density ratios, per-horizon C / D, medians of C / D). Point estimates reproduce the released statistics
exactly. These are conditional test-identity intervals for the fixed trained models, not training-seed uncertainty.

| statistic | estimate | 95 % interval |
|---|---|---|
| median C / D, 49 pairs | 10.46 | [10.07, 10.71] |
| median C / D, fixed 13 pairs | 10.74 | [10.44, 11.12] |
| pairs with C > D | 49 of 49 | [49, 49] in every replicate |
| median C / D by PDE: darcy | 13.45 | [10.72, 15.60] |
| poisson | 16.58 | [16.00, 17.18] |
| helmholtz | 28.61 | [27.75, 29.44] |
| heat | 3.62 | [3.56, 3.71] |
| reaction-diffusion | 3.36 | [3.29, 3.44] |
| burgers | 3.63 | [3.51, 3.76] |
| navier-stokes | 7.39 | [7.12, 7.68] |

Every pair's C / D interval excludes 1 (smallest lower bound 1.48). Of the 3,528 transfer cells, 3,232 have a ratio interval
entirely above 1 and 224 entirely below 1 (transfers that beat the destination specialist). No zero denominators occurred.
The 7,542 rows are in `bootstrap_intervals.csv`.

## 5. Simple baselines under the formal protocol: not run

What exists: `src/pdeobs/methods/interpolation.py` (zero fill, observed-mean fill, nearest, and the piecewise-linear
interpolation with nearest extrapolation registered as `bilinear`) and `src/pdeobs/methods/reduced_order.py` (Gappy POD
with rank candidates 8/16/32/64/128 and ridge 1e-6; Gappy POD-DMD forecasting). The strict-inference CLI can run them on
explicit identity manifests. The only recorded strict reports for these methods are demo-scale acceptance runs on 16 x 16
synthetic data (`acceptance/round2/T3.json`: e.g. piecewise-linear recovery rel-L2 0.064, forward 49.4, inverse 1.000; the
fitted `gappy_pod` and `rbf` flows valid). No baseline was evaluated on the paper's 200-identity blocks under the nine
frozen views, and no Gappy POD rank was selected on the paper's training partitions. Any baseline number for the paper
would have to be produced first; none is claimed here.

## 6. Independent numerical checks: not run

What exists: `results/numerical_quality/` summarises the generation-time quality object of the original corpus (digest
b0141eec...): 560,000 quality passes, family-specific discrete residual, boundary/initial-condition, geometry and
finite-array checks; for the paper slice the maximum normalised residuals are 1.6e-5 (Darcy), 2.8e-5 (Poisson),
3.5e-5 (Helmholtz), 1.5e-8 (heat), 1.6e-4 (reaction-diffusion), 3.9e-2 (Burgers) and 1.4e-4 (Navier-Stokes), all below the
configured 0.05 gate. `docs/numerical_solvers.md` documents the seven generator routes, the cell-centred unit-square grid,
the elliptic tolerance 1e-9 and the boundary conventions. These are discrete consistency records of the production route,
not manufactured-solution errors, reference-solver comparisons or grid-convergence studies; no such evidence exists in the
repository, and none is claimed.

## 7. Anonymous access of the public deposits: re-published and verified

The deposits that the paper points to had been removed on 2026-09-24 because the first upload embedded
author-identifying generation provenance (paths, host and node names, scheduler jobs, the non-anonymous source
commit) in shard attributes, sidecars, the corpus summary and 117 model record files. On 2026-09-26 they were
re-published from de-identified copies:

- **Dataset `PDE-OBS/pdeobs-data`**: every shard re-emitted with the provenance fields replaced (numerical chunks
  copied verbatim, bit-identical arrays), every sidecar and the corpus summaries rewritten, structured and byte audits of
  all 3,360 shards with zero findings; `scrub-manifest.json` maps every generation-time digest to the published digest.
  The paper slice (84 shards, 510 files, 6.9 GB) was uploaded first; the complete corpus (20,167 files, 250.6 GB,
  `release_manifest_full.json`) followed the same night.
- **Models `PDE-OBS/pdeobs-models`**: 441 checkpoints (bytes unchanged) with a second de-identification pass over
  1,028 record files (scheduler blocks, job ids, node/host names, GPU identifiers, partitions); the one byte-level hit
  inside a checkpoint (`GPU-`, four printable bytes inside tensor data) was confirmed a chance match.

`tools/revision/check_access.py` (plain HTTPS, no token, no cookies, no client library) after the re-publication:

| deposit | API | README | declared files |
|---|---|---|---|
| dataset `PDE-OBS/pdeobs-data` | 200 | 200 | 200 (both release manifests, scrub manifest, summary, shards and sidecars) |
| models `PDE-OBS/pdeobs-models` | 200 | 200 | 200 (three cluster manifests, scrub manifest, checkpoints, records) |

Integrity was checked without credentials, file by file, against the staged digests: dataset 20,167 of 20,167 files
(3,363 large files by SHA-256, the rest by git blob id; the paper slice of 510 files was verified separately first),
models 6,477 of 6,477 files (441 checkpoints by SHA-256). The
downloaded cards and the local cards contain no machine paths or author-specific strings. `verify_archived_predictions.py`
accepts the de-identified corpus through the scrub-manifest mapping of `summary.json`. The execution smoke test (score
one stationary and one temporal model from the downloaded artifacts) has not been run yet.

### Third de-identification pass and history squash (2026-09-26, after an external audit)

An external audit of the re-published deposits (raw git objects and every public file) found four residues: (i) the raw
git author of 118 historical commits (89 dataset, 28 model, 1 in a stray empty repository) carried a personal display name,
(ii) 80 model record files kept raw scheduler strings with numeric user/group ids, 18 of them a cluster account name and 8 a
login-node name (`device.allocation.scheduler|parent|step` in `evaluation/completion.json`), (iii) linkage clues: the private
source-commit identifier in 1,038 files, environment and runtime-directory names in 441 `provenance.json`, the data-root
version label in 441 `resolved.yaml` and in the corpus summaries, run-directory names in continuation records, and (iv) the
public manifests listed the digests of the private original files next to the published ones. The current release was
updated as follows; the Git checks below cover the advertised branch histories:

- **Records rewritten (third pass, 2,366 files, checkpoint bytes unchanged):** raw scheduler strings reduced to their resource
  fields (`<SCHEDULER_RECORD> JobState=... NumCPUs=... TRES=...`); source-commit identifiers relabelled `<COMMIT-1>`..`<COMMIT-5>`;
  `numerics-full-t15-<id>` -> `numerics-full-t15`; interpreter paths `<PYTHON>`, environment names `<ENV>`, runtime directories
  `<RUNTIME_DIR>`, run directories `<RUN>`, job numbers in continuation proofs `<JOB>`; login-node names `<NODE>`. The corpus
  `summary.json` (6,721 occurrences) and `summary.quality.json` carry the same relabelling. A byte audit over every staged
  record and top-level file with the audit's token list (ids, account, node names, commit ids, environment names, digest
  field names) reports zero hits.
- **Manifests v3:** `models_manifest.cluster-{A,B,C}.json` and both `scrub-manifest.json` files list the digests of the
  published files only. The correspondence to the archived digests (the identities used by `results/archived_campaign` and
  `results/prediction_verification_20260924`) now lives in this repository, `results/public_deposits/release_map.json`
  (441 checkpoints, all bindings agree with both indexes; `summary.json` b0141eec... -> 671a185d...). The verification tools read
  it (`tools/verify_archived_predictions.py --release-map`, `pdeobs.archived_verification.released_digests`).
- **Histories squashed:** `super_squash_history` on `PDE-OBS/pdeobs-models` (15274b8e3704), `PDE-OBS/pdeobs-data`
  (b17797d39805) and the stray empty `PDE-OBS/pdeobs-model`; each current `main` branch then had exactly one commit. Fresh unauthenticated
  bare clones of all three (`git clone --bare`, no credentials) list 3 commits in total
  with author PDE-OBS <PDE-OBS@users.noreply.huggingface.co> and committer `system <system@users.noreply.huggingface.co>`; the
  string that the audit flagged does not occur in the raw commit objects reachable from these `main` branches. This
  verification does not certify removal of unadvertised historical objects or third-party caches.
- **This repository:** the same labels applied to 42 files (receipts, inventories, `results/archived_campaign/index.json`,
  prose) and five digest references re-pointed; receipt `results/public_deposits/relabel_receipt_20260926.json`. The frozen
  bundle is untouched (its recorded digest of the archived index predates the relabelling; see
  `results/public_deposits/README.md`). JUnit local-time offsets and PDF creation metadata were removed. The remaining
  audit-log paths and hardware/source identifiers in the current code release were relabelled in the follow-up receipt
  `results/public_deposits/relabel_receipt_20260926_followup.json`; its scope excludes historical remote Git objects.

Unauthenticated re-verification after the squash: models 6477 of 6477 files match the staged digests
(441 checkpoints by SHA-256, the rest by git blob id); dataset 20166 of
20167 before the card rewrite (the one difference was the card itself, rewritten during the run) and
7 of 7 top-level files after it; every file was already covered by the full check of the morning, and only the
files listed above changed since. `tools/revision/check_access.py` was re-run afterwards (`access_check.json`).

## Deletions performed in this revision

Removed from the bundle on request: the budget-comparison audit, the study-attempt evidence (failed first launches,
seed-subset rows) and the receipts of the failed `helmholtz/gnot/mixed_nine_views` row. The inventories cover the 441
credited models only. The public paper text was not modified by this bundle.

## Regeneration

```bash
python tools/revision/build_results.py
python tools/revision/audit_training.py
python tools/revision/static_diagnostics.py aggregate --scored results/revision_v2/static_scored --out results/revision_v2
python tools/revision/bootstrap.py
python tools/revision/check_access.py
python -m unittest discover -s tools/revision -p 'test_*.py'
```

`mix_strict/` and `static_scored/` were produced on the host that holds the retained prediction files (see
`tools/revision/README.md`); their receipts carry the digests of every file read and written there.
