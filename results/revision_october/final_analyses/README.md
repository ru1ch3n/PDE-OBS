# Final analyses (E1-E5) of the October 2026 revision

Machine-readable outputs of the analyses described in `docs/october_revision.md` (section "Final analyses"). They were
computed from the stored predictions and result files of the campaign; no number here was retyped. The run-level rows
of E2 (cohort `E2ev`) and E6 (`E6`, `E6ev`) are in `runs_long.part-A.csv`, the E5 forecast rows in `runs_long.part-B.csv`
(cohort `E5compose`).

| File | Analysis | Content |
|---|---|---|
| `e1_rpb_ood_diagnostics.json` | E1 | RPB in-distribution and OOD test sets: norms and spectra of the target fields, representation error of the training basis, per-view error statistics of the covariance and gappy-POD references at the reported setting (condition numbers, coefficient errors), and their sensitivity curves over the regularisation grid |
| `e2_data_consistent_transfer.json` | E2 | raw / data-consistent / hidden-region split of the matched (D) and off-pattern (C) errors per run and per cohort (old Poisson / Darcy matrices, V2 slices, RPB) |
| `e3_stored_vs_finer_targets.json` | E3 | per slice: stored-vs-finer target difference, the stored route's reproduction error, and every method's error against the stored and the finer target with paired differences |
| `e4_rpb_paired_bootstrap.json` | E4 | paired per-record differences on RPB (MXR2 mixtures vs destination specialists; neural methods vs covariance, gappy POD and interpolation) with parent-bootstrap 95 % intervals, per seed, ID and OOD separately |
| `e5_composition_reference.json` | E5 | NS rough: u0 reconstruction error and forecast error of the composition reference (covariance or nearest fill -> frozen full-state FNO) per view and seed, with the regularisation selection on validation |

The public library implements the same definitions: `pdeobs.revision.covariance` (E1, E5), `pdeobs.revision.dc` (E2) and
`pdeobs.revision.stats` (E4); `reproduce/october/score_method.py --method covariance` re-fits the covariance reference on
the published V2 data.
