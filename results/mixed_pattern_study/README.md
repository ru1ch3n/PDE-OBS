# Mixed-pattern training study (v0.2.1 observation specs)

One model per (PDE, method) trained on a deterministic mixture of the nine frozen observation views, one view per training
sample with exactly the campaign arguments, under the same epoch budget, optimizer, schedule and identity split as the
per-pattern rows of the archived campaign, then scored on the nine frozen views (same 200 held-out identities, same scorer).
These are study rows, not paper cells; the archived campaign index is untouched.

`mask_spec`: `random_3pct(ratio=0.5) + random_3pct(ratio=0.65) + random_3pct(ratio=0.8) + block_missing(missing_fraction=0.49) + line_sensors(num_lines=76, orientation=both) + line_sensors(num_lines=64, orientation=horizontal) + line_sensors(num_lines=64, orientation=vertical) + boundary_sensors(width=19) + clustered_sensors(ratio=0.5)`

| PDE / method | protocol | epochs | mean rel-L2, mixed | mean rel-L2, specialists | geomean(mixed / specialist) |
|---|---|---|---|---|---|
| navier_stokes/ufno_2d | pdeobs.fixed500-last-checkpoint/v7 | 500 | 0.0349 | 0.0255 | 1.19 |
| poisson/ufno_2d | pdeobs.max200-min120-then-train-data-patience10/20260920 | 200 | 0.0712 | 0.1165 | 0.46 (recipe-matched subset block_observed_50pct, boundary_band_50pct, clustered_50pct: 3.05) |
| helmholtz/deeponet | pdeobs.fixed500-last-checkpoint/v7 | 500 | 0.1214 | 0.0501 | 2.56 |
| heat/deeponet | pdeobs.fixed500-last-checkpoint/v7 | 500 | 0.1329 | 0.0983 | 1.32 |
| helmholtz/fno | pdeobs.fixed500-last-checkpoint/v7 | 500 | 0.0531 | 0.0212 | 2.08 |
| helmholtz/gnot | pdeobs.fixed500-last-checkpoint/v7 | running | pending | | |

Note: the mixed row ran the settings-v3 recipe (one_cycle, lr 5e-4, 200-epoch ceiling); only the specialists for block_observed_50pct, boundary_band_50pct, clustered_50pct ran that recipe, the other six ran the step schedule with patience stops, so the recipe-matched ratio is the like-for-like number.

Files: `results_long.csv` (one row per scored view, nine frozen views plus the extra density and mixture views),
`comparison.csv` (mixed vs the specialist trained on the tested view and vs the other eight specialists), `summary.json`.
Specialist values come from `results/archived_campaign` (cohort and epochs recorded per row).
