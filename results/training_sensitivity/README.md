# Original-500 configuration-matched sensitivity analysis

`original500_full_config_audit.json` covers 13 PDE-method pairs, 117 models and
1,053 archived evaluation blocks. Selection uses `training_cohort=original500`
and `actual_epochs=500` for **all nine views**, not performance.

Every released `resolved.yaml` was hash-checked against the release manifest and
its original hash matched to the archived row. Original checkpoint identities,
dataset summary, split, seed and repository revision were cross-checked. The
complete parsed configuration agrees within all 13 pairs after removing only
`name`, `data.mask` and `data.root`: the row label, intended mask intervention and
filesystem location. Architecture, batch, optimizer, learning-rate schedule,
loss, stopping, precision and loader configuration were not excluded.

This is stronger than comparing the archived `training_config` field alone.
It does not assert identical hardware or bitwise numerical execution. Each
pair's normalized configuration and all 117 binding hashes are retained.

The numerical sensitivity results in this artifact use the **historical scalar
archive**, not the still-running complete prediction-verification result set.
The metadata-defined subset will be retained unchanged for the new-score cut.
All 13 pairs have larger transfer than matched error; equal-count and
training-pattern-dependent density findings also persist. The magnitude is not
identical to the full archive. The subset is historical, single-seed and
conditioned on completion, not a randomized architecture benchmark.

No training, inference, test-driven selection or configuration modification was
performed for this audit. Historical results remain unchanged.
