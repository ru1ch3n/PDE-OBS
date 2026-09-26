# Recorded production numerical quality

`production_qc_summary.json` summarizes the generation-time `dataset.quality`
object of the original corpus `summary.json`, not a newly generated dataset or
an inference result. Its source SHA-256 is
`b0141eec2199615a27facc75e2517ec4ec7e00913638e48c84ab2c88a2be77c1`,
the same summary digest recorded by all 441 archived settings.

The summary records 560,000 quality passes, zero missing or invalid quality
entries, and 80,000 records per family. It includes family-specific discrete
PDE residual, boundary/initial-condition, geometry and finite-array checks.
The separately filtered paper slice contains 14,000 records: exactly 2,000 per
family, smooth random-field conditions, Dirichlet stationary and periodic
temporal boundaries, with regime counts 667/667/666.

| Family | Selected records | Mean normalized residual | Maximum normalized residual |
|---|---:|---:|---:|
| Darcy | 2,000 | 1.1408363e-5 | 1.5674091e-5 |
| Poisson | 2,000 | 1.6883657e-5 | 2.8426042e-5 |
| Helmholtz | 2,000 | 1.8015463e-5 | 3.5152672e-5 |
| Heat | 2,000 | 1.3239560e-8 | 1.5109281e-8 |
| Reaction-diffusion | 2,000 | 3.3926813e-5 | 1.5511597e-4 |
| Burgers | 2,000 | 1.2630053e-2 | 3.8529777e-2 |
| Navier-Stokes | 2,000 | 5.0796310e-5 | 1.3748056e-4 |

All selected maxima are below the configured normalized-residual threshold
0.05. Counts, weighted moments and extrema are combined across disjoint
calibration keys after exact boundary/condition filtering; the source streaming
accumulator uses population variance (`quality.py`, `m2/count`). No held-out
model score enters these statistics. Raw full-corpus arrays were not all
re-solved during this summary audit; the 84 selected shards were rehashed
separately before model verification.

**Interpretation:** these are actual recorded production consistency results,
not merely implemented checks. They are nevertheless not independent
reference-solution error bounds or a grid-convergence study. Production
publication-readiness flags are not promoted by this extraction.
