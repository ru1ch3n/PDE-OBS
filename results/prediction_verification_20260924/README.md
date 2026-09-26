# New prediction-level result cut

This directory uses only the new prediction scores in
`pdeobs-prediction-verification/20260925-v1`. It does not overwrite or upgrade
the legacy scores in `../archived_campaign/`. All 441 retained checkpoints and
3969 view blocks are required; each block uses the same 200 held-out physical
identities. Bad finite outcomes are retained. No training or model selection is
part of this verification.

## Contents

- `index.json`: exact checkpoint identities, training cohorts, new cell means and
  standard deviations, three dynamic horizon summaries, and file bindings.
- `statistics.json`: all matrices, destination-referenced contrasts, density
  responses and the frozen original500 configuration-matched subset.
- `per_identity_<pde>.jsonl.gz`: all 113400 per-record cell errors for each PDE,
  including each dynamic horizon. Repeated records across masks/checkpoints are
  paired observations, not independent physical examples.
- `dataset_bindings.json`: selected source-shard hashes, split identities and
  per-identity physical-time mappings. No raw tensor data are embedded here.
- `fixed117_config_recheck.json`: fresh input bindings to the existing 13-pair
  complete-configuration audit, excluding only name, data.mask and data.root.
- `historical_score_comparison.json`: audit-only old/new comparisons. These old
  scores are not inputs to the new manuscript's tables or conclusions.
- `manifest.json`: file digests and precise result denominators.

## Statistics

The cell score is the arithmetic mean of 200 per-record joint relative-L2 errors.
Its standard deviation is the sample SD with ddof=1 across those 200 errors.
Forecasting's joint three-frame relative error is not the mean of three separate
horizon errors. D and C summarize 9 diagonal and 72 off-diagonal cell means;
their SDs are over those 9/72 cell means. The compact main table additionally
summarizes 63/504 cell means per PDE across seven methods and labels that separate
population explicitly. None of these SDs measures random-seed uncertainty or is
a confidence interval. Ratios, medians and counts have no invented SD.

All computation uses full-precision scores. Main tables show two decimals,
complete appendix cells four, with equal precision and trailing zeros for
mean/SD. Ratios use the destination-trained diagonal E[w,w], not E[v,v].
The fixed 13-pair, 117-model subset was selected from training metadata before the
new scores; it is not an optimized subset or a uniform architecture leaderboard.

## CPU-only reproduction

From the repository root, in an environment with NumPy:

```bash
python tools/prediction_analysis/analyze_new.py \
  --index results/prediction_verification_20260924/index.json \
  --out /path/to/new-statistics.json
python -m unittest discover -s tools/prediction_analysis -p 'test_*.py'
```

The output must not already exist. The regression tests also reproduce the
unaltered historical aggregate; that branch is not a source for current results.
Plotting additionally needs pandas, Matplotlib, pdflatex with the manuscript's
Times packages, and pdftoppm. Generating tables and figures does not perform
inference, training, downloads or remote access.

## Audit boundary

The strict scorer checked all raw prediction/target entries for finiteness and
computed the new per-record errors. The terminal audit freshly rehashed all 3969
prediction HDF5 files and the checkpoint/configuration/data inputs, tied them to
score and completion receipts, and independently checked complete identities,
shapes, source/physical times, exact regenerated masks and source geometry.
It did not run the models again. This establishes the new prediction artifacts'
integrity, not retrospective access to nonexistent old prediction arrays or
independent numerical-solver accuracy.

The old strict comparison rule remains atol=1e-7, rtol=1e-4. A separate 1% relative
difference summary is a post hoc reporting convention, not a preregistered
tolerance or a validity gate. All new valid scores are used regardless of the
size or direction of their historical differences. Causes of numerical
differences have not been established.

Raw data, weights and prediction tensors are not part of this Git repository.
Their external anonymous accessibility must be checked separately. Production
PDE residuals show discrete consistency, not analytic/reference-error or
grid-convergence validation. Training histories remain heterogeneous and
single-seed; stronger causal or cross-method claims need additional controls.
