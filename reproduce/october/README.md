# Rebuilding the October 2026 tables

```bash
python reproduce/october/rebuild_tables.py --results results/revision_october --out reproduce/october/out
```

The script uses the standard library only. It rebuilds nine tables from the run-level results:
1. static calibration
2. observation transfer (D3 / C3 per seed)
3. grid jitter, with the data-consistency split
4. NS medium sensor density
5. resolution
6. external RPB in-distribution and OOD
7. normalisation-only attribution
8. confirmation-set neural results
9. training failures

"Best neural" columns are descriptive envelopes over every trained model of the listed cohorts, not methods selected on
test data. The outputs are generated files; they are not committed.

# Score your own method on the October protocol

`score_method.py` builds the test (or confirmation / OOD) split of a V2 view with the same per-parent sensor layouts,
input representation and float64 full-field scorer as the replay, and takes the predictions from a callable instead of
a checkpoint. Three predictors are built in: `zero` (the masked field; persistence for the temporal families),
`nearest` (the nearest-observation fill, `pdeobs.revision.fill`) and `covariance` (the covariance-weighted
reconstruction of `pdeobs.revision.covariance`, fitted on the training split with the regularisation chosen on the
validation split from the frozen candidate list in `configs/paper/october/classical.yaml`; static families only).

```bash
python reproduce/october/score_method.py --dataset-id v2_poisson_ell009_ref128 --data-root <V2 store> --method covariance --out cov.json
python reproduce/october/score_method.py --dataset-id v2_ns_ell009_ref128 --data-root <V2 store> --method my_package.my_module:predict
python reproduce/october/score_method.py --dataset-id v2_poisson_rpb_ref128 --data-root <RPB store> --part ood --method nearest
```

The RPB store is the output of `python -m pdeobs.revision.external.rpb`; its view is 64^2 (`v2_poisson_rpb_ref128__grid64/`) and the script selects it for the rpb slice. `--grid 64` / `--grid 256` select the N8 resolution views (`<dataset_id>__grid<N>/`) of the V2 store.

A custom predictor is `module:callable` with the signature `predict(inputs, mask, geometry, context) -> prediction`:
static families get `inputs` [B, C, H, W] (observed values, hidden cells 0, or nearest-filled with `--fill nearest`),
`mask` [B, 1, H, W] and `geometry`; temporal families get the observed initial state [B, 1, C, H, W] and must return
the next three frames. `context` is one dict per record with the task context only (`PREDICTOR_CONTEXT_KEYS` in the
script: record identity, grid, time axis, layout id and sensor count); the target, the generating parameters, the
parent seed, the regime and the reference-solver diagnostics of a record are never passed, and the covariance built-in
reads the observed values from `inputs`. Every prediction is validated before it is scored: a wrong shape, a non-finite
value, a predictor that raises, or a record set with a missing or repeated identity ends the run with `status: invalid`
and exit code 2, and no mean is computed from the remaining records. Reported per view: mean / median / p90 / max of the
per-record relative L2, the number of records and the SHA-256 of the sorted record identities (equal to the split
receipt's), the same statistic as `mean` in `results/revision_october`; a complete run reports `status: valid`.
`--max-records N` scores the first N records for a smoke test and reports `status: partial`. The report JSON refuses
non-finite numbers. The RPB store is the output of `python -m pdeobs.revision.external.rpb` (the data are not
redistributed; download them from the authors' Zenodo record and convert them).

# Weights-inference replay (path C)

`replay_inference.py` rebuilds a run's network from its frozen recipe (`configs/paper/october/recipes.yaml`), loads the run's
checkpoint and re-predicts the run's test views on the same parents, with the same per-parent sensor layouts and the same
input representation. It scores every record with the full-field relative L2 (float64) and compares each view mean with the
stored score in `results/revision_october`.

```bash
python reproduce/october/replay_inference.py --run V4C__poisson__ell009__fno__U256__s1101     --checkpoint <run>/checkpoints/last.pt --data-root <V2 store> --device cuda      # --suite n5nested for nested-density runs
```

Needs the `train` extra (PyTorch), the V2 views of the slice (`<V2 store>/<dataset_id>[__grid<N>][__t4]/`) and the
checkpoint. Both are on Hugging Face: the V2 views in the dataset `huggingface.co/datasets/ru1ch3n/PDE-OBS` under
`october2026/v2/` (that directory is the `--data-root`), the weights of every scored October run in the model repository
`huggingface.co/ru1ch3n/PDE-OBS` as `october2026/models/<cohort>/<run_id>/weights.pt` (pass that file as `--checkpoint`).

```python
from huggingface_hub import snapshot_download
data = snapshot_download("ru1ch3n/PDE-OBS", repo_type="dataset", allow_patterns=["october2026/v2/v2_poisson_ell009_ref128/*"])
model = snapshot_download("ru1ch3n/PDE-OBS", allow_patterns=["october2026/models/V4C/V4C__poisson__ell009__fno__U256__s1101/*"])
```
The verdict is `status`: `agrees` (exit 0) when every view of the suite has a stored score, every replayed mean is finite
and within `--rtol` (default 1e-4) of it with the stored record count, and, with `--per-record <file>`, every record of
every view is found once in the per-record export and within `--per-record-rtol` (default 1e-3) of its stored error;
`disagrees` (exit 1) when a difference exceeds its tolerance or a replayed value is not finite; `incomplete` (exit 1)
when a stored score, a record or a per-record row is missing or repeated. The per-record exports are the
`per_record/<suite>.csv.gz` files under `october2026/results/part-A/update_20261021/` and `part-B/update_20261021/` of
the dataset repository (columns `run_id, view, sample_id, error`); `--per-record-out` writes the replayed per-record
errors in the same layout.

Recorded checks (2026-10-16; fresh virtual environments from the built wheel, PyTorch 2.14.1, 1,800 test parents):

| Run | Suite (views) | CUDA, NVIDIA H200: max relative difference | CPU: max relative difference |
|---|---|---|---|
| V4C Poisson medium FNO U256 s1101 | v2eval7 (7) | 1.0e-6 | 1.0e-6 (`--fft auto`); 1.2e-2 with `--fft native` |
| P4 Poisson medium Voronoi-CNN U256 s1101 | v2eval7 (7) | 2.7e-5 | 0 (every view mean identical) |
| P4 NS medium FNO U256 s1101 | n5nested (18) | 6.9e-8 | 1.8e-7 (`--fft auto`); 6.9e-5 with `--fft native` |

**FNO checkpoints and CPU inference.** The FNO spectral layer has independent weights for +ky and -ky, so the half spectrum it
inverts is not conjugate-symmetric in the kx = 0 and Nyquist columns. The real inverse FFT then has to drop or keep the
imaginary parts of those bins, and libraries differ: CPU FFTs drop them, cuFFT keeps them for large batches (from about 4,096
rows of 1-D transforms, which every October training and evaluation shape exceeds). The October FNOs were trained and scored
on NVIDIA GPUs, so on a CPU the replay uses `pdeobs.revision.fftcompat` (`--fft auto`, the default), which reproduces the
large-batch cuFFT result. The FNO layer itself is unchanged. The batch threshold was measured with cuFFT 13.0 on an RTX PRO
6000 and may differ with the cuFFT version or GPU; the replay's own CUDA check ran on an H200.

# Training reproduction (path D)

`train_run.py` retrains a run from the public package and the frozen recipes (`configs/paper/october/recipes.yaml`):
the same training split, per-parent sensor layout, input representation, network, optimiser settings, epochs and seed;
for the method-tuned (V4) recipes the training-set normaliser is recomputed from the training split. `--set KEY=VALUE`
overrides one recipe field for a controlled variant (recorded in `completion.json`; replay such a model with the same
`--set`). A run whose `checkpoints/last.pt` exists is resumed exactly, so a training can span several jobs.

```bash
python reproduce/october/train_run.py --run P4__poisson__ell009__voronoi_cnn__U256__s1101 --data-root <V2 store> --out <dir> --device cuda
python reproduce/october/replay_inference.py --run P4__poisson__ell009__voronoi_cnn__U256__s1101 --checkpoint <dir>/checkpoints/last.pt --data-root <V2 store> --device cuda
```

Recorded retrains (2026-10-18 to 2026-10-20; fresh virtual environment from the built wheel, PyTorch 2.14.1, NVIDIA B40,
several trainings sharing one GPU; the campaign ran on H200 / B40 GPUs with PyTorch 2.7.1). Poisson medium, U256, seeds
1101-1103, nine runs of the three recipes the script covers. Values are retrain / original campaign run:

| Run | Outcome | Final training loss | U256 test error | G256 test error |
|---|---|---|---|---|
| P4 fno U256 s1101 | failed: constant-output collapse at epoch 40 (original: epoch 35) | 1.00 / 1.00 (never left the trivial plateau) | - | - |
| P4 fno U256 s1102 | completed, 250 epochs | 0.0586 / 0.0712 | 0.0717 / 0.0809 | 0.0616 / 0.0617 |
| P4 fno U256 s1103 | failed: constant-output collapse at epoch 36 (original: epoch 33) | 1.00 / 1.00 (never left the trivial plateau) | - | - |
| P4 voronoi_cnn U256 s1101 | completed, 250 epochs | 0.0200 / 0.0200 | 0.0186 / 0.0216 | 0.0253 / 0.0284 |
| P4 voronoi_cnn U256 s1102 | completed, 250 epochs | 0.0204 / 0.0203 | 0.0198 / 0.0193 | 0.0268 / 0.0257 |
| P4 voronoi_cnn U256 s1103 | completed, 250 epochs | 0.0200 / 0.0205 | 0.0227 / 0.0190 | 0.0303 / 0.0260 |
| V4C fno U256 s1101 | completed, 250 epochs | 0.0249 / 0.0253 | 0.0264 / 0.0255 | 0.0180 / 0.0215 |
| V4C fno U256 s1102 | completed, 250 epochs | 0.0265 / 0.0269 | 0.0292 / 0.0262 | 0.0198 / 0.0168 |
| V4C fno U256 s1103 | completed, 250 epochs | 0.0259 / 0.0260 | 0.0256 / 0.0276 | 0.0215 / 0.0243 |
| P4n fno U256 s1101 (P4 recipe + training-set normaliser) | completed, 250 epochs | 0.0332 / - | 0.0333 / - | 0.0244 / - |
| P4n fno U256 s1102 (P4 recipe + training-set normaliser) | completed, 250 epochs | 0.0329 / - | 0.0313 / - | 0.0264 / - |
| P4n fno U256 s1103 (P4 recipe + training-set normaliser) | completed, 250 epochs | 0.0328 / - | 0.0328 / - | 0.0246 / - |

- **Matched view.** On the training layout (U256) every completed retrain lands within about 10-20 % of the original
  seed's score, in both directions; the 250-epoch training losses agree to about 2 % for the Voronoi-CNN and V4C
  runs and differ by 18 % for the P4 FNO seed 1102 (0.0586 against 0.0712). Retraining is not
  bit-identical (different GPU, PyTorch 2.14 against 2.7, several trainings per GPU), so `replay_inference.py` reports
  `agrees: false` for a retrain by design (its tolerance, 1e-4, is for replaying the original checkpoint).
- **Off-pattern views.** The transfer errors of a specialist (C256 / L256 / U1024 / C1024 / L1024 of a U256-trained
  model) differ between the retrain and the original by factors of up to 2-3 in either direction, as they do between
  seeds of the original campaign: the off-pattern failure is large and unstable, the matched accuracy is not.
- **The P4 FNO failure reproduces.** The frozen P4 recipe (no normaliser, lr 1e-4) trains the FNO on raw Poisson medium
  solutions of standard deviation about 0.012. Seeds 1101 and 1103 never leave the trivial plateau (training loss 1.00
  from the first epoch), the output standard deviation decays geometrically and the health check stops them with
  `constant_output_collapse` at epochs 40 / 36 (original campaign: 35 / 33) with the same per-epoch trajectory; seed
  1102 escapes the plateau at epoch 8-10 in both. A controlled variant with the training-set normaliser as the only
  change (`--set v4_norm=train_scalar`, output name P4n, rows above) trains all three seeds to 250 epochs (final
  training loss 0.033) and scores 0.031-0.033 on U256 and 0.024-0.026 on G256 (2026-10-21): the normaliser removes the
  collapse and most of the gap to the V4C recipe (0.0255-0.0276 on U256), whose remaining difference is the learning
  rate (3e-4 against 1e-4).

**Published retrains.** The ten completed retrains (the three P4n runs included) are in the model repository
`huggingface.co/ru1ch3n/PDE-OBS` under `october2026/retrains/<name>/`: `weights.pt` (the format of the October models),
`model.json` (recipe run, `--set` overrides, final training loss, the seven v2eval7 view means of the retrain) and
`history.json`; `october2026/retrains/manifest.json` lists them. The two collapsed P4 FNO retrains have no model. Replay a
retrain with its recipe run and the same `--set`, for example `--run P4__poisson__ell009__fno__U256__s1101 --set
v4_norm=train_scalar --checkpoint october2026/retrains/P4n__poisson__ell009__fno__U256__s1101/weights.pt`.
