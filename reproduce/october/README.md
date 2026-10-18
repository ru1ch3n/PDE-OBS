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

# Weights-inference replay (path C)

`replay_inference.py` rebuilds a run's network from its frozen recipe (`configs/paper/october/recipes.yaml`), loads the run's
checkpoint and re-predicts the run's test views on the same parents, with the same per-parent sensor layouts and the same
input representation. It scores every record with the full-field relative L2 (float64) and compares each view mean with the
stored score in `results/revision_october`.

```bash
python reproduce/october/replay_inference.py --run V4C__poisson__ell009__fno__U256__s1101     --checkpoint <run>/checkpoints/last.pt --data-root <V2 store> --device cuda      # --suite n5nested for nested-density runs
```

Needs the `train` extra (PyTorch), the V2 views of the slice (`<V2 store>/<dataset_id>[__grid<N>][__t4]/`) and the
checkpoint. The October checkpoints and V2 data are not yet public; they are planned for `huggingface.co/ru1ch3n/PDE-OBS`.
The script exits 1 when a view mean differs from the stored one by more than `--rtol` (default 1e-4).

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
large-batch cuFFT result. The FNO layer itself is unchanged.
