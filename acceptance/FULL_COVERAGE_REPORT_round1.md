# PDE-OBS v0.2.0 - full-coverage sweep report, round 1 (software acceptance)

- Sweep UTC: 2026-09-21T14:52:23Z (finished Mon Sep 21 14:55:20 UTC 2026)
- Node: a single CUDA node, NVIDIA GeForce RTX 5090, 595.71.05, 32607 MiB, 12.0; torch 2.7.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 5090 sm_120
- Code under test: `PDE_OBS_v020_code.zip` (commit ANON_ACCEPTANCE_COMMIT_1) sha256 `4a7f59b054561492b3af915608694f0ba63c21d4e906a450461f2c8e153cc6bd`; harness `remote_full_coverage.py` sha256 `9403bbd4...` (round-1 harness)
- Inputs (written on the node under `qa_full_r1/` and `logs_r1/`; shipped in this tree as `acceptance/round1/`): `T*.json`, `coverage.json`, `essentials.txt`, `coverage.log`
- Scope: software acceptance only; nothing here enters paper tables.

## Headline

- Frozen scope intact: **521/521** core tests, API tests: **38 passed in 2.59s** (T12).
- Numerical generation coverage **840/840** combos (T1); paper masks exact at 128^2 (T2); CPU/CUDA prediction parity (T10); upstream wrappers gated (T14).
- 500-epoch cost estimates for all 14 paper structures obtained (T13, table below).
- **Not clean.** Facade defects and harness defects were found; see *Findings*. Essentials: 2 failed, 32 passed in 2.45s.

## Per-tier results

| Tier | What | Status | Summary | Classification |
|---|---|---|---|---|
| T1 | numerical generation coverage | done | `{"combos": 840, "supported": 840, "unsupported_or_error": 0, "seconds": 31.8}` | pass |
| T2 | observation protocol coverage | done | `{"general_ok": 81, "general_total": 86, "paper_exact": true, "errors_rejected": true}` | 4/86 general combos error (`block_shape` given as int) - facade parameter coercion/validation; 1 informational (`stored`) |
| T3 | task interfaces x models (neural + classical) | done | `{"passed": 9, "total": 26}` | 9 pass / 2 rejected-ok / 15 fail - **facade bug**: weight-free classical models fail in `load_predictor` (manifest `weights: null`, `api/artifacts.py:56`) |
| T4 | every public model: presets, custom, illegal, geometry variants, multichannel | done | `{"passed": 14, "total": 16}` | 14/16 - 1 expected rejection counted as failure (pino paper modes=20 on 32^2 grid: harness), 1 **facade bug** (fno on navier_stokes velocity: 6 vs 5 input channels) |
| T5 | 7 main models x all observation views at 128^2 (paper nine + general) | done | `{"passed": 0, "total": 14}` | 0/14 - **facade bug**: `evaluate_views` output dir collides for two general views of the same protocol (`FileExistsError`, `api/evaluate.py:126`) |
| T6 | fno smoke across boundary x regime for a static and a temporal PDE | done | `{"passed": 24, "total": 24}` | pass |
| T7 | splits and budgets (t7_splits_budgets: plan_split paper, stable_split receipt) | crashed | `ValueError: production data have wrong regimes: ['low']` | crashed - harness bug: generated a single regime (`low`) but `stable_split` requires the full regime set (kernel constraint, correct) |
| T8 | artifacts and inference (t8_artifacts_inference: package/legacy checkpoint load, target-free predict) | crashed | `FileNotFoundError: Checkpoint does not exist: <run-root>/work/full/t8/base/model/checkpoints/last.pt` | crashed - `checkpoints/last.pt` absent after the base training run; needs investigation (api.train checkpoint writing vs harness path assumption) |
| T9 | every easy CLI subcommand + legacy CLI entry points from the installed package | done | `{"as_expected": 30, "total": 32}` | 30/32 - 2 error-path commands exited 2 as expected; `identity-manifest` and `paper-row demo` failed on missing `poisson.h5` manifest shard (harness staging) |
| T10 | CPU vs CUDA prediction parity for the same weights | done | `{"all_within_tol": true}` | pass |
| T11 | same seed -> same losses and weights (deterministic mode); masks deterministic | done | `{"gpu_bitwise": false, "cpu_bitwise": true}` | CPU bitwise OK; GPU not bitwise (non-deterministic CUDA kernels under `warn_only=True`, expected). But see essentials `test_I` below |
| T12 | frozen 521 portable + updater tests and the 38 API tests | done | `{"core_passed": 521, "core_failed": 0, "api": "38 passed in 2.59s"}` | pass |
| T13 | paper structures at 128^2 with the paper batch sizes: peak memory + per-step timing -> epoch / 500-epoch estimate, then stop (no training) | done | `{"passed": 14, "total": 14, "max_peak_gpu_mb": 19930.1}` | pass (timing only, no training) |
| T14 | optional exact upstream wrappers: dependency-gated, never substituted | done | `{"gated": true}` | pass (dependency-gated, never substituted) |

## Benchmark essentials (`tests/test_benchmark_essentials.py`, on the node)

- Result: **2 failed, 32 passed in 2.45s**
- FAILED `test_E_trivial_baselines_have_expected_order`
- FAILED `test_I_same_seed_same_weights_cpu`
  - `test_E_trivial_baselines_have_expected_order`: same root cause as T3 - `ModelArtifact.weights` does `manifest.get('weights', {}).get('file')` but the manifest stores `weights: null` for zero/mean/nearest -> `AttributeError`.
  - `test_I_same_seed_same_weights_cpu`: two CPU runs (`fno`, preset `smoke`, `max_steps=2`, `batch_size=2`, `seed=9`) produced different weight sha256. T11 reports CPU bitwise-equal under its own conditions, so seeding is path-dependent - must be investigated before v0.2.0 is called reproducible via the facade.

## Findings (ordered by severity)

### Facade defects (fix in `src/pdeobs/api/`, add tests; kernels untouched)
1. **Weight-free models cannot be loaded back** (T3 x15, essentials `test_E`): `api/artifacts.py:56` assumes `manifest['weights']` is a dict. Affects bilinear, mean, nearest, rbf, zero, persistence on every task. `gappy_pod(fitted)` scored (rel_l2 2.1e-6) but was also marked failed (steps=0 / train_s=null) - check the harness pass criterion.
2. **`api.evaluate_views` cannot evaluate two general views of the same protocol** (T5 x14): output directory is `out/<spec.name>` and `evaluate_records` uses `mkdir(exist_ok=False)`; the second `random` view raises `FileExistsError`. Directory should key on `observation_id` (or the harness must give distinct names). This blocked the whole 7-model x all-views tier at 128^2.
3. **CPU seed reproducibility through the facade** (essentials `test_I`): differing weights for identical `api.train` calls on CPU. Contradicts T11 (`cpu_bitwise: true`); find which of preset / `max_steps` / `batch_size` / `seed` routes differ.
4. **fno on navier_stokes velocity fields** (T4): conv expects 5 input channels, got 6 - input-channel inference for 2-component fields is off by one (`geometry_channels` / coordinate channels double-counted?).
5. **`block_shape` given as a scalar int** (T2 x4): `TypeError: 'int' object is not iterable` instead of a clean `ValueError` (or broadcast to `(6, 6)`). Contract says unknown/illegal parameters raise cleanly.

### Harness defects (fix `remote_full_coverage.py`, rerun tiers T4, T7, T8, T9 only)
- T7: generate all regimes before `api.plan_split(..., 'paper')`; `stable_split` rejecting a single-regime production set is correct behaviour.
- T8: the tier assumes `<artifact>/checkpoints/last.pt`; confirm what `api.train` actually writes under a `max_steps` budget before treating this as a facade bug.
- T9: stage the `poisson.h5` manifest shard for `identity-manifest` and `paper-row demo`; the two `rc=2` error-path commands are correct rejections and should be counted as expected.
- T4: running the `paper` pino preset on a 32^2 grid is correctly rejected (modes=20 > 16); count as `rejected_ok`.

### Expected / informational
- T11 GPU non-bitwise: `upsample_bilinear2d_aa_backward` and `adaptive_avg_pool2d_backward` have no deterministic CUDA implementation (warnings in `coverage.log`); the CPU path is the reproducibility contract.
- T1: `source='download'` is refused with a clear message (no verified endpoint registered) - as designed. `create_dataset_from_arrays` correctly limited to poisson/darcy/helmholtz/heat.
- T14: `paper_cno` / `paper_fno` / `paper_unet` are dependency-blocked and not served via the easy API; direct registry use requires the 8 attestation kwargs.

## T13 - paper structures at 128^2, paper batch sizes (RTX 5090)

Extrapolated from 3x2 timed optimizer steps per configuration; no training was performed.

| Model:task | bs | params | peak GPU MB | step s | epoch s | 500 epochs h | optimizer | physics loss |
|---|---|---|---|---|---|---|---|---|
| cno:recovery@128 | 8 | 103,617 | 524 | 0.0123 | 2.8 | 0.39 | adamw | - |
| cno:rollout@128 | 8 | 103,617 | 1248 | 0.0249 | 5.6 | 0.78 | adamw | - |
| deeponet:recovery@128 | 4 | 221,121 | 342 | 0.006 | 2.7 | 0.37 | adam | - |
| deeponet:rollout@128 | 4 | 221,121 | 682 | 0.0097 | 4.4 | 0.61 | adam | - |
| fno:recovery@128 | 8 | 2,366,145 | 442 | 0.014 | 3.1 | 0.44 | adam | - |
| fno:rollout@128 | 8 | 2,366,145 | 1027 | 0.0231 | 5.2 | 0.72 | adam | - |
| gnot:recovery@128 | 4 | 2,099,975 | 6739 | 0.1041 | 46.9 | 6.51 | adamw | - |
| gnot:rollout@128 | 4 | 2,099,975 | 19930 | 0.2469 | 111.1 | 15.43 | adamw | - |
| pino:recovery@128 | 4 | 32,798,273 | 728 | 0.0243 | 11.0 | 1.52 | adam | pino_static_fd_v1 |
| pino:rollout@128 | 4 | 32,798,273 | 1706 | 0.0465 | 20.9 | 2.91 | adam | pino_rollout_spectral_v1 |
| transolver:recovery@128 | 4 | 2,811,457 | 5218 | 0.114 | 51.3 | 7.12 | adamw | - |
| transolver:rollout@128 | 2 | 11,196,737 | 9685 | 0.2796 | 251.6 | 34.95 | adamw | - |
| ufno:recovery@128 | 4 | 5,049,545 | 426 | 0.0337 | 15.1 | 2.1 | adam | - |
| ufno:rollout@128 | 4 | 5,049,545 | 956 | 0.0543 | 24.4 | 3.39 | adam | - |

Sum over the 14 structures: **77.2 GPU-hours** for 500 epochs each on one RTX 5090; transolver:rollout@128 (35 h, 9.7 GB peak) and gnot:rollout@128 (15 h, 19.9 GB peak) dominate.

## Verdict

The frozen kernel scope is untouched and green (521/521, 38/38) and the facade's happy paths across 7 PDEs x models x tasks work on sm_120. v0.2.0 should **not** be declared fully covered until facade defects 1-3 are fixed with regression tests added to `tests/test_api_workflow.py` / `tests/test_benchmark_essentials.py`, and tiers T3, T4, T5, T7, T8, T9 are rerun (~15 min of GPU time; T5 is the only one that needs the GPU seriously).
