# PDE-OBS v0.2.0 - full-coverage sweep report, round 2 (software acceptance)

- Sweep UTC: 2026-09-21T15:47:06Z (finished Mon Sep 21 15:56:06 UTC 2026)
- Node: a single CUDA node, NVIDIA GeForce RTX 5090, 580.159.03, 32607 MiB, 12.0; torch 2.7.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 5090 sm_120
- Code under test: `PDE_OBS_v020_code_r2.zip` (commit ANON_ACCEPTANCE_COMMIT_2, facade fixes) sha256 `58e7e059df7e490bf2fca925647156162ede706577d1c0f14afcc4e4eb3d1b5a`; harness `remote_full_coverage.py` sha256 `d0b7b74f...` (full sweep) / `64c750da...` (T7 rerun after the 64^2 fix)
- Inputs (written on the node under `qa_full/` and `logs/`; shipped in this tree as `acceptance/round2/`): `T*.json`, `coverage.json`, `essentials.txt`, `coverage.log`
- Scope: software acceptance only; nothing here enters paper tables.

## Headline

- Frozen scope intact: **521/521** core tests, API tests: **42 passed in 14.78s** (T12).
- Numerical generation coverage **840/840** combos (T1); paper masks exact at 128^2 (T2); CPU/CUDA prediction parity (T10); upstream wrappers gated (T14).
- 500-epoch cost estimates for all 14 paper structures obtained (T13, table below).
- **All 14/14 tiers completed**; essentials: 34 passed in 12.88s. Every round-1 defect is closed (see *Findings*).

## Per-tier results

| Tier | What | Status | Summary | Classification |
|---|---|---|---|---|
| T1 | numerical generation coverage | done | `{"combos": 840, "supported": 840, "unsupported_or_error": 0, "seconds": 79.2}` | pass |
| T2 | observation protocol coverage | done | `{"general_ok": 86, "general_total": 86, "paper_exact": true, "errors_rejected": true}` | pass |
| T3 | task interfaces x models (neural + classical) | done | `{"passed": 26, "total": 26}` | pass |
| T4 | every public model: presets, custom, illegal, geometry variants, multichannel | done | `{"passed": 16, "total": 16}` | pass |
| T5 | 7 main models x all observation views at 128^2 (paper nine + general) | done | `{"passed": 14, "total": 14}` | pass |
| T6 | fno smoke across boundary x regime for a static and a temporal PDE | done | `{"passed": 24, "total": 24}` | pass |
| T7 | splits (paper 2000-record, holdout, explicit, stored), budgets, identities, overwrite | done | `{"paper_split_1800_200": true, "all_rejections_ok": true}` | pass (paper_split_1800_200, all_rejections_ok) |
| T8 | artifact / inference / evaluation edge cases and refusals | done | `{"refusals_ok": true, "invalid_scores_flagged": true}` | pass (refusals_ok, invalid_scores_flagged) |
| T9 | every easy CLI subcommand + legacy CLI entry points from the installed package | done | `{"as_expected": 32, "total": 32}` | pass |
| T10 | CPU vs CUDA prediction parity for the same weights | done | `{"all_within_tol": true}` | pass |
| T11 | same seed -> same losses and weights (deterministic mode); masks deterministic | done | `{"gpu_bitwise": true, "cpu_bitwise": true}` | CPU bitwise OK; GPU bitwise |
| T12 | frozen 521 portable + updater tests and the 38 API tests | done | `{"core_passed": 521, "core_failed": 0, "api": "42 passed in 14.78s"}` | pass |
| T13 | paper structures at 128^2 with the paper batch sizes: peak memory + per-step timing -> epoch / 500-epoch estimate, then stop (no training) | done | `{"passed": 14, "total": 14, "max_peak_gpu_mb": 19955.3}` | pass |
| T14 | optional exact upstream wrappers: dependency-gated, never substituted | done | `{"gated": true}` | pass (dependency-gated, never substituted) |

## Benchmark essentials (`tests/test_benchmark_essentials.py`, on the node)

- Result: **34 passed in 12.88s**

## Findings

### Round-1 defects and their status in this round

| # | Round-1 finding | Fix (commit ANON_ACCEPTANCE_COMMIT_2 unless noted) | Evidence in this round |
|---|---|---|---|
| 1 | weight-free models could not be reloaded (`weights: null`, `api/artifacts.py:56`) | `ModelArtifact.weights/fitted_state` and `load_predictor` treat a null entry as *no file*; `test_parameter_free_models_round_trip` | T3 **26/26** (zero/mean/nearest/bilinear/rbf on recovery/forward/inverse, persistence rollout, gappy_pod fitted); essentials `test_E` passes |
| 2 | `evaluate_views` collided on two general views of one protocol | output dir / result key fall back to the observation id; duplicate observations rejected; `test_evaluate_views_accepts_same_protocol_at_two_settings` | T5 **14/14**: 7 models x {recovery, rollout} x (9 paper + 8 general views) at 128^2 |
| 3 | CPU seed reproducibility through the facade (`test_I`) | RNGs are seeded before the structure is built (`api/train.py`) | essentials `test_I` passes; T11 `cpu_bitwise: true` (and `gpu_bitwise: true` on this host) |
| 4 | fno on Navier-Stokes velocity: 6 vs 5 input channels | in/out channels follow the served representation (`effective_channels`, `api/train.py`; `api/models.py` no longer lets a preset pin `out_channels`); `test_navier_stokes_served_representation_sets_channels` | T4 **16/16** incl. `fno:navier_stokes[velocity]` and `[vorticity]` |
| 5 | `block_shape=6` leaked a `TypeError` | public int side length is translated to `(side, side)` for the mask factory; `test_block_shape_side_length_is_a_public_int` | T2 **86/86** general combos, paper exact, errors rejected |
| H1 | T7 harness: single-regime 2000-record set | 667/667/666 records over low/medium/high (harness) | T7 `paper_split_1800_200: true`, per-regime 600/67, 600/67, 600/66 |
| H2 | T7 harness: paper fno preset (modes=12) on a 16^2 grid (found in this round's first pass) | identity/origin checks moved to the 64^2 records (harness); T7 rerun with `--tiers T7` | T7 `all_rejections_ok: true`, paper/custom origins and recipe hashes differ |
| H3 | T8 harness: wrong legacy-checkpoint path (`<artifact>/checkpoints`) | `<run>/checkpoints/last.pt` (harness) | T8 `refusals_ok: true`, `invalid_scores_flagged: true`, legacy adapter round trip |
| H4 | T9 harness: `poisson.h5` shard name; error-path exits | `low.h5 medium.h5 high.h5`; rc=2 expected for the two bad-input commands (harness) | T9 **32/32** incl. `identity-manifest` and the `paper-row` demo on CUDA |
| H5 | T4 harness: paper presets on a 32^2 grid; classical runs lacked `loss_finite` | paper presets run on 64^2 records; `loss_finite` defaults to true for non-neural methods (harness) | T4 16/16, T3 26/26 |

### Observations (no action required)

- T13 per-step times are 1.6-3x slower than round 1 on this host (same GPU model, driver 580.159.03 vs 595.71.05, host load average ~6 at start). Peak memory is unchanged, so the difference is host/CPU-side (data loading, launch overhead); the 500-epoch figures are node-specific estimates, not benchmark numbers.
- T11 `gpu_bitwise` was false in round 1 and true here: the non-deterministic CUDA kernels (`upsample_bilinear2d_aa_backward`, `adaptive_avg_pool2d_backward`) run under `warn_only=True`, so bitwise GPU equality is host dependent. The CPU path remains the reproducibility contract.
- T1 generation took 79 s vs 32 s in round 1 (same 840 combos): CPU-side, consistent with the T13 observation.

## T13 - paper structures at 128^2, paper batch sizes (RTX 5090)

Extrapolated from 3x2 timed optimizer steps per configuration; no training was performed.

| Model:task | bs | params | peak GPU MB | step s | epoch s | 500 epochs h | optimizer | physics loss |
|---|---|---|---|---|---|---|---|---|
| cno:recovery@128 | 8 | 103,617 | 655 | 0.0265 | 6.0 | 0.83 | adamw | - |
| cno:rollout@128 | 8 | 103,617 | 1318 | 0.0446 | 10.0 | 1.39 | adamw | - |
| deeponet:recovery@128 | 4 | 221,121 | 411 | 0.0158 | 7.1 | 0.99 | adam | - |
| deeponet:rollout@128 | 4 | 221,121 | 757 | 0.0312 | 14.0 | 1.95 | adam | - |
| fno:recovery@128 | 8 | 2,366,145 | 439 | 0.0422 | 9.5 | 1.32 | adam | - |
| fno:rollout@128 | 8 | 2,366,145 | 1024 | 0.082 | 18.5 | 2.56 | adam | - |
| gnot:recovery@128 | 4 | 2,099,975 | 6813 | 0.1751 | 78.8 | 10.94 | adamw | - |
| gnot:rollout@128 | 4 | 2,099,975 | 19955 | 0.3164 | 142.4 | 19.78 | adamw | - |
| pino:recovery@128 | 4 | 32,798,273 | 725 | 0.0774 | 34.8 | 4.84 | adam | pino_static_fd_v1 |
| pino:rollout@128 | 4 | 32,798,273 | 1703 | 0.1021 | 45.9 | 6.38 | adam | pino_rollout_spectral_v1 |
| transolver:recovery@128 | 4 | 2,811,457 | 5218 | 0.1742 | 78.4 | 10.89 | adamw | - |
| transolver:rollout@128 | 2 | 11,196,737 | 9718 | 0.356 | 320.4 | 44.51 | adamw | - |
| ufno:recovery@128 | 4 | 5,049,545 | 1224 | 0.1007 | 45.3 | 6.3 | adam | - |
| ufno:rollout@128 | 4 | 5,049,545 | 1026 | 0.1458 | 65.6 | 9.11 | adam | - |

Sum over the 14 structures: **121.8 GPU-hours** for 500 epochs each on one RTX 5090; transolver:rollout@128 (45 h, 9.7 GB peak) and gnot:rollout@128 (20 h, 20.0 GB peak) dominate.

## Verdict

**Green.** With commit ANON_ACCEPTANCE_COMMIT_2 (`api/` only; kernels, masks, splits, scorer untouched) all 14 tiers complete, the frozen scope is 521/521, the API suite is 42/42 (38 + 4 regression tests), the 34 benchmark essentials pass, and every round-1 defect has a regression test. Round-1 evidence is shipped under `acceptance/round1/` with `FULL_COVERAGE_REPORT_round1.md`.
