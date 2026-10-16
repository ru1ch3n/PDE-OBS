# The October 2026 revision

The September release evaluated a 441-row grid on smooth fields:
- 7 PDEs x 7 learned methods x 9 frozen views;
- 1,800 train / 200 test records per PDE;
- no validation split.

The October revision asks how far that picture holds once three things change:
- the fields are harder (new V2 data);
- the neural baselines are method-tuned on validation data;
- strong observation-conditioned linear references are added.

It also checks every main claim on a sealed confirmation set and on an external dataset. The historical grid, its configs
and its results are unchanged; the revision is a separate, versioned layer.

## Data (V2)

| | |
|---|---|
| PDEs | Poisson, Darcy (static recovery); Navier-Stokes (forecast u0 -> frames 1-3, vorticity) |
| Slices | Gaussian-random-field constructions with length scales 0.22 / 0.09 / 0.025 (smooth / medium / rough) |
| Regimes | low / medium / high, per regime 3,000 train / 300 validation / 600 test parents |
| Confirmation | 200 further parents per regime (600 per slice), sealed until every model, view list and setting was frozen |
| Size | 35,100 unique parents (3 PDEs x 3 regimes x 3,900); 105,300 slice records (each parent once per slice); addon 64^2 / 256^2 views of Poisson and NS rough are views of the same parents, not new samples |
| Grid | 128 x 128 |
| Not included | Burgers V2 (did not pass the numerical gate) |

The definitions are in [`configs/paper/october/data_v2.yaml`](../configs/paper/october/data_v2.yaml). The generator and
the reference solvers are in `pdeobs.revision` (`v2spec`, `v2data`, `reference`, `numchecks`).

## Observations

| Layout | Definition |
|---|---|
| U\<K\> | uniform random over the eligible cells |
| C\<K\> | four Gaussian clusters |
| L\<K\> | K/64 rows x 64 distinct columns per row |
| G\<K\> | centre-of-bin grid |

- Sensor counts are K = 256 / 1,024. The Dirichlet boundary is never observed.
- Layouts belong to the parent and its replicate, never to the model or the training seed.
- Mixtures:
  - MXR3 = U256 / C256 / L256;
  - MXR6 adds the three 1,024-sensor layouts;
  - MXR2 = U256 / C256, the 64^2 external data where L256 cannot exist.
- Nested uniform sets: K = 64 ... 4,096.
- Grid perturbations: coherent shifts and within-bin jitter of 1-3 cells.

See [`observations_v2.yaml`](../configs/paper/october/observations_v2.yaml) and `pdeobs.revision.v2masks`.

## Methods and recipes

Learned models:
- FNO
- Voronoi-CNN (nearest-observation fill)
- CNO
- Senseiver

Two recipe cohorts are kept apart:
- **Frozen V2 recipes** (cohorts P4 / P5), chosen on a development screen.
- **Method-tuned V4 recipes** (cohort V4-C). Each adds a fixed training-set scalar normaliser where needed. The recipe is
  the top-2 variant with the best validation score at 20,000 updates.

Test data never chose a recipe. Every recipe that ran is listed in
[`recipes.yaml`](../configs/paper/october/recipes.yaml).

Classical references (fitted on the training split, settings chosen on the validation split):
- gappy POD;
- a **covariance-weighted linear reconstruction**, the Gaussian conditional mean under the training covariance on 2,048
  POD modes ([`pdeobs.revision.covariance`](../src/pdeobs/revision/covariance.py));
- linear interpolation;
- for NS, POD affine-dynamics references.

## Studies

| Block | What | Size |
|---|---|---|
| V4-C | method-tuned formal comparison | 96 runs |
| V4-D / V4-E | layout / density specialists; NS rough information control (partial vs full initial state x viscosity hidden / provided) | 33 + 24 runs |
| N1 | independent re-check of the covariance reference | rank x weighting, full covariance, nested training size |
| N2 | sensor noise (clean-tuned vs noise-aware regularisation); stored vs finer reference targets | |
| N3 | fixed-count grid jitter; jitter-augmented training (GJ2) | 6 runs + evaluations |
| N4 | construction shift medium <-> rough | evaluations |
| N5 | NS medium sensor density; two-density mixture (MXR6) | 3 runs + nested evaluations |
| N6 | normalisation-only control | 3 runs |
| N8 | resolution 64^2 / 128^2 / 256^2 | 48 runs |
| N7 + T7 | sealed confirmation set; paired parent statistics | 237 evaluations |
| T8 | fit / inference cost | |
| RPB | external data: the CNO authors' Representative PDE Benchmarks, Poisson | 18 runs + 9-run normaliser control |

Single-factor controls are listed in [`controls.yaml`](../configs/paper/october/controls.yaml).

## Main findings

The numbers below are taken from the result files. Errors are per-record relative L2 averaged over the test parents and
over three training seeds, unless stated otherwise.

1. **A strong linear reference changes the picture on static data.**
   - On all 16 static (Poisson / Darcy) cells, the covariance reconstruction beats every neural model, often by orders of
     magnitude (Poisson medium U256: about 1e-4 against 0.020).
   - On NS the neural models lead on 6 of 8 cells.
   - An independent solve reproduces the reference (agreement 5e-11). The gain over gappy POD comes from the eigenvalue
     weighting.
   - On the confirmation set it beats every named neural model in all 1,425 paired per-seed comparisons (95 %
     parent-bootstrap intervals exclude 0).
2. **The linear reference is not robust by default.**
   - With regularisation tuned on clean data it diverges under sensor noise.
   - Re-tuned on noisy validation data it stays at or below the neural models.
   - Out of distribution (RPB OUT) it extrapolates badly on 256-sensor views, while neural predictions and interpolation
     stay bounded.
3. **Changing the observation pattern is costly and asymmetric for every neural model.** On the confirmation set the
   off-pattern / matched error ratio is 1.2-2.2 in every seed.
4. **Measurement geometry and density are predictor-specific failure modes, and training exposure removes them.**
   - A grid-trained FNO goes from 0.031 to 0.24 / 0.39 / 0.51 under 1 / 2 / 3-cell jitter. Jitter-augmented training
     keeps it at 0.045-0.070.
   - A zero-fill FNO trained on 256 NS sensors scores 0.038 at 256 sensors and 0.58 at 1,024. Two-density training gives
     0.054 / 0.027.
   - Voronoi-CNN improves monotonically with more sensors.
5. **NS rough is limited by partial observation of the initial state, not by the learned dynamics.**
   - With the full initial state FNO forecasts at 0.014-0.015 (linear dynamics: 0.44).
   - With partial observation every model sits near 0.75.
6. **Frozen recipes caused several baseline failures.**
   - A training-set normaliser turns the Darcy FNO from a training-mean predictor into a working model.
   - It brings Poisson medium CNO from 0.836 to 0.059 (learning rate alone: 0.167).
   - On RPB the frozen Voronoi-CNN recipe failed 9 / 9 with a gradient pathology (fields of size about 1e-3) and trained
     normally with the normaliser.
7. **External data.** RPB in-distribution solutions lie in a low-dimensional span:
   - linear references are exact on U1024 / G256 and reach 0.072 on U256;
   - the best neural models (MXR2 training) reach 0.31-0.38 on U256.
8. **Cost.** The covariance reference fits in about 18 CPU-seconds but needs a fresh solve per sensor layout (11-126 ms on
   8 threads). Neural predictors take 0.24-4.0 GPU-h to train, then 1-4 ms per record on a GPU.

## What is not claimed

- That the covariance reconstruction solves static recovery in general. It needs noise-aware regularisation and fails
  outside the training distribution.
- That neural methods are necessary for nonlinear dynamics in general. Only NS was compared against linear dynamics
  references.
- That observation effects vanish after density matching. Layout transfer gaps remain at matched density.
- Seed uncertainty from parent intervals. Three training seeds per configuration are reported separately.

## Where things are

| | |
|---|---|
| Library | `src/pdeobs/revision/`: V2 data, layouts, references, `covariance`, `dc` (raw / observed / hidden split), `stats` (paired parent bootstrap), `external/rpb` |
| Frozen configs | `configs/paper/october/` |
| Tests | `tests/test_revision_october.py` |
| Results and table rebuilds | `results/revision_october/` (separate review step) |
