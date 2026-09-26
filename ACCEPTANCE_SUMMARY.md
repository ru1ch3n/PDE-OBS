# v0.2.0 acceptance summary

Generated 2026-09-21T14:07:21Z.

## Environment

- device: NVIDIA GeForce RTX 5090 capability [12, 0] (31.37 GB)
- torch 2.7.1+cu128 / CUDA 12.8; pdeobs 0.2.0

## CUDA primitives

- all_ok: True; peak 0.067 GB; fft roundtrip err 1.9073486328125e-06

## L0 CPU regression (on the GPU host, v0.2.0 tree)

- frozen portable core + updater tests: **521/521 passed** (failed 0, skipped 0); rc=0
- new API tests (specs + workflows): 38 passed in 7.60s

## L1 parameter coverage

| model | status | default params | alt params | params differ | illegal rejected | reload identical |
|---|---|---|---|---|---|---|
| cno | passed | 6897 | 26465 | True | True | True |
| deeponet | passed | 1505 | 5057 | True | True | True |
| fno | passed | 75009 | 445705 | True | True | True |
| gnot | passed | 5554 | 54981 | True | True | True |
| pino | passed | 75009 | 445705 | True | True | True |
| transolver | passed | 6519 | 47957 | True | True | True |
| ufno | passed | 189497 | 1139921 | True | True | True |

## L2 GPU model × PDE matrix (49 minimal full-coverage flows)

device: cuda; resolution 64², 4 samples, 2 optimizer steps, views ['random', 'block']; **passed 49/49**

Each flow: paper architecture → train a few steps → save → fresh reload (identical outputs) → target-free predict → strict evaluate. Software acceptance only; not training to convergence, not a performance number.

| model \\ pde | darcy | poisson | helmholtz | heat | reaction_diffusion | burgers | navier_stokes |
|---|---|---|---|---|---|---|---|
| fno | ok | ok | ok | ok | ok | ok | ok |
| pino | ok | ok | ok | ok | ok | ok | ok |
| ufno | ok | ok | ok | ok | ok | ok | ok |
| cno | ok | ok | ok | ok | ok | ok | ok |
| deeponet | ok | ok | ok | ok | ok | ok | ok |
| gnot | ok | ok | ok | ok | ok | ok | ok |
| transolver | ok | ok | ok | ok | ok | ok | ok |

Highest peak GPU memory (bytes) / seconds:

- transolver:navier_stokes: 3263951360 / 5.4s
- gnot:navier_stokes: 3202617344 / 2.74s
- transolver:burgers: 3126422528 / 5.36s
- gnot:burgers: 3050928640 / 2.67s
- transolver:reaction_diffusion: 3041909760 / 5.28s

## Boundaries

All numbers above are software acceptance under `acceptance/` and do not enter any paper table. No production run, multi-seed study or convergence result is claimed.
