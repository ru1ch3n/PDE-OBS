# Protocol map: do not conflate these scopes

## Selected paper protocol (description in `configs/paper/protocol.yaml`)

Stationary: Darcy, Poisson, Helmholtz; Dirichlet; smooth_grf; partial solution -> same solution.
Temporal: Heat, reaction_diffusion, scalar 2-D Burgers, NS vorticity; periodic; smooth_grf;
initial state -> three future frames with predicted-state feedback. Stored source frames are 0..14,
input 0, targets 1,2,3; physical times come from each record, not a guessed interval.

Per PDE: 2000 physical records, train1800/test200; regime counts train600/600/600,
test67/67/66; no validation. The corpus contains many more settings. Seeds for generated physical
records, split/mask definitions and model training are not interchangeable. In `paper-row`, the
campaign seed controls split/masks; the row training_seed controls model training.

| Namespace/view | Count on 128x128 | Meaning |
|---|---:|---|
| paper:R50 | 8192 | random distinct points |
| paper:H / paper:V | 8192 | 64 whole rows / columns |
| paper:CL | 8192 | four Gaussian centers; nonwrapped distances |
| paper:BL | 8284 | outside a missing 90x90 square |
| paper:LI | 8284 | 38 rows union 38 columns |
| paper:BD | 8284 | width19 perimeter band |
| paper:R65 | 10650 | random distinct points |
| paper:R80 | 13107 | random distinct points |

Random densities are NOT nested. BL is a missing block, not a visible patch. BD is not an actual
physical wall on a periodic domain. Use `api.make_observation('paper:R50')`; do not construct a
replacement mask from nominal percentage or a substring such as `random_3pct`.

Public methods: ufno_2d (easy alias ufno), fno, cno (CNO-inspired adaptation), deeponet, gnot,
transolver, pino. A withheld eighth learned entry exists in the full campaign registry: do not count
504 scheduled rows as 441 public completed rows. Original500 and demo are portable paper-row
cohorts; fixed200/min120/recovered metadata are NOT executable cohort implementations here.

## General easy API

Config schema: `pdeobs-pipeline/v1`. Model parameters nest inside `model.params`.
Observation parameters are **flat**: `{name: random, ratio: 0.5}`, not
`{name: random, params: {ratio: 0.5}}`. `easy validate` validates structure; it is not evidence of an
executed benchmark or proof that all data/resources will be available.

`preset=smoke` changes architecture. `preset=paper` does not imply paper data, split, budget, or scores.
`split=all` is the default; use explicit train/test IDs for a held-out custom workflow.
Forward/inverse/velocity/custom-observation extensions are executable interfaces, not all evaluated
claims of the selected paper. `mean` uses each example's observed mean, not a fitted TRAIN mean.

## Input and scoring layouts

Canonical stored records: condition NHWC; trajectory NTHWC; geometry NHW1. Target-free
input packages: observations NHWC or NTHWC, binary mask with one trailing channel, explicit
schema. Do not provide hidden targets to `predict`. Integer source-frame labels must not be
silently rounded. `stored` reuses the supplied mask.

Strict scorer accepts recovery/forward/inverse NHWC and rollout NTHWC. Main score per identity:
L2(pred-target) / max(L2(target),1e-12); arithmetic mean over the full expected set; float64.
Rollout primary norm covers all target frames jointly. A relative ratio is not a percentage unless
explicitly multiplied by100 for display. MSE is not relative error.

For static noiseless recovery, raw mask/observation are mandatory for diagnostics; projection is
separate and cannot repair an invalid raw prediction. All200 explicit identities are required for
`paper-test`; general/demo scoring can have other counts and is never automatically paper-certified.
Matrix D averages9 matched cells; C averages72 transferred cells. Delta(v,w)=E(v,w)-E(w,w).
Cells share records/checkpoints, so they are not independent repetitions.

## What is not included or certified

The ZIP has no production HDF5 dataset, published data endpoint, production checkpoints, or paper
result tables. Numerical generators and tiny checks do not independently validate the full corpus.
Archived acceptance reports are software evidence from other environments, not this audit's GPU
measurements. Strict valid scores are not proof of all provenance or numerical reference accuracy.
The chain from a paper cell to its artifacts, and the local checker for author-supplied materials
(`python -m pdeobs.evidence`), are described in `docs/paper_evidence.md`; which acceptance record applies
to what is indexed in `docs/ai/ACCEPTANCE_INDEX.md`.
