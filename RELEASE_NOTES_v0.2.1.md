# PDE-OBS v0.2.1 — observation specs: parameters, mixtures and a pinned mask seed

This revision adds one interface and changes no science. A mask may now be requested as a
**protocol with a parameter** (`uniform 1`, `block 50`, `line 30`, `boundary 50`), and several such
components may be joined with `+` into a **deterministic mixture** (`uniform 1 x2 + sensor 20 +
block 50`). The same string is accepted for **training** (`data.mask.protocol`) and for a **test
view** (`evaluation.observation_protocols`, `--obs`, pipeline `observation:`), so a model can be
trained on one mixture of patterns and scored on another. A dataset-level **mask seed**
(`data.mask_seed`) and the trainer's own seed (`training.seed`) are now independent, which gives a
controlled multi-seed configuration: split membership and sensors fixed, initialisation and batch
order varying.

Package version `0.2.1`; numerical kernel version unchanged at `0.1.0`; the nine frozen views,
the registered mask factories, every seed derivation of an existing configuration, and the
`pdeobs-strict-v1` scorer are unchanged. `src/pdeobs/masks.py` and `src/pdeobs/evaluation.py` are
byte-identical to v0.2.0. The full guide is [`docs/observation_specs.md`](docs/observation_specs.md).

## What is new

| Capability | How to ask for it | Where |
|---|---|---|
| a protocol with a parameter | `uniform 1`, `sensor 20`, `block 50`, `line 30`, `boundary 50`, `grid 25` (the number is the observed percentage) | `data.mask.protocol`, `evaluation.observation_protocols`, `--obs`, `api.make_observation`, pipeline `observation:` |
| explicit arguments | `line_sensors(num_lines=64, orientation=horizontal)` | same |
| a mixture, optionally weighted | `uniform 1 x2 + sensor 20 + block 50` | same; one component is assigned per sample, deterministically |
| a mixture as a test view | `line 30 + random 20` | scored as one view over the common test identity set, keyed `line_sensors-orientation-both-p30__random_3pct-p20` |
| sensors pinned across training seeds | `data.mask_seed: <int>` (defaults to `seed`) | runner configuration; paper rows already wrote this key and it is now honoured |
| an independent training seed | `training.seed: <int>` | runner configuration (`TrainingConfig.seed`); the numpy fallback loader now shuffles from it as well |

## What is guaranteed

- A plain registered protocol name never enters the new path; the legacy dictionary form
  (`{protocol: random, ratio: 0.2}`) is untouched, and `uniform 20` reproduces it **bit-for-bit**
  (same per-sample seed, same cells), because the per-sample mask seed is derived from the resolved
  protocol name alone, exactly as before.
- A component gives the same mask for a given sample whether it is requested alone or inside a
  mixture; a mixture's assignment of components to samples is a deterministic function of the mask
  seed, the canonical spec and the sample's factor stratum, exact up to rounding across a regime.
- The percentage is always the fraction of cells observed; at 128 x 128, `boundary 50` solves to the
  frozen paper band (width 19), `block 50` is `missing_fraction = 0.5`, and `line 50` solves to
  75 lines (50.02 %; the paper's 76 lines cover 50.6 % and remain available as an explicit argument).
- Every record carries the spec, the component that produced its mask and the selection provenance;
  every scored view carries the spec, its components and the weight-averaged target observation ratio.
- The paper protocol (`paper_row`, the nine views, `VIEW_ORDER`) is unchanged and still rejects
  anything outside the nine names. Specs are a study interface, not a paper cell.

## Evidence

`tests/test_mask_specs.py` (35 cases: grammar, realised fractions at 128 x 128, weighted rotation,
legacy equivalence, mask-seed pinning, runner views, the `easy` API) plus the existing suite,
722 tests in total on CPU. The protected-source manifest (`docs/ai/frozen_science_sha256.json`)
records the revised hashes of `dataset.py`, `runner.py` and `__init__.py` under `revisions`, with
the previous hashes and the reason; `python tools/ai_assist.py doctor --json` passes.

## Files

New: `src/pdeobs/mask_specs.py`, `tests/test_mask_specs.py`, `docs/observation_specs.md`,
this note. Modified: `src/pdeobs/dataset.py` (`mask_seed`, the spec branch), `src/pdeobs/runner.py`
(`mask_seed` pass-through, canonical view ids, spec-aware view context, fallback shuffle seed),
`src/pdeobs/api/observation.py` (specs in the `general` namespace), `src/pdeobs/__init__.py`
(version), `pyproject.toml`, `docs/observations.md`, `docs/cli_reference.md`,
`docs/repository_map.md`, `README.md`, `llms.txt`, `AI_PATCH_NOTES.md`,
`docs/ai/frozen_science_sha256.json`.
