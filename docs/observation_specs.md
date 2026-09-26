# Observation specs: parameters, mixtures and seeds

*Added in v0.2.1. This page is the user guide; the mask semantics and the nine frozen views are in
[Observation views](observations.md), the grammar's implementation is
[`src/pdeobs/mask_specs.py`](../src/pdeobs/mask_specs.py).*

PDE-OBS trains one model per observation pattern and scores it on the nine frozen views. Two
questions that design cannot answer by itself are whether a model trained on a *mixture* of patterns,
under the same budget, does as well as the per-pattern models, and how much of a reported gap is
run-to-run variance. Version 0.2.1 answers both without touching the nine views: an observation may
be written as a **spec**, and the training seed may be moved while the split and the sensors stay
fixed.

## 1. Writing a spec

A spec is a short string accepted wherever a protocol name is accepted.

```
uniform 1                                  1 % of the cells observed, uniformly at random
sensor 20                                  clustered point sensors covering 20 %
block 50                                   one missing block, 50 % of the cells still observed
line 30                                    line sensors covering 30 % (both orientations)
boundary 50                                a boundary band covering 50 %
line 30 + random 20                        a mixture of two components
uniform 1 x2 + sensor 20 + block 50        a weighted mixture: uniform drawn twice as often
line_sensors(num_lines=64, orientation=horizontal) + random_1pct
                                           registered protocols with explicit arguments
```

Rules:

- Components are separated by `+`. Each is a *name*, then optionally either an **observed
  percentage** or a parenthesised `key=value` list, then optionally an integer weight (`x2` or `*2`).
- The number after a name is **always the fraction of cells that end up observed**, in percent. It
  is converted to the protocol's own parameter at generation time from the field's shape, so the
  same spec means the same coverage at any resolution.
- Names are the registered protocols (`random_3pct`, `block_missing`, `line_sensors`, ...), their
  registry aliases, and these shorthand families:

| Family | Resolves to | The percentage becomes |
|---|---|---|
| `uniform`, `random`, `rand`, `u` | `random_3pct` | `ratio = p/100` |
| `sensor(s)`, `cluster(s)`, `clustered` | `clustered_sensors` | `ratio = p/100` |
| `block(s)`, `missing` | `block_missing` | `missing_fraction = 1 - p/100` |
| `grid`, `regular` | `regular_grid` | `ratio = p/100` |
| `boundary`, `edge`, `band` | `boundary_sensors` | `width` solved so the band covers `p` % |
| `line(s)` | `line_sensors`, both orientations | `num_lines` solved for `p` % |
| `hline(s)`, `horizontal` / `vline(s)`, `vertical` | `line_sensors`, one orientation | `num_lines = round(p/100 * H)` (or `W`) |
| `full` | every cell observed | not allowed |

- A percentage may be combined with arguments that do not set the same quantity
  (`line(orientation=horizontal) 30` is fine; `uniform(ratio=0.2) 20` is refused).
- Duplicate components are refused; use a weight instead. `full` may appear as a mixture component.
- At 128 x 128: `boundary 50` gives width 19 (the frozen paper band, 50.6 %); `block 50` gives
  `missing_fraction = 0.5`; `line 50` gives 75 lines (50.02 %; the paper's 76 lines, 50.6 %, are
  available as `line_sensors(num_lines=76)`).

Every spec has a **canonical text**, which re-parses to the same spec and is what the records store
(`random_3pct 1 x2 + clustered_sensors 20 + block_missing 50`), and a **view id** used as a result key
and a directory name (`random_3pct-p1-x2__clustered_sensors-p20__block_missing-p50`). Registered
names keep their own keys.

## 2. Where a spec is accepted

| Interface | Field | Example |
|---|---|---|
| runner configuration, training | `data.mask.protocol` | `mask: {protocol: "uniform 1 x2 + sensor 20 + block 50"}` |
| runner configuration, test views | `evaluation.observation_protocols` | `["full", "uniform 20", "line 30 + random 20"]` |
| `easy` CLI | `--obs` | `--obs "line 30 + random 20"` (no `--obs-param`) |
| Python API | `api.make_observation` | `api.make_observation("uniform 1 + sensor 20")` |
| pipeline file | `observation:` | `observation: {protocol: "uniform 1 x2 + block 50"}` |
| mapping form | `data.mask` | `{protocol: mixture, components: [{protocol: uniform, percent: 1, weight: 2}, "block 50"]}` |

Not accepted, by design: the paper protocol (`pdeobs paper-row`, `paper_row.py`, `VIEW_ORDER`),
which still runs exactly the nine frozen views. A spec is a study interface; a score obtained with
one is not a paper cell and must be reported as such.

## 3. Train on a mixture, test on other patterns

A complete runner configuration (the ordinary `pdeobs train` / `pdeobs eval` path):

```yaml
schema_version: pdeobs.experiment/v1
name: poisson-fno-mixed-pattern
seed: 20260804              # split membership and, by default, the sensors
task: recovery

data:
  root: ${PDEOBS_DATA}
  train_glob: "poisson/**/*.h5"
  split: iid
  filters: {pde: poisson}
  mask:
    protocol: "uniform 1 x2 + sensor 20 + block 50"   # training patterns, one per sample
  mask_seed: 20260804       # sensors pinned to this seed (defaults to `seed`)

training:
  epochs: 500               # match the per-pattern rows you compare against
  batch_size: 4
  learning_rate: 0.001
  seed: 20260804            # trainer seed; move only this for a multi-seed subset

method: {name: fno, kwargs: {}}

evaluation:
  observation_protocols:
    - full
    - "uniform 20"
    - "line 30 + random 20"
    - random_3pct            # registered names are still accepted verbatim
```

```bash
pdeobs train --config poisson-fno-mixed.yaml --output runs/poisson-fno-mixed
pdeobs eval  --config poisson-fno-mixed.yaml \
             --checkpoint runs/poisson-fno-mixed/checkpoints/last.pt \
             --output runs/poisson-fno-mixed/metrics.json
```

`metrics.json` then holds one entry per view under `observation_views`, keyed
`full`, `random_3pct-p20`, `line_sensors-orientation-both-p30__random_3pct-p20` and `random_3pct`.

The same thing through the one-line interface, one test view per call:

```bash
pdeobs easy train --data records/poisson --task recovery --model fno --preset paper \
                  --obs "uniform 1 x2 + sensor 20 + block 50" --epochs 500 --out runs/mixed
pdeobs easy eval  --model-from runs/mixed/model --data records/poisson \
                  --obs "line 30 + random 20" --out runs/mixed/eval-line30-random20
```

**Matched budget.** To compare a mixed-pattern model against the per-pattern rows of the grid, give
it the same `epochs`, `batch_size`, optimizer and scheduler as those rows and the same dataset split
(same `seed`). The spec changes only which cells each sample exposes; the runner records everything
else in `row_resolved.json` / the run's resolved configuration, so the comparison can be audited.

## 4. A controlled multi-seed subset

Three seeds are independent:

| Seed | Controls | Default |
|---|---|---|
| campaign / experiment `seed` | which identities are train and test (the split) | required |
| `data.mask_seed` | where the sensors are, per sample | `seed` |
| `training.seed` | parameter initialisation and batch order | `seed` |

To measure run-to-run variance on a subset, copy a row's configuration and change **only**
`training.seed`; keep `seed` and `data.mask_seed` as they are. Each run then sees the same test
identities under the same masks and differs only in initialisation and batch order. Paper rows
already record `split_seed` and `training_seed` separately and write `data.mask_seed` equal to the
campaign seed; v0.2.1 makes the dataset honour that key instead of ignoring it.

```yaml
seed: 20260804            # unchanged across the subset
data:
  mask_seed: 20260804     # unchanged across the subset
training:
  seed: 1                 # 1, 2, 3, ... one run each
```

## 5. What is recorded

Per sample (in the record metadata, so it travels with every prediction bundle):

| Key | Content |
|---|---|
| `mask_policy` | `parameterized` (one component) or `mixture` |
| `mask_spec` | the canonical, re-parseable spec text |
| `mask_view_id` | the view id (result key / directory name) |
| `mask_component_id`, `mask_component` | the component that produced this sample's mask |
| `mask_id`, `mask_seed` | the resolved protocol and the per-sample seed, exactly as for a fixed view |
| `observation_count`, `observation_ratio` | realised, not nominal |
| `mask_mixture_selection_seed`, `mask_mixture_stratum_position`, `mask_mixture_selection_index`, `mask_mixture_candidates` | mixtures only: how the component was chosen |

Per scored view (in the evaluation output under `observation_views[<view id>]`): `mask_protocol`
(the view id), `mask_spec`, `mask_mixture`, `mask_components` (each with its protocol, arguments,
percentage, weight and target fraction) and, when every component declares a target,
`observation_ratio` as the weight-averaged target. The realised per-sample ratio is in the records.

## 6. Determinism and equivalence

- The per-sample mask seed is `derive_seed(mask_seed, "mask", sample_id, protocol)`, the same
  derivation as a fixed view, from the **resolved protocol name only**. Hence `uniform 20` and the
  legacy `{protocol: random, ratio: 0.2}` produce identical masks, and a component produces the same
  mask for a sample whether requested alone or inside a mixture.
- A mixture assigns components by rotating the weight-expanded cycle (`uniform 1 x2 + block 50`
  becomes `[uniform, uniform, block]`) by `derive_seed(mask_seed, "mask_mixture", canonical_spec,
  pde, boundary, setting, regime)` and indexing it by the sample's stratum position. Two runs with the
  same mask seed and spec assign identical components; the split across components within a regime is
  exact up to rounding.
- Two `observation_protocols` entries that resolve to the same view are refused.
- A plain registered name never enters this path, so every existing configuration, hash and seed is
  unchanged; `masks.py` and `evaluation.py` are byte-identical to v0.2.0.

## 7. Suggested wording for a paper

> *Mixed-pattern baseline.* For each (PDE, method) pair we additionally train one model on a
> deterministic mixture of observation patterns, each training sample receiving one pattern drawn
> from {uniform 1 %, clustered sensors 20 %, block 50 %} with weights 2:1:1 (`uniform 1 x2 +
> sensor 20 + block 50` in the released code), under the same epoch budget, optimizer, schedule and
> identity split as the per-pattern models. The pattern assigned to a sample is a deterministic
> function of the dataset seed and the sample's stratum, and the per-sample sensor placement is the
> same one the per-pattern models see. The mixed model is scored on the nine frozen views and on
> held-out mixtures (e.g. `line 30 + random 20`).
>
> *Seed variance.* For a subset of S settings we repeat training with three trainer seeds while
> holding the identity split and the sensor placements fixed (`training.seed` varied,
> `seed` and `data.mask_seed` unchanged), and report the range across seeds beside the single-seed
> grid value.

Replace the pattern list, weights and S with what you actually ran; the records and the evaluation
output carry the exact spec and seeds, so the numbers can be traced.

## 8. What this is not

It adds no learned or adaptive sampler; weights are static integers; the assignment rotates rather
than samples independently (by design, for exact balance and reproducibility). It does not change the
`full_to_partial` study, whose `mixed_partial` policy over the nine raw protocols is kept as is. And
it does not turn a study run into a paper cell.
