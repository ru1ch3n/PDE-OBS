# Observation views

An observation is a boolean mask over the stored spatial grid, applied to a
physical record that has already been generated. The mask is the axis this
benchmark varies while the physical record, its identity and its data split are
held fixed.

![The nine frozen observation views at 128 x 128, each with its realized observed-cell count.](figures/observation_views.png)

*The nine frozen views, drawn by the mask code itself at 128 x 128. Dark is observed, light is
hidden. The counts under each panel are realized, not nominal, and match the table below; regenerate
them with `python docs/figures/make_figures.py`.*

## Mask semantics

A mask is `bool[H, W]` with `True` meaning observed. The mask is supplied to a
model separately from the filled values, so a physical zero is never confused
with a missing value. Unobserved entries are filled with `0.0` by default
(`apply_mask(values, mask, fill_value=0.0)` in
[`src/pdeobs/masks.py`](../src/pdeobs/masks.py)).

`broadcast_mask` expands one spatial mask over the channel and time axes. It
accepts exactly four target layouts — `HW`, `HWC`, `THW`, `THWC` — and raises
`target must have HW, HWC, THW, or THWC shape matching mask` for anything else.

Changing the mask does not change the stored solution, the physical identity, or
the data split, and it does not invoke the numerical solver again. This is
recorded machine-readably as `reuse_physical_record_without_resolving: true` and
`true_means: observed` in
[`configs/paper/observations.yaml`](../configs/paper/observations.yaml).

The observation mask and the geometry (solid/obstacle) field are different
inputs and are never interchanged.

## The nine frozen views at 128 x 128

The nine views are frozen under schema `pdeobs.paper-observations/v1` in
[`configs/paper/observations.yaml`](../configs/paper/observations.yaml) and
mirrored as `PAPER_VIEWS` in
[`src/pdeobs/api/specs.py`](../src/pdeobs/api/specs.py), which is the object the
resolver actually reads. The grid has `128 * 128 = 16,384` cells.

| Code | Canonical view | Mask protocol | Parameters |
|---|---|---|---|
| R50 | `random_50pct` | `random_3pct` | `ratio: 0.50` |
| R65 | `random_65pct` | `random_3pct` | `ratio: 0.65` |
| R80 | `random_80pct` | `random_3pct` | `ratio: 0.80` |
| BL | `block_observed_50pct` | `block_missing` | `missing_fraction: 0.49` |
| LI | `line_sensors_50pct` | `line_sensors` | `num_lines: 76`, `orientation: both` |
| H | `horizontal_lines_50pct` | `line_sensors` | `num_lines: 64`, `orientation: horizontal` |
| V | `vertical_lines_50pct` | `line_sensors` | `num_lines: 64`, `orientation: vertical` |
| BD | `boundary_band_50pct` | `boundary_sensors` | `width: 19` |
| CL | `clustered_50pct` | `clustered_sensors` | `ratio: 0.50` |

The exact observed-cell count of each view is deliberately not repeated here.
[`docs/observation_reference.md`](observation_reference.md) emits it from the
live specifications, so that table cannot drift from the code; use it as the
authoritative list.

The labels are nominal, not exact fractions. Only R50, H, V and CL land on
exactly one half of the grid. BL, LI and BD observe slightly more than half,
R65 slightly more than 65 percent and R80 slightly less than 80 percent, because
each count is a discrete cell count derived from geometry rather than a rounded
percentage. The integer counts, not the labels, are the frozen quantity.

The counts are exact and seed-invariant: they are asserted at the raw-protocol
level at seed 33 in
[`tests/test_release_contracts.py`](../tests/test_release_contracts.py) and
through the resolver at seed 0 in
[`tests/test_benchmark_essentials.py`](../tests/test_benchmark_essentials.py).
The count is invariant; the positions are not — see the seed chain below.

Two views carry a machine-readable `semantic:` caveat in the configuration: BL
(`observed_outside_nonwrapping_90_by_90_missing_square`) and BD
(`outer_array_band_not_a_new_physical_boundary_condition`). The remaining
caveats are stated in the section below.

## Per-view geometry

### Random densities (R50, R65, R80)

Uniform sampling without replacement to an exact discrete count,
`count = round(ratio * H * W)` clipped into `[1, H*W]`. The three densities are
drawn independently per requested ratio.

R50, R65 and R80 are not a nested sensor sequence and should not be described as
one fixed sensor set enlarged monotonically. The configuration records this as a
file-level key, `random_density_masks_guaranteed_nested: false`. The contract
promises reproducibility and count, nothing about nesting; particular NumPy
versions and seeds can produce accidentally nested draws, which is why
[`tests/test_release_contracts.py`](../tests/test_release_contracts.py) asserts
reproducibility and count only and its comment says so explicitly. If a
comparison requires exact sensor pairing across densities, construct and save
the realized masks yourself.

### Block (BL)

`block_missing` places one non-wrapping rectangular unobserved region and
observes everything outside it. With `missing_fraction: 0.49` the side length is
`round(128 * sqrt(0.49)) = 90`, so a `90 x 90` hole is removed from the grid.
The block never wraps across an edge; its top-left corner is drawn from the
per-sample RNG, so the hole's position varies per record while its size does
not.

### Line sensors (LI, H, V)

`line_sensors` selects distinct complete rows and/or columns. These are full
sensor lines, not isolated dots placed approximately on a line.

- LI (`num_lines: 76`, `orientation: both`) splits as
  `horizontal = (76 + 1) // 2 = 38` rows and `vertical = 76 // 2 = 38` columns.
  Row/column intersections are counted once, `38*128 + 38*128 - 38*38`, which is
  why LI is not exactly half the grid.
- H is 64 complete rows and V is 64 complete columns, with no intersection term,
  so both land on exactly half.

Which rows and columns are chosen is drawn from the per-sample RNG.

### Boundary band (BD)

`boundary_sensors` with `width: 19` observes an outer array band 19 cells wide
on all four sides, leaving an unobserved `128 - 2*19 = 90` square interior — the
same interior size as BL, hence the same observed count. The factory constrains
`1 <= width <= min(H, W) // 2`, so width 19 is legal at 128 x 128 (limit 64).

This is sensor geometry, not a new physical boundary condition. For a periodic
PDE, BD still means the outer band of the stored array; it is not a claim that
the PDE has a non-periodic boundary, and it does not interact with the
`geometry` channel.

### Clustered (CL)

`clustered_sensors` with `ratio: 0.50` draws an exact count of cells by weighted
sampling without replacement from a mixture of spatial Gaussians. The view
passes no `clusters` or `spread` override, so the factory defaults apply: four
clusters, spread 0.08 (`sigma = spread * min(H, W)`). There is no periodic
wrapping of the cluster kernels. The cluster centres are drawn from the
per-sample RNG.

## Reproducing one mask bit-for-bit

1. **Campaign seed.** For an executable paper row,
   [`src/pdeobs/paper_row.py`](../src/pdeobs/paper_row.py) reads `campaign["seed"]`
   from the campaign file named by the row's `campaign` key, which for the
   packaged row is
   [`configs/campaign/all_pde_one_setting_10method_9x9.yaml`](../configs/campaign/all_pde_one_setting_10method_9x9.yaml)
   (`seed: 20260804`).
   [`configs/paper/protocol.yaml`](../configs/paper/protocol.yaml) records the
   same value, but it is descriptive
   (`purpose: descriptive_protocol_not_job_submission`) and
   [`configs/paper/README.md`](../configs/paper/README.md) states that these
   files are not job submissions; the campaign file is the one the code reads.
2. **Dataset seed.** `BenchmarkDataset(seed=...)` is what the per-sample
   derivation keys on, and the three entry points supply it differently:
   - [`src/pdeobs/paper_row.py`](../src/pdeobs/paper_row.py) passes the campaign
     seed directly to `load_identity_dataset`, so a paper row's masks come from
     the campaign seed.
   - [`src/pdeobs/strict_inference.py`](../src/pdeobs/strict_inference.py) uses
     `data.mask_seed` when present and the top-level config `seed` otherwise.
   - The generic dataset builder in
     [`src/pdeobs/runner.py`](../src/pdeobs/runner.py) passes the top-level
     config `seed` and does **not** read `data.mask_seed`.
3. **Per-sample mask seed.**
   `derive_seed(dataset_seed, "mask", sample_identity, canonical_protocol_name)`
   in [`src/pdeobs/dataset.py`](../src/pdeobs/dataset.py). The sample identity is
   the record's `sample_id` (falling back to its stored `seed`). `derive_seed`
   ([`src/pdeobs/schema.py`](../src/pdeobs/schema.py)) is a *personalized*
   BLAKE2b digest — `blake2b(digest_size=8, person=b"pdeobs-v1")`, no key — over
   the base seed and NUL-separated string parts, read little-endian and reduced
   mod `2**32`, so it does not depend on process-randomized hashing.
4. **Mask construction.** `generate_mask(protocol, (H, W), seed=mask_seed,
   **options)` builds the array with a local `numpy` generator; no global RNG
   state is touched.

### Which seed moves the sensors

`paper_row.py` writes two separate keys into the resolved configuration: the
row's `training_seed` as the top-level `seed`, and the campaign seed as
`data.mask_seed`. Because the same module then hands the campaign seed straight
to the dataset loader, the training seed does not move a sensor along the
executable paper-row path, and the same holds for the strict-inference path,
which honours `data.mask_seed`.

That separation is not enforced by the generic path. A configuration carrying
the same two keys but executed through `runner.py` derives its masks from the
top-level `seed`, that is from the training seed, because `data.mask_seed` is
ignored there. The packaged row hides this, since its `training_seed` equals the
campaign seed, but a row that varies the training seed while expecting fixed
sensors must go through `paper_row.py` or `strict_inference.py`, or pin both
keys to the same value.

Each resolved observation also carries a content-addressed identifier,
`observation_id = "<namespace>:<name>:<12 hex sha256>"` over namespace, name,
mask configuration and schema version — for example
`paper:random_65pct:b37b1133058e`.

### What the dataset records per sample

| Metadata key | Value |
|---|---|
| `mask_id` | the canonical registry protocol name actually used |
| `mask_seed` | the derived per-sample seed (`None` for the `full` policy) |
| `observation_count` | `int(np.count_nonzero(mask))` |
| `observation_ratio` | `float(np.mean(mask))` |

These four keys are written back into the returned task metadata by
[`src/pdeobs/dataset.py`](../src/pdeobs/dataset.py), so a realized mask can be
audited from a stored record without re-deriving anything.

## Namespaces

[`src/pdeobs/api/observation.py`](../src/pdeobs/api/observation.py) resolves four
namespaces; an unknown one raises
`use general, paper, custom or stored`.

| Namespace | What it resolves | Parameters |
|---|---|---|
| `general` | free parameters over the registered mask factories (`random`, `grid`, `block`, `line`, `horizontal`, `vertical`, `boundary`, `clustered`, `full`) | validated against the real factory signatures; unknown keys and conflicting pairs are rejected before any expensive work |
| `paper` | the nine frozen views, passed verbatim to the same mask construction | none |
| `custom` | a factory registered in `MASK_REGISTRY`, or an explicit boolean array handed to `apply_mask` | passed through unvalidated |
| `stored` | the mask an observation package already carries | none |

`custom` also appears as a row in the generated general-protocol table, but it is
not resolvable from the `general` namespace: requesting it there raises
`use namespace='custom' with the registered factory name for custom masks`.

A `seed=` keyword given to a `custom` observation is recorded in the spec's
`params` only. It is not copied into `mask_config`, which is what
`BenchmarkDataset` receives and what the identifier hashes, so it changes
neither the produced mask nor the `observation_id`. Seed a custom mask through
the dataset seed chain above instead.

Early validation covers unknown keys and the mutually exclusive groups, but it
is not exhaustive at the range boundaries. `ratio` is specified with
`minimum=0.0`, so `make_observation("random", ratio=0.0)` passes API validation
and is rejected later by `_count_from_ratio` in
[`src/pdeobs/masks.py`](../src/pdeobs/masks.py) with `ratio must lie in (0, 1]`.

Request forms: `make_observation("paper:R65")`,
`make_observation("random_65pct", namespace="paper")`,
`make_observation("random", ratio=0.5)`, `make_observation("stored")`,
`make_observation({"protocol": "demo_checkerboard"}, namespace="custom")`.

The frozen views accept no parameters, and asking for a paper view from the
general namespace is a hard error, never a silent fallback:

```
>>> make_observation("random_65pct")
ValueError: 'random_65pct' is a paper view; request it with
namespace='paper' (or 'paper:random_65pct') so the frozen definition is
used explicitly

>>> make_observation("paper:R65", ratio=0.6)
ValueError: paper views are frozen; they accept no parameters
(use the general namespace to vary them)
```

A general request that happens to reproduce a frozen view (for example `random`
at `ratio=0.5`) yields the same array but a different `observation_id`, because
namespace, name and schema version are part of the hash. Provenance stays
distinguishable.

For the full general parameter list, defaults, aliases and the mutually
exclusive parameter groups, use the generated table in
[`docs/observation_reference.md`](observation_reference.md) rather than
restating them here — it is produced from the live specifications by
[`gen_docs.py`](../gen_docs.py), so it cannot drift from the code.

Two general protocols deserve a note: `horizontal` and `vertical` are not
separate factories but the single `line_sensors` factory with a fixed
`orientation`; and the public `block_shape` parameter is one integer side
length, expanded internally to a `(height, width)` pair.

Schema versions, each of which is part of the `observation_id` hash:
`pdeobs-observation-general/v1` for the `general` and `custom` namespaces,
`pdeobs.paper-observations/v1` for the frozen views, and
`pdeobs-inference-input/v1` for `stored`.

## Common misreadings

- The unconfigured default is not a paper view. When no mask is configured,
  `BenchmarkDataset` falls back to `{"protocol": "random_3pct"}`. With neither
  `ratio` nor `count` given, that factory has a 128 x 128-only convention of
  exactly 500 sensors (`MAIN_TRAIN_COUNT_128`), roughly 3 percent of the grid,
  and not the paper's 50 percent training protocol. None of R50/R65/R80 can
  reach it, because each passes an explicit `ratio`. At other grid sizes the
  fallback is 3 percent (for example 123 cells at 64 x 64).
- `regular_grid` is used by none of the nine views. The canonical protocol tuple
  `MASK_PROTOCOL_NAMES` has nine entries, and the paper has nine views, but they
  do not correspond one-to-one. The frozen views reference only `random_3pct`,
  `block_missing`, `line_sensors`, `boundary_sensors` and `clustered_sensors`.
- Default-parameter counts are a different table. `OBSERVATION_COUNTS_128` in
  [`src/pdeobs/protocol.py`](../src/pdeobs/protocol.py) records what each factory
  produces at 128 x 128 with its *default* arguments — for example
  `regular_grid` to 441 cells, `block_missing` to 12288, `line_sensors` and
  `boundary_sensors` to 508, `clustered_sensors` to 492. None of these are paper
  views. In particular the 441 there is a cell count and has nothing to do with
  the 441 planned training rows in
  [`docs/reproducing_the_paper.md`](reproducing_the_paper.md).
- The finer random densities work by name. `random_1pct`, `random_5pct` and
  `random_10pct` are registered factories and are accepted verbatim by the
  general resolver, even though the generated table lists a single `random` row
  whose documented aliases are `random_3pct` and `random_sensors`.
- The nine views are 128 x 128-specific. No sanctioned smaller-grid equivalent is
  defined anywhere in this release. Do not copy the fixed counts onto a demo
  grid; use a ratio or a count legal for that grid instead.

## Dataset-level mask policies

Beyond the registered factories,
[`src/pdeobs/dataset.py`](../src/pdeobs/dataset.py) implements two policies that
exist only at dataset level. Neither is a paper view.

`full` (aliases `full_observation`, `all_visible`) observes every cell. It
rejects any extra option and records `mask_id = "full"`, `mask_seed = None`,
`observation_count = mask.size`, `observation_ratio = 1.0`. It is a diagnostic
configuration. The related evaluation switch is `observation_mode` in
[`src/pdeobs/evaluation.py`](../src/pdeobs/evaluation.py), whose `matched`
versus `full` distinction — and the rule that a full-observation control never
replaces a matched-mask score — is documented in
[`docs/results_format.md`](results_format.md).

`mixed_partial` (aliases `balanced_partial`, `all_partial`) rotates one protocol
per stratum position instead of fixing a single view. Its only accepted option is
a `protocols` candidate list, defaulting to all nine entries of
`MASK_PROTOCOL_NAMES`; duplicates are rejected. Selection is deterministic:

```
factor_offset_seed = derive_seed(dataset_seed, "mixed_partial_protocol",
                                 pde, boundary, setting, regime)
stratum_position   = metadata["regime_sample_index"]   # dataset index as fallback
protocol           = protocols[(stratum_position + factor_offset_seed) % len(protocols)]
```

It writes five extra metadata keys: `mask_policy`, `mask_protocol_selection_seed`,
`mask_protocol_stratum_position`, `mask_protocol_selection_index` and
`mask_protocol_candidates`. The per-sample `mask_seed` is then derived from the
selected protocol exactly as for a fixed view. This policy is used by the
full-to-partial study configuration in
[`src/pdeobs/full_to_partial.py`](../src/pdeobs/full_to_partial.py), not by the
nine-view paper protocol.

## Mask specs: a protocol plus a parameter, and mixtures

[`src/pdeobs/mask_specs.py`](../src/pdeobs/mask_specs.py) adds a third dataset-level
policy that is accepted wherever a protocol name is: `data.mask.protocol` for
training and each entry of `evaluation.observation_protocols` for test views. A
*spec* is a short string:

```
uniform 1                                    one percent of the cells, uniformly at random
line 30 + random 20                          a mixture: line sensors at 30 %, random at 20 %
uniform 1 x2 + sensor 20 + block 50          weighted mixture (uniform drawn twice as often)
line_sensors(num_lines=64, orientation=horizontal) + random_1pct
                                             registered protocols with explicit arguments
```

Components are separated by `+`. Each is a name, then either an *observed
percentage* or a parenthesised `key=value` list, then an optional integer weight
(`x2`). Names are the registered protocols, their registry aliases, and these
shorthand families; the number after a family is always the fraction of cells that
end up observed, converted to the protocol's own parameter at generation time from
the field's shape:

| Family | Resolves to | The percentage becomes |
|---|---|---|
| `uniform`, `random`, `rand`, `u` | `random_3pct` | `ratio = p/100` |
| `sensor(s)`, `cluster(s)`, `clustered` | `clustered_sensors` | `ratio = p/100` |
| `block(s)`, `missing` | `block_missing` | `missing_fraction = 1 - p/100` |
| `grid`, `regular` | `regular_grid` | `ratio = p/100` |
| `boundary`, `edge`, `band` | `boundary_sensors` | `width` solved for the band to cover `p` % |
| `line(s)` | `line_sensors` (both orientations) | `num_lines` solved for `p` % |
| `hline(s)`, `horizontal` / `vline(s)`, `vertical` | `line_sensors` | `num_lines = round(p/100 * H)` (or `W`) |
| `full` | every cell observed | not allowed |

At 128 x 128, `boundary 50` solves to width 19 (the frozen paper band, 50.6 %),
`line 50` to 75 lines (50.02 %; the paper's 76 lines cover 50.6 % and remain
available as `line_sensors(num_lines=76)`), and `block 50` is `missing_fraction = 0.5`.

**Seeds and equivalence.** The per-sample mask seed is derived exactly as for a fixed
view, from the dataset mask seed, the sample id and the *resolved protocol name*,
never from the arguments. So `uniform 20` reproduces `{protocol: random, ratio: 0.2}`
bit-for-bit, and a component produces the same mask for a given sample whether it is
requested alone or inside a mixture. A plain registered name never enters this path.

**Mixtures.** One component is assigned per sample, deterministically: the
weight-expanded cycle (`uniform 1 x2 + block 50` -> `[uniform, uniform, block]`) is
rotated by `derive_seed(mask_seed, "mask_mixture", canonical_spec, pde, boundary,
setting, regime)` and indexed by the sample's stratum position, so the split of a
regime over components is exact up to rounding. The record carries `mask_policy`
(`mixture` or `parameterized`), `mask_spec` (the canonical, re-parseable text),
`mask_view_id`, `mask_component_id`, `mask_component`, and for mixtures the
selection seed, stratum position, cycle index and candidate list. Duplicate
components are rejected; use a weight instead.

**Test views.** In `evaluation.observation_protocols` a spec entry is scored as one
view over the common test identity set, keyed by its canonical id
(`line_sensors-orientation-both-p30__random_3pct-p20` for `line 30 + random 20`);
registered names keep their existing keys. The view context records `mask_spec`,
`mask_mixture`, the component list and, when every component declares a target,
the weight-averaged `observation_ratio`. Two entries that resolve to the same view
are rejected. These views are a study interface: the nine paper views and the
paper-row protocol are unchanged and still reject anything outside `VIEW_ORDER`.

### Seeds: split, mask, training

Three seeds are now distinct. The split is drawn from the campaign seed (see
`stable_split`); the sensors are drawn from `data.mask_seed`, which defaults to the
experiment `seed`; the trainer is seeded from `training.seed`, which defaults to the
experiment `seed` (see `_training_config` in
[`src/pdeobs/runner.py`](../src/pdeobs/runner.py)). Setting `training.seed` alone
therefore changes initialisation and batch order while every sample keeps the same
split membership and the same sensors, which is the controlled multi-seed
configuration. Paper rows already record `split_seed` and `training_seed`
separately and write `data.mask_seed = campaign seed`; that key is now honoured by
the dataset instead of being ignored.

## Rollout semantics

For rollout, one spatial mask is built and broadcast over the available history.
The paper uses history length one: the executable row path fixes
`history_steps=1`, `horizon=3` in
[`src/pdeobs/paper_row.py`](../src/pdeobs/paper_row.py), and
[`configs/paper/protocol.yaml`](../configs/paper/protocol.yaml) describes the
same protocol (`history_steps: 1`, `horizon: 3`, `target_frames: [1, 2, 3]`).

The learned free-rollout recurrence applies the observation only at the first
step and then feeds its own predictions forward; the autoregressive wrapper in
[`src/pdeobs/methods/neural.py`](../src/pdeobs/methods/neural.py) sets the mask
to `None` after step one, with the comment that predicted states are fully
specified from then on. The descriptive protocol pins the same contract from the
other side: `teacher_forcing_ratio: 0.0`, `recurrent_input: previous_prediction`,
`future_ground_truth_as_input: forbidden`.

Future masks paired with true future values are never exposed. Masks restrict
predictor inputs only; supervision targets are the complete field or the
complete future trajectory. See
[`docs/tasks_and_permissions.md`](tasks_and_permissions.md) for the full
information-permission statement.

## Reusing one record under several views

```python
from pdeobs.masks import apply_mask, generate_mask

# `field` is one complete HWC state already read from a Sample.
shape = field.shape[:2]
random_mask = generate_mask("random_3pct", shape, seed=5, ratio=0.5)
block_mask = generate_mask("block_missing", shape, seed=5, missing_fraction=0.49)
random_observation = apply_mask(field, random_mask)
block_observation = apply_mask(field, block_mask)
assert random_observation.shape == block_observation.shape == field.shape
```

Only the observation changes. For information-isolation tests, instantiate the
mask once and keep it fixed before changing hidden targets; a changed seed would
test both masking and prediction and could give a misleading leakage result.

Two derivation details are worth knowing when reusing records:

- The per-sample derivation keys on the canonical protocol name, not the view
  name. All three random densities resolve to `random_3pct`, and LI, H and V all
  resolve to `line_sensors`, so those views share a mask seed on a given record;
  they still differ because their keyword arguments differ.
- The `boundary_sensors` factory deletes its generator (`del rng`) before
  building the band, so BD is bit-identical for every seed. Its optional
  `count=` sub-sampling also uses evenly spaced deterministic probes rather than
  a random subset.

## Scope: what these views are, and what they are not

All nine views are noiseless point observations of the stored field. The strict
scorer enforces this for static recovery: it fails closed if any supplied
observed value disagrees with the recovery target at a mask-true entry
(`observed values disagree with the noiseless recovery target`), and it requires
the mask to contain only 0 and 1
([`src/pdeobs/strict_score.py`](../src/pdeobs/strict_score.py)).

There is no sensor-noise, outlier or measurement-error model anywhere in the
observation configuration, and no field in which to declare one. There is also
no claim that the current core includes moving sensors, arbitrary
continuous-coordinate observations, or an active-learning loop. A custom plugin
may implement such a system, but its assumptions and validation are then
additional work.

Geometry-aware filtering is not applied uniformly by every built-in observation
mechanism, so a sensor count alone does not prove that all selected sites lie in
a fluid region. The mask (observation protocol) and the geometry (solid/obstacle
field) remain different inputs and are never interchanged.

## Cross-links

- [`docs/observation_reference.md`](observation_reference.md) — generated
  parameter, alias and conflict tables for the general namespace, plus the
  frozen-view table with the exact observed counts, both emitted from the live
  specifications.
- [`docs/scoring.md`](scoring.md) — how observed and unobserved cells are
  reported separately (`observed_common_denominator`,
  `hidden_common_denominator`, the separately named `hidden_only_rel_l2` and its
  applicability rule), and the optional data-consistency projection.
- [`docs/results_format.md`](results_format.md) — the `matched` versus `full`
  `observation_mode` distinction and the separate training-view/test-view axis.
- [`docs/extending.md`](extending.md) — registering a new mask factory through
  `MASK_REGISTRY` without touching a PDE kernel, and the tests an extension is
  expected to carry.
- [`docs/reproducing_the_paper.md`](reproducing_the_paper.md) — how the nine
  training views v pair with the nine test views w over a fixed 200-identity
  test set.
- [`configs/paper/observations.yaml`](../configs/paper/observations.yaml) and
  [`src/pdeobs/api/specs.py`](../src/pdeobs/api/specs.py) — the two
  hand-maintained copies of the frozen definitions. The YAML is the documented
  source of truth; the Python dictionary is what the resolver reads.
