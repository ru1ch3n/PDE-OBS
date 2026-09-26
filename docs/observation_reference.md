# Observation reference (v0.2.0)

Four namespaces. `general` exposes free parameters over the existing mask factories (`custom` is
listed below but is not resolvable from `general`); `paper` replays the nine frozen paper views
verbatim (same protocol, kwargs and seed policy); `custom` resolves a factory registered in
`MASK_REGISTRY`; `stored` reuses the mask already carried by an inference input package.

General notes:
- counts are exact discrete cell counts; ratios are targets, not guarantees
- grid/line/boundary geometry is discrete: the realized fraction may differ from the requested ratio
- paper views are frozen 128x128 definitions; small demo grids must not reuse the fixed 128 counts
- block observes everything outside a non-wrapping missing square; boundary is an outer array band, not a boundary condition
- mask (observation protocol) and geometry (solid/obstacle field) are different inputs and are never interchanged

## general protocols

| protocol | aliases | parameters | meaning |
|---|---|---|---|
| `random` | random_3pct, random_sensors | `ratio`=None; `count`=None | uniformly random sensors (exact discrete count); default 3% (128x128 default: 500 sensors) |
| `grid` | regular_grid, regular-grid | `ratio`=0.03; `spacing`=None; `random_offset`=False | regular grid of sensors approximating a target ratio (spacing is discrete; exact ratio not guaranteed) |
| `block` | block_missing, missing_block | `missing_fraction`=0.25; `block_shape`=None | everything observed except one missing square block (missing_fraction of the area) |
| `line` | line_sensors, lines | `ratio`=0.03; `num_lines`=None; `orientation`=both | full rows and/or columns of sensors |
| `horizontal` | horizontal_lines | `ratio`=0.03; `num_lines`=None | full rows of sensors (line protocol, horizontal) |
| `vertical` | vertical_lines | `ratio`=0.03; `num_lines`=None | full columns of sensors (line protocol, vertical) |
| `boundary` | boundary_sensors, edge, boundary_band | `width`=1; `count`=None | a band of `width` cells along the array perimeter (deterministic; not a physical boundary condition) |
| `clustered` | clustered_sensors, clusters | `ratio`=0.03; `count`=None; `clusters`=4; `spread`=0.08 | sensors grouped in Gaussian clusters |
| `full` | full_observation, all_visible | (none) | every cell observed (dataset-level pseudo protocol; accepts no parameters) |
| `custom` | registered | `protocol`= | a mask factory registered in pdeobs.registry.MASK_REGISTRY, or an explicit mask array |

Conflicting parameters (give at most one of each group) are rejected:

- `random`: {ratio, count}
- `grid`: {ratio, spacing}
- `block`: {missing_fraction, block_shape}
- `line`: {ratio, num_lines}
- `horizontal`: {ratio, num_lines}
- `vertical`: {ratio, num_lines}
- `clustered`: {ratio, count}

## paper views (128×128, frozen)

Request with `paper:<view>` or the label, e.g. `paper:R65`. These carry no parameters.

| view | label | mask config | observed cells @128² |
|---|---|---|---|
| `random_50pct` | R50 | `{'protocol': 'random_3pct', 'ratio': 0.5}` | 8192 |
| `random_65pct` | R65 | `{'protocol': 'random_3pct', 'ratio': 0.65}` | 10650 |
| `random_80pct` | R80 | `{'protocol': 'random_3pct', 'ratio': 0.8}` | 13107 |
| `block_observed_50pct` | BL | `{'protocol': 'block_missing', 'missing_fraction': 0.49}` | 8284 |
| `line_sensors_50pct` | LI | `{'protocol': 'line_sensors', 'num_lines': 76, 'orientation': 'both'}` | 8284 |
| `horizontal_lines_50pct` | H | `{'protocol': 'line_sensors', 'num_lines': 64, 'orientation': 'horizontal'}` | 8192 |
| `vertical_lines_50pct` | V | `{'protocol': 'line_sensors', 'num_lines': 64, 'orientation': 'vertical'}` | 8192 |
| `boundary_band_50pct` | BD | `{'protocol': 'boundary_sensors', 'width': 19}` | 8284 |
| `clustered_50pct` | CL | `{'protocol': 'clustered_sensors', 'ratio': 0.5}` | 8192 |

general schema: `pdeobs-observation-general/v1` (also used by `custom`); paper schema: `pdeobs.paper-observations/v1`; stored schema: `pdeobs-inference-input/v1`
