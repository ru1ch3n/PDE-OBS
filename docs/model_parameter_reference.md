# Model parameter reference (v0.2.0)

Every parameter is validated by the easy API before any expensive work: unknown keys, wrong types,
out-of-range values, alias confusion (`width` vs `hidden`, `depth` vs `layers`, `physics_slices` vs
`slices`) and illegal combinations are rejected with the list of valid options. A parameter that
cannot be mapped to the real constructor is never silently ignored.

`geometry_channels` defaults to 1 (the paper setting, 1=solid); set 0 for models that accept it to
run without a geometry input. `in_channels`/`out_channels` are taken from the dataset unless overridden.

## fno — Fourier Neural Operator (compact PDE-OBS adaptation)

- family: neural; tasks: recovery, forward, inverse, rollout; rollout adapter: autoregressive; requires torch: True
- aliases: fno2d, compact_fno
- **main paper model**

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|
| `in_channels` | int | 1 | [1, None] | io | input state channels (1 scalar field; 2 for NS velocity) |
| `out_channels` | int | 1 | [1, None] | io | output state channels |
| `geometry_channels` | int | 1 | [0, 1] | io | obstacle/geometry channels appended to the input (0 disables; 1 is the paper setting) |
| `width` | int | 32 | [1, None] | structure | channel width of every spectral layer; must be divisible by min(8, width) (GroupNorm) |
| `modes` | int | 12 | [1, None] | structure | retained Fourier modes per axis; modes > H//2 are clipped at run time |
| `layers` | int | 4 | [1, None] | structure | number of spectral+local layers |

Constraints:
- width % min(8, width) == 0

Presets: `smoke` = {'width': 16, 'modes': 6, 'layers': 2}; `paper` = {'width': 32, 'modes': 12, 'layers': 4}

## pino — Physics-Informed Neural Operator (PDE-OBS adaptation, FNO backbone + residual loss)

- family: neural; tasks: recovery, forward, rollout; rollout adapter: autoregressive; requires torch: True
- aliases: pino_static, physics_informed_fno
- **main paper model**

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|
| `in_channels` | int | 1 | [1, None] | io | input state channels (1 scalar field; 2 for NS velocity) |
| `out_channels` | int | 1 | [1, None] | io | output state channels |
| `geometry_channels` | int | 1 | [0, 1] | io | obstacle/geometry channels appended to the input (0 disables; 1 is the paper setting) |
| `width` | int | 32 | [1, None] | structure | FNO backbone width; must be divisible by min(8, width) |
| `modes` | int | 12 | [1, None] | structure | retained Fourier modes per axis |
| `layers` | int | 4 | [1, None] | structure | number of spectral+local layers |
| `physics_contract` | str | pino_static_fd_v1 | {pino_static_fd_v1, pino_rollout_spectral_v1} | physics | residual contract; static tasks use pino_static_fd_v1, periodic rollout uses pino_rollout_spectral_v1 |

Constraints:
- width % min(8, width) == 0
- physics_contract must match task: static->pino_static_fd_v1, rollout->pino_rollout_spectral_v1
- training requires physics_loss == physics_contract, physics_loss_weight > 0 and amp == False
- inverse task is not supported by any PINO residual

Presets: `smoke` = {'width': 16, 'modes': 6, 'layers': 2}; `paper` = {'width': 64, 'modes': 20, 'layers': 5}

## ufno — U-FNO-2D (PDE-OBS adaptation)

- family: neural; tasks: recovery, forward, inverse, rollout; rollout adapter: autoregressive; requires torch: True
- aliases: u_fno, ufno2d, u_fno_2d, ufno_2d
- **main paper model**

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|
| `in_channels` | int | 1 | [1, None] | io | input state channels (1 scalar field; 2 for NS velocity) |
| `out_channels` | int | 1 | [1, None] | io | output state channels |
| `geometry_channels` | int | 1 | [0, 1] | io | obstacle/geometry channels appended to the input (0 disables; 1 is the paper setting) |
| `width` | int | 36 | [1, None] | structure | channel width (no GroupNorm; any positive width) |
| `modes` | int | 12 | [1, None] | structure | retained Fourier modes per axis |
| `padding` | int | 8 | [0, None] | structure | zero padding applied before the spectral stack |
| `dropout` | float | 0.0 | [0.0, 1.0] | regularization | dropout inside the three U-branches |

Constraints:
- block count is fixed: 6 spectral layers, U-branches on layers 4-6 (no `layers` parameter)
- U-branches use BatchNorm2d: training batch size must be >= 2

Presets: `smoke` = {'width': 12, 'modes': 6, 'padding': 2}; `paper` = {'width': 36, 'modes': 12, 'padding': 8, 'dropout': 0.0}

## cno — CNO-inspired anti-aliased operator (compact PDE-OBS adaptation)

- family: neural; tasks: recovery, forward, inverse, rollout; rollout adapter: autoregressive; requires torch: True
- aliases: cno2d, compact_cno, cno_inspired
- **main paper model**

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|
| `in_channels` | int | 1 | [1, None] | io | input state channels (1 scalar field; 2 for NS velocity) |
| `out_channels` | int | 1 | [1, None] | io | output state channels |
| `geometry_channels` | int | 1 | [0, 1] | io | obstacle/geometry channels appended to the input (0 disables; 1 is the paper setting) |
| `width` | int | 32 | [1, None] | structure | encoder width; the decoder uses 2x width |

Constraints:
- depth is fixed (encoder / mid / decoder); there is no `layers` or `depth` parameter

Presets: `smoke` = {'width': 8}; `paper` = {'width': 32}

## deeponet — Variable-sensor DeepONet (PDE-OBS adaptation)

- family: neural; tasks: recovery, forward, inverse, rollout; rollout adapter: autoregressive; requires torch: True
- aliases: variable_sensor_deeponet
- **main paper model**

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|
| `in_channels` | int | 1 | [1, None] | io | input state channels (1 scalar field; 2 for NS velocity) |
| `out_channels` | int | 1 | [1, None] | io | output state channels |
| `geometry_channels` | int | 1 | [0, 1] | io | obstacle/geometry channels appended to the input (0 disables; 1 is the paper setting) |
| `hidden` | int | 128 | [1, None] | structure | hidden width of branch/trunk MLPs |
| `latent` | int | 128 | [1, None] | structure | branch/trunk latent dimension |
| `branch_layers` | int | 2 | [1, None] | structure | branch depth |
| `trunk_layers` | int | 2 | [1, None] | structure | trunk depth |
| `branch_mode` | str | sensor_set | {sensor_set, spatial_cnn} | structure | sensor_set (permutation-invariant) or spatial_cnn (ordered mask-aware) |
| `branch_grid` | int | 4 | [1, None] | structure | pooling grid of the spatial_cnn branch |
| `branch_width` | int? | None | [1, None] | structure | conv width of the spatial_cnn branch (default min(hidden, 64)) |

Presets: `smoke` = {'hidden': 16, 'latent': 16, 'branch_layers': 1, 'trunk_layers': 1, 'branch_mode': 'spatial_cnn', 'branch_grid': 2, 'branch_width': 8}; `paper` = {'hidden': 128, 'latent': 128, 'branch_layers': 2, 'trunk_layers': 2, 'branch_mode': 'spatial_cnn', 'branch_grid': 4, 'branch_width': 64}

## gnot — GNOT (mask-conditioned PDE-OBS adaptation)

- family: neural; tasks: recovery, forward, inverse, rollout; rollout adapter: autoregressive; requires torch: True
- aliases: mask_conditioned_gnot
- **main paper model**

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|
| `in_channels` | int | 1 | [1, None] | io | input state channels (1 scalar field; 2 for NS velocity) |
| `out_channels` | int | 1 | [1, None] | io | output state channels |
| `geometry_channels` | int | 1 | [0, 1] | io | obstacle/geometry channels appended to the input (0 disables; 1 is the paper setting) |
| `hidden` | int | 128 | [1, None] | structure | token width; must be divisible by heads |
| `layers` | int | 3 | [1, None] | structure | number of attention blocks |
| `heads` | int | 1 | [1, None] | structure | attention heads |
| `experts` | int | 2 | [1, None] | structure | gated MLP experts per block |
| `inner_ratio` | int | 4 | [1, None] | structure | expert MLP expansion ratio |
| `mlp_layers` | int | 3 | [1, None] | structure | depth of the source/query encoders |
| `dropout` | float | 0.0 | [0.0, 1.0] | regularization | attention/MLP dropout |

Constraints:
- hidden % heads == 0

Presets: `smoke` = {'hidden': 16, 'layers': 1, 'heads': 1, 'experts': 1, 'inner_ratio': 2, 'mlp_layers': 1}; `paper` = {'hidden': 128, 'layers': 3, 'heads': 1, 'experts': 2}

## transolver — Transolver (mask-conditioned PDE-OBS adaptation)

- family: neural; tasks: recovery, forward, inverse, rollout; rollout adapter: autoregressive; requires torch: True
- aliases: transolver_2d
- **main paper model**

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|
| `in_channels` | int | 1 | [1, None] | io | input state channels (1 scalar field; 2 for NS velocity) |
| `out_channels` | int | 1 | [1, None] | io | output state channels |
| `geometry_channels` | int | 1 | [0, 1] | io | obstacle/geometry channels appended to the input (0 disables; 1 is the paper setting) |
| `hidden` | int | 256 | [1, None] | structure | token width; must be divisible by heads |
| `layers` | int | 5 | [1, None] | structure | number of physics-attention blocks |
| `heads` | int | 8 | [1, None] | structure | attention heads |
| `slices` | int | 32 | [1, None] | structure | physics slices per head |
| `mlp_ratio` | int | 1 | [1, None] | structure | MLP expansion ratio |
| `dropout` | float | 0.0 | [0.0, 1.0] | regularization | dropout |

Constraints:
- hidden % heads == 0

Presets: `smoke` = {'hidden': 16, 'layers': 1, 'heads': 2, 'slices': 4, 'mlp_ratio': 1}; `paper` = {'hidden': 128, 'layers': 8, 'heads': 8, 'slices': 64, 'mlp_ratio': 1}

Temporal presets (rollout): `paper` = {'hidden': 256, 'layers': 8, 'heads': 8, 'slices': 32, 'mlp_ratio': 1}

## unet — Mask-channel U-Net (compact reference)

- family: neural; tasks: recovery, forward, inverse, rollout; rollout adapter: autoregressive; requires torch: True
- aliases: mask_unet, mask_channel_unet

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|
| `in_channels` | int | 1 | [1, None] | io | input state channels (1 scalar field; 2 for NS velocity) |
| `out_channels` | int | 1 | [1, None] | io | output state channels |
| `geometry_channels` | int | 1 | [0, 1] | io | obstacle/geometry channels appended to the input (0 disables; 1 is the paper setting) |
| `width` | int | 32 | [1, None] | structure | base width; doubled per level |

Presets: `smoke` = {'width': 4}; `paper` = {'width': 32}

Notes: not part of the paper's seven learned models

## unet_paper_guided — Deep paper-guided mask U-Net

- family: neural; tasks: recovery, forward, inverse, rollout; rollout adapter: autoregressive; requires torch: True
- aliases: deep_mask_unet

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|
| `in_channels` | int | 1 | [1, None] | io | input state channels (1 scalar field; 2 for NS velocity) |
| `out_channels` | int | 1 | [1, None] | io | output state channels |
| `geometry_channels` | int | 1 | [0, 1] | io | obstacle/geometry channels appended to the input (0 disables; 1 is the paper setting) |
| `width` | int | 64 | [1, None] | structure | base width |
| `levels` | int | 4 | [1, None] | structure | encoder levels |

Presets: `smoke` = {'width': 4, 'levels': 2}

## convlstm — ConvLSTM rollout baseline

- family: neural; tasks: rollout; rollout adapter: native; requires torch: True
- aliases: conv_lstm

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|
| `in_channels` | int | 1 | [1, None] | io | input channels |
| `out_channels` | int | 1 | [1, None] | io | output channels |
| `hidden_channels` | int | 32 | [1, None] | structure | recurrent hidden channels |

Constraints:
- no geometry input

Presets: `smoke` = {'hidden_channels': 4}

## mae_small — Masked autoencoder (small reference)

- family: neural; tasks: recovery, forward, inverse; rollout adapter: native; requires torch: True
- aliases: masked_autoencoder, masked_autoencoder_small, mae_style_small

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|
| `in_channels` | int | 1 | [1, None] | io | input channels |
| `out_channels` | int | 1 | [1, None] | io | output channels |
| `width` | int | 32 | [1, None] | structure | width |
| `latent_channels` | int | 128 | [1, None] | structure | latent channels |
| `patch_size` | int | 8 | [1, None] | structure | patch size |
| `mask_ratio` | float | 0.75 | [0.0, 1.0] | regularization | pretraining mask ratio |
| `preserve_visible` | bool | True |  | io | copy visible values through |

Constraints:
- no geometry input

Presets: `smoke` = {'width': 4, 'latent_channels': 8, 'patch_size': 4}

## zero — Zero fill

- family: classical; tasks: recovery, forward, inverse; rollout adapter: native; requires torch: False

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|

Presets: `smoke` = {}; `paper` = {}

## mean — Mean of observed values

- family: classical; tasks: recovery, forward, inverse; rollout adapter: native; requires torch: False

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|

Presets: `smoke` = {}; `paper` = {}

## nearest — Nearest-neighbour interpolation

- family: classical; tasks: recovery, forward, inverse; rollout adapter: native; requires torch: False

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|

Presets: `smoke` = {}; `paper` = {}

## bilinear — Bilinear interpolation

- family: classical; tasks: recovery, forward, inverse; rollout adapter: native; requires torch: False

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|

Presets: `smoke` = {}; `paper` = {}

## rbf — Gaussian RBF interpolation

- family: classical; tasks: recovery, forward, inverse; rollout adapter: native; requires torch: False
- aliases: gaussian_rbf

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|
| `smoothing` | float | 1e-06 | [0.0, None] | structure | RBF smoothing |
| `epsilon` | float? | None |  | structure | kernel shape parameter |
| `max_centers` | int | 512 | [1, None] | structure | maximum sensors used as centers |
| `normalized_coordinates` | bool | False |  | structure | use [0,1] coordinates |

Presets: `smoke` = {}; `paper` = {'smoothing': 1e-06, 'epsilon': 2.0, 'max_centers': 512, 'normalized_coordinates': True}

## persistence — Persistence rollout

- family: classical; tasks: rollout; rollout adapter: native; requires torch: False

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|

Presets: `smoke` = {}; `paper` = {}

## gappy_pod — Gappy POD (fitted linear reduced-order method)

- family: fitted; tasks: recovery, forward, inverse; rollout adapter: native; requires torch: False

| parameter | type | default | range/choices | effect | meaning |
|---|---|---|---|---|---|
| `rank` | int? | None | [1, None] | structure | POD rank (None selects among rank_candidates) |
| `ridge` | float | 1e-06 | [0.0, None] | structure | ridge regularization |
| `oversampling` | int | 16 | [0, None] | structure | oversampling |
| `seed` | int | 0 |  | structure | seed |

Presets: `smoke` = {'rank': 4}; `paper` = {'rank': 64, 'ridge': 1e-06}

Notes: trained through fit_dataset, not the Trainer; served through the legacy `train` runner path
