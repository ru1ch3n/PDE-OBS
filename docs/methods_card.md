# Methods card: the seven learned baselines, the classical comparators, and what each adaptation claims

## How to read this card

[`model_parameter_reference.md`](model_parameter_reference.md) is generated from the live
specification tables in [`src/pdeobs/api/specs.py`](../src/pdeobs/api/specs.py) by
[`gen_docs.py`](../gen_docs.py). It is the **parameter contract**: constructor arguments, types,
ranges, enforced constraints and frozen presets.

That file is generated *from* the specs, but nothing in the test suite regenerates it and diffs the
result — no test in [`tests/`](../tests/) refers to `gen_docs.py` or to
`model_parameter_reference.md`. It therefore reflects the specs as of the last time someone ran the
generator, and it can lag. Where the generated reference and the source disagree, the source is
authoritative.

This card is the **provenance and scope contract**. It records what the generated reference cannot:
which upstream work each baseline is anchored on, what the adaptation deliberately does *not* claim,
which upstream revision and licence were inspected, what the frozen training recipe is per method,
and which registered methods are scaffolding rather than reported baselines. It is hand-written and
may lag the code; where the two disagree, the generated reference and the source are authoritative.

Two structural points first.

**Counting rule.** `release/candidate_metadata.json` states it directly: *registry aliases and
wrappers are not distinct algorithms*. The method registry ships 26 built-in factories
([`src/pdeobs/methods/__init__.py`](../src/pdeobs/methods/__init__.py), `BUILTIN_METHODS`); of those,
21 are publicly enumerated by `available_methods()` and five are hidden. The easy API's
`MODEL_SPECS` table holds 21 entries: 18 public model names (`fno`, `pino`, `ufno`, `cno`,
`deeponet`, `gnot`, `transolver`, `unet`, `unet_paper_guided`, `convlstm`, `mae_small`, `zero`,
`mean`, `nearest`, `bilinear`, `rbf`, `persistence`, `gappy_pod`) plus three dependency-gated
upstream wrappers. The paper protocol freezes seven learned methods. Those numbers describe registry
surface, public API surface and scientific claim respectively, and they are not interchangeable. An
autoregressive rollout wrapper is a temporal adapter around a one-step model, not an eighth
architecture; `ufno`, `u_fno`, `ufno2d`, `u_fno_2d` and `ufno_2d` are five spellings of one model.

**The protocol id `ufno_2d` resolves to the registry name `ufno`.**
[`configs/paper/protocol.yaml`](../configs/paper/protocol.yaml) lists
`learned_methods: [ufno_2d, fno, cno, deeponet, gnot, transolver, pino]`, but
[`src/pdeobs/methods/neural.py`](../src/pdeobs/methods/neural.py) registers the class as
`ufno` with aliases `("u_fno", "ufno2d", "u_fno_2d")`. The API alias index in
[`src/pdeobs/api/specs.py`](../src/pdeobs/api/specs.py) adds `ufno_2d`, and
[`src/pdeobs/one_setting.py`](../src/pdeobs/one_setting.py) keeps an explicit `"ufno_2d": "ufno"`
mapping when it resolves a campaign row. A reviewer grepping the method registry for `ufno_2d` will
not find it; that is expected.

---

## The seven learned methods in the paper

Each section gives, in this fixed order: the family, the upstream anchor and whether the entry is a
PDE-OBS adaptation or a pinned external wrapper; what the adaptation takes from the upstream work;
what it explicitly does **not** claim, in the source's own terms; the frozen `paper` structure;
and the constraints enforced before construction.

One property of the public parameter layer applies to all seven and is stated once here. In
`resolve_model` ([`src/pdeobs/api/models.py`](../src/pdeobs/api/models.py)) the **unknown-parameter
check runs first and unconditionally**: any key not in the model's own parameter list raises

> `model '<name>' does not accept ['<key>']; valid parameters: [...]`

An alias-confusion guard (`hidden`/`width`, `depth`/`layers`, `physics_slices`/`slices`,
`blocks`/`layers`) appears a few lines later, but it is only reachable for a key that is *not*
unknown, which the preceding check has already excluded. In practice every misspelled or
wrong-family parameter is rejected by the generic message, which lists the valid keys but does not
name the intended replacement. Rejection is reliable; the wording is generic.

### 1. U-FNO-2D

*Family: operator (spectral). Upstream anchor: the published U-FNO architecture. Status: PDE-OBS
adaptation.*

**Takes from upstream.** The six-block composition — three Fourier blocks followed by three
U-Fourier blocks — realised as `nn.ModuleList(SpectralConv2d(...) for _ in range(6))` plus three
`_UFNOBranch2d` U-branches on layers 4–6.

**Does not claim.** The class docstring states it: *"The official U-FNO release is a 3-D space–time
model. The benchmark uses the paper's six-block composition in two spatial dimensions and adds only
the observation-mask, coordinates, and geometry channels required by PDE-OBS. It must therefore be
reported as `U-FNO-2D (PDE-OBS adaptation)`, not as an exact upstream reproduction."* The campaign
registry repeats this as `adaptation: official_release_is_3d_space_time_this_is_explicit_2d_mask_conditioned_variant`,
and records `local_code_policy: independent_implementation_no_upstream_code_copied_or_modified`.
This is the one method in the table whose upstream release is a 3-D space–time model and which must
therefore be reported as a 2-D adaptation.

**Frozen `paper` structure.** `width=36`, `modes=12`, `padding=8`, `dropout=0.0`.

**Enforced before construction.** The block count is fixed at six spectral layers with U-branches on
layers 4–6, so there is *no* `layers` parameter; the spec minimums require `width >= 1`,
`modes >= 1` and `padding >= 0`; the U-branches use `BatchNorm2d`, so a training batch size of at
least 2 is required, and the easy API emits that warning for every non-rollout task.

### 2. FNO

*Family: operator (spectral). Upstream anchor: the Fourier Neural Operator. Status: PDE-OBS
adaptation.*

**Takes from upstream.** Truncated real-FFT spectral convolution with a parallel local (1×1
convolution) path, stacked `layers` deep.

**Does not claim.** The class docstring: *"Compact mask-channel FNO2d architectural reference. This
is not an exact reproduction of the original FNO paper model."* The module docstring covers every
compact baseline in the file: the networks *"are **not exact reproductions** of corresponding
research papers or official repositories"*, and the shared `_REFERENCE_NOTE` attached to the
capability objects reads *"Compact architectural reference; not an exact paper reproduction."*

**Frozen `paper` structure.** `width=32`, `modes=12`, `layers=4`.

**Enforced before construction.** `width % min(8, width) == 0` (GroupNorm) is checked in
`_check_constraints` in [`src/pdeobs/api/models.py`](../src/pdeobs/api/models.py), which is the only
place that emits the FNO-specific message listing legal widths. `CompactFNO2d.__init__` itself
carries no such guard — its one explicit check is `geometry_channels < 0` — so direct construction
with an illegal width fails with PyTorch's own `num_channels must be divisible by num_groups`
instead. `modes` larger than the grid supports are clipped by the spectral layer at run time;
`check_grid_compatibility` reports them as a pre-flight problem when a resolution is known.

### 3. CNO

*Family: operator (convolutional). Upstream anchor: the Convolutional Neural Operator. Status:
PDE-OBS adaptation.*

**Takes from upstream.** Anti-aliasing applied before learned convolution.

**Does not claim.** The in-source comment in `_AntiAliasedBlock.forward` is explicit: *"A fixed
binomial low-pass before learned convolution reduces aliasing without claiming the complete
continuous-discrete CNO construction."* The class docstring reads *"Compact anti-aliased CNO-like
reference, not an exact CNO reproduction."* The registry records
`adaptation: compact_mask_conditioned_scalar_2d_variant`.

**Frozen `paper` structure.** `width=32`.

**Enforced before construction.** Depth is fixed (encoder / mid / decoder); the public parameter list
is exactly `in_channels`, `out_channels`, `geometry_channels`, `width`, so `depth` and `blocks` are
rejected by the generic unknown-parameter message described above — which names the valid keys but
not a "correct key", and the `depth`→`layers` alias guard can never fire here because `layers` is not
a CNO parameter either.

**Width bookkeeping.** `CompactCNO2d` builds `enc: (in + 1 + geometry + 2) → width`,
`mid: width → 2·width`, `dec: (2·width + width) → width`, `head: width → out`. It is the
bottleneck that doubles the width; the decoder block consumes `3·width` channels and emits `width`.
The generated parameter reference inherits a `width` doc string that says the decoder uses twice the
width — see *Known gaps*.

### 4. DeepONet

*Family: operator-network. Upstream anchor: the branch–trunk operator network. Status: PDE-OBS
adaptation.*

**Takes from upstream.** The canonical branch–trunk inner product.

**Does not claim.** The module docstring for
[`src/pdeobs/methods/operator_networks.py`](../src/pdeobs/methods/operator_networks.py) states that
these implementations *"are independent benchmark adapters, not copies of the upstream repositories
and not claims of exact paper reproduction."* The class docstring explains the one substantive
change: *"The upstream DeepONet branch receives an ordered, fixed sensor vector. A global mean/max
set reduction is too lossy for structured 2-D PDE fields, so the source-faithful adapter can
retain sensor ordering in a small convolutional branch before the canonical branch–trunk inner
product. The legacy set branch remains available for old checkpoints."* The registry records
`local_code_policy: independent_implementation_no_upstream_code_copied`.

Note the default/preset split: the **constructor** default is `branch_mode="sensor_set"` (the
permutation-invariant set branch), while the frozen `paper` preset selects
`branch_mode="spatial_cnn"`. A caller reaching past the preset gets the legacy branch.

**Frozen `paper` structure.** `hidden=128`, `latent=128`, `branch_layers=2`, `trunk_layers=2`,
`branch_mode=spatial_cnn`, `branch_grid=4`, `branch_width=64`.

**Enforced before construction.** `branch_mode` must be `sensor_set` or `spatial_cnn`;
`min(hidden, latent, branch_layers, trunk_layers, branch_grid) >= 1`; `branch_width` applies only to
`spatial_cnn` (supplying it with `sensor_set` is rejected by the API) and otherwise defaults to
`min(hidden, 64)`.

### 5. GNOT

*Family: transformer (neural operator transformer). Upstream anchor: the General Neural Operator
Transformer. Status: PDE-OBS adaptation.*

**Takes from upstream.** Normalized linear cross-attention with geometry-gated experts, named as the
anchor in the module docstring.

**Does not claim.** No upstream code is copied — and in this case that is not only a policy choice.
The campaign registry records `reviewed_license: no_license_file_found_do_not_copy` together with
`local_code_policy: independent_implementation_no_upstream_code_copied`: no licence file was found
in the inspected revision, so the upstream source is never copied and the adapter is implemented
independently from the published description. The registry also records
`adaptation: official_unit_normalization_not_applied_to_preserve_physical_relative_l2_objective` —
the upstream unit normalization is deliberately dropped so that the benchmark's physical
`mean_per_sample_relative_l2` objective is what is actually optimized.

**Frozen `paper` structure.** `hidden=128`, `layers=3`, `heads=1`, `experts=2` (with
`inner_ratio=4`, `mlp_layers=3` at their constructor defaults).

**Enforced before construction.** `hidden % heads == 0`, raised in the constructor and again in the
API pre-flight; all structural dimensions must be at least 1.

### 6. Transolver

*Family: transformer. Upstream anchor: physics-slice attention. Status: PDE-OBS adaptation.*

**Takes from upstream.** Physics-slice attention, and the officially published Darcy and
Navier-Stokes dimensions.

**Does not claim.** The registry records
`adaptation: structured_grid_mask_conditioned_variant_with_official_darcy_and_ns_dimensions`; the
module docstring's blanket statement applies — an independent benchmark adapter, not a copy of the
upstream tree and not a claim of exact reproduction.

**Frozen `paper` structure.** Transolver is the only learned method with a separate rollout
structure. Static: `hidden=128`, `layers=8`, `heads=8`, `slices=64`, `mlp_ratio=1`. Rollout:
`hidden=256`, `layers=8`, `heads=8`, `slices=32`, `mlp_ratio=1`. The campaign registry stores the
same split as `architecture` versus `temporal_architecture`, using the key name `physics_slices`;
the resolver maps it onto the constructor's `slices`.

**Enforced before construction.** `hidden % heads == 0`; all structural dimensions at least 1. The
registry key `physics_slices` is **not** a public parameter, so passing it to the API is rejected —
by the generic unknown-parameter message, not by a dedicated one. The registry key and the public
key therefore cannot be confused silently, but the refusal does not tell the caller to use `slices`.

### 7. PINO

*Family: physics-informed (FNO backbone plus a trainer-side residual). Upstream anchor: the
residual-augmented neural-operator objective. Status: PDE-OBS adaptation.*

**Takes from upstream.** The residual-augmented objective and the official 5:1 data-to-residual
weighting.

**Does not claim.** `PhysicsInformedFNO2d` subclasses `CompactFNO2d`, and its docstring says why:
*"PINO is distinguished from FNO by the trainer-side PDE residual, not by silently changing the
spectral architecture."* Its capability note adds *"PDE-OBS PINO adaptation with an FNO backbone and
separately configured static-recovery or periodic free-rollout residual; not an exact paper
reproduction."* The registry records
`adaptation: masked_sparse_input_and_frozen_dataset_specific_residual_metadata_with_official_five_to_one_data_residual_weighting`
— inputs are masked and sparse, and the residual reads frozen dataset-specific metadata.

**Frozen `paper` structure.** `width=64`, `modes=20`, `layers=5` — wider and deeper than the
compact FNO preset, but the same architecture class.

**Enforced before construction.** `width % min(8, width) == 0` in `_check_constraints`;
`physics_contract` must be one of `pino_static_fd_v1` / `pino_rollout_spectral_v1` and must match the
task (static → static contract, rollout → temporal contract); the easy API refuses `task="inverse"`
outright; the trainer refuses to start unless `physics_loss == physics_contract`,
`physics_loss_weight > 0` and `amp == false`. See the dedicated section below.

---

## Adaptation and provenance table

| Method | Upstream anchor | What is **not** claimed | Inspected upstream revision | Recorded upstream licence | Local code policy |
|---|---|---|---|---|---|
| U-FNO-2D | Six-block composition (3 Fourier + 3 U-Fourier) | Upstream release is a **3-D space–time model**; this is an explicit 2-D mask-conditioned variant and must be reported as "U-FNO-2D (PDE-OBS adaptation)" | `8315fd7b5bd75282b7efe42ee6b8de86543d13cc` | CC-BY-NC-ND-4.0 | *(registry)* Independent implementation; no upstream code copied or modified |
| FNO | Truncated real-FFT spectral convolution with a parallel local path | Not an exact reproduction of the original FNO paper model | `00b7d86f8d74ff0af55da53eb585fe26df9c71f0` | MIT | *(docstring)* Compact architectural reference |
| CNO | Anti-aliasing before learned convolution | A fixed binomial low-pass only; the complete continuous–discrete CNO construction is not claimed. Compact mask-conditioned scalar 2-D variant | `6e765198aa02b56352e0a3437104b9d9e337176e` | MIT | *(docstring)* Compact architectural reference |
| DeepONet | Canonical branch–trunk inner product | Upstream feeds an ordered fixed sensor vector; this adapter substitutes an ordered mask-aware convolutional branch (the set branch is retained for legacy checkpoints) | `8d62345afd39e1df9c2c8c8d0e7c41882b06a9bf` | CC-BY-NC-SA-4.0 | *(registry)* Independent implementation; no upstream code copied |
| GNOT | Normalized linear cross-attention with geometry-gated experts | Official unit normalization deliberately not applied, to preserve the benchmark's physical relative-L2 objective | `5ee2e6925a43f9a340a6016bad4da2c82a452cbe` | **No licence file found in the inspected revision** | *(registry)* `no_license_file_found_do_not_copy`: upstream source is **never** copied; independent implementation only |
| Transolver | Physics-slice attention | Structured-grid mask-conditioned variant using the official Darcy and Navier-Stokes dimensions; not a copy of the upstream tree | `75e0f67643806a81cd1d3f6adc88dd8c02416fe7` | MIT | *(docstring)* Independent benchmark adapter |
| PINO | Residual-augmented objective and the official 5:1 data-to-residual weighting | The spectral architecture is unchanged from the compact FNO; the difference is the trainer-side residual. Not an exact paper reproduction | `3b6bc307c63c64057d0496753bf1af5f44ac5108` | Apache-2.0 | *(docstring)* Independent implementation of the residual; FNO backbone shared with `fno` |

**Where each cell comes from.** `reviewed_revision` and `reviewed_license` are read from
[`configs/method/all_pde_source_faithful_registry.yaml`](../configs/method/all_pde_source_faithful_registry.yaml)
for all seven. The other two columns are mixed, and the table marks which is which:

- `local_code_policy` exists in the registry for **three** methods only — `ufno_2d`, `deeponet` and
  `gnot`. Cells marked *(registry)* quote it. Cells marked *(docstring)* are this card's
  one-line paraphrase of the class or module docstring; the registry says nothing about local code
  policy for `fno`, `cno`, `transolver` or `pino`.
- `adaptation` exists in the registry for six of the seven. There is **no** `adaptation` key for
  `fno`; the "what is not claimed" cell for FNO is the class docstring.

Four of the seven classes additionally carry the inspected revision as an `upstream_revision` class
attribute (`deeponet`, `transolver`, `gnot` in `operator_networks.py`; `ufno` in `neural.py`); FNO,
CNO and PINO carry theirs only in the registry.

The recorded licence describes the upstream repository as inspected at that revision. It is not a
licence grant for this repository, whose own terms are in [`LICENSE`](../LICENSE) and
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). A recorded revision means a reviewer read
that tree; it is not an assertion of bitwise behavioural equivalence.

---

## PINO is a trainer-side difference, not an architecture

`PhysicsInformedFNO2d` is a subclass of `CompactFNO2d` with no architectural change — the same
spectral backbone, only a wider frozen preset. Everything that distinguishes PINO lives in the
objective, in [`src/pdeobs/pino.py`](../src/pdeobs/pino.py) and
[`src/pdeobs/training.py`](../src/pdeobs/training.py). The contract is deliberately narrow, and the
module docstring says so: the two residual contracts are *"intentionally distinct so a final-field
prediction can never be mistaken for a trajectory-level PINO objective."*

**Two residual contracts, and only two.**

- `pino_static_fd_v1` — a finite-difference residual restricted to
  `PINO_STATIC_FAMILIES = ("darcy", "poisson", "helmholtz")`, the three elliptic families whose
  residual is identifiable from one predicted field plus the stored PDE condition. Asked for any of
  the four temporal families it raises, in its own words: *"{family} PINO requires a predicted
  trajectory and reviewed temporal residual; a recovered final field is scientifically
  insufficient."*
- `pino_rollout_spectral_v1` — a **periodic-only** spectral residual for the free autoregressive 2-D
  rollout. A non-periodic canonical boundary raises `pino_rollout_spectral_v1 is periodic-only`, and
  fewer than two predicted future states raises `temporal PINO requires at least two predicted
  future states`.

**Residuals are evaluated strictly between predicted states.** The temporal residual docstring:
*"Residuals are evaluated only between predicted states (t1→t2, t2→t3, ...), so hidden ground-truth
states are never fed back to the model or to the PDE residual."* No hidden target frame enters the
objective through the residual term.

**The inverse task is refused before any work starts — by the generic task gate.** The `pino`
`ModelSpec` declares `tasks=("recovery", "forward", "rollout")`, so `resolve_model` refuses
`task="inverse"` at the same check every model passes through:
`model 'pino' does not support task 'inverse'; supported: ['recovery', 'forward', 'rollout']`. A
PINO-specific raise, `PINO has no physics residual for the inverse task`, appears further down in
`resolve_model` but is unreachable behind that gate. The model spec records the rule as a declared
constraint in different words: *"inverse task is not supported by any PINO residual."* The refusal is
real; the message a caller actually sees is the generic one.

**Mixed precision is rejected while a residual is active.** `TrainingConfig` raises
`PINO residual training requires amp=false`, and additionally requires that `pino_static_fd_v1` be
paired with an elliptic recovery/forward task and `pino_rollout_spectral_v1` with `task=rollout`.

**Model and trainer must agree.** The model carries `required_physics_loss`; the trainer raises
`model/trainer physics contract mismatch` if the configured `physics_loss` differs. A PINO model
trained without its residual is a configuration error, not a silent fallback to FNO.

**Weighting.** `data_loss_weight: 5.0`, `physics_loss_weight: 1.0` — the 5:1 data-to-residual
weighting described in the registry as matching the official weighting. The row resolver reads both
weights straight out of the registry into the generated training configuration, along with
`physics_loss_epsilon: 1.0e-12`.

`configs/paper/protocol.yaml` records the scope of the physical context PINO is given:
`pino_physical_context: declared_training_loss_only`. It is a training-information difference, and
[`docs/reproducing_the_paper.md`](reproducing_the_paper.md) requires that it *"remain visible in
method descriptions and result grouping."*

---

## Shared training contract and per-method deltas

All seven learned methods share one frozen contract, recorded as `shared_contract` in the campaign
registry:

| Shared contract item | Value |
|---|---|
| Objective | `mean_per_sample_relative_l2` |
| Epochs | 500 |
| Validation | none (0 validation records) |
| Checkpoint | final epoch only |
| Temporal adapter | free autoregressive `t0 → t1 → t2 → t3` |
| Inputs | masked field or condition, observation mask, coordinates, geometry where required |
| Scheduler policy | method-specific, source-faithful, rescaled to the 500-epoch budget |

The per-method deltas below are the only differences. Until now they were readable only by opening
[`configs/method/all_pde_source_faithful_registry.yaml`](../configs/method/all_pde_source_faithful_registry.yaml);
this table is a transcription of that file plus the defaults applied by `build_experiment_config` in
[`src/pdeobs/one_setting.py`](../src/pdeobs/one_setting.py).

| Method | Optimizer | Learning rate | Weight decay | Batch size | Scheduler | Gradient clipping |
|---|---|---|---|---|---|---|
| U-FNO-2D | Adam | 1e-3 | 1e-4 | 4 | resolver default (`step`, 100 epochs, γ=0.5) | resolver default 1.0 |
| FNO | Adam | 1e-3 | 1e-4 | **8** | resolver default | resolver default 1.0 |
| CNO | AdamW | 1e-3 | 1e-6 | **8** | resolver default | resolver default 1.0 |
| DeepONet | Adam | 1e-3 | 0.0 | 4 | resolver default | resolver default 1.0 |
| GNOT | AdamW | 1e-3 | 5e-5 | 4 | `one_cycle`, `pct_start=0.2`, `div_factor=1e4`, `final_div_factor=1e4`, cosine | **1000.0** |
| Transolver | AdamW | 1e-3 | 1e-5 | 4 static, **2 for temporal rows** | `one_cycle`, `pct_start=0.3`, `div_factor=25.0`, `final_div_factor=1e4`, cosine | **0.1** static; `null` for temporal rows, with `gradient_warning_fatal: true` |
| PINO | Adam | 1e-3 | 0.0 | 4 | `step`, `step_size_epochs=100`, `gamma=0.5` | 1.0 |

Notes a reviewer should not have to infer:

- **Gradient clipping spans four orders of magnitude between two methods in the same table.** GNOT
  clips at `1000.0` — effectively a warning threshold rather than a constraint — while Transolver
  clips at `0.1`. Both values are deliberate entries in the registry `health` blocks, not defaults.
- **Only GNOT, Transolver and PINO carry explicit `scheduler` and `health` blocks.** For the other
  four, the row resolver supplies its documented defaults: scheduler `step` with
  `step_size_epochs=100` and `gamma=0.5`, `grad_clip=1.0`, and `gradient_warning_fatal=True`
  (the three methods with an explicit health block set that flag to `false`, except Transolver's
  temporal override, which sets it back to `true`).
- **Transolver is the only method with temporal overrides**, and it has three: the wider/half-sliced
  `temporal_architecture`, `temporal_optimizer: {batch_size: 2}`, and
  `temporal_health: {grad_clip: null, gradient_warning_fatal: true}`.
- **Every row sets the same fixed trainer flags** regardless of method: `loss: relative_l2`,
  `teacher_forcing_ratio: 0.0`, `mixed_precision: false`, `deterministic: true`,
  `health_policy: fail`, `early_stopping_patience: null`, `checkpoint_every: 1`. The one-cycle
  `scheduler_steps_per_epoch` is computed as `ceil(records_per_epoch / batch_size)` with
  `records_per_epoch` = 1800 for a 500-epoch production row and 90 for the 3-epoch preflight.
- **Rollout is served by a wrapper, not by a different model.** For the four temporal families the
  resolver wraps the one-step model as `{"name": "autoregressive", "base": <model>}`;
  `AutoregressiveModel.forward` sets `mask = None` at the end of each loop iteration, with the
  comment *"predicted states are fully specified after step one"*. The mask is therefore applied
  only at `t0`, matching `future_ground_truth_as_input: forbidden` in the protocol.
- The learning rate is `1e-3` for all seven. That is a fact about the registry, not a tuned result:
  the release documents no hyperparameter-selection procedure, and the split carries 0 validation
  records.

---

## Classical comparators and their rollout adapters

The frozen campaign has exactly two classical slots —
`CLASSICAL_METHOD_ORDER = ("gaussian_rbf", "gappy_pod")` — and `validate_campaign` raises
`classical method slots changed` on any deviation.

| Slot | Implementation | Frozen setting | Fit scope | Declared rollout adapter |
|---|---|---|---|---|
| Gaussian RBF interpolation | [`interpolation.py::RBFInterpolation`](../src/pdeobs/methods/interpolation.py) | `epsilon=2.0`, `smoothing=1e-6`, `max_centers=512`, `normalized_coordinates=true` | none (no fitting) | `reconstruct_t0_then_persistence_rollout`, implemented by `RBFSpatialPersistence` |
| Gappy POD | [`reduced_order.py::GappyPOD`](../src/pdeobs/methods/reduced_order.py) | `rank=64`, `ridge=1e-6` | once per PDE, on complete training fields only | `gappy_coefficients_at_t0_plus_training_only_linear_DMD_transition`, implemented by `GappyPODDMD` |

Neither adapter reads a target frame at inference. That is the point of both of them.

**The RBF persistence adapter is deliberately weak, and leakage-free by construction.** Its
docstring: *"RBF interpolation is a spatial reconstruction method, not a dynamical model. This
deliberately weak temporal adapter makes that limitation explicit while still providing a
leakage-free comparator for rollout tasks. Every future state is the same reconstruction of the
sparse initial condition; no target frame is ever read by `predict`."* The implementation is a single
`np.repeat` of the `t0` reconstruction across the horizon, so the absence of a target read is visible
in the code rather than established by proof. It should be read as a floor, not as a forecasting
baseline.

**Gappy POD fits on complete training fields only.** The basis is built by a three-pass randomized
range finder — one pass for the mean, one for the sample-space sketch, one for the small projected
matrix — with `O(Nr + rD)` memory rather than `O(ND)`. Fitting with fewer than two fields raises
`GappyPOD needs at least two training fields`.

**Gappy POD's rank rule is pre-registered.** `fit_dataset` refuses to guess: if `rank` is left unset
and no validation dataset is supplied it raises
`validation data is required for preregistered POD rank selection`. The declared candidates are
`(8, 16, 32, 64, 128)` and the selection sweeps `validation_protocols`, which defaults to all nine
entries of `MASK_PROTOCOL_NAMES`, at `validation_samples_per_stratum=3`. The frozen campaign setting
pins `rank=64` directly, so that selection path is not exercised by a paper row.

**The Gappy POD + DMD rollout adapter fits both basis and transition on complete training states
only.** Its docstring: *"The POD basis and transition are fitted exclusively from complete states in
the declared training split. At inference, coefficients are inferred only from sparse `t0`
observations. The transition then advances those coefficients freely, so no future target frame can
enter the recurrence."*

Both adapters are registered `hidden=True` (see the next section) because they are the temporal
service path of an existing classical slot, not additional campaign slots. The classical baselines
are indexed as PDE × method × test view with no trained-view axis: 7 × 2 × 9 = 126 possible blocks,
which the campaign config justifies explicitly — repeating them nine times would be pseudoreplication
— and which must be reported separately from the learned cross-view matrix.
[`docs/reproducing_the_paper.md`](reproducing_the_paper.md) states that this candidate makes no
claim that those classical blocks have been run.

Five further non-learning baselines (`zero`, `mean`, `nearest`, `bilinear`, `rbf`) plus
`persistence` are registered with `requires_torch=False` and degrade gracefully without SciPy, but
only the `gaussian_rbf` and `gappy_pod` slots occupy classical campaign positions.

---

## Registered but not reported

These methods exist in the registry as scaffolding, reference baselines or service adapters. None of
them is among the seven, and **the release records no run, score or result for any of them**. That is
a claim about results, not about configuration files: the tree does ship experiment configs that name
several of them — [`configs/experiment/`](../configs/experiment/) contains `core_pilot_unet.yaml`,
`core_medium_unet.yaml`, `recovery_unet.yaml`, `recovery_unet_smoke.yaml`, `inverse_darcy_unet.yaml`,
`full_to_partial_unet.yaml`, `paper_cno2023_poisson_unet.yaml`,
`paper_cno2023_navier_stokes_unet.yaml` and `pretrain_mae_small.yaml`, among others. A config is a
declared setup; none of them corresponds to a recorded result in this release.

| Registered method | Registry name | What it is for |
|---|---|---|
| Mask-channel U-Net | `unet` | Small U-Net taking values and mask as channels. Its spec carries the note *"not part of the paper's seven learned models"*, and a generic U-Net is a **forbidden slot** in the frozen campaign. |
| Deep paper-guided mask U-Net | `unet_paper_guided` | Four-level symmetric U-Net adapted from segmentation to masked fields; a deeper reference point, parameter-documented but not a paper row. |
| ConvLSTM | `convlstm` | Small recurrent rollout reference accepting BTCHW histories. Native rollout, no autoregressive wrapper, and no geometry input. |
| Masked autoencoder (small) | `mae_small` | Compact convolutional MAE-style sparse-reconstruction anchor. Explicitly *not* a ViT-MAE and not an exact reproduction; no geometry input. |
| Compact residual encoder | `residual_cnn` | Mask-aware residual CNN producing retrieval embeddings and semantic labels. Serves the retrieval / classification / supervised-multitask interfaces and is not exposed through the easy-API model specs at all. |
| RBF + persistence | `rbf_persistence` (hidden) | The declared rollout adapter for the `gaussian_rbf` slot — a service path, not a slot. |
| Gappy POD + DMD | `gappy_pod_dmd` (hidden) | The declared rollout adapter for the `gappy_pod` slot — a service path, not a slot. |
| Autoregressive wrappers | `autoregressive`, `autoregressive_fno` | The mechanism by which a one-step model serves the four temporal PDEs. A wrapper is not an algorithm; the mask is dropped after the first step. |

Hidden registration means the factory is in the registry but excluded from public enumeration:
`rbf_persistence`, `gappy_pod_dmd` and the three upstream wrappers below all pass `hidden=True`, so
`available_methods()` returns 21 of the 26 built-in factories.

---

## The three commit-pinned upstream wrappers

`paper_unet`, `paper_fno` and `paper_cno` in
[`src/pdeobs/methods/paper_official.py`](../src/pdeobs/methods/paper_official.py) are the contrast
case that the generated parameter reference omits entirely: `gen_docs.py` skips every spec with
`upstream_wrapper=True`, so a reviewer reading `model_parameter_reference.md` cannot discover that
exact-author-architecture baselines exist in this tree. They do, with a different claim from
everything above.

**They are the only entries that claim the exact author architecture.** The module docstring: *"The
author source is not vendored into PDE-OBS. Every construction verifies a clean official checkout,
its exact Git revision, and the selected source file digest before importing it. The adapters only
translate PDE-OBS's BCHW sparse recovery interface to the authors' tensor layout; they do not change
the authors' architecture."*

**All three attest against one shared upstream checkout.** `attest_official_checkout` accepts exactly
one revision, `OFFICIAL_CNO_REVISION = bd9362c8c192d6f160129a7a85b1fe00d6c41523`, for all three
wrappers; `OFFICIAL_SOURCE_SHA256` then pins three files inside that single tree
(`_OtherModels/BaselinesModules.py`, `_OtherModels/FNOModules.py`, `CNOModule.py`). This is one
reviewed upstream repository, not three.

**They fail closed on a fixed sequence of checks**, all before any author code is imported. Two are
applied by the constructor first:

- **Per-PDE setting** — `_verify_setting` compares the requested structure against a hard-coded
  reviewed table and raises `Official {paper_pde} setting mismatch` otherwise. The tables are
  `paper_unet`: channels 32 (Poisson) / 64 (Navier-Stokes); `paper_fno`: width 16 / 128 with
  modes 16, layers 5, padding 0; `paper_cno`: layers 3 with channels 16 / 32 and neck/level
  residuals 6/4 and 8/1.
- **Initialization seed** — anything other than 4 raises
  *"Reviewed author-code initialization seed is 4."*

`attest_official_checkout` then runs, in this order:

1. **Checkout exists** — the resolved `upstream_root` must be a directory.
2. **Revision** — the requested revision must equal `OFFICIAL_CNO_REVISION`; only the reviewed
   paper-era revision is accepted at all.
3. **Reviewed file** — `source_file` must be one of the three entries in `OFFICIAL_SOURCE_SHA256`.
4. **Configured digest** — the caller's `source_file_sha256` must equal the reviewed digest.
5. **`HEAD`** — `git rev-parse --verify HEAD` must equal the requested revision.
6. **Checkout cleanliness** — `git status --porcelain --untracked-files=all` must be empty, else
   *"Official checkout is not clean; refusing modified or untracked source."*
7. **Path containment** — the resolved source path must lie inside the checkout and be a file.
8. **Committed blob digest** — the digest is taken from `git show <revision>:<file>`, i.e. from the
   committed bytes, *independent of checkout line-ending filters*, and must match the reviewed entry.

**They are never enumerated publicly and never substituted.** All three register with `hidden=True`;
the easy API refuses them with an explanatory message rather than quietly serving a compact
look-alike: *"'paper_fno' is an exact upstream wrapper requiring a pinned external checkout and
explicit attestation arguments; it is not served through the easy API."* Their spec note states the
same rule: *"fail-closed: needs upstream_root/upstream_revision/source_file_sha256; never
substituted by a compact model."* Direct registry construction is not a shortcut either —
`OfficialPaperFNO2d.__init__` takes eight required keyword-only attestation and structure arguments
(`upstream_root`, `upstream_revision`, `source_file_sha256`, `paper_pde`, `width`, `modes`,
`layers`, `padding`) before `initialization_seed` defaults in. Recorded gating evidence is in
`acceptance/round2/T14.json`, where all three report `dependency_blocked` and the facade refusal is
recorded as `rejected_ok`.

**They cannot be compared against the seven at the benchmark resolution.** All three route their
inputs through `_masked_observations`, which hard-raises *"Reviewed CNO-paper settings require 64x64
inputs"* for anything other than a 64×64 one-channel BCHW tensor. The paper protocol runs at
128×128 (`resolution: [128, 128]` in `configs/paper/protocol.yaml`, `PAPER_RESOLUTION` in
`specs.py`). These wrappers therefore serve the `recovery` task at the reviewed upstream 64×64
setting only; they are dependency-gated, they are not part of the seven, and no head-to-head
comparison with the seven is possible through them without a resolution the adapters refuse.

---

## Slots and prohibitions

- **A generic U-Net is a forbidden campaign slot.**
  [`configs/campaign/all_pde_one_setting_10method_9x9.yaml`](../configs/campaign/all_pde_one_setting_10method_9x9.yaml)
  records `forbidden_slot: generic_unet` under `methods:`, and the `unet` model spec is annotated
  *"not part of the paper's seven learned models"*. The repository records the prohibition; it does
  not record a rationale or a supporting ablation for it.
- **The retained campaign carries an eighth learned slot with no implementation in this
  repository.** `LEARNED_METHOD_ORDER` has eight entries; `PUBLIC_LEARNED_METHODS` is the same tuple
  with `jeno` removed, and `build_experiment_config` raises
  `unknown PDE, public method, or view` for it. The registry records
  `implementation: private_adapter_not_copied_into_public_repository` and
  `public_upstream: none_attested`. That slot is filtered out of every runnable path.
- **This is why two accountings coexist.** The retained campaign file — which is marked
  `status: candidate_preflight_only`, `release_eligible: false`, `formal_credit: 0` — hard-requires
  504 learned training rows (7 PDE × **8** learned methods × 9 views), 4536 learned blocks, 126
  classical blocks and 4662 total, and `validate_campaign` raises `campaign accounting changed` if those
  numbers move. The paper protocol, which counts only the seven public methods, records
  `training_rows: 441` (7 × 7 × 9) and `possible_learned_blocks: 3969` (441 × 9). **441 is the
  runnable public grid; 504 is the retained internal accounting.** Both are denominators, not
  completion counts.

---

## Known gaps

Stated plainly, because a reviewer will look for these and they are not in the tree:

- **No cost or size figures for any of the seven at its `paper` preset.** No parameter count,
  FLOP count, memory footprint or wall-clock figure is recorded for the frozen preset of any of the
  seven. Parameter numbers do exist elsewhere in the tree and should not surprise a grep: a source
  comment in `paper_official.py` gives 7.8M/31.0M for the upstream U-Net at widths 32/64 — a
  commit-pinned wrapper, not one of the seven — and the acceptance JSONs record `"params": 75009`
  for the small smoke-sized configurations they exercised (`acceptance/round2/T3.json`,
  `acceptance/round2/T6.json`). None of those is a `paper` figure.
- **No initialization or seeding policy stated for the seven.** Some models do initialize
  explicitly: `paper_official.py` pins `initialization_seed=4` for the wrappers; the DeepONet adapter
  calls `self.apply(self._initialize)`, applying `nn.init.xavier_normal_` to every `Linear`/`Conv2d`
  weight and `nn.init.zeros_` to biases; and Transolver applies `trunc_normal_(std=0.02)` with zero
  biases plus `nn.init.orthogonal_` on the slice projection. GNOT and the compact `neural.py` models
  rely on framework defaults. What is missing is a document that states any of this as policy. The
  campaign seed (20260804) governs the identity split, mask pairing and the default training seed; it
  is not a per-module initialization policy.
- **No benchmark result in this release.** No score, leaderboard entry or comparison number for any
  method enters any reported table, and `configs/paper/protocol.yaml` sets
  `missing_result_marker: RESULT_PENDING`. `RELEASE_NOTES_v0.2.0.md` states plainly that there was no
  production 500-epoch run, no 441-row sweep, no multi-seed benchmark and no convergence study. The
  `acceptance/` directory *does* contain per-method `rel_l2` values with `"score_status": "valid"`
  (for example in `acceptance/round1/T4.json`, `acceptance/round2/T3.json` and
  `acceptance/round2/T9.json`); these are software-acceptance numbers from a few optimizer steps per
  flow and, by their own declaration, enter no table. A grep of `acceptance/` will find numbers; none
  of them is a benchmark result.
- **No per-method task record.** The generated reference lists the tasks each **model spec** declares
  (`gen_docs.py` prints `spec.tasks`), not the tasks the method was run on — and not the class's own
  `MethodCapabilities`, which can be narrower. The `fno` spec declares
  `("recovery", "forward", "inverse", "rollout")` while `CompactFNO2d.capabilities` is
  `{"recovery", "forward", "inverse"}`; rollout is served by the autoregressive wrapper, not by the
  class itself. The protocol marks forward and inverse as implemented extensions rather than primary
  experiments.
- **`geometry_channels` defaults differ by layer, and the API applies a default rather than a
  forced value.** `resolve_model` calls `kwargs.setdefault("geometry_channels", 1)` for neural
  families, so an explicit `geometry_channels=0` from the caller is honoured. The published reference
  documents `1`, four compact constructors in `neural.py` default it to `0`, and the operator-network
  constructors default to `1`. A caller reaching past the API gets a different default. Geometry is
  never silently discarded, though: a non-zero `geometry_channels` with no geometry tensor is a hard
  error.
- **The generated reference's `cno` width doc is inaccurate.** The `ParamSpec` doc string for the CNO
  `width` reads "encoder width; the decoder uses 2x width". The code doubles at the bottleneck
  (`mid: width → 2·width`) and the decoder block takes `2·width + width` channels and emits `width`.
  The wording should be corrected in `specs.py`, and the generated reference regenerated, before that
  sentence is relied on.
- **Rejection messages at the public parameter layer are generic.** The alias-confusion guard in
  `resolve_model` (`depth`→`layers`, `blocks`→`layers`, `physics_slices`→`slices`, `hidden`↔`width`)
  sits behind an unconditional unknown-parameter check and is therefore unreachable in practice.
  Wrong keys are always rejected, but the message lists the valid parameters instead of naming the
  intended one.
- **Gappy POD's full parameter surface is not in the generated reference.** The published table
  exposes `rank`, `ridge`, `oversampling` and `seed`; `rank_candidates`, `fit_batch_size`,
  `validation_samples_per_stratum`, `validation_protocols` and `state_path` are documented here and
  in the source only — including the rule that omitting `rank` makes a validation dataset mandatory.

---

Sources for everything above:
[`configs/method/all_pde_source_faithful_registry.yaml`](../configs/method/all_pde_source_faithful_registry.yaml),
[`configs/campaign/all_pde_one_setting_10method_9x9.yaml`](../configs/campaign/all_pde_one_setting_10method_9x9.yaml),
[`configs/paper/protocol.yaml`](../configs/paper/protocol.yaml),
[`src/pdeobs/one_setting.py`](../src/pdeobs/one_setting.py),
[`src/pdeobs/pino.py`](../src/pdeobs/pino.py),
[`src/pdeobs/training.py`](../src/pdeobs/training.py),
[`src/pdeobs/api/specs.py`](../src/pdeobs/api/specs.py),
[`src/pdeobs/api/models.py`](../src/pdeobs/api/models.py),
[`src/pdeobs/methods/`](../src/pdeobs/methods/).
Parameter contract: [`model_parameter_reference.md`](model_parameter_reference.md).
Protocol and grid: [`reproducing_the_paper.md`](reproducing_the_paper.md).
Metric definitions and the strict scoring contract: [`scoring.md`](scoring.md).
