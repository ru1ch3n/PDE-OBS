# Extending PDE-OBS

Add your own PDE, method, or observation view.

The preferred extension changes the smallest relevant layer. A new observation
view does not require a new physical dataset, and a new method does not require
a new PDE solver. Five axes are extensible through public registries, and each
one is answered below in the same form: the exact callable or class to supply,
the exact method to implement, the invariant that must hold, and how to check
it. Two of the five — the condition field and the metric — carry narrower
contracts than the other three, and the sections say so explicitly rather than
implying a uniformity that the source does not have.

**The rule that governs every extension: registration is not certification.**
A plugin that registers successfully has made no claim about correctness,
permission compliance, or numerical accuracy. Import success, a valid output
shape and a green smoke test are not evidence of a result. The complementary
rule is that a broken optional plugin must never prevent the built-in baselines
from being used: entry-point discovery records a failing plugin instead of
propagating its exception ([`src/pdeobs/methods/base.py`](../src/pdeobs/methods/base.py),
[`src/pdeobs/registry.py`](../src/pdeobs/registry.py)).

| Axis | Registry | Entry-point group | You supply | You implement |
|---|---|---|---|---|
| Observation view | `pdeobs.registry.MASK_REGISTRY` | `pdeobs.masks` | a callable | `factory(shape, rng, **kwargs) -> bool[H,W]` |
| Condition field | `pdeobs.registry.SETTING_REGISTRY` | `pdeobs.settings` | a callable | `factory(shape, rng) -> float[H,W]` (no options are forwarded) |
| Method | `pdeobs.methods.register_method` | `pdeobs.methods` | a class or factory | `predict(observations, mask=None, **kwargs)` |
| PDE family | `pdeobs.pdes.register_generator` | `pdeobs.pdes` | a callable | `generator(boundary, setting, regime, seed, resolution, time_steps, **options) -> PDEOutput` |
| Metric | `pdeobs.registry.METRIC_REGISTRY` | `pdeobs.metrics` | a callable | no fixed signature; see [the metric section](#how-do-i-add-a-new-metric) |

---

## How do I add a new observation view?

**Register a mask factory into `MASK_REGISTRY` and implement one callable.**
The factory signature is `(shape, rng, **kwargs)` and it must return a
`bool[H, W]` array in which `True` means **observed**. Nothing else is accepted:
[`generate_mask`](../src/pdeobs/masks.py) resolves the name through the registry,
calls the factory with a local generator (`rng=` as a keyword, plus any extra
options), coerces the result to `bool`, and raises `RuntimeError` if the
returned shape differs from the requested grid.

```python
import numpy as np
from pdeobs.registry import MASK_REGISTRY
from pdeobs.masks import generate_mask, apply_mask

@MASK_REGISTRY.register("alternating_rows_demo")
def alternating_rows(shape, rng, **kwargs):
    del rng, kwargs  # deterministic protocol: see the exception below
    mask = np.zeros(shape, dtype=bool)
    mask[::2, :] = True
    return mask

mask = generate_mask("alternating_rows_demo", (16, 16), seed=0)
assert mask.shape == (16, 16) and mask.dtype == np.bool_ and mask.sum() == 128
observations = apply_mask(field, mask)   # unobserved entries filled with 0.0
```

**The invariant: all mask randomness must come from the supplied generator.**
`generate_mask` constructs `np.random.default_rng(int(seed))` and passes it in as
`rng`. A stochastic factory draws from that object and from nothing else; it must
never touch NumPy's global RNG, `random`, the clock, a process id, or a hash of
an unordered container. This is what makes a package reproducible from its
identity alone: for a registry protocol the dataset derives the mask seed
deterministically as `derive_seed(dataset_seed, "mask", sample_id, protocol)` and
writes `mask_id`, `mask_seed`, `observation_count` and `observation_ratio` back
into the record's metadata ([`src/pdeobs/dataset.py`](../src/pdeobs/dataset.py)).
The one exception is the `full` protocol, which observes everything, records
`mask_id = "full"` and `mask_seed = None`, and derives no seed at all. A resolved
observation also carries a content-addressed `observation_id` over its namespace,
name, mask configuration and schema version
([`src/pdeobs/api/observation.py`](../src/pdeobs/api/observation.py)). If your
factory reaches outside `rng`, that identity stops naming one array and every
downstream receipt becomes unverifiable.

**The deterministic-protocol exception.** A factory is allowed to ignore `rng`
entirely when the view has no random component. The built-in `boundary_sensors`
protocol does exactly that: it deletes the generator, sets an outer band of
integer width, and sub-samples with `np.linspace` when a `count` is requested, so
it produces a bit-identical mask at every seed. The rule is not "consume the
generator", it is "derive nothing from an uncontrolled source".

**The mask is never conflated with the geometry field.** `mask` is the
observation protocol; `geometry` is the solid/obstacle field written by the PDE
generator. They are different inputs and are never interchanged
([`src/pdeobs/api/observation.py`](../src/pdeobs/api/observation.py)). A sensor
count alone does not prove that the selected sites lie in a fluid region, and a
geometry ring is not an observation budget. Built-in geometry-aware filtering is
not uniform across mechanisms; if your view needs to respect an obstacle, do it
explicitly inside the factory and say so.

**How to check it.** Generate at two seeds and assert `bool[H, W]` at the
requested grid, a realized count you computed independently, and bit-identical
output at a repeated seed (a deterministic protocol is identical across seeds by
design).

**Using the view.** Pass the registered name to a dataset as
`mask={"protocol": "alternating_rows_demo"}`, or resolve it through the
`custom` namespace of the easy API and check what it actually observes:

```python
from pdeobs import api

spec = api.make_observation("alternating_rows_demo", namespace="custom")
spec.observation_id                      # 'custom:alternating_rows_demo:<12 hex>'
api.realized_count(spec, (128, 128))     # 8192
```

Registration is local to a Python process, so import the extension module before
constructing a dataset. A ratio is a target; the realized count is what the
factory returns. The nine frozen `paper` views accept no parameters and are
not extensible — vary them in the `general` namespace instead, and do not reuse a
frozen label for a redefined view. See [Observation protocols](observations.md)
and [the generated reference](observation_reference.md).

---

## How do I add a new condition field?

**Register a factory into `SETTING_REGISTRY` and implement one callable.** The
factory receives the spatial shape positionally and the generator as the keyword
`rng`, and returns a finite `float[H, W]` field:
[`make_setting_field`](../src/pdeobs/pdes/common.py) calls
`SETTING_REGISTRY.create(canonical, shape, rng=rng)`, then raises `RuntimeError`
on a wrong shape and `RuntimeError` on any non-finite value instead of accepting
it.

```python
import numpy as np
from pdeobs.registry import SETTING_REGISTRY

@SETTING_REGISTRY.register("ramp_demo")
def ramp(shape, rng, **_):
    del rng  # deterministic field
    height, width = shape
    row = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    return np.repeat(row[None, :], height, axis=0)
```

**The contract is narrower than the mask contract: no options are forwarded.**
`make_setting_field` passes the shape and `rng` and nothing else, so on the
built-in generation path a setting factory can never receive keyword options.
Every built-in is declared as `(shape, rng, **_)` and ignores them
([`src/pdeobs/settings.py`](../src/pdeobs/settings.py)); write yours the same way
for safety, but do not design a field whose behaviour depends on a keyword that
will never arrive. Parameterize by registering separate names instead.

**The invariant is the mask invariant, one layer down: all randomness comes from
`rng`.** Each built-in family draws its condition field from a fixed stream of
the sample seed — `make_rng(seed, 200)` for poisson, `make_rng(seed, 100)` for
darcy, and so on — so a field that reaches outside `rng` silently destroys the
reproducibility of every record generated with it.

**How to check it.** Call the factory twice with `np.random.default_rng(0)` and
assert a finite `float64` `[H, W]` array of the requested shape and bit-identical
values; then generate one sample of a family with `setting="<your name>"` twice at
the same seed and compare the stored arrays. A custom registered name resolves
through `normalize_setting`, so it is usable wherever a built-in setting name is.

Registering an eleventh-plus field does not establish that it is physically
sensible for all seven families; there are ten built-in settings today
([`SETTING_NAMES`](../src/pdeobs/pdes/common.py), aliases `s0`–`s9`).

---

## How do I add a new method?

**Register a class with `register_method` and implement `predict`.** The class
declares `name` and `capabilities` (a `MethodCapabilities` dataclass with
`tasks`, `trainable`, `temporal`, `requires_mask`, `supports_multichannel`,
`reference_only`, `notes`) and implements
`predict(self, observations, mask=None, **kwargs)`
([`src/pdeobs/methods/base.py`](../src/pdeobs/methods/base.py)).

```python
import numpy as np
from pdeobs.methods.base import MethodCapabilities, register_method
from pdeobs.methods import create_method

@register_method("visible_copy_demo")
class VisibleCopy:
    name = "visible_copy_demo"
    capabilities = MethodCapabilities(
        tasks=frozenset({"recovery"}), trainable=False, requires_mask=True
    )

    def predict(self, observations, mask=None, **kwargs):
        del kwargs
        if mask is None:
            raise ValueError("this method requires a mask")
        return np.where(np.asarray(mask, dtype=bool), observations, 0.0)

model = create_method("visible_copy_demo")
```

This is an interface example, not a new algorithm and not a competitive
baseline.

**Trap: two different `register_method` / `METHOD_REGISTRY` symbols exist, and
only one of them feeds `create_method`.** `pdeobs.methods.base.METHOD_REGISTRY`
is a plain dict, `pdeobs.methods.register_method` writes into it, and
`create_method` reads only that dict. `pdeobs.registry` separately defines a
`METHOD_REGISTRY` (a `Registry` bound to the same `pdeobs.methods` entry-point
group) and `register_method = METHOD_REGISTRY.register`; `discover_plugins()`
loads the entry-point group into *that* object. The two are not the same object.
A method registered only through `pdeobs.registry` will appear in
`pdeobs list --kind methods` and still make `create_method` raise `KeyError`.
Import `register_method` from `pdeobs.methods` (or `pdeobs.methods.base`).
Importing `pdeobs.methods` mirrors the built-ins into the `pdeobs.registry`
object as well (`install_builtin_methods()`), which is why the built-in names
appear on both surfaces.

**What a method receives: an observation package.** For every task the dataset
hands the predictor the filled values plus the mask as a `float32` indicator
array, plus geometry, plus metadata — and nothing else. Unobserved entries of
`observations` are filled with `0.0`, which is why the mask is supplied
separately: a physical zero must not be read as a missing value. The served mask
is `float32`, not `bool`
([`src/pdeobs/dataset.py`](../src/pdeobs/dataset.py)), so cast it explicitly —
as the example above does with `np.asarray(mask, dtype=bool)` — rather than
relying on a boolean dtype.

| Task | `observations` | `mask` | `target` (supervision only) | Also present |
|---|---|---|---|---|
| `recovery` | masked state at the target step, `[H,W,V_state]` | `[H,W,1]` | complete state | `pde_condition`, `geometry`, `metadata` |
| `forward` | masked condition field, `[H,W,V_cond]` | `[H,W,1]` | state at the target step | `pde_condition`, `geometry`, `metadata` |
| `inverse` | masked state, `[H,W,V_state]` | `[H,W,1]` | complete condition field | `geometry`, `metadata` |
| `rollout` | masked history, `[T_hist,H,W,V_state]` | `[T_hist,H,W,1]` | future frames | `initial_state`, `geometry`, `metadata` |

The mask column keeps a trailing axis of exactly 1 in every row because the
dataset broadcasts one spatial mask across channels; the observation channel axis
is the field's own `V`.

`pde_condition` and `initial_state` are training context only. The dataset's own
comments call them "declared training context" for recovery, the "declared
training loss" input for forward, and "Training-only access" for rollout
([`src/pdeobs/dataset.py`](../src/pdeobs/dataset.py)): the complete coefficient or
source exists for a physics-informed training loss and for classical
reduced-order fitting, never as a test-time model input. A method must return an
array matching the declared target shape for the task it advertises. Read
[Tasks and information permissions](tasks_and_permissions.md) before you decide
what your method is allowed to look at; a plugin can be handed keyword arguments
and metadata, so it must declare and audit its own information access.

**Two escalation tiers.**

1. *Swap a registered built-in by name.* An experiment configuration selects a
   method as `method: {name: <registered name>, kwargs: {...}}`
   ([`configs/experiment/recovery_fno.yaml`](../configs/experiment/recovery_fno.yaml)),
   and a temporal row wraps a one-step model through
   `method: {name: autoregressive, base: {name: ..., kwargs: {...}}}`
   ([`configs/experiment/rollout_fno.yaml`](../configs/experiment/rollout_fno.yaml),
   resolved in [`src/pdeobs/runner.py`](../src/pdeobs/runner.py)).
   The one-line CLI takes `--model <name>`. Nothing else changes.
2. *Supply your own class.* Register it as above, import the module, then use its
   name in exactly the same place. `create_method` raises on an unknown name
   rather than silently choosing another algorithm. The easy-API `--model`
   surface is a separate, fixed table of publicly specified models
   ([`docs/model_parameter_reference.md`](model_parameter_reference.md)); a newly
   registered method is reached through the configuration route, not by being
   added to that table.

**The geometry invariant: geometry is required whenever geometry channels are
non-zero.** A model configured with a non-zero `geometry_channels` and handed no
geometry tensor raises, with the reason stated in the message: *"geometry is
required when geometry_channels is non-zero; obstacle geometry must never be
silently discarded"*
([`src/pdeobs/methods/neural.py`](../src/pdeobs/methods/neural.py)). Batch and
spatial mismatches, and a wrong channel count, raise as well. Note that the
default differs by layer: the compact neural constructors default
`geometry_channels=0`, the operator-network constructors default `1`, and the
easy API *defaults* it to `1` — `kwargs.setdefault("geometry_channels", 1)` for
specs whose family is `neural` and that expose the parameter, so an explicit
`geometry_channels` in `params` still wins
([`src/pdeobs/api/models.py`](../src/pdeobs/api/models.py)). State which one your
method assumes.

**The rollout rule.** A one-step model is served to the four temporal families by
the autoregressive wrapper, which feeds each prediction forward and sets
`mask = None` after the first step, because predicted states are fully specified
([`AutoregressiveModel` in `src/pdeobs/methods/neural.py`](../src/pdeobs/methods/neural.py)).
The paper protocol pins this as history 1, horizon 3, teacher-forcing ratio 0.0,
recurrent input = previous prediction, future ground truth forbidden as input
([`configs/paper/protocol.yaml`](../configs/paper/protocol.yaml)). Any method
that reads a future target frame at inference is disqualified by construction: it
is not a weaker result, it is not a result. If your rollout adapter cannot avoid
a future frame, do not register it as a rollout method.

**How to check it.** Instantiate through `create_method`, assert the declared
target shape for one batch of the task you advertise, assert that a missing mask
raises when `requires_mask` is set, and hold the permitted observations fixed
while changing hidden values to confirm the prediction does not move.

**Practical notes.** Keep optional PyTorch imports inside the neural path; an
interpolation-only or generation-only extension must not force a GPU runtime.
Trainable modules additionally need the expected forward/layout contract and a
declared architecture/configuration so a checkpoint can be reconstructed. Aliases
resolve names and are never counted as separate methods.

---

## How do I add a new PDE or condition generator?

**Register a generator with `pdeobs.pdes.register_generator` and return a
`PDEOutput`.** Unlike the mask and method helpers, `register_generator` is a
plain function, not a decorator — its signature is
`register_generator(name, generator, *, aliases=(), replace=False) -> None`
([`src/pdeobs/pdes/__init__.py`](../src/pdeobs/pdes/__init__.py)), so call it
with the callable as the second argument:

```python
from pdeobs.pdes import generate_sample, register_generator

def my_family(*, boundary, setting, regime, seed, resolution, time_steps=None, **options):
    ...  # build and return a PDEOutput
    
register_generator("my_family_demo", my_family, aliases=("mfd",))
sample = generate_sample("my_family_demo", boundary="periodic", seed=0, resolution=32)
```

The generator is called through `generate_sample(family, ...)` with `boundary`,
`setting`, `regime`, `seed`, `resolution`, `time_steps` and family options, all
passed as keywords; build the return value with
[`pdeobs.pdes.common.build_output`](../src/pdeobs/pdes/common.py) so the layout
and the rejection rules below are enforced for you.

**What a generator must return** — channel-last, unbatched, spatially consistent:

| Array | Shape | Rule |
|---|---|---|
| `condition` | `[H, W, V_cond]` | the family's own reading of the setting field (coefficient, source, or initial state) |
| `trajectory` | `[T, H, W, V_state]` | `T = 1` for a static family; the temporal default is `T = 9` |
| `geometry` | `[H, W, 1]` | last axis must be exactly 1 |

All three must share the same `H, W`; `PDEOutput.__post_init__` raises otherwise.
A static family must reject any request for `T != 1` — `resolve_time_steps`
raises `"static PDE families always have T=1"`. Use `add_channel` to lift an
`[H, W]` field into the canonical `[H, W, 1]` layout.

**Separate deterministic RNG streams.** Draw the condition field and the geometry
from different streams of the same sample seed via `make_rng(seed, stream)`,
which builds a generator from `np.random.SeedSequence([seed, stream])`. The
built-in families use one pair per family — darcy 100/101, poisson 200/201,
helmholtz 300/301, heat 400/401, reaction-diffusion 500/501, burgers 600/601,
navier-stokes 700/701 — so an obstacle draw can never shift the condition field.
Pick an unused pair rather than reusing a built-in one. A condition field itself
should come from `make_setting_field`, which resolves through `SETTING_REGISTRY`
and raises on a wrong shape or a non-finite value instead of accepting it.

**Record a solver identifier and a named quality-residual contract.** Every
built-in family writes provenance into `parameters`: a `solver_id` or
`integrator_id` naming the numerical route actually taken, and a
`quality_residual_contract` such as `pdeobs.quality.poisson.fd2_boundary_v1`,
so the audit that applies to a record is fixed at generation time rather than
guessed later. A family whose route depends on the boundary must record which
route ran for that sample. See [Numerical solvers](numerical_solvers.md) for how
the built-ins do this, including the statement that parameter values are unitless
numbers on the code's unit domain.

**Non-finite or out-of-range output is rejected, never clipped.** `build_output`
documents the reason: *"Non-finite or out-of-range solver output is rejected. It
is never clipped or replaced, because doing so would hide numerical failures in
generated benchmark data."* In practice `_safe_array` raises `FloatingPointError`
on NaN or infinity and `OverflowError` when the magnitude exceeds the storage
dtype's range. Do not add a `np.nan_to_num`, a clamp, or a retry-with-smaller-step
that silently substitutes a different problem inside a generator. A failed solve
is a failed record.

**How to check it.** Generate one sample twice at the same seed and compare every
array bit-for-bit; assert the three shapes and the shared `H, W`; assert that a
static family raises on `time_steps=2`; and assert that a deliberately divergent
parameter set raises instead of returning a clipped field.

**A name is not a family.** A meaningful new family also states: the equation,
state representation, conventions and boundary semantics; deterministic seeding,
dtype and spatial/time shape behaviour; admissible parameters, solver error
handling and solver provenance; finite/IC/BC/residual checks plus an independent
accuracy check; and task and baseline compatibility tests. PDE quality operators
and physical metrics require the corresponding context, and a generic
registration does not certify an arbitrary parameter Cartesian product.

---

## How do I add a new metric?

**Register a callable into `METRIC_REGISTRY` — this axis has no fixed
signature.** The document states that rather than inventing a contract the code
does not enforce. `install_builtin_metrics()` installs twelve built-ins through
`registry.register(name, obj=metric)` — a direct call, not a decorator — and
their signatures are heterogeneous
([`src/pdeobs/metrics.py`](../src/pdeobs/metrics.py)): `mse`, `mae`,
`relative_l2`, `spectral_centroid_error` and `high_frequency_energy_error` take
`(prediction, target)` and return a float; `frequency_band_errors` and
`rollout_horizon` take `(prediction, target)` and return a dict;
`energy_error`, `enstrophy_error` and `vorticity_error` take
`(prediction, target)` with a channel axis; `ood_degradation` takes two scalar
scores; `stability` takes a single rollout array. Your callable must match
whatever *your* caller passes, and you must document that expectation yourself.

```python
from pdeobs.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register("max_abs_error_demo")
def max_abs_error(prediction, target):
    import numpy as np
    return float(np.max(np.abs(np.asarray(prediction, dtype=np.float64)
                               - np.asarray(target, dtype=np.float64))))
```

**The invariant: registering a metric changes no scored number.** The evaluation
path builds a `MetricSuite` over the fixed `DEFAULT_METRICS` set, plus optional
frequency-band, rollout-horizon, stability and physical additions
([`src/pdeobs/evaluation.py`](../src/pdeobs/evaluation.py)), and the strict
scorer computes its own relative L2 in float64
([`src/pdeobs/strict_score.py`](../src/pdeobs/strict_score.py)). Neither consults
`METRIC_REGISTRY`. What the registry buys is discovery and listing; a registered
metric enters a comparison only where your own code calls it, and a changed
metric definition needs a new score version (see [What extension does not
buy](#what-extension-does-not-buy)).

**How to check it.** Call it with the exact arguments your caller uses, assert a
deterministic float64 result on fixed arrays, and confirm it is listed:
`pdeobs list --kind metrics --plugins`.

---

## Custom physical arrays without writing a solver

If you have your own coefficient, source or initial-state arrays and no wish to
write a generator, pass them to the existing kernels. The interface is
deliberately thin: the caller specifies a well-posed problem and chooses the
spacings, boundary and parameter values; nothing is imputed from a mask.

```python
from pdeobs.physical_inputs import advance_custom_heat, solve_custom_elliptic

# `source` (and optional `coefficient`) are finite 2-D fields whose two
# dimensions are each at least 3.
solution, info = solve_custom_elliptic(
    source, boundary="periodic", dx=1.0 / n, dy=1.0 / n, coefficient=coefficient
)
# `initial` is a finite 2-D field with both dimensions at least 3.
next_state, heat_info = advance_custom_heat(
    initial, diffusivity=0.02, dt=0.01, boundary="periodic", dx=1.0 / n, dy=1.0 / n
)
```

Validation at this layer is exactly: a 2-D array, `min(shape) >= 3`, all finite,
and finite positive `dx`, `dy`
([`src/pdeobs/physical_inputs.py`](../src/pdeobs/physical_inputs.py)). It does
**not** check that `coefficient` is positive; well-posedness of the elliptic
problem remains the caller's responsibility.

The returned `SolverInfo` records convergence; preserve and report it rather than
storing a non-converged field. To write canonical records from your own arrays,
`api.create_dataset_from_arrays` covers the four routes it can run with complete
inputs — `poisson` (`source`), `darcy` (`source` + `coefficient`), `helmholtz`
(`source` + `physical["reaction"]`) and `heat` (`initial_state` +
`{diffusivity, dt, steps}`) — and rejects the other families explicitly rather
than approximating them ([`src/pdeobs/api/data.py`](../src/pdeobs/api/data.py)).
A solve that does not converge raises and the record is not stored.

**Scope note.** The E4 release demo exercises this path and checks one resolved
periodic analytic Heat mode to `1e-12`; its own return value labels that scope as
*"one resolved periodic Fourier mode; not a seven-family convergence study"*. A
single resolved analytic mode is a reference check, not a convergence study, and
a complete-input solver is not a generally fair partial-input baseline.

---

## Registering through entry points

A reusable package publishes its factories in one of five entry-point groups, so
no fork of this repository is required:

| Group | Registry | Loaded by |
|---|---|---|
| `pdeobs.pdes` | `PDE_REGISTRY` | `pdeobs.pdes.discover_generators()`, and implicitly on the first generator lookup |
| `pdeobs.settings` | `SETTING_REGISTRY` | `Registry.discover()` / `discover_plugins()` |
| `pdeobs.masks` | `MASK_REGISTRY` | `Registry.discover()` / `discover_plugins()` |
| `pdeobs.methods` | the `pdeobs.methods` dict via `discover_methods()`; `pdeobs.registry.METHOD_REGISTRY` via `discover_plugins()` | either, and implicitly by `create_method` on a cache miss |
| `pdeobs.metrics` | `METRIC_REGISTRY` | `Registry.discover()` / `discover_plugins()` |

The `pdeobs.methods` row is the one to read twice: the same group name feeds two
different objects, and only the `discover_methods()` one is what `create_method`
resolves against (see the trap noted in the method section).

Importing `pdeobs` never imports third-party packages; discovery happens on an
explicit `discover_*` call, on the first generator lookup
(`get_generator` calls `discover_generators`), or on a `create_method` cache miss.
**A broken plugin cannot break the built-ins.** `Registry.discover` takes
`on_error` in `{"warn", "ignore", "raise"}` and defaults to `warn`;
`discover_methods` catches a failing entry point and stores its repr, so that
*"Broken optional plugins do not prevent built-in baselines from being used"*.
Read the recorded failures with `pdeobs.methods.method_discovery_errors()`.

Verify that your plugin is visible:

```bash
pdeobs list --kind masks --plugins
pdeobs list --plugins --json
```

Installation success is not validation of a plugin, and an entry point that
loads is not an entry point that is correct. Visibility in `pdeobs list` is not
the same as reachability from `create_method`.

---

## Required extension tests

An extension is not usable in a comparison until it passes this checklist. Write
the tests in your own file; they are your evidence, not this repository's.

| # | What must be shown | Concretely |
|---|---|---|
| 1 | **Shape and dtype contract** | a mask returns `bool[H,W]` at the requested grid; a setting returns a finite `float[H,W]`; a generator returns `[H,W,V]` / `[T,H,W,V]` / `[H,W,1]` with matching spatial shape and `T=1` for a static family; a method returns the declared target shape |
| 2 | **Determinism from the declared seed** | two calls at the same seed produce bit-identical output, and two different seeds do not; a deterministic protocol is identical across seeds by design |
| 3 | **No target leakage** | with permitted observations held fixed, changing hidden values or a future target must not change an ordinary prediction; a rollout method never reads a frame beyond its history |
| 4 | **A strict score that validates** | an end-to-end `pdeobs-strict-v1` block over disjoint train/test identities returns `status: "valid"`, with a save/reload round trip for any trainable method |
| 5 | **Explicit failure** | an unsupported input, a missing mask, a missing geometry with non-zero geometry channels, and a non-finite solver result each raise, rather than degrading quietly |
| 6 | **Reachability** | the name you registered resolves through the surface you will actually use — `create_method` for a method, `generate_mask` / a dataset `mask={"protocol": ...}` for a view, `generate_sample` for a family — not merely through `pdeobs list` |

A strict block is *valid* when its arrays, identities, shapes and time indices
line up; validity is not a quality claim, and a finite poor prediction stays
valid and stays in the score ([Scoring](scoring.md)). Record a passing smoke test
as a smoke test, never as paper performance or general numerical validation.

Run the built-in contract suites together with your own tests in one command
(install the optional training dependency, `pip install '.[train]'`, to execute
rather than skip the neural cases):

```bash
python -m pytest -q -p no:cacheprovider tests/test_registry.py tests/test_masks.py \
  tests/test_methods.py tests/test_pdes.py tests/test_strict_score.py \
  tests/test_benchmark_essentials.py path/to/your_extension_tests.py
```

The low-cost, already-tested end-to-end extension path is the E6 release demo,
which registers a custom mask factory and a custom method through the public
registries and scores the result without changing a PDE kernel. It does run one:
E6 generates a small Poisson dataset with the built-in generator
([`src/pdeobs/release_demo.py`](../src/pdeobs/release_demo.py)), and its own
returned label is *"registry factory without changing the PDE kernel"*.

```bash
pdeobs demo --example E6 --data-root ./demo-data --output-root ./demo-out
```

Use fresh roots: the demo creates its per-example directories with
`exist_ok=False` and refuses to overwrite prior evidence, so a second run against
the same roots fails with a directory-exists error rather than replacing it.

---

## Contributing a result

The format for a third-party number is fixed in advance so that it cannot be
negotiated after the number exists. A submission is five things together:

| Part | Requirement |
|---|---|
| **Long-format rows** | one row per scored cell, not a pre-aggregated table. Identify the cell by PDE family, method, task, boundary, setting, regime, split, training view `v`, test view `w` and — for rollout — the horizon, then the metric name and its value. Each strict block already records its own `train_view` and `test_view` in the contract ([`src/pdeobs/paper_row.py`](../src/pdeobs/paper_row.py)), and the benchmark runner aggregates flattened rows on `(method, task, pde, boundary, setting, regime, ood_view, split, mask_protocol, rollout_horizon)` ([`src/pdeobs/runner.py`](../src/pdeobs/runner.py), [`src/pdeobs/reports.py`](../src/pdeobs/reports.py)) |
| **Resolved configuration** | the configuration as actually resolved and executed, not the template it came from. The `plan`, `generate` and `generate-case` commands ([`src/pdeobs/cli.py`](../src/pdeobs/cli.py)) and the train, infer and eval routes ([`src/pdeobs/runner.py`](../src/pdeobs/runner.py)) each save a resolved configuration beside their outputs; read-only commands such as `list`, `doctor`, `protocol` and `demo` do not, so record the invocation yourself when you use them |
| **Contract hash** | the strict contract and its canonical SHA-256 (`config_sha256`), together with the identity-set hash and the expected identity order that the scorer recorded ([`src/pdeobs/strict_score.py`](../src/pdeobs/strict_score.py)) |
| **Scorer version** | the scoring version string, `pdeobs-strict-v1` for the release default. Historical results keep their own recorded version; a strict test passing does not retroactively make a legacy number strict-validated |
| **Run bundle** | the artifacts that produced the rows: the resolved row, the split receipt, the identity manifest, the checkpoint, the training completion record, and per test view a `contract.json`, `predictions.h5` and `score.json`, plus the run receipt ([`docs/reproducing_the_paper.md`](reproducing_the_paper.md)) |

**A number without its contract hash and scorer version is not a PDE-OBS
result.** The same rules that govern the benchmark's own rows govern a
contributed one: a paper-test contract requires exactly the 200 declared test
identities, relative L2 is computed in float64 per identity and then averaged
over the complete expected set, a missing cell stays `RESULT_PENDING` or an
explicit absence — never zero, never an interpolated curve, never another row's
value — and a table must say which cohorts and scoring versions are pooled.

**This release accepts no submissions channel.** There is no public leaderboard,
no upload endpoint and no verified public download endpoint in this candidate
([`src/pdeobs/download.py`](../src/pdeobs/download.py),
[`src/pdeobs/api/data.py`](../src/pdeobs/api/data.py)). A local benchmark run
does write `leaderboard.json` and `leaderboard.csv` into its own benchmark root
([`src/pdeobs/runner.py`](../src/pdeobs/runner.py)); those are local aggregates
of your own rows and are not a ranking of anyone else's. The format is published
here so that a result produced independently is already in the shape a future
channel would require, and so that the format cannot be adjusted to fit a number
after the fact.

---

## What extension does not buy

- **Registration is not certification.** A registered plugin has been imported,
  not validated. The release's leakage and permission tests exercise the built-in
  data/predictor boundary with fixed masks; they are targeted tests, not a proof
  about arbitrary third-party plugins.
- **Implemented is not paper-evaluated.** The release records four evidence
  levels — implemented, smoke-tested, reference-checked, paper-evaluated — and a
  small-grid smoke test is not an independent convergence study
  ([README.md](../README.md)).
- **Optional interfaces, registry aliases and wrappers are not extra primary
  algorithms and not completed experiments.** Aliases resolve names; they do not
  multiply the method count. Retrieval, routing, upstream wrappers and broader
  task support are explicitly outside the paper's selected recovery/rollout
  comparison ([`configs/extensions/README.md`](../configs/extensions/README.md)).
- **There is no universal task-plugin registry.** The task layer has explicit
  `recovery`, `forward`, `inverse` and `rollout` branches. A genuinely new task
  needs coordinated dataset, training, prediction, evaluation and permission
  changes, and must not be disguised under an existing task name.
- **A changed metric needs a new score version.** The historical metric registry
  is extensible, but the strict score contract is versioned separately. If an
  extension changes denominators, validity rules, identity interpretation or
  projection, give it a distinct score version. Never use broad exception
  handling to drop failing identities out of a fixed-denominator aggregate.
- **This release does not add** distributed training, a new model family, moving
  sensors, a continuous-coordinate observation API, noise or outlier models, an
  active-learning or solver-query loop, a website backend, or an online
  leaderboard.
