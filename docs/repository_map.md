# Repository map and documentation index

This page is a directory tour: what each part of the tree is *for*, so that the
architecture is visible without cloning and reading source. Read the tree in five
layers — **source** (`src/pdeobs/`, the library and its CLI), **configuration**
(`configs/`, where each subdirectory is one axis of a run command), **documentation**
(`docs/`, indexed below by the question each file answers), **tests** (`tests/`), and
two different kinds of **evidence** (`release/` and `acceptance/`, both software
evidence only). No benchmark result is shipped in this tree; the campaign that fills
the planned grid is not part of this release, and
[`docs/results_format.md`](results_format.md) states the same from the results side.

A sixth thing is worth stating about packaging: the release is the tracked file set, not a
working directory. See [Packaging a copy of this tree](#packaging-a-copy-of-this-tree).

---

## Top level

`ls -a` at the root, including the entries that are conventionally hidden:

```text
pdeobs-anon/
├── .git/                        VCS metadata — RESIDUE, delete before shipping
├── .gitignore                   ignores __pycache__/, *.pyc, .pytest_cache/, *.egg-info/, build/, dist/
├── .pytest_cache/               pytest run cache — RESIDUE, delete before shipping
├── ACCEPTANCE_SUMMARY.md        software-acceptance summary (L0/L1/L2 levels)
├── LICENSE                      MIT licence text for the code
├── README.md                    entry point: the grid, the quickstart, scope limits
├── RELEASE_NOTES_v0.2.0.md      what v0.2.0 adds and what it explicitly does not do
├── THIRD_PARTY_NOTICES.md       retained upstream notices and attribution
├── gen_docs.py                  regenerates two docs/ files from the live specs
├── pyproject.toml               package metadata, dependency floors, extras, entry points
├── acceptance/                  two 14-tier software sweeps (round 1 incomplete) on a CUDA node
├── configs/                     the run axes: paper protocol, campaign, methods, data, demos
├── docs/                        reference documentation (this file included)
├── examples/                    one portable script that forwards to the demo entry point
├── release/                     candidate metadata, frozen test scope, validation record, test runner
├── src/pdeobs/                  the library: generators, storage, masks, methods, scoring, API, CLI
└── tests/                       38 `test_*.py` modules plus one base-only acceptance script
```

`__pycache__/` directories also exist under `src/pdeobs/`, `src/pdeobs/pdes/`,
`src/pdeobs/methods/`, `src/pdeobs/api/` and `tests/`. They are residue too, and they
are listed in that section rather than here.

| Path | One line |
|---|---|
| [`ACCEPTANCE_SUMMARY.md`](../ACCEPTANCE_SUMMARY.md) | The three acceptance levels (CPU regression, parameter coverage, short GPU flows) and their explicit boundaries. Every number in it is software acceptance — including per-model parameter counts and peak-memory/seconds tables — not a paper table entry. |
| [`LICENSE`](../LICENSE) | The MIT licence text for the code and nothing else. It is silent about data; the statement that the code licence does not extend to external assets lives in [`release/candidate_metadata.json`](../release/candidate_metadata.json) (`code_license_automatically_licenses_external_data: false`) and [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) ("does not redistribute third-party datasets, pretrained weights, or upstream source code"). |
| [`README.md`](../README.md) | A short front page: the grid as an arithmetic claim, the first-run constraints, a twelve-line quickstart, one reproduced grid cell, and what the release does not claim. It delegates the detail to the documents indexed below. |
| [`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md) | What the v0.2.0 facade adds (no numerical algorithm, no changed loss/mask/split/scoring) and the "explicitly not done this round" list. |
| [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) | Upstream notices and required attribution retained for the adapted methods, and the statement that no third-party dataset, weight or source tree is redistributed here. |
| [`gen_docs.py`](../gen_docs.py) | Regenerates `docs/model_parameter_reference.md` and `docs/observation_reference.md` from the live specs. |
| [`pyproject.toml`](../pyproject.toml) | Package name `pdeobs`, version `0.2.0`, `requires-python >=3.10`, core dependencies, the `train` / `paper` / `analysis` / `site` / `test` / `dev` extras, the `pdeobs` console script, and the third-party entry-point groups. |
| `acceptance/` | Software-acceptance evidence for the v0.2.0 facade: two sweeps, their raw per-tier JSON, and the defect record. Contains environment strings that must be scrubbed; see [`acceptance/`](#acceptance--two-software-sweeps-of-the-v020-facade). |
| `configs/` | Every configurable axis, one subdirectory per axis (see [configs/](#configs)). |
| `docs/` | Reference documentation, indexed below. |
| `examples/` | A single portable script; see [examples/](#examples). |
| `release/` | Release-candidate metadata and the frozen, re-runnable CPU test scope. |
| `src/pdeobs/` | The installable package. |
| `tests/` | The test suite, mapped to the twelve mandatory categories in [`testing_checklist.md`](testing_checklist.md). |

---

## `src/pdeobs/` — a grouped walk

The package is one flat module namespace plus three subpackages (`pdes/`, `methods/`,
`api/`). The grouping below is by role, not by import order.

### Physics: `src/pdeobs/pdes/`

Deterministic, NumPy-only reference generators for the seven PDE families. The
directory holds ten `.py` files: the seven family modules plus `__init__.py`,
`common.py` and `numerics.py`.

Every family module — static and temporal alike — records a
`quality_residual_contract` string per sample, naming the residual definition in
`pdeobs.quality` against which that sample is checked.

| Module | Purpose |
|---|---|
| [`__init__.py`](../src/pdeobs/pdes/__init__.py) | Registers the seven family generators under the canonical names `darcy`, `poisson`, `helmholtz`, `heat`, `reaction_diffusion`, `burgers`, `navier_stokes`; declares `STATIC_FAMILIES` (the three with `T=1`), `TEMPORAL_FAMILIES` (the other four) and `BOUNDARY_NAMES = ("dirichlet", "neumann", "periodic", "robin")`. |
| `darcy.py`, `poisson.py`, `helmholtz.py` | The three static elliptic families. Each builds its condition channel (coefficient or source), applies the regime parameter, and calls the shared elliptic solver. |
| `heat.py`, `reaction_diffusion.py`, `burgers.py`, `navier_stokes.py` | The four temporal families. In addition to the shared residual contract, each records its own `integrator_id` per sample — the field that is specific to the temporal four. |
| [`common.py`](../src/pdeobs/pdes/common.py) | Everything the seven modules share: the cell-centred unit-square grid, family, boundary, setting and regime aliases, per-sample RNG streams, the scalar boundary relation, the geometry mask, and the output validator that **rejects** rather than clips non-finite solver output. |
| [`numerics.py`](../src/pdeobs/pdes/numerics.py) | The shared numerical kernels: the sparse elliptic solve (preconditioned CG, or MINRES for the indefinite case), the Crank-Nicolson / Fourier diffusion step, the Strang-split reaction-diffusion advance, the CFL-controlled Burgers advance, and the vorticity routes. Non-convergence raises; it never returns a partial solve. |

### Observation and data

| Module | Purpose |
|---|---|
| [`settings.py`](../src/pdeobs/settings.py) | The ten condition-field constructions (`S0`–`S9`, from `smooth_grf` to `front_ring_shock`) shared by all seven families; the family decides whether the field is a coefficient, a source or an initial state. The ten are `SETTING_REGISTRY.register` decorators in this file. |
| [`masks.py`](../src/pdeobs/masks.py) | The nine registered mask factories (`random_1pct` … `clustered_sensors`) plus `broadcast_mask` / `apply_mask`. `True` means observed. The nine are `MASK_REGISTRY.register` decorators in this file. |
| [`mask_specs.py`](../src/pdeobs/mask_specs.py) | v0.2.1: a protocol with an observed percentage and `+`-joined deterministic mixtures, resolved to the registered factories at generation time; never replaces a registered name. |
| [`schema.py`](../src/pdeobs/schema.py) | The canonical in-memory record: channel-last `HWC` / `THWC`, validated for shape, dtype and finiteness before anything is stored. |
| [`storage.py`](../src/pdeobs/storage.py) | Atomic, resumable HDF5 shards (lock file, `.partial` append target, one atomic rename, five sidecars) and the process-safe lazy reader. |
| [`generation.py`](../src/pdeobs/generation.py) | Turns a generation plan into shards: identity strings, derived seeds, stored-frame selection, per-sample metadata and the quality record. |
| [`dataset.py`](../src/pdeobs/dataset.py) | Builds the four task views (recovery, forward, inverse, rollout) *at access time* over immutable shards; derives and records the observation mask per sample. |
| [`splits.py`](../src/pdeobs/splits.py) | Balanced physical regimes, the generic IID split, the official OOD labels, and the nested release tiers. |
| [`download.py`](../src/pdeobs/download.py) | Manifest-driven, checksum-verified downloader. There is **no** default endpoint in this candidate; an explicit manifest is required. |

### Methods: `src/pdeobs/methods/`

| Module | Purpose |
|---|---|
| [`base.py`](../src/pdeobs/methods/base.py) | The method interface and a registry deliberately local to the method layer, plus entry-point discovery so a broken optional plugin cannot break the built-ins. |
| [`interpolation.py`](../src/pdeobs/methods/interpolation.py) | Transparent non-learning baselines (zero, mean, nearest, bilinear, RBF, persistence) that need no PyTorch. |
| [`reduced_order.py`](../src/pdeobs/methods/reduced_order.py) | Leakage-free reduced-order baselines: Gappy POD, fitted only on complete training fields, and its rollout adapter. |
| [`neural.py`](../src/pdeobs/methods/neural.py) | Compact PyTorch reference architectures (U-Net, U-FNO, FNO, CNO, PINO, ConvLSTM, MAE-style, residual encoder) and the autoregressive rollout wrapper. The module docstring states they are **not** exact reproductions of the corresponding papers. |
| [`operator_networks.py`](../src/pdeobs/methods/operator_networks.py) | Paper-guided operator-network adapters (DeepONet, Transolver, GNOT, a deeper guided U-Net) — independent adapters, not copies of upstream trees. |
| [`paper_official.py`](../src/pdeobs/methods/paper_official.py) | Fail-closed adapters for exact, commit-pinned author implementations. The author source is not vendored; construction verifies the checkout, revision and file digest before importing anything, and these entries are not publicly enumerated. |

### Training and the two physics-aware paths

| Module | Purpose |
|---|---|
| [`training.py`](../src/pdeobs/training.py) | The reproducible recovery and rollout training loops over ordinary PyTorch data loaders. |
| [`pino.py`](../src/pdeobs/pino.py) | The differentiable physics residuals for the PINO adapter: a static contract for the three elliptic families and a periodic-only temporal contract evaluated strictly between predicted states. |
| [`physical_inputs.py`](../src/pdeobs/physical_inputs.py) | Small public adapters that validate a caller's own coefficient / source / initial arrays and delegate to the existing kernels. They do not reconstruct unknown coefficients. |

### The scoring contract

| Module | Purpose |
|---|---|
| [`strict_score.py`](../src/pdeobs/strict_score.py) | The versioned, fail-closed scorer: raw arrays validated before any metric, explicit identity sets, no NaN-skipping reduction, standard JSON, and score files written exclusively-create so an old score is never overwritten. |
| [`strict_inference.py`](../src/pdeobs/strict_inference.py) | The inference-to-score bridge and the bound-artifact writer: an identity manifest pins shards by digest, and the exported HDF5 carries the contract JSON and its hash, so a relabelling of the artifact under a different contract is **detectable**. The module docstring is explicit about the limit of that guarantee: hash binding establishes consistency, not independent provenance authentication. |

### The frozen campaign and the executable row

| Module | Purpose |
|---|---|
| [`one_setting.py`](../src/pdeobs/one_setting.py) | Protocol materialization for the frozen all-PDE one-setting cross-observation campaign: the frozen PDE order, the public method tuple, the deterministic identity split, and the campaign validator. It submits no jobs. |
| [`paper_row.py`](../src/pdeobs/paper_row.py) | Executes exactly one row: resolve, split, train under the declared budget, discard the model, reload the final checkpoint in a fresh instance, then score each test view into its own block. |
| [`campaign.py`](../src/pdeobs/campaign.py) | Strict expansion and execution helpers for training campaigns, including the separate full-observation diagnostic path. |
| [`full_to_partial.py`](../src/pdeobs/full_to_partial.py), [`full_to_partial_execution.py`](../src/pdeobs/full_to_partial_execution.py) | Materialization and fail-closed execution/promotion for the all-factor full-to-partial campaign, with no implicit rows. |
| [`protocol.py`](../src/pdeobs/protocol.py) | The frozen, machine-verifiable contract behind `pdeobs protocol --check`. |
| [`migration.py`](../src/pdeobs/migration.py) | Fail-closed planning for cross-domain load balancing. It never calls a scheduler; it only turns captured attestations into an immutable plan. |

### The legacy metric path

These four modules predate the strict scorer and remain available for provenance and
compatibility. Their aggregates are **not** retroactively labelled strict-validated.

| Module | Purpose |
|---|---|
| [`evaluation.py`](../src/pdeobs/evaluation.py) | Inference and evaluation runners for learning and non-learning methods, including the diagnostic full-observation control that may never replace a matched-mask score. |
| [`metrics.py`](../src/pdeobs/metrics.py) | The numerical, spectral, physical, rollout and OOD metric definitions. |
| [`reports.py`](../src/pdeobs/reports.py) | Dependency-free CSV/JSON aggregation (count, mean, std, min, max per group). |
| [`aggregate.py`](../src/pdeobs/aggregate.py) | Shard validation plus result aggregation behind `pdeobs aggregate`. |

### The v0.2.0 facade: `src/pdeobs/api/`

One-line operations over everything above. It adds no numerical algorithm and changes
no loss, mask, split or scoring definition.

| Module | Purpose |
|---|---|
| [`specs.py`](../src/pdeobs/api/specs.py) | The declarative tables: model specs and their parameters, the nine frozen paper views and their expected counts, the task list. Nothing here constructs a model or a mask. |
| [`models.py`](../src/pdeobs/api/models.py) | Validates a model request against the public schema *before* anything expensive, then constructs through the existing factory. |
| [`observation.py`](../src/pdeobs/api/observation.py) | Resolves an observation spec and gives each resolved observation a content-addressed id. Its module docstring and the generated [`observation_reference.md`](observation_reference.md) both document **four** namespaces: `general` (free parameters over the existing factories), `paper` (the nine frozen views), `custom` (a factory registered in `MASK_REGISTRY`) and `stored` (the mask carried by an inference input package). `README.md` still describes two; the resolver is the authority, and the README line is stale — see [Known stale pointers](#known-stale-pointers-and-what-no-test-enforces). |
| [`data.py`](../src/pdeobs/api/data.py) | Data sources: numerical generation, local canonical records, identity manifests, and the observation-only inference package that keeps the targets with the caller. |
| [`train.py`](../src/pdeobs/api/train.py) | Training under an explicit budget (epochs or steps — there is no default long run), plus identity-split planning. |
| [`infer.py`](../src/pdeobs/api/infer.py) | Target-free inference: observations and mask in, raw predictions out. No score is computed here. |
| [`evaluate.py`](../src/pdeobs/api/evaluate.py) | Strict evaluation through the existing `pdeobs-strict-v1` scorer; nothing is scored without an explicit target. Includes the multi-view helper. |
| [`artifacts.py`](../src/pdeobs/api/artifacts.py) | The reusable model-artifact directory: write it, read it back, and refuse a partial load on a structure mismatch. |
| [`pipeline.py`](../src/pdeobs/api/pipeline.py) | The versioned six-stage pipeline config (data → observation → model → train → predict → evaluate), resolvable without executing anything. |
| [`cli.py`](../src/pdeobs/api/cli.py) | The `pdeobs easy ...` command namespace; it shares the resolver with the Python API. |

### Command line, checks and provenance

| Module | Purpose |
|---|---|
| [`cli.py`](../src/pdeobs/cli.py) | The single `pdeobs` console entry point: generation, training, evaluation, strict scoring, the demos and the local checks. Documented command by command in [`cli_reference.md`](cli_reference.md). |
| [`__main__.py`](../src/pdeobs/__main__.py) | Lets `python -m pdeobs` behave like the console command. |
| [`release_demo.py`](../src/pdeobs/release_demo.py) | The six small CPU-only release workflows (E1–E6). Its own docstring states they are interface demonstrations, not benchmark performance evidence. |
| [`quality.py`](../src/pdeobs/quality.py) | Scientific and structural quality control for generated data — deliberately separate from model evaluation. It measures whether generated arrays satisfy the family equation and its stored constraints, under the `quality_residual_contract` each family records. |
| [`provenance.py`](../src/pdeobs/provenance.py) | The reproducibility metadata captured beside every run and every shard. **It records `socket.gethostname()`** (`provenance.py:110`). Receipts, shard sidecars and run metadata produced by this code therefore carry the machine name of whoever generated them; scrub or regenerate them before any artifact is uploaded for anonymous review. No such receipt is shipped in this tree. |
| [`coverage.py`](../src/pdeobs/coverage.py) | Fail-closed audits for factorial experiment coverage; the receipt is written even when the audit fails. |
| [`doctor.py`](../src/pdeobs/doctor.py) | Local and Slurm-allocation preflight checks behind `pdeobs doctor`. The module accepts only the modes `local` and `slurm`, probes for `sbatch` and reads `SLURM_JOB_ID`; `pdeobs doctor --help` names Slurm directly. |
| [`config.py`](../src/pdeobs/config.py) | Configuration loading with environment expansion and `--set` overrides. |
| [`registry.py`](../src/pdeobs/registry.py) | The small dependency-free registries (PDEs, settings, masks, methods, metrics) and their entry-point groups, so another package can extend the benchmark without forking it. |
| [`presets.py`](../src/pdeobs/presets.py) | Code-defined presets so the CLI keeps working from an installed wheel where repository-level `configs/` are absent. |
| [`runner.py`](../src/pdeobs/runner.py) | The config adapters the high-level CLI commands sit on. |
| [`difficulty.py`](../src/pdeobs/difficulty.py) | Deterministic, dependency-free difficulty summaries; it produces tables rather than figures. |
| [`retrieval.py`](../src/pdeobs/retrieval.py), [`routing.py`](../src/pdeobs/routing.py) | Optional anchor baselines for semantic retrieval and solver routing. They are not counted as extra primary algorithms. |

---

## `configs/`

Each subdirectory is one axis of a run command, so listing a directory enumerates the
legal values of that axis. The schemas are deliberately *descriptive* where they
describe a protocol, and only some directories hold runnable job configurations —
see [`configs/paper/README.md`](../configs/paper/README.md) for that distinction. File
counts below include each directory's `README.md` where one exists.

| Directory | Files | What this axis means in a run command | Not a claim of |
|---|---|---|---|
| [`configs/paper/`](../configs/paper/) | 5 | The frozen descriptive protocol: [`protocol.yaml`](../configs/paper/protocol.yaml) (families, tasks, split, temporal contract, scoring), [`observations.yaml`](../configs/paper/observations.yaml) (the nine frozen views and their exact realized counts), [`training_cohorts.yaml`](../configs/paper/training_cohorts.yaml) (budget-cohort semantics) and `row_original500.yaml`. | A completed campaign, or an executable job scheduler. These files are descriptive schemas; do not hand them to `pdeobs train`. |
| [`configs/campaign/`](../configs/campaign/) | 8 | The retained campaign and the historical campaign files. [`all_pde_one_setting_10method_9x9.yaml`](../configs/campaign/all_pde_one_setting_10method_9x9.yaml) is marked `status: candidate_preflight_only`, `release_eligible: false`, `formal_credit: 0`. | Credit for any row. The file is candidate-only by its own header. |
| [`configs/method/`](../configs/method/) | 10 | The method axis. [`all_pde_source_faithful_registry.yaml`](../configs/method/all_pde_source_faithful_registry.yaml) is the source-faithful registry, recording per method its inspected upstream revision, its upstream licence, its optimizer and its declared adaptation; the remaining files are per-method and historical registries. | Bitwise upstream reproduction. The registry records what was inspected and what was deliberately changed. |
| [`configs/dataset/`](../configs/dataset/) | 8 | The generation axis: tiers and numerics profiles, from [`smoke.yaml`](../configs/dataset/smoke.yaml) and the numerics demos through [`numerics_validation20.yaml`](../configs/dataset/numerics_validation20.yaml) (full-factor coverage at 20 samples per macro case) to [`numerics_full_t15.yaml`](../configs/dataset/numerics_full_t15.yaml) (the paper-size design). [`default.yaml`](../configs/dataset/default.yaml) is the reference factorial. | That any tier has been generated here. The shipped default profile deliberately leaves the PDE-loss and divergence thresholds non-gating. |
| [`configs/experiment/`](../configs/experiment/) | 48 | Preserved historical experiment configurations, kept so their original scientific parameters are not silently rewritten. | A uniform protocol, current defaults, or current resource policy. |
| [`configs/demo/`](../configs/demo/) | 3 | Small install and interface checks: two reduced `paper-row` demo rows and [`workflows.yaml`](../configs/demo/workflows.yaml), a descriptive manifest of the six fixed CPU demos. | Paper performance or numerical convergence. `workflows.yaml` documents the demos; it is not a training-runner YAML. |
| [`configs/extensions/`](../configs/extensions/) | 4 | Forward/inverse entry points and [`interfaces.yaml`](../configs/extensions/interfaces.yaml), which names the implemented optional interfaces and the status of their CLI paths. | Inclusion in the paper's main experiment. As [`its README`](../configs/extensions/README.md) puts it, "Implemented is not the same as paper-evaluated." |
| [`configs/analysis/`](../configs/analysis/) | 1 | The difficulty-analysis axis consumed by `pdeobs analyze`. | A result. It ranks records that a run has already produced. |

---

## `docs/` — index by the question each file answers

Twenty files, this one included.

| File | The question it answers | Generated or hand-written |
|---|---|---|
| [`repository_map.md`](repository_map.md) | Where does everything live, and what is each part for? | Hand-written (this page) |
| [`benchmark_overview.md`](benchmark_overview.md) | What is the benchmark, which four pieces are kept apart, and how is its grid counted? | Hand-written |
| [`installation.md`](installation.md) | How do I install it, and what are the tested version floors? | Hand-written |
| [`cli_reference.md`](cli_reference.md) | What does every `pdeobs` command take, and what exit codes does it return? | Hand-written |
| [`numerical_solvers.md`](numerical_solvers.md) | What equation does each family solve, by what numerical route, and what does the release *not* claim about its accuracy? | Hand-written |
| [`data_schema.md`](data_schema.md) | What is stored in a record, how is its identity derived, and how does a shard get written, resumed and declared complete? | Hand-written |
| [`dataset_card.md`](dataset_card.md) | What is in the corpus, how large is the factorial design, how does a reader obtain it, and under what terms? | Hand-written |
| [`data_and_inference_schema.md`](data_and_inference_schema.md) | What objects does the easy API pass between data, model and score, and how are the three resource kinds kept apart? | Hand-written |
| [`observations.md`](observations.md) | What is an observation, what do the nine paper views mean, and what is deliberately out of scope (no moving sensors, no noise model, no active-learning loop)? | Hand-written |
| [`observation_reference.md`](observation_reference.md) | Which observation protocols exist in which of the four namespaces, with what parameters and defaults, and what are the frozen views' exact counts? | **Generated by [`gen_docs.py`](../gen_docs.py)** |
| [`methods_card.md`](methods_card.md) | Which baselines and comparators exist, and what does each adaptation claim relative to its source? | Hand-written |
| [`model_parameter_reference.md`](model_parameter_reference.md) | Which models exist, and what is every validated parameter, default, range, constraint and preset? | **Generated by [`gen_docs.py`](../gen_docs.py)** |
| [`tasks_and_permissions.md`](tasks_and_permissions.md) | What is each task allowed to see, and why is a target being present in a batch not authorization to feed it to the model? | Hand-written |
| [`scoring.md`](scoring.md) | What does `pdeobs-strict-v1` compute, what invalidates a block, and what does a result record about its own provenance? | Hand-written |
| [`results_format.md`](results_format.md) | What will a result artifact look like, what is the long-format schema, and why does this tree contain no scored result file and no leaderboard? | Hand-written |
| [`reproducing_the_paper.md`](reproducing_the_paper.md) | What exactly is the selected protocol, how is the identity split constructed, and how is one row executed end to end? | Hand-written |
| [`checkpoint_compatibility.md`](checkpoint_compatibility.md) | What is inside a model artifact, what is preserved across versions, and how is a legacy checkpoint loaded? | Hand-written |
| [`claims_and_limits.md`](claims_and_limits.md) | What does the release claim, what does it not, and which documentation strings in this tree are known to be stale or inconsistent? | Hand-written |
| [`extending.md`](extending.md) | How do I add a mask, a model, a PDE or a metric without forking, and what must my extension test? | Hand-written |
| [`testing_checklist.md`](testing_checklist.md) | Which twelve test categories must exist for a PDE benchmark, where does each live here, and what is deliberately *not* a software test? | Hand-written |

> **`model_parameter_reference.md` and `observation_reference.md` are generated.**
> [`gen_docs.py`](../gen_docs.py) imports `pdeobs.api.specs` and `pdeobs.api.observation`
> and writes each file from a different source: the model reference comes from
> `specs.MODEL_SPECS` in [`src/pdeobs/api/specs.py`](../src/pdeobs/api/specs.py), the
> observation reference from `observation.describe_observations()` in
> [`src/pdeobs/api/observation.py`](../src/pdeobs/api/observation.py). Do not hand-edit
> them: an edit would be lost on the next regeneration, and the point of generating them
> is that the published tables cannot drift from the validated constructor contract.
> Change the spec, then re-run `python gen_docs.py`.

---

## `tests/`

38 `test_*.py` modules, plus [`check_base_only.py`](../tests/check_base_only.py), which
is not a pytest module but a script for a genuine no-PyTorch environment (it refuses to
run if torch is installed). `ls tests/` also shows a `__pycache__/` directory, which is
residue rather than test material.

These files cover the contracts a reviewer is most likely to want to check first:

| File | What it pins |
|---|---|
| [`test_benchmark_essentials.py`](../tests/test_benchmark_essentials.py) | The minimal must-have set: one fast case per mandatory category, from a manufactured-solution solver check to a tampered-weight refusal. |
| [`test_strict_score.py`](../tests/test_strict_score.py) | The scoring contract: known answers, and the failure modes that invalidate a block rather than shrink its denominator. |
| [`test_release_contracts.py`](../tests/test_release_contracts.py) | The release-level promises, including that the random-density views are *not* claimed nested. |
| [`test_paper_row.py`](../tests/test_paper_row.py) | The executable row: identity manifest, disjoint split, final-checkpoint reload. |
| [`test_one_setting.py`](../tests/test_one_setting.py) | The frozen campaign: PDE order, method slots, free-rollout contract, preflight subset. |
| [`test_splits.py`](../tests/test_splits.py) | Regime balance and the exact deterministic identity split. |
| [`test_numerics.py`](../tests/test_numerics.py) | The shared numerical kernels underneath every family. |

The mapping from the twelve mandatory categories (A–L) to the minimal case and the
deeper file is the table in [`docs/testing_checklist.md`](testing_checklist.md).

---

## `release/` and `acceptance/` — two kinds of evidence, neither of them a result

**Neither directory contains a paper table entry.** Both hold software evidence about
the code; no number in either is a benchmark result. Numbers do appear — training
`loss` values in the T13 tier files, per-model parameter counts, peak memory and
per-step seconds in [`ACCEPTANCE_SUMMARY.md`](../ACCEPTANCE_SUMMARY.md) — and they come
from runs of a few optimizer steps, not from a trained model on a held-out split.

### `release/` — what this candidate is, and the frozen CPU scope

| File | What it is | What it is *not* |
|---|---|---|
| [`candidate_metadata.json`](../release/candidate_metadata.json) | Candidate metadata: status, the scoring version, the four evidence-level names, the rule that registry aliases and wrappers are not distinct algorithms, and the explicit `full_dataset_included: false` / `production_checkpoints_included: false` / `public_upload_verified: false` flags. | A publication record. No download URL is presented as available. |
| [`test_scope.json`](../release/test_scope.json) | The frozen selection manifest for the portable core: the selected files and nodes, and the redacted exclusions with their categories. Its own `status` field reads `selection_manifest_not_an_execution_receipt`. | A pass count. The file states that the selection count is not a pass count and that excluded tests are never counted as passed. |
| [`validation_status.json`](../release/validation_status.json) | The CPU validation record: the portable-core run and a smaller minimum-dependency-floor run, each with its counts, its interpreter version and a free-text `platform` string; which interface flows were executed and which were not; and a preserved record of an initial failing run. **Maintainer check:** the two `platform` strings name the author's operating system (`"Windows CPU, Python 3.12.8"`, `"Windows CPU, Python 3.10.21"`). That is environment detail, not a result; decide deliberately whether it stays in an anonymous artifact. | A claim about the full campaign. It states that counts across scopes are overlapping, not additive. |
| [`run_core_tests.py`](../release/run_core_tests.py) | The runner that executes that frozen scope reproducibly: it pins numerical and Torch threads to one, hashes itself, the scope file and every selected test file, embeds the not-executed exclusions in the receipt, and downgrades its own status if anything failed or did not run. | A CI system. It is a command a reviewer runs, writing a log, JUnit XML and an execution receipt without deleting earlier output. Note that the receipt it writes goes through `provenance.py` and will contain the running machine's hostname. |

### `acceptance/` — two software sweeps of the v0.2.0 facade

Run on a CUDA node. Every training run in this directory is a few optimizer steps.

**Maintainer check before publication.** Unlike the rest of the tree, this directory
carries environment and operational detail copied from the node that ran the sweeps:
[`FULL_COVERAGE_REPORT_round1.md`](../acceptance/FULL_COVERAGE_REPORT_round1.md) quotes
a crash traceback containing an absolute remote-node path under `/workspace/...`; both
round reports name the exact GPU model, its driver build and its memory size, refer to
the node's hosting arrangement, and cite the harness by its path inside the authoring
workspace. None of that is a result, and none of it is needed to read the evidence.
Review and scrub it before the tree is shipped for anonymous review.

| Path | What it is |
|---|---|
| [`README.md`](../acceptance/README.md) | The scope statement and the T1–T14 tier index (generation coverage, observation protocols, task × model interfaces, parameter validation, all views at 128², boundary × regime smoke, splits and budgets, artifact/inference refusals, every CLI command, CPU/CUDA parity, determinism, the frozen scope plus the API tests, timing/memory, and upstream-wrapper gating). |
| [`FULL_COVERAGE_REPORT_round1.md`](../acceptance/FULL_COVERAGE_REPORT_round1.md) | The first sweep. Its own headline is "Not clean": it enumerates the facade defects it found and, separately, the harness defects, judging the kernel-side rejections the rig hit to be correct behaviour. Two tiers (T7, T8) crashed. |
| [`FULL_COVERAGE_REPORT_round2.md`](../acceptance/FULL_COVERAGE_REPORT_round2.md) | The same 14-tier sweep after the fixes, which are confined to `src/pdeobs/api/`; kernels, masks, splits and the strict scorer were untouched. Each round-1 defect is paired with a named regression test and the tier result cited as its closing evidence. |
| [`round1/`](../acceptance/round1/), [`round2/`](../acceptance/round2/) | The raw per-tier JSON, the essentials output and the sweep log for each round. The two rounds are asymmetric by construction: the two round-1 crashes left no per-tier JSON, so `round1/` has twelve tier files and `round2/` has fourteen. |
| [`harness/`](../acceptance/harness/) | The sweep itself and the launcher used on the node. |

The 500-epoch figures in the round reports are **time estimates**, not measured training
runs: `round2/T13.json` records them as "extrapolated from 3x2 timed steps at 128^2 on
this GPU; no training performed beyond that", and the fields are wall-clock (`epoch_s`,
`epochs_500_h`). The two rounds produce different estimates for the same model
structures — round 2 is 1.6–3× slower per step — and the round-2 report attributes the
gap to host-side load, labelling both totals node-specific estimates. The two rounds are
not, however, two runs of identical code: round 1 tested the released tree at one commit
and round 2 tested the same tree plus the `api/`-only facade fixes, as each report's
header states. [`ACCEPTANCE_SUMMARY.md`](../ACCEPTANCE_SUMMARY.md) at the top level
summarizes a separate, earlier three-level acceptance and predates both sweeps.

---

## `examples/`

`examples/` contains exactly one file,
[`release_workflows.py`](../examples/release_workflows.py). It is a thin argument
parser that forwards to the `pdeobs demo` entry point, so
`python examples/release_workflows.py E1 --data-root ... --output-root ...` runs the
same demo as `pdeobs demo --example E1 ...`. It is not a larger example suite, and it
contains no standalone worked example of the Python API — for that, read the module
docstring of [`src/pdeobs/api/__init__.py`](../src/pdeobs/api/__init__.py) and the
demos in [`src/pdeobs/release_demo.py`](../src/pdeobs/release_demo.py).

---

## Packaging a copy of this tree

The release is the tracked file set. Three kinds of local artifact can appear in a working
directory and are **not** part of it: version-control internals, Python bytecode caches
(`__pycache__/`, `*.pyc`) and a test cache (`.pytest_cache/`). [`.gitignore`](../.gitignore)
excludes the last two from version control, which does not exclude them from a directory copy or a
zip export, so a copy made by archiving a working directory should exclude all three explicitly:

```bash
git archive --format=zip --output ../pdeobs.zip HEAD   # tracked files only
# or, when copying a directory:
#   exclude .git/, __pycache__/, *.pyc and .pytest_cache/
```

Two further items are not deletions and are noted in place above: the `platform` strings in
[`release/validation_status.json`](../release/validation_status.json), and the node, driver and
path strings in the two `acceptance/` round reports, which are kept deliberately so the evidence
stays checkable ([Claims and limits](claims_and_limits.md)). Separately,
[`src/pdeobs/provenance.py`](../src/pdeobs/provenance.py) writes `socket.gethostname()` into every
receipt it produces, so any evidence *generated later* from this tree needs the same check before
it is attached to an anonymous submission.

---

## Known stale pointers, and what no test enforces

### Stale evidence paths

Some documentation in this tree points at evidence directories that do not exist here.
[`README.md`](../README.md), [`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md),
[`ACCEPTANCE_SUMMARY.md`](../ACCEPTANCE_SUMMARY.md) and
[`docs/testing_checklist.md`](testing_checklist.md) have been repointed at
[`acceptance/`](../acceptance/), and both round reports now give the shipped location
(`acceptance/round1/`, `acceptance/round2/`) next to the `qa_full/` and `logs/` paths the
harness wrote on the node, so a reader can reconcile the two. Outside Markdown,
[`acceptance/harness/remote_full_coverage.py`](../acceptance/harness/remote_full_coverage.py)
writes its tier JSON to a `qa_full/` path on the remote node, which is where those names
came from. The evidence as shipped is under [`acceptance/`](../acceptance/). Those
pointers are stale, not missing evidence;
[`claims_and_limits.md`](claims_and_limits.md) gives the full reconciliation.

### Namespace count

Four observation namespaces are resolved by
[`src/pdeobs/api/observation.py`](../src/pdeobs/api/observation.py), and the generated
[`observation_reference.md`](observation_reference.md) documents four. Read the generated
reference; an earlier README line said two and predated `custom` and `stored`.

### What no test enforces

**No test asserts the correspondence between this page and the filesystem.** There is no
`tests/test_repository_map.py`, and nothing in the suite fails when a file named here is
renamed, moved or deleted. This page is therefore hand-maintained documentation that can
drift, and a mismatch between it and the tree should be treated as a defect in this page
until such a test exists.

What a reader can check independently, and how:

| Claim on this page | How to check it |
|---|---|
| Ten `.py` files in `src/pdeobs/pdes/`: seven families plus `__init__.py`, `common.py`, `numerics.py` | directory listing |
| One file in `examples/` | directory listing |
| 38 `test_*.py` modules plus `check_base_only.py` in `tests/` | directory listing |
| Per-directory file counts in the `configs/` table | directory listing |
| Twelve tier files in `acceptance/round1/`, fourteen in `acceptance/round2/` | directory listing |
| Ten condition settings | count `SETTING_REGISTRY.register` in [`src/pdeobs/settings.py`](../src/pdeobs/settings.py) — not visible from a directory listing |
| Nine mask factories | count `MASK_REGISTRY.register` in [`src/pdeobs/masks.py`](../src/pdeobs/masks.py) — not visible from a directory listing |
| Seven PDE families, three static and four temporal, four boundary names | read `STATIC_FAMILIES`, `TEMPORAL_FAMILIES` and `BOUNDARY_NAMES` in [`src/pdeobs/pdes/__init__.py`](../src/pdeobs/pdes/__init__.py) |
| Four observation namespaces | read [`observation_reference.md`](observation_reference.md) or the resolver in [`src/pdeobs/api/observation.py`](../src/pdeobs/api/observation.py) |
