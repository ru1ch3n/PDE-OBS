# Install and first run

The shortest path to a real PDE solution: Python 3.10 or newer, a clean virtual
environment, a non-editable install of the core, and one command that solves and
scores a small field on a laptop CPU.

```bash
python -m venv .venv
# Activate .venv with the command appropriate to your shell.
python -m pip install .
pdeobs --help
pdeobs demo --example E2 --data-root ./demo-data --output-root ./demo-runs --threads 1
```

That demo generates a Poisson record with the built-in numerical solver, masks
it, runs an untrained classical baseline, and writes a strict `pdeobs-strict-v1`
score plus a receipt. No download, no GPU, no cluster account, no configuration
file. Nothing about installation requires the full corpus.

## Supported environment

| Requirement | Declared floor | Where it is declared | Why |
|---|---|---|---|
| Python | `>=3.10` | `pyproject.toml` (`requires-python`) | Classifiers list 3.10, 3.11 and 3.12 |
| NumPy | `>=1.24` | `pyproject.toml` | Array layer for every record, mask and score |
| SciPy | `>=1.12` | `pyproject.toml` | See the note below |
| h5py | `>=3.8` | `pyproject.toml` | HDF5 is the only canonical record format |
| PyYAML | `>=6.0` | `pyproject.toml` | Configuration and protocol files |
| PyTorch (optional) | `>=2.1`, via the `train` extra | `pyproject.toml` | Neural construction and training only |

**Why the SciPy floor exists.** Both iterative elliptic routes call
`scipy.sparse.linalg.cg` and `scipy.sparse.linalg.minres` with the `rtol`
keyword (`src/pdeobs/pdes/numerics.py`). The release does not catch an
unsupported solver signature and quietly switch algorithms or keywords;
`tests/test_dependency_floor.py` asserts that the public Darcy, Poisson and
Helmholtz entry points reach CG/MINRES with `rtol` present and `tol` absent. What
this repository can attest is exactly two things: `scipy>=1.12` is the declared
floor in `pyproject.toml`, and the floor environment recorded in
`release/validation_status.json` ran that assertion at SciPy 1.12.0. Whether an
older SciPy accepts the same keyword is an upstream question this repository does
not test. Pinning below a declared floor is unsupported.

**The two environments recorded in `release/validation_status.json`, in which the
test suites were actually run:**

| Recorded scope | Interpreter | Versions | Result |
|---|---|---|---|
| `portable_core_source` | CPU, Python 3.12.8 | NumPy 1.26.4, SciPy 1.16.0, h5py 3.16.0, PyYAML 6.0.3, pytest 9.1.1, torch 2.7.1+cpu | 521 collected / 521 passed / 0 failed / 0 skipped / 0 not-run |
| `declared_base_dependency_floor_source_subset` | CPU, Python 3.10.21 | NumPy 1.24.0, SciPy 1.12.0, h5py 3.8.0, PyYAML 6.0, no PyTorch | 106 passed / 0 failed / 0 skipped over `test_dependency_floor.py`, `test_numerics.py`, `test_pdes.py`, `test_doctor.py` |

(The JSON also carries a `platform` string per row; it is left there rather than
repeated here.) Those two rows are the whole of the environment evidence recorded
in `release/validation_status.json`. They are not the whole of the release's
evidence: [`ACCEPTANCE_SUMMARY.md`](../ACCEPTANCE_SUMMARY.md) records a separate
GPU software-acceptance environment on a CUDA node, and
[`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md) describes the same one.
That summary states its own boundary — software acceptance only, no paper table,
no convergence, no performance number.

**No broader platform, optional-backend or minimum-version matrix was run.** The
floor run covers the declared base lower bounds only, not every combination of
Python, dependency version, optional accelerator backend and operating system;
the release notes state the same under "Explicitly not done this round" in
[`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md). The two recorded counts
overlap and must not be added (`counts_are_overlapping_not_additive: true`).

Package and kernel versions are deliberately separate: `pyproject.toml` and
`src/pdeobs/__init__.py` declare package version `0.2.0`, while
`__solver_version__` stays `0.1.0`, so a packaging bump never relabels the
numerical implementation that produced a record. (The `release_id` string inside
`release/validation_status.json` still carries an earlier candidate label; the
version in the source tree is the one that identifies this code.)

## Extras, and what each unlocks

| Install | Declares | Unlocks |
|---|---|---|
| `pip install .` (core) | NumPy, SciPy, h5py, PyYAML | Numerical generation (`pdeobs generate-case`, `pdeobs easy data --source generate`), the classical baselines in `src/pdeobs/methods/interpolation.py` and `src/pdeobs/methods/reduced_order.py`, demos E1–E4 and E6, and the strict scorer (`src/pdeobs/strict_score.py` imports NumPy only) |
| `pip install ".[train]"` | `torch>=2.1` | Construction and training of the neural methods, demo E5, and the three neural categories of the fast subset below. PyTorch is imported behind `try/except ImportError` in `src/pdeobs/methods/neural.py` and `src/pdeobs/evaluation.py`, so an interpolation-only install stays usable |
| `pip install ".[analysis]"` | `pandas>=2.0`, `matplotlib>=3.7` | Downstream analysis and plotting of exported records. No module under `src/pdeobs/` imports either package, and `pdeobs analyze` reads JSON/CSV metric records without them |
| `pip install ".[paper]"` | `torch>=2.1`, `setuptools>=68,<81`, `requests==2.31.0` | The pinned external wrappers `paper_unet` / `paper_fno` / `paper_cno` in `src/pdeobs/methods/paper_official.py` |
| `pip install ".[test]"` | `pytest>=7.4`, `pytest-cov>=4.1` | The test suites below |

`pyproject.toml` also declares `site` and `dev`; neither is needed for anything
on this page.

**The `paper` extra does not supply an upstream checkout.** It installs the
dependencies those wrappers need and nothing else. Each wrapper's constructor
requires keyword-only attestation arguments — `upstream_root`,
`upstream_revision`, `source_file_sha256`, `paper_pde` and the per-PDE
architecture values — and verifies a clean commit-pinned checkout, its exact
revision and the committed blob digest before importing author code. Without a
separately obtained upstream tree the construction fails; it is never
substituted by a compact local model, and the high-level API refuses these names
outright (`src/pdeobs/api/models.py` raises for any spec with
`upstream_wrapper` set: "not served through the easy API").

## First run: the six demos

Each demo writes its own `E1`…`E6` subdirectory under both roots, forces
`CUDA_VISIBLE_DEVICES=""`, runs on one or two CPU threads, and writes a
`receipt.json` carrying the environment, package and dependency versions, the
command, duration, peak resident memory and `gpu_requested: 0`. A failure is
recorded in the receipt and then re-raised — never converted into a success
(`src/pdeobs/release_demo.py`). The fixed parameters are also listed
descriptively in [`configs/demo/workflows.yaml`](../configs/demo/workflows.yaml).

The quickstart above already ran E2 under `./demo-data` and `./demo-runs`, and a
demo refuses to re-create a directory it has already written (see the first
constraint below), so this block uses a second pair of roots and runs all six:

```bash
pdeobs demo --example E1 --data-root ./demo-data-all --output-root ./demo-runs-all --threads 1
pdeobs demo --example E2 --data-root ./demo-data-all --output-root ./demo-runs-all --threads 1
pdeobs demo --example E3 --data-root ./demo-data-all --output-root ./demo-runs-all --threads 1
pdeobs demo --example E4 --data-root ./demo-data-all --output-root ./demo-runs-all --threads 1
pdeobs demo --example E6 --data-root ./demo-data-all --output-root ./demo-runs-all --threads 1
python -m pip install ".[train]"
pdeobs demo --example E5 --data-root ./demo-data-all --output-root ./demo-runs-all --threads 1
```

| Demo | What it proves | What it does **not** prove | Requirement |
|---|---|---|---|
| **E1** | Poisson generation at 16×16 (3 samples, seed 1729, periodic, `smooth_grf`, low regime), lazy HDF5 read-back, and three masks — the `random_3pct` protocol with `count=128`, `block_missing` at 0.5, horizontal `line_sensors` at 0.5 — applied to one record; then asserts the stored field is unchanged, element for element (`np.array_equal` against a pre-mask copy) | Nothing about mask difficulty or recovery quality: no model runs and no score is produced. The receipt labels its own scope "small workflow demonstration" | Core, CPU |
| **E2** | Static recovery with the untrained `nearest` interpolator through a complete strict-v1 contract → prediction bundle → score round trip | Not a trained-model result and not a baseline comparison: the contract (`score-contract.json`) records `checkpoint_id: "untrained:nearest"` and scope "workflow demonstration, not a paper performance estimate", and the demo's own `receipt.json` records `test_role: "untrained workflow demonstration"` under `details` | Core, CPU |
| **E3** | Heat rollout (4 frames, history 1, horizon 3) with `persistence`, scored jointly and at horizons 1/2/3 | A persistence baseline is not a forecasting result; it carries the same untrained, demo-scale scope as E2 | Core, CPU |
| **E4** | User-supplied source, coefficient and initial-state arrays passed to the existing elliptic and Heat kernels, with one periodic analytic Heat mode checked to a maximum absolute error of 1e-12 | Its own return value states the scope: "one resolved periodic Fourier mode; not a seven-family convergence study". It is not a grid-convergence study and covers no other family | Core, CPU |
| **E5** | A tiny U-Net (`width=4`) trained for 2 epochs on 4 identities, checkpoint saved, then reloaded in a **separate Python process** that must report a different PID and a matching checkpoint SHA-256 before strict-scoring 3 disjoint test identities | An interface test, not a convergence result — the return value says "interface validation only; not convergence". Two epochs on four identities is not evidence about model quality | `train` extra, CPU |
| **E6** | A custom mask factory (`demo_checkerboard`) and a custom method (`demo_visible_identity`) registered through the public registries and then scored: "registry factory without changing the PDE kernel" | Registration is not certification. The method is a trivial visible-identity predictor and makes no performance claim | Core, CPU |

The demos declare their identity sets explicitly and score under the `demo`
split. The requirement of exactly 200 test identities applies only to the
`paper-test` split in `src/pdeobs/strict_score.py`; **it is not imposed on the
tiny demos.** A command being documented or implemented does not, by itself,
imply it passed a test.
[`examples/release_workflows.py`](../examples/release_workflows.py) forwards to
the same entry point.

## Two constraints to know before you re-run

**1. Each demo creates its own per-example directories, and they must not already
exist.** `run_demo` first appends the example name to both roots, then rejects a
data and an output directory *for the same example* that are equal or nested
("data and output roots must be separate non-nested locations"), then creates
`<data-root>/<E#>` and `<output-root>/<E#>` with `exist_ok=False`. Because the
check runs after the example name is appended, it compares `<data-root>/<E#>`
against `<output-root>/<E#>` and not the two roots themselves. Repeating the same
demo into a root that already holds that example fails by design, so an earlier
receipt stays identifiable rather than being silently overwritten. Use fresh
roots when repeating a demo.

**2. The core test runner's two destination directories must be new and outside
the source tree, and one test needs a version-control binary.**
`release/run_core_tests.py` refuses `--output-dir` or `--basetemp` if the path
already exists (nothing is ever deleted), if it lies inside the source tree or is
an ancestor of it, or if the two overlap each other. Separately,
`tests/test_methods.py::test_official_source_hash_uses_committed_blob` builds a
synthetic local repository, so a `git` binary must be on the `PATH`; this is the
"Git must be available for a synthetic local repository provenance test"
requirement recorded in [`release/test_scope.json`](../release/test_scope.json).
No cluster account, scheduler, production dataset or download endpoint is
required. Execution receipts embed the caller-supplied filesystem paths and a
platform string — review them before sharing.

## Generate real data without any download

This single-case entry point needs no dataset endpoint and no manifest:

```bash
pdeobs generate-case --pde poisson --boundary periodic --setting smooth_grf \
    --param-regime low --num-samples 4 --resolution 16 --seed 17 --root ./my-data
```

For a temporal family, add a legal time-step count:

```bash
pdeobs generate-case --pde heat --boundary periodic --setting smooth_grf \
    --param-regime low --num-samples 4 --resolution 16 --time-steps 5 \
    --seed 17 --root ./my-data
```

`--time-steps` must be positive (`src/pdeobs/schema.py`). For a built-in static
family it is not rejected but overridden: `generate_job` passes `time_steps=1`
whenever the family is a built-in one outside `TEMPORAL_FAMILIES` (`heat`,
`reaction_diffusion`, `burgers`, `navier_stokes`), because the built-in elliptic
families are canonically T=1; an external plugin family receives the configured
value and defines its own static/temporal contract
(`src/pdeobs/generation.py`). Note also that the `generate-case` defaults are
`--resolution 128`, `--num-samples 100`, `--shard-size 100` and `--seed
20260804`, so an unqualified invocation is far larger than the two commands
above.

A full tier is **not** an installation test, and the full-corpus configuration is
deliberately not the quickstart. `pdeobs plan` followed by `pdeobs generate`
exists for planned campaigns; see [data schema](data_schema.md). There is no
default download endpoint in this candidate: `pdeobs download` fails before
making any network request when `--manifest` is omitted ("Supply an explicit
local or HTTPS release manifest with `--manifest`"), and
`api.load_dataset(..., source="download")` raises rather than regenerating data
under a download label.

## Verify the install

Fast subset — one fast case per mandatory test category
([testing checklist](testing_checklist.md)). Install the `train` extra as well,
or three of the twelve categories will not run:

```bash
python -m pip install ".[test,train]"
python -m pytest -q -p no:cacheprovider tests/test_benchmark_essentials.py
```

Without PyTorch, `test_H_neural_model_forward_backward_is_finite`,
`test_I_same_seed_same_weights_cpu` and
`test_J_artifact_records_weight_hash_and_detects_tampering` each call
`pytest.importorskip("torch")` and are skipped, so the model-coverage,
reproducibility and provenance/tamper categories are silently not covered. The
testing checklist quotes roughly a minute on CPU with the `train` extra
installed.

**This file sits outside the recorded portable-core scope.**
`tests/test_benchmark_essentials.py` is not among the 31 files in
`release/test_scope.json`, and it is not one of the four files in the
`declared_base_dependency_floor_source_subset` row of
`release/validation_status.json`. Neither recorded run executed it, so it is not
a subset of the 521 and carries no recorded pass evidence in this release.

Full portable-core scope, from the extracted source directory:

```bash
python release/run_core_tests.py --source-root . --output-dir ../core-results --basetemp ../core-temp
```

The runner pins NumPy/BLAS and Torch threads to one, executes exactly the
selection in `release/test_scope.json` (31 whole files plus 20 individually named
nodes — a *selection* count, not a pass count), and writes `pytest.log`,
`junit.xml` and `execution.json` into the output directory without deleting
earlier output. The `execution.json` receipt records the runner's own SHA-256,
the scope file's SHA-256, a SHA-256 per selected test file, the resolved pytest
arguments, the pinned thread environment, the not-executed exclusions, the
per-outcome counts, and a status that degrades to `failed_or_incomplete` if
anything failed, errored during collection or did not run. The demo receipts
described above play the same role for the six workflows.

That same `execution.json` also records `source_root`, `output_dir`, `basetemp`,
`python_executable` and a platform string — the caller-supplied paths and host
details of whoever ran it. It is the receipt most likely to be attached to a
submission, so review it before sharing.

Previously excluded deployment, website and missing-historical-fixture tests are
omitted from this export and are **never** counted as passed. This is a
source-tree test, not a wheel-only installation test, not a complete deployment
suite, and not an independent numerical convergence study. As above: a command
being documented or implemented does not, by itself, imply it passed a test.

## Where to next

- [`README.md`](../README.md) — what the benchmark is, the pipeline from physical
  configuration to strict evaluation, and the unified `pdeobs easy` interface.
- `pdeobs --help`, `pdeobs easy --help`, and `pdeobs easy list models |
  observations | pdes | presets` — the live command surface. The
  [model parameter reference](model_parameter_reference.md) and the
  [observation reference](observation_reference.md) are generated from the same
  specifications the code resolves against, by `gen_docs.py`, which imports
  `pdeobs.api.specs` and `pdeobs.api.observation`. They are checked-in generated
  artefacts: no test regenerates and diffs them, and `gen_docs.py` is not part of
  `release/test_scope.json`, so re-run it after changing
  `src/pdeobs/api/specs.py` and treat the live `--help` and `list` output as
  authoritative if the two ever disagree.
- [Numerical solvers](numerical_solvers.md), [observations](observations.md),
  [data schema](data_schema.md), [tasks and permissions](tasks_and_permissions.md)
  and [scoring](scoring.md) — what each layer computes and what it does not
  assert.
- [Reproducing the paper](reproducing_the_paper.md) — the frozen protocol, the
  deterministic identity split and the cross-view evaluation grid.
- [`release/validation_status.json`](../release/validation_status.json),
  [`release/candidate_metadata.json`](../release/candidate_metadata.json),
  [`ACCEPTANCE_SUMMARY.md`](../ACCEPTANCE_SUMMARY.md) and
  [`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md) — what was executed,
  what was not, and what this release explicitly does not claim.
- [Extending PDE-OBS](extending.md) — custom masks, custom methods and the
  optional registry entry points.
