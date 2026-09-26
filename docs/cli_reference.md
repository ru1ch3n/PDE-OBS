# Command reference — every `pdeobs` command, its arguments and its exit codes

The package installs one console entry point, `pdeobs`
([`pyproject.toml`](../pyproject.toml), `[project.scripts]` → `pdeobs.cli:main`).
That entry point carries two surfaces over the same machinery:

* the **task-level commands** — generation, training, inference, evaluation,
  aggregation, the strict scoring path and the local checks
  ([`src/pdeobs/cli.py`](../src/pdeobs/cli.py));
* the **`easy` namespace** — `pdeobs easy ...`, a one-line surface that shares
  its resolver with the Python API
  ([`src/pdeobs/api/cli.py`](../src/pdeobs/api/cli.py)).

Neither surface adds a numerical algorithm, and neither changes a loss, mask,
split or scoring definition. `easy` assembles the existing generators, datasets,
masks, models, `Trainer` and `pdeobs-strict-v1` scorer; it does not reimplement
them.

## Global conventions

| Convention | Behaviour | Source |
|---|---|---|
| A subcommand is required | `pdeobs` with no subcommand is an argparse usage error | `build_parser()`: `add_subparsers(dest="command", required=True)` |
| `pdeobs --version` | prints `pdeobs <version>` and exits 0 | `parser.add_argument("--version", action="version", ...)` |
| `--set KEY=VALUE` (task level) | repeatable dotted-key override of a resolved YAML configuration | `_overrides()`, attached to exactly six parsers: `plan`, `generate`, `train`, `infer`, `eval`, `benchmark` |
| `--set KEY=VALUE` (`easy train`) | a **different** flag with the same spelling: repeatable `key=value` overrides of the `TrainingConfig`, not dotted YAML keys | `p.add_argument("--set", dest="training", ...)` in `src/pdeobs/api/cli.py` |
| Resolved configuration | the **configured-run** commands save the configuration they actually resolved: `plan` writes `<output>.resolved.yaml` and `.provenance.json`, `generate` and `generate-case` write `resolved.yaml` (plus `provenance.json`), `train` writes `resolved.yaml`, `infer` writes `inference.resolved.yaml`, `eval` writes `evaluation.resolved.yaml`, `benchmark` does so through the train/eval runs it launches, `paper-row` writes `row_resolved.json`, and `api.run` / `pdeobs easy run` write `pipeline_receipt.json` | `save_resolved_config` call sites in `src/pdeobs/cli.py` and `src/pdeobs/runner.py`; `write_report` in `src/pdeobs/paper_row.py` |
| Not every command resolves a configuration | `list`, `doctor`, `download`, `identity-manifest`, `aggregate`, `quality` and `analyze` print a listing, a check table or a result summary and save no resolved configuration; the strict commands write a `contract.json` / `score.json` report pair instead | the matching `_cmd_*` handlers in `src/pdeobs/cli.py`; `run_strict_infer` in `src/pdeobs/strict_inference.py` |
| Output format in `easy` | `--json` prints indented, sorted JSON; without it, YAML | `_emit()` in `src/pdeobs/api/cli.py` |
| Loading is not training | reading data or a model never trains, and training always needs an explicit budget | [`README.md`](../README.md); `Budget.parse` in `src/pdeobs/api/train.py` |

## Exit codes

| Code | Condition |
|---|---|
| `0` | success |
| `1` | a failed environment check (`doctor`), detected protocol drift (`protocol --check`), or an `easy` argument-shape refusal raised as `SystemExit("…")` (for example `easy infer` with `--data` but the default `--obs stored`) |
| `2` | an invalid strict score, a failed quality gate, a `paper-row` that did not complete, a failed `demo`, any uncaught error reaching `main()`, and argparse usage errors |
| `130` | keyboard interrupt (`main()` catches `KeyboardInterrupt`, prints `Interrupted` and returns 130) |

`main()` catches `Exception` and returns 2 after printing `pdeobs: error: <message>`;
for the `easy` namespace it additionally prints the offending subcommand and the
exception type. `SystemExit` is not an `Exception`, so the `easy` handlers that
raise `SystemExit("…")` for a malformed argument combination exit 1 with that
message.

An unreadable or malformed prediction bundle is recorded as an **explicit invalid
receipt plus a non-zero exit**, never as a smaller successful run:
`_cmd_strict_score` returns `0 if result["status"] == "valid" else 2`, and
`runner.run_strict_score` always writes a report, including a
`contract loading failed` invalid report. `_cmd_strict_infer` and `_cmd_paper_row`
use the same pattern. A finite but poor prediction stays valid — validity is not
a quality claim (see [Scoring](scoring.md)).

## Top-level commands

Twenty-one subcommands, grouped by the job they belong to rather than
alphabetically. All parser details are from `build_parser()` in
[`src/pdeobs/cli.py`](../src/pdeobs/cli.py).

### Data

| Command | Purpose | Required | Key optional | Exit |
|---|---|---|---|---|
| `plan` | write an explicit manifest for generation jobs (a JSONL plan plus `.resolved.yaml` and `.provenance.json`) | `--output` | `--config`, `--tier {tiny,debug,signal,medium,full}`, quality-profile flags, `--set` | 0 / 2 |
| `generate` | generate deterministic, resumable data shards | — | `--config`, `--output` \| `--root` (mutually exclusive), `--tier`, `--plan`, `--array-index`, `--array-bundle-size` (1), `--force`, `--dry-run`, `--num-workers` (1), quality-profile flags, `--set` | 0 / 2 |
| `generate-case` | generate one explicit PDE / boundary / setting / regime case | `--pde`, `--boundary`, `--setting`, `--param-regime` | `--num-samples` (100), `--root` (`data`), `--resolution` (128), `--time-steps`, `--shard-size` (100), `--seed` (20260804), `--tier` (see below), `--force`, `--dry-run`, quality-profile flags | 0 / 2 |
| `download` | download and verify a published dataset tier | `--tier` | `--manifest`, `--output` \| `--root`, `--force` | 0 / 2 |
| `identity-manifest` | hash explicit existing canonical shards and record their identities | `--data-root`, `--shards` (one or more), `--output` | — | 0 / 2 |

`download` has **no default endpoint**. Omitting `--manifest` raises
`Supply an explicit local or HTTPS release manifest with --manifest` and exits 2
**before any network request is made** (`_cmd_download`; `DEFAULT_RELEASE_MANIFEST_URL`
is `None` in `src/pdeobs/download.py`). The Python API refuses the matching case
outright: `api.load_dataset(..., source="download")` raises rather than
regenerating data under a download label.

`generate-case --tier` accepts `{tiny,debug,signal,medium,full,custom}` and has
**no default**. When it is omitted, the tier is inferred by matching
`--num-samples` against `TIER_SIZES`, falling back to `custom` when no tier size
matches (`_cmd_generate_case`, `src/pdeobs/cli.py`). `custom` is therefore both a
selectable value and the fallback label.

`--set` is attached to six parsers only (`plan`, `generate`, `train`, `infer`,
`eval`, `benchmark`). In this Data group that means `plan` and `generate` accept
it while `generate-case`, `download` and `identity-manifest` do not; `aggregate`,
`quality` and `analyze` do not either.

### Run a configured experiment

| Command | Purpose | Required | Key optional | Exit |
|---|---|---|---|---|
| `train` | train a configured recovery or rollout baseline | `--config`, **or** `--task` + `--model` + `--data` | the selector flags, `--output`, `--resume`, `--dry-run`, `--set` | 0 / 2 |
| `infer` | run checkpoint inference and stream predictions to HDF5 | `--checkpoint` / `--ckpt` (argparse-required); plus `--config` or the selectors | the selector flags, `--output` (`.h5` / `.hdf5`), `--dry-run`, `--set` | 0 / 2 |
| `eval` | evaluate a method or checkpoint | `--pred`, **or** `--config`, **or** `--task` + `--model` + `--data` | `--checkpoint` / `--ckpt` (optional), the selector flags, `--metrics`, `--output`, `--dry-run`, `--set` | 0 / 2 |
| `benchmark` | run a local configured method/split suite | `--preset` or `--config` (not both — see below) | `--tier` (`medium`), `--data`, `--output`, `--dry-run`, `--set` | 0 / 2 |

Without `--config`, `train` / `infer` / `eval` require `--task`, `--model` and
`--data`; the missing names are listed back
(`_runner()`: *"without --config, the one-line interface requires …"*).
`eval --checkpoint` is optional and is never a required alternative: with no
`--config`, `_runner()` still demands the three selector flags.
`eval --pred` is a separate prediction-file path and cannot be combined with
`--config` or `--checkpoint`; with `--pred` and no `--task`, the task defaults to
`sparse_recovery`.

`benchmark --preset` and `benchmark --config` are **not** in an argparse
mutually-exclusive group (unlike `--output` / `--root` on `generate` and
`download`, which use `add_mutually_exclusive_group()`). The rule is enforced at
runtime in `_cmd_benchmark`, which raises
`ValueError("--preset and --config are mutually exclusive")`; passing both
therefore exits 2 with `pdeobs: error: --preset and --config are mutually exclusive`,
not with an argparse usage message. Passing neither raises
`benchmark requires --preset or --config` the same way.

`benchmark` without `--config` needs one of exactly four built-in presets
(`benchmark_preset_names()` in `src/pdeobs/presets.py`):
`cno_sparse_recovery`, `fno_sparse_recovery`, `fno_world_modeling`,
`unet_sparse_recovery`.

### The strict path

| Command | Purpose | Required | Key optional | Exit |
|---|---|---|---|---|
| `strict-infer` | infer explicit manifest identities and write bound `pdeobs-strict-v1` artifacts | `--config`, `--contract`, `--data-root`, `--manifest`, `--output` | `--checkpoint`, `--device` (`cpu`) | 0 valid / 2 invalid |
| `strict-score` | validate raw predictions and score explicit identities (strict-v1) | `--predictions`, `--contract`, `--output` | — | 0 valid / 2 invalid |
| `paper-row` | execute one `original500` or explicitly reduced `demo` row | `--config`, `--data-root`, `--manifest`, `--output` | `--device` (`cpu`) | 0 complete / 2 otherwise |
| `paper-row-demo-data` | explicitly generate nine small physical records (plus `identities.json`) for row integration demos | `--pde {poisson,heat}`, `--output` | — | 0 / 2 |

`paper-row` accepts only the `original500` and `demo` cohorts; `fixed200`,
`budget200_min120` and recovered cohorts are refused until their separately
versioned attempt adapter exists (`resolve_row` in `src/pdeobs/paper_row.py`).
See [Scoring](scoring.md) and [Reproducing the paper](reproducing_the_paper.md).

### Inspection and local checks

| Command | Purpose | Required | Key optional | Exit |
|---|---|---|---|---|
| `list` | list registered extensible components | — | `--kind {all,pdes,settings,masks,methods,metrics}` (`all`), `--plugins`, `--json` | 0 / 2 |
| `protocol` | print or validate the frozen benchmark-paper contract | — | `--config`, `--check`, `--json` | 0 pass / 1 drift |
| `doctor` | verify runtime, storage, scheduler and GPU setup | — | `--cluster {local,slurm}` (`local`), `--gpu`, `--offline` | 0 pass / 1 fail |
| `demo` | run one small CPU release workflow | `--example {E1..E6}`, `--data-root`, `--output-root` | `--threads {1,2}` (1) | 0 passed / 2 failed |
| `easy` | one-line data / observation / model / train / infer / eval / run workflows | one of the nine subcommands below | see the `easy` table | 0 / 1 / 2 |

`demo` forces `CUDA_VISIBLE_DEVICES=""` and pins the thread environment variables
to `--threads`. It requires the data root and the output root to be **separate and
non-nested**, then creates `<data-root>/E<n>` and `<output-root>/E<n>` with
`mkdir(parents=True, exist_ok=False)` (`run_demo` in
[`src/pdeobs/release_demo.py`](../src/pdeobs/release_demo.py)). It is therefore the
per-example `E<n>` subdirectory — not the root — that must not already exist: the
same pair of roots is meant to be shared across E1–E6, and only re-running the
*same* example needs its two `E<n>` directories removed (or fresh roots). A failure
is recorded in `receipt.json` and re-raised — never converted into a success.

### Analysis and audit

| Command | Purpose | Required | Key optional | Exit |
|---|---|---|---|---|
| `aggregate` | validate shards and aggregate result records | `--input`, `--output` | `--validate-shards`, `--skip-checksums`, `--expected-plan`, `--group-by` (`method,task,split`), quality-gate flags | 0, **2** on gate fail |
| `quality` | audit and aggregate stored PDE losses and dataset-quality checks (also writes a sibling `.csv`) | `--input`, `--output` | `--recompute`, quality-gate flags | 0, **2** on gate fail |
| `analyze` | summarize problem difficulty from JSON/CSV metric records | `--input`, `--output` | `--config`, `--primary-metric`, `--top-k` | 0 / 2 |

`aggregate` discovers result files by four fixed names only — `metrics.json`,
`results.json`, `metrics.csv`, `results.csv` — so strict `score.json` blocks are
not collected by it.

## Shared flag groups

### The nine-flag experiment selector

Attached by `_experiment_selector()` to `train`, `infer` and `eval`:

`--task  --model  --data  --split  --mask  --pde  --boundary  --setting  --param-regime`

Defaults chosen **in code**, not by the user (`_runner()` in `src/pdeobs/cli.py`):

| Flag / output | Coded default | Note |
|---|---|---|
| `--split` | `test` for `infer`, `iid` otherwise | not the paper split; see the caveats below |
| `--mask` | `random_3pct` | the generic random protocol, **not** a paper view |
| `train` output | `runs/<model>_<normalized task>` when `--output` is absent and no `--config` is given | the task is normalized first (`normalize_task(args.task)`), so `--model fno --task recovery` writes to `runs/fno_sparse_recovery`, not `runs/fno_recovery` |
| `infer` output | `<run_root>/preds.h5`, where `<run_root>` is the checkpoint's parent, or its grandparent when the checkpoint sits in a `checkpoints/` directory | |
| `eval --pred` task | `sparse_recovery` when `--task` is absent | |

With `--config`, the selector values become dotted overrides
(`task`, `data.root`, `data.split`, `data.mask.protocol`, `data.filters.*`);
`--model` may not be combined with `--config` — use `--set method.name=…`.

### Quality-profile options (generation)

`_quality_profile_options()`, attached to `plan`, `generate` and `generate-case`:

* `--quality-profile {report,strict,publication}` — report measurements, reject
  failed calibrated checks, or require publication-grade solver/threshold evidence.
* `--max-pde-loss FLOAT` — maximum normalized PDE loss; it must be calibrated for
  the selected protocol.

The shipped default dataset profile is `report`, with the PDE-loss and divergence
thresholds deliberately left unset, because family-, boundary-, resolution- and
dtype-specific thresholds have not been independently calibrated
([`configs/dataset/default.yaml`](../configs/dataset/default.yaml)).

### Quality-gate options (audit)

`_quality_gate_options()`, attached to `aggregate` and `quality`:

* `--quality-strict` / `--strict` — fail on stored quality failures.
* `--max-pde-loss FLOAT` — fail if any PDE family's maximum normalized loss exceeds it.
* `--require-all-pdes` — require quality records for all seven built-in families.
* `--require-validated-solvers` — require every present PDE to use a solver marked
  independently validated.

A failing gate exits 2. The gate never reports publication readiness on its own:
`publication_ready` is hard-coded `False` in `assess_quality_gate`.

## The `easy` namespace

Nine subcommands (`add_easy_parser()` in
[`src/pdeobs/api/cli.py`](../src/pdeobs/api/cli.py)). Each mirrors a call in the
public Python API (`from pdeobs import api`).

| Subcommand | Required | Key options | Python API it mirrors |
|---|---|---|---|
| `easy data` | `--pde` + `--out` for `--source generate`; `--path` for `local` / `manifest` | `--source {generate,local,manifest}` (`generate`), `--boundary` (`periodic`), `--setting` (`smooth_grf`), `--regime` (`low`), `--samples` (4), `--resolution` (32), `--frames`, `--seed` (0), `--format` (`hdf5`), `--shard-size`, `--numeric k=v`, `--manifest`, `--json` | `api.create_dataset`, `api.load_dataset` |
| `easy train` | `--data`, `--task`, `--model`, `--obs`, `--out` | `--preset {smoke,paper}`, `--model-param k=v`, `--obs-param k=v`, `--epochs` \| `--max-steps`, `--batch-size`, `--split` (`all`), `--history` (1), `--horizon`, `--seed` (0), `--device` (`auto`), `--num-workers` (0), `--max-samples`, `--set k=v` (TrainingConfig), `--label`, `--json` | `api.train` |
| `easy infer` | `--model-from`, `--out`, and one of `--input` / `--data` | `--obs` (`stored`), `--obs-param k=v`, `--horizon`, `--batch-size` (4), `--device` (`auto`), `--free-geometry`, `--max-samples`, `--json` | `api.load_predictor`, `api.predict` |
| `easy eval` | `--out`, plus `--predictions` + `--targets` **or** `--model-from` + `--data` + `--obs` | `--obs-param k=v`, `--batch-size` (4), `--device` (`auto`), `--max-samples`, `--json` | `api.evaluate`, `api.evaluate_records`, `api.evaluate_views` |
| `easy run` | `--config` | `--dry-run`, `--json` | `api.run` |
| `easy validate` | `--config` | `--json` | `api.validate_pipeline` |
| `easy plan` | `--config` | `--json` | `api.plan` |
| `easy list` | positional `kind` ∈ {`models`, `observations`, `pdes`, `presets`} | `--json` | `api.list_models`, `api.describe_observations`, `api.list_pdes`, `api.list_presets` |
| `easy describe` | positional model name | `--json` | `api.describe_model` |

`--task` is restricted to `recovery`, `forward`, `inverse`, `rollout`.
`--model-param`, `--obs-param`, `--numeric` and `--set` take scalar
`key=value` pairs only; a mapping or list value, a missing `=`, an empty key or a
repeated key is refused.

`--obs` spans four namespaces, as its own help text states: a `general` protocol
name (`random`/`grid`/`block`/`line`/`horizontal`/`vertical`/`boundary`/`clustered`/`full`),
a frozen view as `paper:<view>`, a registered mask factory as `custom:<factory>`,
or `stored` (the mask carried inside an inference input package). `make_observation`
rejects any other namespace with
*"unknown observation namespace …; use general, paper, custom or stored"*
(`src/pdeobs/api/observation.py`).

Since v0.2.1 the `general` namespace also accepts a **mask spec**: a protocol with
an observed percentage or a `+`-joined mixture, e.g.
`--obs "uniform 1 x2 + sensor 20 + block 50"` or `--obs "line 30 + random 20"`.
A spec carries its own parameters, so `--obs-param` must be empty with it; the
same string is valid in a pipeline `observation: {protocol: "..."}` section and
in `data.mask.protocol` / `evaluation.observation_protocols` of a runner
configuration. See [Observation specs](observation_specs.md).

### Rules that are enforced, not defaulted

| Rule | Behaviour | Source |
|---|---|---|
| Training refuses to start without an explicit budget | `easy train` with neither `--epochs` nor `--max-steps` passes `budget=None` and fails with *"training needs an explicit budget: `{'epochs': N}` or `{'max_steps': N}`; no default long run is started"* (exit 2). | `Budget.parse`, `src/pdeobs/api/train.py`; `_cmd_train`, `src/pdeobs/api/cli.py` |
| …but the two CLI flags are not cross-checked | `_cmd_train` builds `{"epochs": args.epochs} if args.epochs else ({"max_steps": args.max_steps} if args.max_steps else None)`, so passing **both** `--epochs` and `--max-steps` silently uses `--epochs` and never reaches the *"give exactly one of budget.epochs or budget.max_steps"* branch, and `--epochs 0` is falsy and produces the missing-budget message rather than *"budget must be positive"*. The *exactly one* and *must be positive* rules are properties of `Budget.parse`, i.e. of `api.train(budget={...})`. | `_cmd_train`, `src/pdeobs/api/cli.py`; `Budget.parse`, `src/pdeobs/api/train.py` |
| An observation package carries its own mask | with `--input`, `--obs` must stay `stored`: *"an observation package carries its own mask: --obs must stay 'stored'"*. Generating another mask would hide the user's actual sensor positions. | `_cmd_infer`; `api.predict` |
| Benchmark-mode inputs need an explicit protocol | with `--data`, `--obs stored` is refused: *"benchmark-mode inputs need an explicit --obs protocol"* (exit 1). | `_cmd_infer`; `api.predict` |
| Inference never receives targets | `api.predict` takes observations only; in benchmark mode the hidden targets stay with the caller for a later `evaluate`. An untrained predictor is refused outright. | `api.predict`, `api.inference_input_from_dataset` |
| Evaluation returns a non-zero exit for a non-valid report | `easy eval` returns `0 if status == "valid" else 2`; in the nine-view mode the status is `valid` only when `all_valid` holds. | `_cmd_eval` |
| Temporal records must declare their frames | `--pde heat` without `--frames` fails: *"heat is temporal: give time_steps (number of stored frames, >= 2; the paper uses 15)"*; a static family rejects `time_steps != 1`. | `api.create_dataset` |
| Frozen views accept no parameters | `--obs paper:R65` replays the frozen definition; asking for a paper view name from the `general` namespace is a hard error, not a silent fallback. | `api.make_observation` |

In records mode, `easy eval --obs paper` (or `paper`) runs
`api.evaluate_views` over all nine frozen views and writes a `views.json` index;
any other `--obs` value evaluates that single view through `api.evaluate_records`.

## Optional extras, per command

Declared in [`pyproject.toml`](../pyproject.toml). The core dependencies are
NumPy ≥ 1.24, SciPy ≥ 1.12, h5py ≥ 3.8 and PyYAML ≥ 6.0.

| Extra | Declares | Commands that need it | Note |
|---|---|---|---|
| *(core only)* | — | `plan`, `generate`, `generate-case`, `download`, `identity-manifest`, `list`, `protocol`, `doctor`, `aggregate`, `quality`, `analyze`, `strict-score`, `easy data`, `easy list`, `easy describe`, `easy validate`, `easy plan`, demos E1–E4 and E6, and any `train` / `infer` / `eval` / `strict-infer` run whose method is a classical baseline | The classical interpolators and reduced-order methods import only NumPy, with SciPy-optional fallbacks. |
| `train` | `torch>=2.1` | anything that constructs or runs a **neural** model: `train`, `infer`, `eval` and `benchmark` with a neural method, `paper-row`, `strict-infer` with a neural checkpoint, `easy train` / `easy infer` / `easy eval` (records mode) with a neural model, and demo **E5** | `api.build` states the requirement explicitly: *"model `<name>` requires PyTorch (install pdeobs[train]); the configuration itself is valid"*. Torch is imported under a guard, so the rest of the package stays usable without it. |
| `analysis` | `pandas>=2.0`, `matplotlib>=3.7` | the analysis and plotting tooling a user builds on top of the reports | `README.md` describes this extra as adding analysis/plotting tools. No module in this tree imports `pandas` or `matplotlib`; `pdeobs analyze`, `aggregate` and `quality` write their JSON and CSV with the standard library, so they run on the core alone. |
| `paper` | `torch>=2.1`, `setuptools>=68,<81`, `requests==2.31.0` | the pinned external-model wrappers (`paper_unet`, `paper_fno`, `paper_cno`) | **The extra does not supply an upstream checkout.** Those wrappers require the documented, version-pinned external source and fail closed if it is absent or incompatible: they verify a clean checkout, its exact revision and the committed source digest before importing author code, are never substituted by a compact model, and are not served through the `easy` API. |
| `test` | `pytest>=7.4`, `pytest-cov>=4.1` | the test suites, including `release/run_core_tests.py` | Install `.[train,test]` to execute the optional neural tests rather than skip them. |
| `site` | `bibtexparser`, `beautifulsoup4` | — | Declared for tooling outside this tree; no module here imports either package. |
| `dev` | the union of the above, plus `ruff` | development | |

## Three journeys, kept apart

### 1. Install check — the six demos

```bash
pdeobs demo --example E1 --data-root ./demo-data --output-root ./demo-runs --threads 1
```

Each demo creates its own `E1`–`E6` subdirectory under **both** roots and writes a
single `receipt.json` — under the output root only
(`<output-root>/E<n>/receipt.json`; `_json(out / "receipt.json")` in
`src/pdeobs/release_demo.py`). No receipt is written under the data root. The
receipt records the environment (Python version, platform, package and dependency
versions, `cpu_threads`), the resolved data and output directories, the status,
duration, peak process RSS, `gpu_requested: 0`, `demo_source_sha256` (a digest of
the demo module) and `command` — the full `sys.argv` of the invocation. Since
`command` and the directory fields capture the invoking paths verbatim, run demos
from a location whose path you are willing to publish alongside the receipt.

The two roots must be separate and non-nested; each example's `E<n>` subdirectory
is created with `exist_ok=False`, so repeating one example requires removing its
`E<n>` directories (or using fresh roots), while running a *different* example
against the same two roots is the intended usage.

| Example | Workflow | Requirement |
|---|---|---|
| E1 | Poisson generation, HDF5 read-back, three masks on one record; asserts the stored field is unchanged | core, CPU |
| E2 | static recovery with the untrained `nearest` interpolator plus a full strict contract/score round trip | core, CPU |
| E3 | Heat rollout with `persistence`, scored jointly and per horizon | core, CPU |
| E4 | user-supplied coefficient / source / initial-state arrays passed to the existing kernels | core, CPU |
| E5 | tiny neural training, checkpoint save, reload in a **separate Python process**, strict scoring on disjoint demo test identities | `train` extra, CPU |
| E6 | a custom mask factory and a custom method registered through the public registries | core, CPU |

**What it proves:** the installed package runs end to end on CPU — generation,
observation construction, task interfaces, the checkpoint round trip, registry
extension and strict scoring all execute and produce receipts.

**What it does not prove:** nothing here is a result. The demos run at 16×16 with
a handful of records; E2 and E3 score *untrained* baselines with a
`checkpoint_id` of `untrained:<method>`; E4 declares its own scope as *"one
resolved periodic Fourier mode; not a seven-family convergence study"*; E5
declares *"interface validation only; not convergence"*. The paper's requirement
of 200 test identities is not imposed on tiny demos, and a command being
documented or implemented does not by itself imply it passed a test.

### 2. Reproduce a protocol cell — the strict row entry point

`paper-row` executes one row with the retained method registry, the exact
deterministic paper split, no validation records and a final-checkpoint reload,
then runs strict cross-view evaluations. **The number of scored views depends on
the cohort:** `resolve_row` uses
`views = list(VIEW_ORDER) if cohort == "original500" else list(row.get("demo_test_views", [row["train_view"]]))`
(`src/pdeobs/paper_row.py`), so `original500` scores all nine frozen views while a
`demo` row scores only its declared `demo_test_views` (defaulting to its own
training view). The demo configuration below declares two views and therefore
writes exactly two `evaluation/<view>/` blocks.

`paper-row` requires real records and an identity manifest; missing data fails
explicitly rather than falling back.

```bash
# integration-sized rehearsal of the same execution path (demo cohort: two views)
pdeobs paper-row-demo-data --pde poisson --output ./row-demo-data
pdeobs paper-row --config configs/demo/paper_row_static.yaml \
    --data-root ./row-demo-data --manifest ./row-demo-data/identities.json \
    --output ./row-demo-run
```

For a real cell, supply existing canonical shards, pin them with
`pdeobs identity-manifest`, and point a row configuration at them.
A row configuration uses schema `pdeobs-paper-row-v1` and accepts only the keys
`schema_version`, `campaign`, `registry`, `cohort`, `pde`, `method`,
`train_view`, `training_seed`, `run_id`, `attempt_id`, `demo_epochs` and
`demo_test_views` (see [`configs/demo/paper_row_static.yaml`](../configs/demo/paper_row_static.yaml)).

**What it proves:** that one (PDE, method, training view) cell can be executed
under the frozen protocol, and that the artifacts it produces — `row_resolved.json`,
`split.json`, `identity_manifest.json`, `training_completion.json`,
`evaluation/<view>/{contract.json,predictions.h5,score.json}` and `receipt.json` —
bind the checkpoint, manifest, configuration and seed to every scored block.

**What it does not prove:** the `demo` cohort is an explicitly reduced split and
is *not* the 1800/200 algorithm, and it does not exercise the nine-view sweep;
only `original500` and `demo` are executable at all. Budget completion is recorded as
`scientific_acceptance: not_inferred_from_budget_completion`. This release
performed no production 500-epoch run, no 441-row sweep, no multi-seed benchmark
and no convergence study, and it bundles neither the production corpus nor
trained checkpoints. See [Reproducing the paper](reproducing_the_paper.md).

### 3. Run your own configuration

Either the `easy` chain:

```bash
pdeobs easy data  --source generate --pde heat --resolution 32 --samples 12 --frames 4 --out demo/heat
pdeobs easy train --data demo/heat --task rollout --model fno --preset smoke \
                  --obs random --obs-param ratio=0.5 --max-steps 5 --out demo/fno
pdeobs easy infer --model-from demo/fno/model --data demo/heat \
                  --obs random --obs-param ratio=0.5 --out demo/preds.npz
pdeobs easy eval  --model-from demo/fno/model --data demo/heat \
                  --obs random --obs-param ratio=0.5 --out demo/score
```

or one pipeline configuration (next section):

```bash
pdeobs easy validate --config pipeline.yaml   # resolve only
pdeobs easy plan     --config pipeline.yaml   # data scale, model, budget
pdeobs easy run      --config pipeline.yaml
```

**What it proves:** that a chosen PDE, observation protocol, task, model and
budget resolve against the real constructors and run through the same masks,
`Trainer` and strict scorer the protocol path uses, with every resolved
configuration saved.

**What it does not prove:** a self-chosen configuration is not the paper
protocol. The generic defaults (`random_3pct`, `--split iid`, the generic
rollout horizons) are not the nine frozen views or the 1800/200 identity split,
so a score obtained this way is not comparable with a protocol cell. `easy train`
uses `--split all` by default, i.e. no held-out identities at all unless the
caller says otherwise.

## The pipeline configuration

`api.run` / `pdeobs easy run` execute the stages in fixed order
(`STAGES` in [`src/pdeobs/api/pipeline.py`](../src/pdeobs/api/pipeline.py)):

```text
data -> observation -> model -> train -> predict -> evaluate
```

* `schema_version` must be exactly `pdeobs-pipeline/v1`; any other value is rejected.
* Accepted top-level keys: `schema_version`, `name`, `seed`, `device`, `out`,
  `task`, `data`, `observation`, `model`, `train`, `predict`, `evaluate` and
  `stages`. Unknown top-level keys are rejected together with the allowed list.
* `stages` selects which of the six are declared; if omitted, it is the declared
  stages in canonical order.
* `model` is **resolved, not executed**: `run()` has execution blocks for `data`,
  `observation`, `train`, `predict` and `evaluate` only, and the model section is
  validated during resolution and then consumed by the `train` stage. A run that
  declares all six therefore reports six entries in `resolved["stages"]` but five
  executed entries in the receipt's `stages` map.
* `validate_pipeline` (and `api.run(..., dry_run=True)`, which returns status
  `validated`) **resolves every stage without generating data, training or
  writing**: cheap checks only — model parameters against the real constructor
  schema, grid compatibility, the observation namespace, and the budget.
* `data.source: download` is refused; the `train` stage requires the `data`,
  `observation` and `model` sections; `evaluate` in bundle mode requires a
  `predict` stage in the same run (or `mode: records`).

No example pipeline file ships in this repository, although the README's pipeline
commands name one. Here is a minimal, runnable configuration:

```yaml
schema_version: pdeobs-pipeline/v1
name: minimal-recovery
seed: 0
device: cpu
out: out/minimal
task: recovery

data:
  source: generate
  pde: poisson
  boundary: periodic
  setting: smooth_grf
  regime: low
  num_samples: 8
  resolution: 32
  seed: 0
  out: out/minimal/records

observation:
  protocol: random
  ratio: 0.5

model:
  name: fno
  preset: smoke

train:
  budget: {max_steps: 2}
  batch_size: 2
  split: all
  out: train

predict:
  out: predictions.npz
  batch_size: 4

evaluate:
  mode: bundle
  out: evaluation
```

Save it as `pipeline.yaml` and run `pdeobs easy run --config pipeline.yaml`.
The run writes `out/minimal/records/` (canonical shards plus their sidecars),
`out/minimal/train/` (checkpoints, `split.json`, `health.json`, and the model
artifact under `train/model/`), `out/minimal/predictions.npz`,
`out/minimal/evaluation/{contract.json,score.json}` and
`out/minimal/pipeline_receipt.json` — that is, the five executed stages complete
and the `model` section is resolved into the training stage.

For a temporal PDE, add `time_steps` to the `data` section (≥ 2) and set
`task: rollout`; validation refuses a temporal family without it. This
configuration is a two-optimizer-step interface exercise, not a result.

## Known interface caveats

* **The generic defaults are not the paper protocol.** The selector's
  `--mask random_3pct` is the generic random protocol, which at 128×128 has a
  convention of exactly 500 sensors — it is not the paper's 50 % training view.
  `--split iid` is not the 1800/200 identity split. The nine frozen views are
  in [`configs/paper/observations.yaml`](../configs/paper/observations.yaml) and
  are requested explicitly as `paper:<view>` (or by label, e.g. `paper:R65`).
* **The generic rollout horizons differ from the paper contract; the history does
  not.** The protocol fixes `history_steps: 1` and `horizon: 3`
  ([`configs/paper/protocol.yaml`](../configs/paper/protocol.yaml)). The generic
  rollout preset also sets `input_horizon: 1` — so the one-frame history matches —
  but it trains on `training_horizons: [1, 2]` and evaluates
  `horizons: [1, 2, 4, 8]` (`src/pdeobs/presets.py`), none of which is the fixed
  three-step contract. `easy train --history` likewise defaults to 1.
* **Both surfaces are supported; nothing is marked deprecated.** The task-level
  `train` / `infer` / `eval` / `benchmark` commands and the `easy` namespace are
  both present, and the CLI source carries no deprecation marker for either. The
  release states only that the historical `eval` / `infer` and metric APIs
  *"remain available for provenance and compatibility"*.
* **Older scores are not retroactively relabelled.** Legacy results keep their
  scoring version: the protocol records
  `legacy_results_automatically_strict_validated: false`. A legacy aggregate is
  not a substitute for a strict raw-prediction check, and the two paths differ —
  legacy `relative_l2` uses `nansum` / `nanmean`, while strict v1 invalidates a
  block that contains a non-finite entry.
* **Strict export covers three of the four tasks.** `recovery`, `forward` and
  `rollout` export; `inverse` raises *"inverse needs an explicit
  condition-coordinate adapter"*, even though strict array scoring accepts
  `inverse`.
* **The `easy` API never serves the exact upstream wrappers.** `paper_unet`,
  `paper_fno` and `paper_cno` require a pinned external checkout and explicit
  attestation arguments; requesting one through `easy` is rejected with that
  reason rather than silently substituted.
* **`geometry_channels` defaults differ by layer.** The `easy` API and the model
  specs use `1` (the paper setting); four compact constructors in
  `src/pdeobs/methods/neural.py` default to `0`. A caller reaching past the API
  gets the constructor default.

## `python -m pdeobs.evidence` (review overlay, separate module)

The evidence checker is not a `pdeobs` subcommand: the legacy command module is a protected scientific
source file, so the tool lives in its own module. `plan` reads a `paper-row` output directory and writes a
bundle manifest; `check` verifies every link read-only (exit 0 complete, 3 incomplete, 2 failed or bad
input; JSON on stdout, diagnostics on stderr); `export` copies the small artifacts and a hashed file index
into a new directory and refuses demo bundles unless `--allow-demo`. Status semantics and the two worked
blocks are in [Paper evidence](paper_evidence.md).

## See also

* [Benchmark overview](benchmark_overview.md) — what the benchmark is, in one page.
* [Reproducing the paper](reproducing_the_paper.md) — the protocol, split and cohorts.
* [Results format](results_format.md) — what a scored block contains and how tables are assembled.
* [Claims and limits](claims_and_limits.md) — what this release does and does not assert.
* [Scoring](scoring.md) — the strict contract, the manifest and the command sequence.
* [Observation protocols](observations.md) and the generated
  [observation reference](observation_reference.md).
* [Model parameters](model_parameter_reference.md) — generated from the live specs.
* [Data schema](data_schema.md) and [data and inference schema](data_and_inference_schema.md).
* [Installation and compatibility](installation.md); [Testing checklist](testing_checklist.md).
* [Tasks and permissions](tasks_and_permissions.md); [Extending PDE-OBS](extending.md).
