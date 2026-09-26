# What this release claims, what it does not, and how the evidence is labelled

Claims and limits — how to read this page.

This page consolidates, into one place, statements that are otherwise spread across
[`README.md`](../README.md), [`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md),
[`ACCEPTANCE_SUMMARY.md`](../ACCEPTANCE_SUMMARY.md),
[`acceptance/README.md`](../acceptance/README.md),
[`docs/numerical_solvers.md`](numerical_solvers.md),
[`docs/testing_checklist.md`](testing_checklist.md),
[`docs/results_format.md`](results_format.md),
[`docs/benchmark_overview.md`](benchmark_overview.md),
[`docs/methods_card.md`](methods_card.md),
[`docs/checkpoint_compatibility.md`](checkpoint_compatibility.md),
[`docs/reproducing_the_paper.md`](reproducing_the_paper.md),
[`docs/installation.md`](installation.md),
[`release/candidate_metadata.json`](../release/candidate_metadata.json),
[`release/test_scope.json`](../release/test_scope.json) and
[`release/validation_status.json`](../release/validation_status.json).

Every entry below names the file that backs it. That is deliberate: a reviewer should be
able to verify a **non-claim** exactly as easily as a claim. Where this page states
something the rest of the tree does not, it says so and marks it as supplied here.

This page adds no result. It reports no accuracy number, no leaderboard and no ranking,
because this release contains none.

> **On anonymity.** This copy is prepared for double-blind review. What is deliberately absent,
> what is deliberately retained, and the one category of artifact that must be redacted before it
> travels with a review copy are set out in
> [Anonymity: what this copy withholds](#anonymity-what-this-copy-withholds).

---

## Before you start — the things that will actually bite

*Section current as of 2026-09-22.*

Four practical constraints cause almost every first-run failure. They are hoisted above
everything else here because this repository states these facts but scatters them.

**1. The SciPy floor is load-bearing, not cosmetic.**
The core requires SciPy `>= 1.12` ([`pyproject.toml`](../pyproject.toml), line 26) because
both elliptic iterative routes call `scipy.sparse.linalg.cg` and
`scipy.sparse.linalg.minres` with the `rtol` keyword
([`docs/installation.md`](installation.md); [`README.md`](../README.md), "Dependencies and
optional features"). Pinning below a declared bound is unsupported. The release does not
catch an unsupported solver signature and silently switch algorithms, and non-convergence
raises rather than returning a partial solve
([`src/pdeobs/pdes/numerics.py`](../src/pdeobs/pdes/numerics.py), `solve_elliptic`).
**Solver errors are never caught and relabelled as successful solutions.**

**2. The core test runner needs two new directories, outside the source tree, and
separate from each other.**

```bash
python release/run_core_tests.py --source-root . --output-dir ../pdeobs-core-results --basetemp ../pdeobs-core-temp
```

Both destinations must be new, outside the source tree, and distinct
([`README.md`](../README.md), "Tests and evidence"). The runner pins numerical and
Torch threads to one, writes a log, JUnit XML and an execution receipt, and does not
delete earlier output ([`release/run_core_tests.py`](../release/run_core_tests.py)).
A small bookkeeping note: the directory names above are the ones
[`README.md`](../README.md) documents, but the run actually recorded in
[`release/validation_status.json`](../release/validation_status.json) used the shorter
spelling `../core-results` / `../core-temp`, and
[`acceptance/README.md`](../acceptance/README.md) uses `../core` / `../tmp` again. The
names are arbitrary; only the three constraints matter.

**3. A version-control binary must be on `PATH`.**
One test builds a synthetic *local* repository to check provenance collection
([`release/test_scope.json`](../release/test_scope.json) → `requirements`). The same file
states the network position precisely, and it is worth quoting rather than paraphrasing:
"No cluster account, scheduler, production dataset, or public download endpoint is
**required**." That is a statement about requirements, not about behaviour. No file in
this tree asserts the stronger property that nothing in the frozen scope *contacts* a
network, and this page does not assert it either.

**4. Every demo needs fresh, non-nested, not-yet-existing roots.**
`run_demo` resolves `<data-root>/<Ex>` and `<output-root>/<Ex>`, rejects roots that are
equal or nested, and creates both with `exist_ok=False`, so re-running a demo into an
existing directory fails rather than overwriting an earlier receipt
([`src/pdeobs/release_demo.py`](../src/pdeobs/release_demo.py)). It also forces the run
onto CPU and accepts only one or two threads. Use a fresh pair of roots each time.

One more item that is not an error but is routinely misread: `pdeobs download` has **no
default endpoint**. Omitting `--manifest` fails before any network request
([`src/pdeobs/download.py`](../src/pdeobs/download.py), `DEFAULT_RELEASE_MANIFEST_URL = None`;
[`src/pdeobs/cli.py`](../src/pdeobs/cli.py)), and `api.load_dataset(..., source="download")`
raises instead of regenerating records under a download label
([`src/pdeobs/api/data.py`](../src/pdeobs/api/data.py)). Local regeneration via
`pdeobs generate-case` is the supported route.

---

## The four evidence levels, defined

The release uses a four-level vocabulary — **implemented**, **smoke-tested**,
**reference-checked**, **paper-evaluated** — in three places, with three different degrees
of precision:

* [`release/candidate_metadata.json`](../release/candidate_metadata.json) lists the four
  terms as a bare array under `evidence_levels`, with no definitions.
* [`README.md`](../README.md) names the four terms and attaches two non-inferences ("A
  small-grid solver smoke test is not an independent convergence study. A strict scorer
  passing synthetic tests is not proof that every historical experiment was rescored."),
  but does not define them.
* [`docs/numerical_solvers.md`](numerical_solvers.md), section *What constitutes numerical
  evidence*, **does** define all four — but only for numerical/generator evidence. Nothing
  in the tree extends those definitions to models, tasks, observation views or interface
  surfaces, and nothing assigns a level to a component.

The definitions below extend the numerical-evidence ladder to every component of the
release. **The wording below and the per-component assignment in the next section are
supplied by this page**, not stated elsewhere in the tree. One of them is not merely a
generalisation and must be flagged as such: the source ladder's level 1 is "a numerical
route and error handling exist in source"
([`docs/numerical_solvers.md`](numerical_solvers.md)), **with no test requirement**. This
page adds one. That is a stricter bar than the tree's own, adopted here deliberately so
that "implemented" cannot be satisfied by unexercised code; a reader comparing the two
should expect this page's `implemented` to be the narrower label.

| Level | Definition used on this page | What it does *not* establish |
|---|---|---|
| **implemented** | The code path exists together with its error handling, and at least one unit test in the selected scope exercises it. (Stricter than the source ladder, which requires only that the route and its error handling exist.) | That it was ever run end to end, or that its output is numerically accurate. |
| **smoke-tested** | A stated small-grid or few-step end-to-end run completed and passed the selected checks: valid shapes, finite values, declared refusals honoured. This is the role of the release demos, most unit tests, and every run recorded under `acceptance/`. | Convergence, accuracy, or any performance ordering between methods. |
| **reference-checked** | A specific solution was compared against an **independent** reference — an analytic solution or an independent numerical solution — on a **stated narrow scope**, with the grid, the code path and the error reported. | A convergence rate, an order of accuracy, or accuracy at any other grid, family, boundary, code path or regime. |
| **paper-evaluated** | A result was produced under the frozen protocol — 128 × 128 fields, the `pdeobs-strict-v1` scorer, exactly 200 held-out identities per block, the declared cohort — with its scoring provenance preserved ([`configs/paper/protocol.yaml`](../configs/paper/protocol.yaml)). | — |

One accounting rule interacts with all four: "Registry aliases and wrappers are not
distinct algorithms" ([`release/candidate_metadata.json`](../release/candidate_metadata.json)
→ `algorithm_counting`). An autoregressive wrapper around a one-step model is not an
eighth method.

---

## Evidence level per component

**Nothing in this tree reaches `paper-evaluated`.** That follows from
[`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md) ("No production C500 run, 441-row
sweep, multi-seed benchmark or convergence study"),
[`release/validation_status.json`](../release/validation_status.json)
(`paper_scale_original500_training: not_executed`) and
[`acceptance/README.md`](../acceptance/README.md) ("nothing in this directory enters the
paper's tables"). The `paper-evaluated` column is therefore omitted from the tables below
rather than printed as a column of "no".

### Per PDE family

T1 is one sweep over all 840 combinations (7 families × 4 boundaries × 10 settings ×
3 regimes), at **resolution 16², one sample, and three time steps for the temporal
families** ([`acceptance/harness/remote_full_coverage.py`](../acceptance/harness/remote_full_coverage.py),
`t1_generation`). Each family therefore accounts for 120 of the 840, and "840/840" below
is the whole-sweep result, not a per-family count.

| Family | implemented | smoke-tested | reference-checked |
|---|---|---|---|
| Darcy (F0) | yes | yes — its 120 combinations inside T1's 840/840 at 16² | **no** |
| Poisson (F1) | yes | yes — T1; demos E1, E2 | **narrow, periodic, custom-array path only** — see below |
| Helmholtz (F2) | yes | yes — T1 | **no** |
| Heat (F3) | yes | yes — T1; demo E3 | **narrow, periodic, custom-array path only** — see below |
| Reaction-diffusion (F4) | yes | yes — T1 | **no** |
| Burgers (F5) | yes | yes — T1 | **no** |
| Navier-Stokes (F6) | yes | yes — T1 | **no** |

The two reference checks, stated with the code path they actually exercise:

* `test_A_poisson_manufactured_solution_periodic` imports **`solve_custom_elliptic` from
  `pdeobs.physical_inputs`** and checks the manufactured solution `sin(2πx)cos(2πy)` at
  32 × 32, asserting relative error `< 0.02`
  ([`tests/test_benchmark_essentials.py`](../tests/test_benchmark_essentials.py), line 24).
* `test_A_heat_single_mode_decay_periodic` imports **`advance_custom_heat` from the same
  module** and checks single-mode decay at 16 × 16, asserting max absolute error `< 1e-6`
  (same file, line 38).

**Neither test touches `src/pdeobs/pdes/poisson.py` or `src/pdeobs/pdes/heat.py`.** They
validate the custom-array entry points, which share kernels with the family generators but
are a different code path from the one the benchmark uses to produce datasets. Filing them
under "Poisson" and "Heat" above is a statement about the *physics* checked, not about the
generator having been reference-checked. Demo E4 separately asserts one resolved periodic
Fourier mode to `<= 1e-12` and labels its own scope "one resolved periodic Fourier mode;
not a seven-family convergence study".

Read both checks conservatively. Each is one grid, one closed-form solution, one boundary
(periodic), one family, one entry point. Neither is a convergence study, neither reports an
order of accuracy, and neither covers a bounded or obstacle route.
[`docs/numerical_solvers.md`](numerical_solvers.md) states the general form of the caution:
linear-system residuals, finite/shape checks and operator-matched PDE residuals "do not on
their own prove continuous-solution accuracy or a grid-convergence rate. In particular,
forming `b=A*u` and solving it back only demonstrates algebraic consistency." Note that
that document names only the Heat check as the release's narrow reference check; the
Poisson manufactured-solution case is also present in the essentials suite and is counted
here.

### Per observation view

All nine frozen views (R50, R65, R80, BL, LI, H, V, BD, CL) are **implemented** and
**smoke-tested**, and their exact observed-cell counts at 128 × 128 are frozen by unit test
and re-verified in the sweeps (`tests/test_benchmark_essentials.py::test_C_paper_view_counts_are_frozen`;
[`configs/paper/observations.yaml`](../configs/paper/observations.yaml); T2 `paper_exact: true`,
T5 at 128 × 128). "reference-checked" does not apply: a mask is a definition, not a solution
compared against a reference. None is paper-evaluated. The file-level caveat that the
random densities are **not** guaranteed nested
(`random_density_masks_guaranteed_nested: false`) is part of the definition, not a bug.

### Per method

| Method group | implemented | smoke-tested | reference-checked |
|---|---|---|---|
| The seven learned methods (U-FNO-2D, FNO, CNO, DeepONet, GNOT, Transolver, PINO) | yes | yes — 49 short flows, 7 models × 7 families, each "train a few steps → save → fresh reload (identical outputs) → target-free predict → strict evaluate" at 64², 4 samples, 2 optimizer steps, views `['random', 'block']` ([`ACCEPTANCE_SUMMARY.md`](../ACCEPTANCE_SUMMARY.md)); separately T3, T4, T5 in both sweeps | **no** — no output of any adapted model is compared against an upstream implementation's output |
| Classical baselines (zero, mean, nearest, bilinear, RBF, persistence, Gappy POD) | yes | yes — demos E2/E3; T3 26/26 in round 2 (9/26 in round 1, before the artifact-loader fix) | **no** |
| Commit-pinned upstream wrappers (`paper_unet`, `paper_fno`, `paper_cno`) | yes, fail-closed | **not executed in this tree**: all three are recorded `dependency_blocked`, refused by the easy API, and require eight attestation keyword arguments for direct construction (`acceptance/round2/T14.json`) | **no** |

The 49-flow wording above is [`ACCEPTANCE_SUMMARY.md`](../ACCEPTANCE_SUMMARY.md)'s own.
[`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md) describes the same 49 flows at
128 × 128 with nine observation scores and speaks of a fresh-*process* reload; inconsistency
#4 below rules that account incorrect, so it is not used here.

The adapted learned models are documented in source as adaptations: "not exact
reproductions of corresponding research papers or official repositories"
([`src/pdeobs/methods/neural.py`](../src/pdeobs/methods/neural.py)) and "independent
benchmark adapters, not copies of the upstream repositories"
([`src/pdeobs/methods/operator_networks.py`](../src/pdeobs/methods/operator_networks.py)).
A recorded upstream revision identifies **what was read**, not that the local output
matches it.

### Per task

[`configs/paper/protocol.yaml`](../configs/paper/protocol.yaml) sets
`tasks.primary: [recovery, rollout]` and
`tasks.implemented_extensions_not_primary_experiments: [forward, inverse]` — **both**
forward and inverse carry the second marking.

| Task | Level in this tree |
|---|---|
| recovery | implemented, smoke-tested (demos, T3/T4/T5, essentials); `primary` |
| rollout | implemented, smoke-tested (demo E3, T3/T5); `primary` |
| forward | implemented, smoke-tested; marked `implemented_extensions_not_primary_experiments` |
| inverse | marked `implemented_extensions_not_primary_experiments`, and **implemented only, not end to end**: strict array scoring accepts `inverse`, but strict export refuses it — "strict inference currently supports recovery, forward and rollout; inverse needs an explicit condition-coordinate adapter" ([`src/pdeobs/strict_inference.py`](../src/pdeobs/strict_inference.py), line 121). No adapter and no timeline is shipped. |

### Per interface surface

| Surface | Level |
|---|---|
| `pdeobs easy` (nine subcommands) | smoke-tested — every subcommand invoked from the installed package; T9 **30/32 in round 1, 32/32 in round 2**. The two round-1 shortfalls were `identity-manifest` and `paper-row demo` failing on a missing manifest shard, judged a harness-staging defect (`acceptance/round1/T9.json`, `acceptance/round2/T9.json`) |
| Top-level CLI (21 subcommands) | partially smoke-tested — 9 of the 21 appear in T9 (`easy`, `demo`, `doctor`, `list`, `protocol`, `generate-case`, `identity-manifest`, `paper-row`, `paper-row-demo-data`). The other 12 (`plan`, `generate`, `download`, `train`, `infer`, `eval`, `benchmark`, `aggregate`, `quality`, `analyze`, `strict-score`, `strict-infer`) have no entry in T9. `strict-score`/`strict-infer` are exercised indirectly inside the demos and by `tests/test_strict_score.py` / `tests/test_strict_inference.py` in the frozen scope. [`acceptance/README.md`](../acceptance/README.md) labels T9 "every CLI command"; on the evidence of T9's own JSON that is an overclaim, and it is listed as inconsistency #8 below |
| Python API (`from pdeobs import api`) | smoke-tested — `tests/test_api_specs.py`, `tests/test_api_workflow.py`, T3–T5 |
| Pipeline config (`pdeobs-pipeline/v1`) | implemented and validated by `api.validate_pipeline`; **no example pipeline config ships in this tree**, although [`README.md`](../README.md) shows `--config pipeline.yaml` (and [`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md) shows the same flag with the filename `pipe.yaml`) |
| The six demos E1–E6 | smoke-tested interface demonstrations. The module says so in its own docstring: "The examples are interface demonstrations, not benchmark performance evidence." Each receipt records `"scope": "small workflow demonstration"`, `"device": "cpu"`, `"gpu_requested": 0`; E5's return value says "interface validation only; not convergence"; E2, E3 **and E6** score **untrained** baselines with `checkpoint_id` literally prefixed `untrained:` ([`src/pdeobs/release_demo.py`](../src/pdeobs/release_demo.py), line 170 for the shared classical helper behind E2/E3, line 320 for E6's `untrained:demo_visible_identity`) |

---

## Software acceptance, precisely scoped

[`acceptance/`](../acceptance/) holds **two** full-coverage sweeps of the same 14 tiers,
run on a CUDA node on 2026-09-21. They are not two runs of an identical harness: three
distinct harness SHA-256 values are recorded across the rounds — one for round 1, one for
round 2's full sweep, and a third for round 2's separate T7 rerun (both
`FULL_COVERAGE_REPORT` headers, line 5). Nor is the code identical; see the round-2
comparison caveat below.

**What the 14 tiers cover** ([`acceptance/README.md`](../acceptance/README.md)):
T1 numerical generation coverage (840 family × boundary × setting × regime combinations at
16²); T2 observation protocols (general namespace plus the nine frozen views);
T3 task interfaces × models; T4 every public model with preset / custom / illegal
parameters; T5 seven main models × all views at 128²; T6 boundary × regime smoke;
T7 splits and budgets; T8 artifact and inference refusals; T9 CLI commands (the tier index
says "every CLI command"; see the interface table above); T10 CPU/CUDA parity;
T11 determinism; T12 the frozen 521-test scope plus the API tests; T13 128² timing and
memory; T14 upstream-wrapper gating.

**Round 1 was not clean.** Its own headline says so
([`acceptance/FULL_COVERAGE_REPORT_round1.md`](../acceptance/FULL_COVERAGE_REPORT_round1.md)).
**Seven of the fourteen tiers were not at full pass**: T2 81/86, T3 9/26, T4 14/16,
T5 0/14, T7 crashed, T8 crashed, T9 30/32. Five facade defects were found:

| # | Defect found in round 1 | Closing regression test | Round-2 evidence |
|---|---|---|---|
| 1 | Weight-free (classical) models could not be loaded back — the artifact loader assumed a dict where the manifest stores a null `weights` entry | `test_parameter_free_models_round_trip` | T3 26/26; essentials `test_E` passes |
| 2 | `api.evaluate_views` collided on the output directory for two general views of one protocol (`FileExistsError`) | `test_evaluate_views_accepts_same_protocol_at_two_settings` | T5 14/14 |
| 3 | Identical CPU training calls produced different weight hashes through the facade (RNG seeded after structure construction) | covered by essentials `test_I_same_seed_same_weights_cpu` | essentials `test_I` passes; T11 `cpu_bitwise: true` |
| 4 | Channel inference off by one for two-component velocity fields (6 vs 5 input channels) | `test_navier_stokes_served_representation_sets_channels` | T4 16/16, both velocity and vorticity |
| 5 | A scalar `block_shape` side length leaked a `TypeError` instead of being accepted or cleanly rejected | `test_block_shape_side_length_is_a_public_int` | T2 86/86 |

All four named regression tests are present in
[`tests/test_api_workflow.py`](../tests/test_api_workflow.py) (lines 169, 184, 203, 215),
which accounts for the API suite growing from 38 to 42 collected cases. Round 2 reports all
14 tiers complete. The fixes are confined to `src/pdeobs/api/`; numerical kernels, masks,
splits and the strict scorer are unchanged.

Round 1's report also lists **four** *harness* (test-rig) defects — against T7, T8, T9 and
T4 — and judges the kernel-side rejections they hit to be correct behaviour rather than
product bugs: `stable_split` refusing a single-regime production set, and the PINO
`paper` preset being correctly rejected at a grid too small for `modes=20`.
[`acceptance/README.md`](../acceptance/README.md) says round 1 "found five facade defects
and five harness defects"; the fifth harness defect is labelled in
[`acceptance/FULL_COVERAGE_REPORT_round2.md`](../acceptance/FULL_COVERAGE_REPORT_round2.md)
as found in round 2's own first pass, not in round 1. The correct reading is **four harness
defects in round 1, one more in round 2**; see inconsistency #7.

### What the acceptance evidence is not

* **Every training run recorded in the sweeps and in the acceptance summary is a few
  optimizer steps.** The 49-flow L2 matrix is "2 optimizer steps" per flow
  ([`ACCEPTANCE_SUMMARY.md`](../ACCEPTANCE_SUMMARY.md)). That file sits at the repository
  **root**, not under `acceptance/`, and is a separate artifact from the two sweeps — it
  predates both (see inconsistency #3). Nothing in either place is training to convergence.
* **Nothing in the acceptance directory enters a paper table.** Stated three times:
  [`acceptance/README.md`](../acceptance/README.md), the scope line of both sweep reports,
  and the boundaries paragraph of [`ACCEPTANCE_SUMMARY.md`](../ACCEPTANCE_SUMMARY.md).
* **The cost figures are extrapolations, not measurements.** Tier T13 times optimizer steps
  per configuration at 128² with the paper batch sizes and extrapolates
  `epoch ≈ step_time × ceil(1800 / batch)`, `500 epochs ≈ 500 × epoch`
  ([`docs/testing_checklist.md`](testing_checklist.md), "Resource envelope instead of
  training"). **No training was performed to produce those hours.** The tree does not agree
  with itself on how many steps were timed; see inconsistency #9.
* **The two sweeps disagree, and the code was not identical between them.** Summed over the
  same 14 structures, round 1 extrapolates **77.2 GPU-hours** and round 2 **121.8
  GPU-hours**, with peak memory essentially unchanged (19 930 MB vs 19 955 MB). The two
  rounds are built from two different commits — round 2 is the facade-fix build — so the
  correct phrasing is **"with only `src/pdeobs/api/` changed between them"**, not "on
  identical code"; the numerical kernels, masks, splits and scorer are the same. Round 2
  attributes the gap to host-side load and concludes that "the 500-epoch figures are
  node-specific estimates, not benchmark numbers". Treat them as order-of-magnitude
  planning aids on one host, nothing more. The same host effect shows up in T1: the same 840
  generation combinations took 32 s in round 1 and 79 s in round 2.
* **Bitwise accelerator determinism is host-dependent.** T11 reported `gpu_bitwise: false`
  in round 1 and `true` in round 2 — again across the two builds, with the kernels
  unchanged. The non-deterministic CUDA kernels run under `warn_only=True`, so bitwise GPU
  equality cannot be promised. **The CPU path is the reproducibility contract**
  ([`acceptance/FULL_COVERAGE_REPORT_round2.md`](../acceptance/FULL_COVERAGE_REPORT_round2.md),
  Observations).
* **Round 2's "14 of 14" was not one uninterrupted sweep.** One harness defect was found
  during round 2's own first pass and forced a separate single-tier rerun of T7 with a
  different harness build; the report's header carries two harness hashes. No single-
  invocation green sweep artifact exists.
* **Round 1's raw evidence is asymmetric.** T7 and T8 crashed and therefore left no
  per-tier JSON — only an error and traceback inside `acceptance/round1/coverage.json`.
  Round 2 ships `T7.json` and `T8.json`; round 1 does not.

---

## Test counts, and why they may not be added

| Surface | Count | Recorded in | Qualifier |
|---|---|---|---|
| Frozen portable-core scope (CPU) | 521 collected / 521 passed / 0 failed / 0 skipped / 0 not-run / 0 collection errors | [`release/validation_status.json`](../release/validation_status.json), scope `portable_core_source`, with log and JUnit SHA-256 | Pre-freeze **source** evidence; not a claim that historical deployment tests or the full campaign passed |
| Same frozen scope re-run on the accelerator node | 521 passed, 0 failed, both rounds | `acceptance/round1/T12.json`, `acceptance/round2/T12.json` | Confirms the facade work left the frozen scope untouched |
| "37 updater tests" | **no count is recorded anywhere, and no file under `tests/` is named for an updater** | [`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md) lines 42 and 49; [`docs/checkpoint_compatibility.md`](checkpoint_compatibility.md) line 49; T12's `what` string in both sweep reports | Unverifiable in this tree. See inconsistency #6 |
| Minimum-dependency-floor subset | 106 passed / 0 failed / 0 skipped, four files, **no PyTorch installed** | [`release/validation_status.json`](../release/validation_status.json), scope `declared_base_dependency_floor_source_subset` | "Base lower bounds only; not all Python/optional-backend version combinations." This subset is **contained in** the 521 scope |
| Minimal essentials set | 2 failed / 32 passed (round 1) → 34 passed (round 2) | `acceptance/round1/essentials.txt`, `acceptance/round2/essentials.txt` | **16 test functions across 12 checklist categories A–L**, parameterized up to 34 collected cases — see the note below |
| Interface test suites (`test_api_specs.py` + `test_api_workflow.py`) | 38 collected → 42 after the four regression tests | `acceptance/round1/T12.json`, `acceptance/round2/T12.json` | 30 test functions across the two files; the four added cases are exactly the round-1 regressions |
| Selection manifest for the frozen scope | 31 whole files + 20 individually named test nodes | [`release/test_scope.json`](../release/test_scope.json) | The file's own status is `selection_manifest_not_an_execution_receipt` |
| Exclusions | 5 files + 7 nodes, redacted to opaque ids with a category and a reason | [`release/test_scope.json`](../release/test_scope.json) | — |

On the essentials set, the tree's own summary is slightly wrong and this page does not
repeat it. [`tests/test_benchmark_essentials.py`](../tests/test_benchmark_essentials.py)
opens "one fast test per category", but it defines **16** functions over **12** categories:
A, C, D and G have two each; B, E, F, H, I, J, K and L have one. The accurate statement is
*at least* one fast test per category.

Three rules govern how these numbers may be used, and all three are the release's own:

1. **The counts overlap and are not additive.**
   [`release/validation_status.json`](../release/validation_status.json) sets
   `counts_are_overlapping_not_additive: true`. The 106-case dependency-floor run is a
   subset of the files in the 521-case scope. The essentials and interface suites are
   *separate* files that are **not** in
   [`release/test_scope.json`](../release/test_scope.json)'s `selected_files`, so they are
   not inside the 521 either. Do not print a single summed total.
2. **A selection count is not a pass count.** "Parameterized pytest cases determine the
   actual collected count; 31 files plus 20 nodes is a selection count, not a pass count."
   ([`release/test_scope.json`](../release/test_scope.json) → `reporting_rules`.)
3. **Excluded tests are never counted as passed.** "Excluded tests are not executed by this
   command and are never counted as passed." (same file.) Optional-dependency skips,
   collection errors, failures and unexecuted cases stay distinct in the execution receipt.

The release also preserves the record of an initial **failing** run rather than only its
green successor: one solver-version provenance regression after a package bump, resolved by
separating the package version from the unchanged numerical-kernel version
([`release/validation_status.json`](../release/validation_status.json) →
`preserved_initial_failure`).

---

## Explicitly not done

Consolidated from the places that each state one piece of it.

| Not done | Source |
|---|---|
| No production-scale (500-epoch) training run | [`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md); [`release/validation_status.json`](../release/validation_status.json) → `paper_scale_original500_training: not_executed` |
| No completed sweep of the 441-row grid; 441 is a denominator, not a completion count | [`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md); [`docs/reproducing_the_paper.md`](reproducing_the_paper.md) |
| No multi-seed study, and therefore no error bars, no confidence intervals and no significance test. One seed (20260804) governs split, mask pairing and the default training seed | [`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md); [`configs/paper/protocol.yaml`](../configs/paper/protocol.yaml) |
| No convergence study | [`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md); [`docs/testing_checklist.md`](testing_checklist.md) ("experiments … never asserted by unit tests") |
| No grid-convergence study or independent reference solution for the other five families; the two narrow periodic custom-array checks above are the whole of it | [`docs/numerical_solvers.md`](numerical_solvers.md); [`release/test_scope.json`](../release/test_scope.json) |
| No per-family accuracy against analytic solutions for Darcy or Helmholtz, and none through the Poisson/Heat **family generators** as opposed to the custom-array entry points; only algebraic residuals are stored | [`docs/numerical_solvers.md`](numerical_solvers.md); [`tests/test_benchmark_essentials.py`](../tests/test_benchmark_essentials.py) |
| No stated order of accuracy for any route. Solver ids say `fd2` / `fv2` / `spectral`; no file asserts or measures second-order convergence | [`src/pdeobs/pdes/`](../src/pdeobs/pdes/) |
| No hyperparameter-selection protocol. Validation is **zero records by design**; architecture, optimizer, scheduler and batch size come from the frozen method registry, and how those values were chosen is not documented | [`configs/paper/protocol.yaml`](../configs/paper/protocol.yaml) (`validation: 0`); [`configs/method/all_pde_source_faithful_registry.yaml`](../configs/method/all_pde_source_faithful_registry.yaml) |
| No statistical comparison procedure — no significance test, no interval, no rule for declaring one method better on the cross-view matrix | nothing in [`configs/paper/`](../configs/paper/) defines one |
| No **measured** compute or wall-clock budget for the 441-row grid. Two sets of figures exist and neither is a measurement of that grid: the extrapolated T13 envelopes, which the release labels node-specific estimates, and a `compute_planning` block for a *different* campaign, explicitly marked `status: unmeasured_planning_scenario` | [`acceptance/FULL_COVERAGE_REPORT_round2.md`](../acceptance/FULL_COVERAGE_REPORT_round2.md); [`configs/campaign/core_observation_medium.yaml`](../configs/campaign/core_observation_medium.yaml) lines 106–115 and the hard-coded copy in [`src/pdeobs/protocol.py`](../src/pdeobs/protocol.py) lines 306–315 |
| No classical-baseline execution adapter. The 126 classical blocks are an indexing scheme only; `pdeobs paper-row` raises "row must select one of the seven public learned methods" | [`src/pdeobs/paper_row.py`](../src/pdeobs/paper_row.py) line 107; [`docs/reproducing_the_paper.md`](reproducing_the_paper.md); [`docs/methods_card.md`](methods_card.md) |
| No end-to-end inverse-task evaluation. Strict scoring accepts `inverse`; strict export refuses it pending a condition-coordinate adapter | [`src/pdeobs/strict_inference.py`](../src/pdeobs/strict_inference.py); [`docs/results_format.md`](results_format.md) |
| No strict aggregator. [`docs/results_format.md`](results_format.md) states it is unwritten | [`docs/results_format.md`](results_format.md) |
| No dataset distribution. `full_dataset_included: false`, `public_download_url: null`, `public_upload_verified: false`; the downloader has no default endpoint | [`release/candidate_metadata.json`](../release/candidate_metadata.json); [`src/pdeobs/download.py`](../src/pdeobs/download.py) |
| No pre-trained checkpoints. `production_checkpoints_included: false` | [`release/candidate_metadata.json`](../release/candidate_metadata.json) |
| No continuous-integration configuration. There is no CI directory in the tree, so nothing re-verifies the 521 scope, the essentials or the API suites on each change; every count is a one-time recorded receipt | tree contents |
| No coverage measurement. `pytest-cov` is declared in the `test` extra, but no line or branch coverage figure is reported anywhere | [`pyproject.toml`](../pyproject.toml) |
| No platform matrix beyond one CPU host and one accelerator generation. "No full optional-backend / minimum-version / platform matrix"; the floor run covers base lower bounds only | [`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md); [`release/validation_status.json`](../release/validation_status.json) |
| No machine-readable failure inventory for the strict generation gate, and no quality-gate publication claim: `publication_ready` is hard-coded `False` | [`src/pdeobs/quality.py`](../src/pdeobs/quality.py) |
| `RESULT_PENDING` is declared as a protocol marker but has **no implementation**: no module in `src/` emits or reads it | [`configs/paper/protocol.yaml`](../configs/paper/protocol.yaml); [`docs/results_format.md`](results_format.md) states the same |
| **No mechanical anonymity scan in the release gate.** The tracked tree was reviewed by hand and the checks below are documented, but no scan runs alongside the other release gates | this page, [Anonymity](#anonymity-what-this-copy-withholds) |

Two further scope statements worth repeating verbatim, because they bound how the facade
may be described:

* "It adds **no** numerical algorithm and changes no loss, mask, split or scoring
  definition, does not rescore historical results, and does not run a paper-scale training
  campaign." ([`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md))
* "A command being documented or implemented does not, by itself, imply it passed a test."
  ([`README.md`](../README.md))

---

## Known internal inconsistencies a reviewer will notice

These are stated here rather than hidden. Each names the files that carry the conflict and
the reading this page treats as correct.

**1. The package version and the release-record identifiers disagree.**
[`pyproject.toml`](../pyproject.toml) and `src/pdeobs/__init__.py` both declare `0.2.0`;
[`release/validation_status.json`](../release/validation_status.json) carries
`release_id: pdeobs-anonymous-candidate-0.1.1-20260921` and its `preserved_initial_failure`
note discusses "Package 0.1.1";
[`release/candidate_metadata.json`](../release/candidate_metadata.json) uses a different,
undated-version id, `pdeobs-anonymous-candidate-20260921`.
*Correct reading:* the artifact under review is package **0.2.0**. The `0.1.1` strings are
retained text from the preceding candidate. The separately tracked **numerical kernel
version is `0.1.0`** and is deliberately independent of the package version — that
separation is itself asserted by a test (checklist category L,
`test_L_kernel_version_is_independent_of_package_version`).
[`docs/installation.md`](installation.md) is **not** part of this conflict: it already
states this page's correct reading, in the same terms, and notes that the `release_id`
carries an earlier candidate label.

**2. Some documentation points at evidence directories that do not exist in this tree.**
The front-page pointers have been repointed: [`README.md`](../README.md),
[`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md),
[`ACCEPTANCE_SUMMARY.md`](../ACCEPTANCE_SUMMARY.md) and
[`docs/testing_checklist.md`](testing_checklist.md) now name
[`acceptance/`](../acceptance/), and both sweep reports now name the shipped location
(`acceptance/round1/`, `acceptance/round2/`) alongside the `qa_full/` and `logs/` paths the
harness wrote on the node, so the two can be reconciled. What remains stale:
[`docs/testing_checklist.md`](testing_checklist.md) still names `result_update/`, and
[`release/validation_status.json`](../release/validation_status.json) refers to an
"accompanying `release_manifest.json`".
*Correct reading:* **the evidence in this tree lives under [`acceptance/`](../acceptance/)** —
`FULL_COVERAGE_REPORT_round1.md`, `FULL_COVERAGE_REPORT_round2.md`, `round1/`, `round2/`,
`harness/`. None of the other paths exists here. A consequence worth stating plainly: the
post-freeze (installed-archive) half of the validation story is **not available** in this
tree, and the original `execution.json`, `pytest.log` and `junit.xml` for the 521-case run
are not shipped — only their SHA-256 values are recorded.

**3. The acceptance summary predates both sweeps.**
[`ACCEPTANCE_SUMMARY.md`](../ACCEPTANCE_SUMMARY.md) is timestamped 2026-09-21T14:07:21Z;
round 1 started at 14:52:23Z and round 2 at 15:47:06Z. It reports "38 passed" for the
interface tests, which is the **pre-fix** count.
*Correct reading:* the shipped tree's interface suite is **42** collected cases
(`acceptance/round2/T12.json`), 38 plus the four round-1 regression tests. Read
`ACCEPTANCE_SUMMARY.md` as a snapshot of an earlier state, not as current.

**4. The release notes and the acceptance summary disagree about the L2 accelerator
matrix.**
[`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md) describes the 49 flows as
"128×128, batch 1 … nine paper observation strict scores" and as ending in a
fresh-*process* reload; [`ACCEPTANCE_SUMMARY.md`](../ACCEPTANCE_SUMMARY.md) records
"resolution 64², 4 samples, 2 optimizer steps, views `['random', 'block']`" and describes
the step as "save → fresh reload (identical outputs)".
*Correct reading:* the recorded evidence is the summary's — **64², 4 samples, 2 optimizer
steps, two views, a fresh reload**. The nine-view coverage at 128² is a *different* tier
(T5), which round 2 reports at 14/14. The release-notes sentence conflates the two.

**5. The descriptive protocol and the retained campaign disagree on row and block counts.**
[`configs/paper/protocol.yaml`](../configs/paper/protocol.yaml) states `training_rows: 441`
and `possible_learned_blocks: 3969` (7 families × 7 public learned methods × 9 training
views, each checkpoint evaluated on 9 test views).
[`configs/campaign/all_pde_one_setting_10method_9x9.yaml`](../configs/campaign/all_pde_one_setting_10method_9x9.yaml)
states `training_rows: 504`, `unique_blocks: 4536`, classical `unique_blocks: 126`,
`unique_blocks_total: 4662`, and `validate_campaign` in
[`src/pdeobs/one_setting.py`](../src/pdeobs/one_setting.py) hard-raises "campaign accounting
changed" if those four numbers differ.
*Correct reading:* **441 / 3969 is the public denominator.** The campaign file's `learned`
array (line 105) contains an **eighth** learned slot, registry key `jeno`, and the public
code path filters it out (`PUBLIC_LEARNED_METHODS` in
[`src/pdeobs/one_setting.py`](../src/pdeobs/one_setting.py), line 55). The statement that
its adapter is deliberately absent —
`implementation: private_adapter_not_copied_into_public_repository` — is in
[`configs/method/all_pde_source_faithful_registry.yaml`](../configs/method/all_pde_source_faithful_registry.yaml)
(line 117), **not** in the campaign file; the campaign file only names the slot. The
campaign file is itself marked `status: candidate_preflight_only`,
`release_eligible: false`, `formal_credit: 0`. The 126 classical blocks have no
trained-view axis and must be reported separately from the learned matrix, never merged
into it. That eighth slot also carries an anonymity cost, recorded below.

**6. "37 updater tests" is an unbacked number.**
[`RELEASE_NOTES_v0.2.0.md`](../RELEASE_NOTES_v0.2.0.md) (lines 42, 49),
[`docs/checkpoint_compatibility.md`](checkpoint_compatibility.md) (line 49) and T12's own
`what` string in both sweep reports all refer to "the frozen 521 portable tests and 37
updater tests". **No file under [`tests/`](../tests/) is named for an updater, and no count
of 37 is recorded in any receipt in this tree.**
*Correct reading:* the only verifiable figure is **521**. Treat "37 updater tests" as
retained text from an internal tree until a receipt for it is shipped. This page does not
count it.

**7. The harness-defect count in the acceptance index does not match the round-1 report.**
[`acceptance/README.md`](../acceptance/README.md) says round 1 "found five facade defects
and five harness defects". Round 1's own *Harness defects* section lists **four** (T7, T8,
T9, T4). The fifth — the paper preset on a 16² grid — is labelled in round 2's report as
found in that round's first pass.
*Correct reading:* **five facade defects and four harness defects in round 1**, plus one
further harness defect in round 2.

**8. "Every CLI command" overstates T9.**
[`acceptance/README.md`](../acceptance/README.md)'s tier index describes T9 as "every CLI
command". T9's own JSON covers the nine `easy` subcommands plus a subset of the top-level
entry points — 9 of the 21 top-level subcommands.
*Correct reading:* T9 covers **every `easy` subcommand**, not every CLI command. The
interface table above gives the 9-of-21 breakdown.

**9. The tree disagrees with itself on how many optimizer steps T13 timed.**
[`docs/testing_checklist.md`](testing_checklist.md) (line 33) says the sweep measures "one
optimizer step per paper configuration at 128²". Both sweep reports say "Extrapolated from
3x2 timed optimizer steps per configuration" (line 65 / line 64).
*Correct reading:* the sweep reports are the execution receipts and win —
**3×2 timed steps**. Either way the extrapolation multiplier dwarfs the measured interval,
which is why the resulting hours are labelled estimates rather than measurements.

**10. Smaller, but worth knowing.** [`README.md`](../README.md) previously stated
"Observations use two namespaces" while the resolver accepts four (`general`, `paper`,
`custom`, `stored`); the README now states four, and
[`docs/observation_reference.md`](observation_reference.md) opens "Four namespaces". [`configs/paper/observations.yaml`](../configs/paper/observations.yaml)
and `PAPER_VIEWS` in [`src/pdeobs/api/specs.py`](../src/pdeobs/api/specs.py) are parallel
hand-maintained copies of the nine frozen views; no module parses the YAML and no test
asserts the two agree. The count 441 also appears in
[`src/pdeobs/protocol.py`](../src/pdeobs/protocol.py) as an unrelated constant — the
observed-point count of a regular-grid protocol — and must not be conflated with the
441-row training grid.

---

## Anonymity: what this copy withholds

This copy is prepared for double-blind review. Published content is the tracked file set:
no version-control internals, no compiled bytecode and no test caches are part of it.

### Absent by design

There are no author names, no affiliations, no acknowledgements, no funding statement and no
citation or BibTeX block anywhere in this tree, and no `CITATION.cff`. The package metadata and
the licence attribute the work to a generic project name. Author identity, acknowledgements,
funding and a citation block are **deferred to the camera-ready version**; they are not omitted
as unattributed work.

### Withheld: the unreleased eighth learned slot

The campaign's learned-method array carries eight slots while the public grid has seven
(inconsistency #5 above). The eighth is an unreleased in-house method. Its registry entry in
[`configs/method/all_pde_source_faithful_registry.yaml`](../configs/method/all_pde_source_faithful_registry.yaml)
retains only the key, the category and a note; the project label, the private-archive digest, the
digests of its private source files, and its architecture and optimizer settings are withheld,
because they describe unpublished work rather than anything a reviewer can run. The key itself is
kept so that the campaign's row accounting stays checkable, and
[`src/pdeobs/one_setting.py`](../src/pdeobs/one_setting.py) filters it out of
`PUBLIC_LEARNED_METHODS`, so no public code path can reach it.

### Retained on purpose: third-party attribution

Upstream repository URLs, papers and licence terms are kept in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) and in the method registry (including one
entry recorded as `no_license_file_found_do_not_copy`). An upstream project's URL is not author
identity, and dropping required attribution would be a licence problem, not an anonymity
improvement.

### Retained on purpose: the acceptance environment

The sweep reports and the raw tier JSON under [`acceptance/`](../acceptance/) record the GPU model,
the driver build, the framework build and the container working directory of the node the sweep ran
on, together with the SHA-256 of the archive under test. Those are commodity hardware strings and a
rented container's own paths; they are kept so the evidence stays checkable, and they carry no
account, host, site or institution name. The provider, the instance and the cost were removed.

### No distribution channel is implied

The downloader has no author endpoint (`DEFAULT_RELEASE_MANIFEST_URL = None`), and the release
metadata records `public_download_url: null` with `public_upload_verified: false`. No unverified URL
is presented as an available public release.

### One runtime risk that is not a source risk

Provenance collection is deliberately detailed: `collect_provenance` in
[`src/pdeobs/provenance.py`](../src/pdeobs/provenance.py) records `runtime.hostname` (from
`socket.gethostname()`), `runtime.executable`, `runtime.platform` and any scheduler environment
variables present; the shard writer's lock file records a host environment variable, a process id
and scheduler job/array ids ([`src/pdeobs/storage.py`](../src/pdeobs/storage.py)). None of that is in
the source — it is produced **when the code runs**. Consequently:

> Any run artifact generated on a real machine — a provenance file, a demo `receipt.json`, a run
> receipt, a shard lock, a sweep log — must be checked, and redacted where needed, before it is
> shipped with a review copy.

The shard writer already strips volatile execution context from the *content identity* used for
resume (`cwd`, `executable`, `hostname`, `path`, `working_directory`, timestamps, scheduler blocks),
so redacting these fields does not change which shards are considered complete.

### The leak check that should be a release gate

The release already gates on an evidence record
([`release/validation_status.json`](../release/validation_status.json)) and a test-scope manifest
([`release/test_scope.json`](../release/test_scope.json)). A mechanical scan for host names, account
names, absolute build and run paths, institution names, e-mail addresses, and compiled or
version-control artifacts — across the shipped tree *and* across any run artifact bundled with it —
should run alongside those gates and be recorded the same way. That scan is **not** yet part of the
release gate, and adding it is listed under [Explicitly not done](#explicitly-not-done).

```bash
# no compiled bytecode or test caches in the shipped tree
find . \( -name '__pycache__' -o -name '*.pyc' -o -name '.pytest_cache' \) -print
# no absolute build paths from an author machine
grep -rIn -e 'site-packages' --include='*.md' --include='*.json' --include='*.log' .
```

---

## How a reviewer can check this page in about ten minutes

Nothing on this page needs a GPU to verify. The four fastest checks, in order:

```bash
# the two narrow reference checks and the essentials set (CPU, ~1 min with the train extra)
python -m pytest -q tests/test_benchmark_essentials.py

# the interface suites: expect 42 collected, matching acceptance/round2/T12.json
python -m pytest -q tests/test_api_specs.py tests/test_api_workflow.py

# the frozen scope: expect 521 collected / 521 passed, with a written receipt
python release/run_core_tests.py --source-root . \
  --output-dir ../pdeobs-core-results --basetemp ../pdeobs-core-temp

# the refusals this page relies on, each of which should raise rather than degrade
python -c "from pdeobs import api; api.load_dataset(source='download')"   # refuses
python -m pdeobs download                                                 # refuses: no default endpoint
python -m pdeobs paper-row --method gappy_pod                             # refuses: not one of the seven
```

The counts recorded in [`acceptance/`](../acceptance/) cannot be reproduced without the
accelerator node, but every tier's raw JSON is shipped for round 2 (and all but T7/T8 for
round 1), so each summary line on this page can be checked against its receipt by reading.

---

## Summary in one paragraph

This release ships a benchmark's **machinery**: seven PDE generator families, ten condition
settings, four boundary protocols, nine frozen observation views, seven adapted learned
methods plus classical baselines, four task interfaces, a deterministic identity split, and
a fail-closed strict scorer. It ships **evidence that the machinery runs**: a 521-case
frozen CPU scope, a minimum-dependency-floor subset, six CPU demos, and two 14-tier software
sweeps on a CUDA node whose every training run is a few optimizer steps. It ships **no
results**: no production run, no completed grid, no multi-seed study, no convergence study,
no dataset and no checkpoints. Nothing in this tree is `paper-evaluated`. Two narrow
periodic checks against closed-form solutions are the whole of the reference-checked
evidence, and both exercise the custom-array entry points rather than the family
generators. Where a number appears, this page has tried to say exactly what it is a number
*of* — and where the tree cannot back a number, as with the "37 updater tests", this page
says that instead of repeating it. The tree is **not yet anonymous**: seven findings,
including a live version-control directory whose remote names a personal account, must be
cleared before it is shown to reviewers.
