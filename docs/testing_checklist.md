# Testing checklist for a PDE benchmark (what must exist, and where it lives here)

A benchmark's software claims are only as good as the tests that pin them. The categories below are
the ones that reviewers and downstream users rely on; each has a **minimal** test in
`tests/test_benchmark_essentials.py` (one fast case per category) and, where the repository already
had deeper coverage in the frozen 521-test scope, that file is named too. Full-coverage evidence from
the single-GPU sweep is under `acceptance/` (`FULL_COVERAGE_REPORT_round1.md`, `FULL_COVERAGE_REPORT_round2.md`).

| # | Category | Why it is mandatory | Minimal test (essentials) | Deeper coverage already in repo |
|---|---|---|---|---|
| A | **Solver correctness** | The data *is* the benchmark; a wrong solver poisons every result | manufactured Poisson solution (2nd-order FD error), single-mode heat decay exact | `test_numerics.py`, `test_pdes.py`, `test_quality*.py`, demo E4 |
| B | **Data determinism** | Others must regenerate byte-identical records from a seed | same seed → identical arrays and identity hash; different seed → different | `test_generation.py`, `test_dataset.py`, shard SHA-256 sidecars |
| C | **Protocol freeze** | Split, masks and identity sets must never drift silently | paper nine-view counts at 128² exact; `stable_split` 1800/200, receipt hash stable, seed-sensitive | `test_protocol.py`, `test_one_setting.py`, `test_masks.py`, `protocol --check` |
| D | **Scorer known answers** | The metric must be trusted more than any model | perfect → 0, zero → 1, NaN/missing identity → *invalid* (denominator never shrinks), batch-invariant per-identity scores | `test_strict_score.py`, `test_metrics.py`, `test_evaluation.py` |
| E | **Baseline sanity** | Trivial baselines anchor the scale; a learned model must at least be comparable | zero-fill scores exactly 1; nearest < zero; mean ≤ zero | `test_methods.py` (interpolation/ROM) |
| F | **Leakage** | Train ∩ test must be empty for every split path | paper / holdout / stored splits disjoint; explicit overlap rejected | `test_paper_row.py` (identity manifest, disjoint demo split) |
| G | **Interface contracts** | Every public entry point must resolve and reject invalid input with a useful message | every model resolves for its declared tasks and is refused for others; every observation protocol resolves and rejects conflicting params | `test_api_specs.py`, `test_cli.py`, `test_config.py`, `test_registry.py` |
| H | **Model coverage** | Every model × task must instantiate, produce the target shape, and back-propagate finite gradients | forward+backward finite for all neural models (recovery/rollout) | `test_methods.py`, `test_training.py`, `test_pino.py`; GPU sweep T4/T5 |
| I | **Reproducibility** | Same seed → same weights (CPU bitwise; CUDA within tolerance and documented) | two CPU runs give identical weight hashes | GPU sweep T10 (CPU/CUDA parity), T11 (determinism) |
| J | **Provenance / tamper** | Results must bind to exact code, data and weights | manifest carries weight SHA-256 and kernel version; a tampered weight file is refused | `test_release_contracts.py`, strict contracts (`checkpoint_id`, `dataset_manifest_sha256`) |
| K | **Docs-as-tests** | README commands must actually run | every README one-liner parses; GPU sweep T9 executes each command from the installed package | demo E1–E6 receipts, `release/run_core_tests.py` |
| L | **Version separation** | A packaging/API bump must not relabel numerical provenance | package `0.2.0` vs kernel `0.1.0` vs API `v1` asserted | `test_dependency_floor.py`, `release/candidate_metadata.json` |

## What is *not* a software test (and is kept out of these suites)

- Convergence studies, multi-seed benchmark numbers, paper-scale (500-epoch) training: those are
  **experiments** with their own provenance (`result_update/`), never asserted by unit tests.
- Optional exact upstream wrappers: they are dependency-gated; a missing dependency is recorded as
  `dependency_blocked`, never as a pass and never substituted by a compact model.

## Resource envelope instead of training

For GPU acceptance the sweep measures **one optimizer step per paper configuration at 128²** with
the paper batch sizes and extrapolates: `epoch ≈ step_time × ceil(1800 / batch)`, `500 epochs ≈ 500 ×
epoch`. This gives an honest cost estimate for anyone planning a real run without spending GPU time
training here.

## How to run the minimal set

```bash
python -m pytest -q tests/test_benchmark_essentials.py            # ~1 min on CPU with pdeobs[train]
python release/run_core_tests.py --source-root . --output-dir ../core --basetemp ../tmp   # frozen 521
```
