# Software-acceptance evidence for the v0.2.0 facade

This directory documents the **software** acceptance of the `pdeobs.api` / `pdeobs easy` facade on a
CUDA node (RTX 5090, sm_120, torch 2.7.1+cu128).  It is not a benchmark result: every training run here is a
few optimizer steps, and nothing in this directory enters the paper's tables.

* `FULL_COVERAGE_REPORT_round1.md` - first sweep of the released tree.  It found five facade defects and five
  harness defects; all are listed with root causes.
* `FULL_COVERAGE_REPORT_round2.md` - the same 14-tier sweep after the facade fixes that are included in this
  tree (`src/pdeobs/api/` only; numerical kernels, masks, splits and the strict scorer are unchanged).
  All tiers complete; frozen scope 521/521, API tests 42/42, benchmark essentials 34/34.
* `round1/`, `round2/` - the raw per-tier JSON written by the harness, the essentials test output and the
  sweep log for each round.
* `harness/` - the sweep itself (`remote_full_coverage.py`, 14 tiers T1-T14) and the launcher used on the node.

Tier index: T1 numerical generation coverage (840 PDE x boundary x setting x regime combos); T2 observation
protocols (general namespace + the nine frozen paper views); T3 task interfaces x models; T4 every public
model with presets / custom / illegal parameters; T5 seven main models x all views at 128^2; T6 boundary x regime
smoke; T7 splits and budgets; T8 artifact / inference refusals; T9 every CLI command; T10 CPU/CUDA parity;
T11 determinism; T12 the frozen 521-test scope plus the API tests; T13 128^2 timing and memory of the paper
structures; T14 upstream-wrapper gating.

Reproduce locally (CPU): `python -m pytest -q tests/test_benchmark_essentials.py tests/test_api_specs.py
tests/test_api_workflow.py`; frozen scope: `python release/run_core_tests.py --source-root . --output-dir ../core
--basetemp ../tmp`.

Two mechanical substitutions are applied to this directory when the review copy is built: the container run identifier in the on-node working path is replaced by `sweep-run`, so paths read `/workspace/pdeobs/sweep-run/...`.  Nothing else in the tier JSON, the logs or the reports is altered, and the archive SHA-256 values are the ones actually under test.
The second is the internal iteration label: the campaign it exercised was tracked internally as a numbered iteration, and that number is replaced throughout the release by the paper's own naming, so the namespace the tiers exercise reads `paper` here. Only the label changed; the tiers, the counts, the pass and fail outcomes and the archive SHA-256 values are the ones actually produced.
