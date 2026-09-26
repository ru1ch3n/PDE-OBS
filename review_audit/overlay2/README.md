# Overlay 2 logs (CPU, this tree)

Real logs of the overlay-2 verification, copied verbatim except that local absolute paths, the user
name and the host name are replaced by the placeholders in `PATH_MAPPING.json`. Software evidence
only; nothing here is a paper result.

| file | what |
|---|---|
| `final-pytest.log`, `final-junit.xml` | the full shipped suite on this tree (environment header, per-test results, summary, exit code) |
| `baseline-before-overlay2-pytest.log` | the same suite on the unmodified overlay-1 input, run first |
| `ai-smoke-003.log`, `ai_smoke_receipt-003.json` | the bounded CPU smoke flow after the code changes (fresh directory) |
| `evidence-demo/` | `python -m pdeobs.evidence` plan / check / refused paper export / demo export on the Poisson (recovery) and Heat (rollout) demo rows |
| `anon-tree-pytest.log`, `anon-tree-junit.xml` | the full suite on the anonymous review tree itself (overlay 2 plus the archived-results tool and index), CUDA masked |
| `r2-full-pytest.log`, `r2-full-junit.xml` | the full suite after the second-audit corrections (final committed tree), CUDA masked |
| `second-audit-reviewer-tests.log` | the independent reviewer's unmodified regression file plus the shipped evidence tests, run against the corrected tree |
| `install-check.log` | wheel build from a cache-free copy, non-editable install into a fresh venv, installed-package checks with PYTHONPATH unset |
| `final-doctor.json`, `baseline-doctor.json` | read-only doctor: 143 protected science files unchanged, indexed paths present |
| `probe-save-launders-float-frame.txt`, `probes-*.json` | the probes that re-verified the overlay-1 fixes and exposed the PredictionBundle gap fixed in overlay 2 |

Counts in these logs overlap with every earlier acceptance record and are never added
(`docs/ai/ACCEPTANCE_INDEX.md`).
