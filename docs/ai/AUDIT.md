# Independent review of the uploaded ZIP (overlay 1)

English rendering of the reviewer's report on the first uploaded archive. Numbers and file
references are the reviewer's own; the report describes what was executed on the reviewer's machine
and nothing else.

## Verdict

**As the software body of a benchmark, this package is usable; as a complete review package able to
reproduce the paper's conclusions, it is not yet sufficient.** This is not an empty shell with a README:
the generators, observation operators, task adapters, models, training, target-free inference, strict
scoring and short flows all exist and all executed in this review. But the package explicitly does not
distribute production data, formal checkpoints or paper numbers, so passing software tests alone cannot
move the paper's assessment. Only the uploaded ZIP was reviewed; the authors' remote jobs were not
queried, and "not in the ZIP" does not mean "not done elsewhere".

## Scope and versions

- Input: `pdeobs-review-main.zip`, 268 files, about 2.87 MiB unpacked; no production data, weights or
  paper PDF.
- SHA-256: `ece1024b4d32b699e80a2918e77f3cd569ed57b4102dad2eb008035916ce0b79`.
- Software version 0.2.0, numerical kernel version 0.1.0; the added AI overlay is identified separately
  and does not pose as an official author release.
- Environment: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, SciPy 1.17.0, CPU only, no CUDA.
- Executed in that environment; not a minimum-dependency install and not an independent GPU
  reproduction. The RTX 5090 logs in the package are the authors' historical evidence.
- The venue's official pages were consulted but no verifiable complete guideline was obtained; this
  report is not an official admission or track certification.

## Actually executed

| Check | Result | Explanation |
|---|---|---|
| Frozen portable core | 521 passed / 0 failed / 0 skipped, 17.26 s | actually run, not copied from the acceptance documents |
| Existing easy-API tests | 42 passed, 5.57 s | overlaps the full suite |
| All shipped tests of the original package | 597 passed, 21.67 s, 5 warnings | every currently distributed test; the removed internal deployment tests are not counted |
| Non-editable build and install | succeeded | `--no-deps --no-build-isolation`, reusing already installed dependencies |
| Installed-package demos E1 to E6 | 6/6 commands succeeded | started outside the source directory; E5 includes a fresh-process reload |
| All tests of the final enhanced package | **630 passed, 23.12 s, 0 failed / 0 skipped, 5 warnings** | original 597 + 29 interface regressions + 4 tool tests; no original assertion changed |
| New AI smoke flow | Poisson recovery and Heat rollout both passed | 16 x 16, 4 train + 2 test per PDE, 2 steps, CPU; not paper results |

These counts overlap and cannot be added to claim more independent verification. The first
non-editable install failed because the fresh venv did not inherit the current environment's
dependencies; after the installed dependency path was added explicitly, build and install succeeded.
That is not attributed to a project defect, and no clean from-scratch dependency install is claimed.
The final enhanced package passed 630/630; the 143 frozen science source files and configurations are
hash-identical to the original ZIP. After the non-editable install, an easy-API evaluation against an
independent targets file also passed. Details are in `evidence_status.json`.

## Concrete strengths of the original package

1. The small quickstart in `README.md` (lines 76 to 98) is executable and needs no production data.
2. `src/pdeobs/strict_score.py` (lines 110 to 270) strictly checks identities, dimensions and
   non-finite values while keeping poor but finite predictions; the definition is not quietly changed.
3. `src/pdeobs/api/` provides unified generation, loading, parameter resolution, training, save/load,
   inference and scoring; the small flow genuinely runs end to end.
4. `configs/paper/protocol.yaml` and `observations.yaml` separate the descriptive paper protocol from
   runnable tasks, with clear fixed counts.
5. `LICENSE` and `THIRD_PARTY_NOTICES.md` are present; the statement that the learned models are
   adaptations rather than upstream reproductions is retained. The literal Markdown relative-link check
   found 0 missing targets (which does not mean every web link is valid).

## Interface problems confirmed and fixed in the copy

| # | Original location | Measured or checked | Fix |
|---|---|---|---|
| F1 | `src/pdeobs/api/data.py` 352 to 366 | a raw mask with NaN became 0, Inf became 1, 0.5 or 2 became 1, and scoring still reported valid | the mask must be finite and binary before conversion; non-finite geometry is rejected too |
| F2 | `src/pdeobs/api/pipeline.py` 53 to 57, 217 | a misspelled `stages: [trian]` ran zero steps and still returned complete | unknown, duplicate or empty stages are rejected up front |
| F3 | `src/pdeobs/api/evaluate.py` 62 to 80 | correct `targets[::-1]` with reordered IDs was misreported as an observation inconsistency | mask and observation are aligned to the independent target order |
| F4 | `src/pdeobs/api/evaluate.py` 63 to 65 | NumPy target IDs triggered a truth-value ambiguity; a `[0.9]` frame label was truncated to 0 and reported valid | explicit None checks; type-strict validation preserved |
| F5 | `src/pdeobs/api/infer.py` 75 to 101 | the prediction-bundle loader did not verify its own schema version and truncated frame labels | version and integer time labels are checked |
| F6 | `src/pdeobs/api/pipeline.py`, after evaluate | the code wrote `complete` even when the report was invalid; a failure-injection test was added | an invalid score propagates to a failed pipeline receipt |

F1 to F4 have before/after probe JSON; F5 and F6 have targeted regression tests. The F6 failure
injection is a state-propagation unit test, not a claim that real training showed that fault. These
problems sit at the API boundary; they do not mean the original paper's predictions were all wrong,
nor that the underlying strict scorer is invalid. The patch changes only five `api/` files; no numerical
kernel, mask generation, split algorithm, training loss, formal configuration or `strict_score.py`.

## Not yet closed by this ZIP

### P0: a real paper-evidence path for reviewers

The README states that no data is bundled, that there is no default download endpoint, that the data
licence is not covered by the MIT code licence, and that the release carries no paper result numbers.
Provide at least accessible materials for the evaluated data slice, or a verified generation manifest
that recovers the same physical identities and checksums; running a handful of newly generated samples
does not reproduce the old paper data. Also provide the production checkpoints linked to the main
table, per-sample predictions and scores, manifests and configurations. The whole 560k-record corpus
need not be in the archive. Without these materials, what can be verified is the software route, not
the 1620 blocks of the original main table.

### P1: the acceptance records need one unified entry

`ACCEPTANCE_SUMMARY.md` describes 38 API tests, 64 x 64, 49 flows and two general views;
`RELEASE_NOTES_v0.2.0.md` describes 521 + 37 tests, 128 x 128, batch 1 and nine paper views;
`acceptance/FULL_COVERAGE_REPORT_round2.md` describes 42 API tests and a tiered acceptance. These may
come from different rounds, but they must not be merged into one stronger complete-acceptance claim,
especially since the U-FNO documentation requires a batch size of at least 2. The added evidence map
keeps each source separate; the authors should update the root summary to point at concrete tiers and
file hashes without overwriting the historical logs.

### P1: experimental controls and numerical validation are research evidence, not more unit tests

840 small-grid generations succeeding does not prove the independent numerical accuracy of the
original production targets; the ZIP has no complete reference-comparison report that can be linked to
the main table. Control experiments must be judged from actual run outputs. Note that `mean` in
`methods/interpolation.py` (lines 87 to 103) is a per-sample observed-mean fill, not a training-population
mean fitted over full fields; do not report the earlier suggested TRAIN-mean control as done.

### P2: the default interfaces hold semantic traps for assistants

`easy train` defaults to `split=all`; `preset=paper` is not a C500 paper result; general, paper, custom
and stored are different observation namespaces; the internal key `random_3pct` may carry
`ratio=0.5`; most of `configs/paper/` are descriptions, not pipelines. These distinctions are now
consolidated into the short entry documents, the machine-readable index and the AGENTS rules.

## Added material for assistants

- `AI_START_HERE.md`, `AGENTS.md`, `llms.txt`: what to read, which entry to call, no expensive jobs by default.
- `docs/ai/PROTOCOLS.md`: exact shape, namespace, count, metric and seed distinctions, so the paper
  protocol is not confused with another protocol.
- `docs/ai/repository_index.json`: task to implementation and documentation map.
- `docs/ai/evidence_status.json`: separates the authors' historical reports, this execution, and the
  scientific conclusions not yet established.
- `tools/ai_assist.py doctor --json`: read-only status and frozen science-file hash check.
- `tools/ai_assist.py context --task scoring --max-chars 24000`: bounded, line-numbered context per
  task; no scanning of secrets or logs.
- `tools/ai_assist.py smoke --out ../pdeobs-ai-smoke-001`: an explicitly authorized fixed CPU flow that
  refuses to overwrite or to write into the source directory.
- `tests/test_ai_facade_regressions.py`: 29 boundary regression cases.

Assistant support here is not another fifty-thousand-character document but short entries, real
configurations, deterministic commands, machine-readable status, and no pretence of success on failure.

## Suggested next steps

Trial the patch in a separate copy with a CPU regression run, without replacing source beneath running
jobs. Then link the completed real results into hashed manifests and supply access to the evaluated
slice and the production checkpoints. If numerical-reference and control results already exist
externally, provide those reports rather than re-running unrelated unit tests.

Judgement: the software body can support a benchmark paper; the complete scientific reproduction
materials still need to be supplied. No new theory and no foundation-model training from scratch are
required. No assessment is declared for results that have not been provided.
