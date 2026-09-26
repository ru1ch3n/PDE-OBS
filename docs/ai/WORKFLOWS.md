# Bounded assistant workflows

## 1. Environment and navigation (read-only)
```
python tools/ai_assist.py doctor --json
python tools/ai_assist.py context --task overview --max-chars 18000
python tools/ai_assist.py context --task scoring --max-chars 24000
```
The context command includes allowlisted text only and marks truncation; it never recursively dumps
private results or checkpoint bytes. `doctor` is a static environment/file check, not a model test.

## 2. Install (human-authorized dependency operation)
From this source root, use an appropriate isolated Python environment, then:
```
python -m pip install ".[train,test]"
python -m pdeobs easy list models --json
python -m pdeobs easy describe fno --json
```
This installation may download dependencies. Select a CUDA-compatible torch build separately for
GPU work. A version satisfying a lower bound is not proof of GPU compatibility.

## 3. Safe tiny end-to-end validation (explicit opt-in)
```
python tools/ai_assist.py smoke --out ../pdeobs-ai-smoke-001
```
Hard-coded scope: CPU, one thread, Poisson+Heat, 16x16, six generated records per PDE,
four TRAIN and two disjoint TEST demo identities, FNO smoke architecture, two optimizer steps
per PDE. Saves the observation-only package, reloads model weights, predicts without targets,
scores against independently supplied held-out targets, checks reload equivalence, writes a receipt.
No network, scheduler, GPU, full-corpus generation, overwrite, or paper-table insertion.
The command refuses an existing output directory and refuses the source tree or its ancestors.
Choose a new suffix for each run. Do not delete previous outputs merely to make a rerun pass.

## 4. Change a model without changing the paper protocol
Read `easy describe`, add a valid constructor mapping, and test default/legal-alternative/illegal
cases plus save-reload. Explicitly label changed architectures custom. Do not guess that `depth`,
`layers`, `width`, and `hidden` are aliases. Preserve the old test assertions.

## 5. Review a newly produced scientific result
Locate data/checkpoint/config/mask identity bindings first. Recompute strict scores from actual
predictions. Report all required identities or technical failure; don't use NaN-skipping, crop targets,
post-hoc best checkpoints, or reclassify interrupted runs as C500. Match split, observations and
metric before comparing. Uncertainty requires per-record data, not invented errors around means.

## 6. Handoff template
```
Task:
Changed paths:
Scientific semantics changed? (yes/no, explain):
Commands actually executed + exit status:
Pass/fail/skip counts (overlap disclosed):
New measurements vs archived claims:
Remaining blockers:
Next bounded command:
```

## 7. Check and export existing paper materials (read-only check, explicit export)
```
python -m pdeobs.evidence plan  --root <paper-row output dir> --out <new dir>/bundle.json
python -m pdeobs.evidence check --bundle <new dir>/bundle.json --root <paper-row output dir> --out <new dir>/check.json
python -m pdeobs.evidence export --bundle <new dir>/bundle.json --root <paper-row output dir> --out <new dir>/export
```
`plan` reads a `paper-row` receipt and lists the artifacts (existence only). `check` hashes every present
file, verifies schema versions, identity counts, checkpoint and manifest bindings, re-scores each
`predictions.h5` with the frozen `pdeobs-strict-v1` scorer against `score.json`, and checks the split receipt
against the declared cohort (original500: 1800/200, regimes 600/600/600 and 67/67/66). Each category is
reported as passed / blocked / failed on its own; a missing file is `blocked`, never fabricated. Exit 0 =
complete, 3 = incomplete, 2 = failed or bad input; stdout is one JSON document, stderr carries diagnostics.
`export` refuses an existing directory and refuses a software-demo bundle unless `--allow-demo`; large
binaries are referenced by SHA-256 unless `--copy-large`. Nothing is trained, generated or downloaded.
No production materials ship in this tree, so here the commands can only be exercised on the demo rows of
`docs/reproducing_the_paper.md` (*Small static and rollout integration runs*); see `docs/paper_evidence.md`.

## Avoid these shortcuts

Do not label `easy train split=all` + same-dataset eval as held-out accuracy. Do not use the public
mean-fill method to claim the proposed TRAIN-population mean control was completed. Do not copy
software acceptance scores or 500-epoch time extrapolations into empirical paper tables. Do not
resolve a documentation conflict by making a more impressive claim: inspect the concrete receipt.
