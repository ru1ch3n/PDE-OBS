# Quick start for humans and assistants

Four levels, in increasing cost. Levels 1 to 3 touch no GPU, scheduler, network or paper-scale
data; level 4 lists the expensive entry points only so that nobody starts them by accident.
Every command that writes does so only into a new directory you name.

## 0. Install (one isolated environment; downloads dependencies, never data)

```bash
python -m pip install ".[train,test]"
```

## 1. Read-only checks (seconds)

```bash
python tools/ai_assist.py doctor --json                    # 143 protected science files unchanged? indexed paths present?
python tools/ai_assist.py context --task scoring --max-chars 24000
python -m pytest -q -p no:cacheprovider tests/test_ai_facade_regressions.py tests/test_ai_overlay2_regressions.py tests/test_ai_tools.py
```

Full suite, CPU, about one to two minutes (mask the GPU explicitly; on Windows use `-1`, an empty
string does not mask it):

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -p no:cacheprovider --basetemp=../pdeobs-pytest-tmp
```

## 2. Small CPU workflows (minutes; software validation, never paper results)

```bash
# bounded smoke: Poisson recovery + Heat rollout, 16x16, 4 TRAIN + 2 TEST demo identities, 2 optimizer steps
python tools/ai_assist.py smoke --out ../pdeobs-ai-smoke-001

# two demo paper rows (cohort demo, one epoch, nine 16x16 records each), then the evidence checker on them
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
pdeobs paper-row-demo-data --pde poisson --output ../row-demo-data/poisson
pdeobs paper-row --config configs/demo/paper_row_static.yaml --data-root ../row-demo-data/poisson \
  --manifest ../row-demo-data/poisson/identities.json --output ../row-demo-runs/poisson --device cpu
pdeobs paper-row-demo-data --pde heat --output ../row-demo-data/heat
pdeobs paper-row --config configs/demo/paper_row_rollout.yaml --data-root ../row-demo-data/heat \
  --manifest ../row-demo-data/heat/identities.json --output ../row-demo-runs/heat --device cpu
python -m pdeobs.evidence plan  --root ../row-demo-runs/poisson --out ../evidence-checks/poisson-bundle.json
python -m pdeobs.evidence check --bundle ../evidence-checks/poisson-bundle.json --root ../row-demo-runs/poisson
python -m pdeobs.evidence plan  --root ../row-demo-runs/heat --out ../evidence-checks/heat-bundle.json
python -m pdeobs.evidence check --bundle ../evidence-checks/heat-bundle.json --root ../row-demo-runs/heat
```

Expected: `verification_status: complete`, `purpose: software-demo`, exit 0. The scores are
meaningless one-epoch demo numbers.

## 3. Check and export existing paper materials

Point `--root` at an author-supplied `paper-row` output directory (cohort `original500`: checkpoint,
`identity_manifest.json`, `split.json`, `evaluation/<view>/{contract.json,predictions.h5,score.json}`,
`receipt.json`). Read-only until `export`.

```bash
python -m pdeobs.evidence plan  --root <row dir> --out <new dir>/bundle.json \
  --paper-ref '{"table": "2", "row": "poisson/fno", "cell": "R50"}'
python -m pdeobs.evidence check --bundle <new dir>/bundle.json --root <row dir> --out <new dir>/check.json
python -m pdeobs.evidence export --bundle <new dir>/bundle.json --root <row dir> --out <new dir>/export
```

Exit 0 = every link verified (`complete`); 3 = something missing (`incomplete`, listed per file);
2 = a check failed or the input is bad (one JSON line on stderr with `category`, `reason`,
`next_step`). A demo bundle is refused by `export` unless `--allow-demo`; `purpose_guard` fails a
paper-evidence claim over demo or smoke artifacts. Semantics: `docs/paper_evidence.md`.

## 3b. Index the archived campaign runs (read-only scan on the archive host; no re-run)

```bash
python tools/archived_results.py scan --training-root <root>/_preliminary-training/<campaign> \
  --evaluation-root <root>/_preliminary-evaluation/<campaign> --label <machine> --out ./scan-<machine>.json
python tools/archived_results.py pair --scan ./scan-*.json --selection ./selection.json --out ./paired.json
python tools/archived_results.py index --paired ./paired.json --out-dir ./results
```

Legacy campaign-evaluator metrics, cohort per row, `RESULT_PENDING` for missing cells; semantics in
`docs/archived_results.md`.

## 4. Entry points that need explicit authorization (never auto-run)

- `pdeobs paper-row` with `cohort: original500`: one grid cell, 500-epoch training plus nine strict
  evaluations, hours on a GPU (README, *Reproducing one grid cell*).
- `pdeobs plan` and `pdeobs generate` at corpus scale (dataset card).
- `pdeobs download --manifest <file>`: network access, author-supplied manifest only.
- Any GPU, scheduler, cloud or campaign command (`docs/ai/repository_index.json` lists them under
  `never_auto_execute`).

What each level proves is in `README.md` (*Provided features, this-round acceptance, paper evidence,
author materials*) and `docs/ai/ACCEPTANCE_INDEX.md`.
