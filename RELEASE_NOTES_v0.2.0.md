# PDE-OBS v0.2.0 — unified user interface and model-level acceptance

This revision is a **software facade** over the frozen v0.1.1 science. It lets an external
researcher choose data source, PDE, observation, task and model parameters and run the full
generate → train → load → infer → evaluate flow without touching internal scripts. It adds **no**
numerical algorithm and changes no loss, mask, split or scoring definition, does not rescore
historical results, and does not run a paper-scale training campaign.

Package version `0.2.0`; numerical kernel version unchanged at `0.1.0`; observation protocol and
`pdeobs-strict-v1` scoring unchanged. Versions are managed separately (`pdeobs-easy-api/v1`,
`pdeobs-model-artifact/v1`, `pdeobs-inference-input/v1`, `pdeobs-prediction-bundle/v1`,
`pdeobs-pipeline/v1`).

## What a user can now do in one line

| operation | CLI | Python |
|---|---|---|
| create data (solver) | `pdeobs easy data --source generate --pde heat …` | `api.create_dataset(...)` |
| load existing data | `pdeobs easy data --source local --path …` | `api.load_dataset(...)` |
| configure observation | `--obs random --obs-param ratio=0.5` (or `paper:R65`) | `api.make_observation(...)` |
| create model | `--model fno --model-param width=32 …` | `api.create_model(...)` / `api.resolve_model(...)` |
| train (explicit budget) | `pdeobs easy train … --max-steps 5` (or `--epochs`) | `api.train(...)` |
| load + target-free infer | `pdeobs easy infer --model-from … --input obs.npz` | `api.load_predictor(...)` / `api.predict(...)` |
| strict evaluate (target) | `pdeobs easy eval --predictions … --targets …` | `api.evaluate(...)` |
| whole pipeline | `pdeobs easy run --config pipe.yaml` | `api.run(...)` |
| introspect | `pdeobs easy list|describe|validate|plan` | `api.list_models()` etc. |

## Design boundaries enforced by the code

- **Three resources never conflated**: numerical solver vs. existing dataset vs. trained model. No
  `solver=pretrained`; random weights are never a pretrained predictor; an unregistered download
  fails explicitly instead of regenerating data under a download label.
- **Target-free inference is real**: `predict` needs only observations + mask (+ geometry); a
  supervised score is produced only when an independent target is supplied to `evaluate`. `stored`
  observations use the input package's own mask and never generate a new one.
- **Parameters are validated, not assumed**: every public parameter maps to a real constructor;
  unknown keys, wrong types, out-of-range values, alias confusion (`width`/`hidden`,
  `depth`/`layers`) and illegal combinations (`fno` width not divisible by `min(8,width)`,
  `hidden` not divisible by `heads`) are rejected with the valid options. PINO's physics contract
  must match the task; U-FNO needs `batch_size ≥ 2`; the fixed U-FNO/CNO depths are surfaced, not
  faked.
- **Compatibility is a hard constraint**: the frozen 521 portable tests and 37 updater tests are
  preserved; new tests add coverage. Overrides of paper structure/observation/budget produce a
  `custom` identity, never a paper-row identifier. New-API predictions equal the underlying
  `runner`/`evaluate` path for the same weights.

## Acceptance (software, under `acceptance/` — never a paper table)

- **L0** CPU regression: the 521 portable + 37 updater tests (on both the frozen v0.1.1 tree and the
  v0.2.0 tree) plus the new API tests.
- **L1** parameter coverage: each main model — default + a legal non-default (parameter count
  actually changes) + an illegal combination rejected + save/reload output identity.
- **L2** GPU (RTX 5090 / Blackwell) 7 models × 7 PDEs = 49 short flows: paper structure, 128×128,
  batch 1, a few optimizer steps → save → fresh-process reload → nine paper observation strict
  scores. PINO runs its real residual training path.

The RTX 5090 environment is a dedicated PyTorch 2.7 / CUDA 12.8 (Blackwell `sm_120`) build; the CPU
minimum-dependency environment (PyTorch 2.1 CPU) is **not** reused for the GPU tests. Concrete
numbers, per-flow logs and the per-tier acceptance matrix are written under `acceptance/` and summarized in
`ACCEPTANCE_SUMMARY.md`.

## Explicitly not done this round

- No production C500 run, 441-row sweep, multi-seed benchmark or convergence study.
- `fixed200` / `min120` / `recovered` historical cohorts are not implemented; the C500 paper-table
  adapter still requires unthinned first source frames.
- No full optional-backend / minimum-version / platform matrix; optional exact upstream wrappers
  (`paper_fno` etc.) remain dependency-gated and are not served through the easy API.
- GPU results are software acceptance, not paper performance; historical results keep their status.
