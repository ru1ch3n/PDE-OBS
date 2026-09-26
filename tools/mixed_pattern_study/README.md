# Mixed-pattern study adapters

The scripts that trained and scored the mixed-pattern rows in `results/mixed_pattern_study/` (Appendix G of the paper).
They wrap the frozen one-setting runner and evaluator of the archived campaign without editing any science module:

- `mixed_builder.py` patches `pdeobs.one_setting.build_experiment_config` in-process so that the study view
  `mixed_nine_views` (or a seed view `<paper view>@seed<k>`) is accepted; the row is the frozen builder's output for the
  base view with only `name` and `data.mask` (the nine-view mixture spec) replaced, or only `training.seed` for seed views.
- `mixed_train.py` runs the pinned production runner under the row's recorded protocol (fixed 500 epochs, or the
  settings-v3 budget adapter of the recovered rows) and writes `study-protocol.json`; its `audit` mode records the
  per-regime component assignment of the 1,800 training identities without opening any field or test record.
- `mixed_eva.py` runs the frozen nine-view evaluator with its epoch constant set to the row ceiling; `extra_views_eva_v2.py`
  scores additional views (density sweep, held-out mixtures) through the same code path.
- `compat_mixed.py`, `study_guard.py`, `study_worker.py` are the execution gate and the one-GPU worker; host names and
  paths are replaced by placeholders here.

The campaign's frozen evaluator and budget adapter (`base200.py`, `budget200.py`, `budget_eva.py`, `same_job_eva.py`,
`acceptance_policy.py`) are the pinned files whose SHA-256 values appear in the receipts; the study runs used them
byte-identically. The overlay package used for training is the released `src/pdeobs` of this repository (v0.2.1
`dataset.py` and `mask_specs.py` over the archived production modules); its manifest hash is recorded in every receipt.
