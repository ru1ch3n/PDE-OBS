# Revision tools

Entry points added for the repair-and-analysis release. None of them trains a model.

| tool | role | needs |
|---|---|---|
| `mix_strict_inference.py` | inference-only strict rescoring of one mixed-pattern study checkpoint on the nine frozen views plus declared extra views; retains contract-bound prediction files and scores them with the unchanged `pdeobs-strict-v1` scorer (raw and, for static tasks, data-consistency projection) | GPU host with the study checkpoint, the data shards and this package on `PYTHONPATH` |
| `build_results.py` | rebuilds every mixed-versus-specialist comparison from `configs/revision/result_sources.yaml`; fail-closed alignment checks; explicit populations | CPU |
| `audit_training.py` | training inventory of the 441 credited models (one row per attempt, evidence source per field) and the mixture assignment tables reproduced with the release assignment logic | CPU |
| `static_diagnostics.py` | `score`: reopens the 1,701 retained stationary prediction files and rescores them with projection (host with the files, CPU); `aggregate`: release tables of the observed/hidden decomposition and raw-versus-projected errors | CPU |
| `bootstrap.py` | paired, regime-stratified test-identity bootstrap of the 441-grid statistics (C-D, C/D, transfer differences and ratios, density ratios, horizon C/D, medians) | CPU |
| `check_access.py` | unauthenticated access / integrity / execution checks of the public deposits declared in `configs/revision/public_deposits.yaml`, plus an anonymity inspection of the cards | network |
| `test_revision.py`, `test_bootstrap.py` | acceptance tests on synthetic fixtures | CPU |
| `common.py` | shared constants, digests, loaders | |

Outputs go to `results/revision_v2/` (see its README and REPORT). Inputs are declared, digest-checked and listed in
`results/revision_v2/mixed_comparison_inputs.json`, `audit_summary.json`, `bootstrap_config.json` and
`stationary_diagnostics_checks.json`.

## Commands (CPU)

```bash
python tools/revision/build_results.py
python tools/revision/audit_training.py
python tools/revision/static_diagnostics.py aggregate --scored results/revision_v2/static_scored --out results/revision_v2
python tools/revision/bootstrap.py                 # 5000 replicates, seed 20260926
python tools/revision/check_access.py             # exit 2 when a deposit is not reachable without credentials
python -m unittest discover -s tools/revision -p 'test_*.py'
```

## On the host that holds the retained prediction files

```bash
PYTHONPATH=<frozen verification source>/src python tools/revision/static_diagnostics.py score \
  --controller <verification controller root> --source <frozen verification source>/src --pde poisson --out <fresh dir>

PYTHONPATH=src python tools/revision/mix_strict_inference.py \
  --identity poisson/ufno_2d/mixed_nine_views \
  --model-root <study output dir with resolved.yaml, completion.json, split_manifest.json, checkpoints/last.pt> \
  --expected-checkpoint-sha256 <checkpoint_sha256 from the study completion receipt> \
  --data-root <original data root> \
  --bindings results/prediction_verification_20260924/dataset_bindings.json \
  --campaign configs/campaign/all_pde_one_setting_10method_9x9.yaml \
  --output <fresh directory> --device cuda:0 \
  --extra-view "random_3pct 20" --extra-view "random_3pct 35" \
  --extra-view "line_sensors(orientation=both) 30 + random_3pct 20"
```

Both drivers refuse inputs whose digests differ from the receipts or bindings; the diagnostics step aborts on the first
identity whose retained arrays do not reproduce the sealed score or violate the decomposition and projection bounds.

## Acceptance tests

Covered: missing strict score fails; duplicate or missing identities fail; identity sets that differ between the
mixed and specialist blocks fail; forbidden (legacy) score sources are refused; restricted populations use exactly
the declared destinations; the frozen-view contract checks (checkpoint, task, mask); mixture assignments reproduce
across reruns and identity order; unknown evidence stays `unknown`; bootstrap draws are stratified and paired, identical
errors give zero differences, simultaneous reordering changes nothing, a fixed seed reproduces, ratios of means are
never replaced by means of ratios, zero denominators are flagged, incomplete identity sets fail.
