# Public deposits: release map and relabel receipt

Records that bind this repository's archived, digest-anchored results to the files in the public
deposits (`PDE-OBS/pdeobs-data`, `PDE-OBS/pdeobs-models`), whose manifests list published digests only.

| file | contents |
|---|---|
| `release_map.json` | per identity: the archived checkpoint digest (`checkpoint_original_sha256`, the identity used by `results/archived_campaign/index.json` and `results/prediction_verification_20260924/index.json`) and the digest of the published weights-only file (`checkpoint_sha256`) with the digests of every published record file; for the corpus: the generation-time and published digests of `summary.json`, `summary.quality.json` and `summary.quality.csv` |
| `relabel_receipt_20260926.json` | the label map applied on 2026-09-26 to private source-commit identifiers, the data-root version label and environment names in this repository (the same map as the third de-identification pass of the deposits), every relabelled file with its digest before and after, every file whose digest references were re-pointed, and the one reference left untouched |
| `relabel_receipt_20260926_followup.json` | follow-up labels for source commits, physical GPU identifiers and local paths in the current code release, with before/after digests and updated live references; the earlier receipt remains a historical record |

The frozen verification bundle `results/prediction_verification_20260924/` was not modified. Its
`index.json` records the digest of `results/archived_campaign/index.json` as it was when the
verification ran (`f4afdac7...`); the archived index now carries `<COMMIT-1>` in place of the source
commit identifier and therefore has the digest listed in the receipt (`56cbe439...`). Nothing else in
the archived index changed. `tools/verify_archived_predictions.py` reads `release_map.json` by default.
