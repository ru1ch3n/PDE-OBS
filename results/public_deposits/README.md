# Public deposits: release map and relabel receipt

Records that bind this repository's archived, digest-anchored results to the files in the public
deposits (the `ru1ch3n/PDE-OBS` dataset and model repositories), whose manifests list published digests only.

| file | contents |
|---|---|
| `release_map.json` | per identity: the archived checkpoint digest (`checkpoint_original_sha256`, the identity used by `results/archived_campaign/index.json` and `results/prediction_verification_20260924/index.json`) and the digest of the published weights-only file (`checkpoint_sha256`) with the digests of every published record file; for the corpus: the generation-time and published digests of `summary.json`, `summary.quality.json` and `summary.quality.csv` |
| `relabel_receipt_20260926.json` | the label map applied on 2026-09-26 to private source-commit identifiers, the data-root version label and environment names in this repository (the same map as the third de-identification pass of the deposits), every relabelled file with its digest before and after, every file whose digest references were re-pointed, and the one reference left untouched |
| `relabel_receipt_20260926_followup.json` | follow-up labels for source commits, physical GPU identifiers and local paths in the current code release, with before/after digests and updated live references; the earlier receipt remains a historical record |
| `relabel_receipt_20261026.json` | the 2026-10-26 pass: the package now points to the `ru1ch3n/PDE-OBS` deposits (byte-identical copies of the recorded files, verified without credentials), review-archive names relabelled `<SNAPSHOT-1>` / `<SNAPSHOT-2>`, deposit history identifiers relabelled, `access_check.json` regenerated; before/after digests of every changed file |
| `release_test_receipt_20261028.json` | the release test of source tree `6b063cd` on 2026-10-28 (fresh virtual environments from the built wheel, CPU; data, weights and per-record exports fetched from the public repositories without credentials): every step with its exit code and the expected one, pytest counts, the built-in scores against the 2026-10-27 values, both weights-inference replays with their per-record comparison, the negative cases (tampered manifest, NaN / Inf / wrong-shape / raising predictors, missing stored score, tolerance), the voided-attempt check of the part-B per-record export and the digests of the fetched assets; paths and account names scrubbed |
| `hf_inventory/` | both public repositories listed at pinned commits by `reproduce/october/hf_inventory.py`: `inventory_dataset_<commit7>.csv.gz` and `inventory_model_<commit7>.csv.gz` (every path with its size, LFS flag and SHA-256 or git blob id), `summary.json` (commits, file counts, bytes, the digest of every manifest), `reconciliation.json` (per manifest: entries present, digests verified, aggregates recomputed, declared absences, the committed result files checked against this checkout; the files no manifest covers, by class) and `release_bindings.json` (commits, inventory and manifest digests, the result files of this repository); see [Public release](../../docs/public_release.md) |

The frozen verification bundle `results/prediction_verification_20260924/` was not modified. Its
`index.json` records the digest of `results/archived_campaign/index.json` as it was when the
verification ran (`f4afdac7...`); the archived index now carries `<COMMIT-1>` in place of the source
commit identifier and therefore has the digest listed in the receipt (`56cbe439...`). Nothing else in
the archived index changed. `tools/verify_archived_predictions.py` reads `release_map.json` by default.
