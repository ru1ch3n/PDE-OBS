# Public release: data and trained checkpoints

This page records the data and model release inventory and verification procedure.
The 2026-09-26 archived report audited the original neutral-namespace deposits
after their de-identified re-emission (`results/revision_v2/access_check.json`,
`results/revision_v2/REPORT.md`, section 7). The current user-owned copies are
public, have the same file counts and byte-identical root manifests, and representative
data and checkpoint objects have matching size and ETag. A second complete per-file
audit of the copies has not been performed. The copied dataset manifests still
contain absolute URLs to the original deposit, so downloading through a manifest
at the new address can still fetch shards from the old address. Historical audit
records are retained unchanged.

| Resource | Location | Size |
|---|---|---|
| Physical records | `https://huggingface.co/datasets/ru1ch3n/PDE-OBS` | 3,360 shards, 244 GB with their verification sidecars under `data/`; `scrub-manifest.json` lists the rewritten files with their published digests; `summary.json` and `summary.quality.*` de-identified likewise (the correspondence to generation-time digests is `results/public_deposits/release_map.json` here) |
| Trained checkpoints | `https://huggingface.co/ru1ch3n/PDE-OBS` | 441 checkpoints (bytes unchanged) with de-identified training records, about 13 GB; `models_manifest.cluster-{A,B,C}.json` carry the published digests only (manifest v3); `results/public_deposits/release_map.json` binds them to the archived checkpoint identities |

These user-owned locations are public, not anonymous reviewer links. The
historical review-access configuration remains under `configs/revision/`.

## Records

The dataset repository holds the **complete factorial design**, not only the slice the paper
evaluates: seven PDE families x four boundary protocols x ten condition-field constructions, 2,000
records per combination divided 667/667/666 across the three physical regimes, at 128 x 128. The
paper's evaluated slice (one boundary and the `smooth_grf` construction per family, 14,000 records,
84 shards) is the subset under the seven `data/<family>/<paper boundary>/smooth_grf/` paths.

Files are the canonical HDF5 shards described in [Physical records, identities, and storage](data_schema.md),
unchanged from generation, with their `.sha256`, `.manifest.json`, `.metadata.json`, `.metadata.csv`
and `.quality.json` sidecars, plus the corpus-level `summary.json`, `summary.quality.json` and
`summary.quality.csv`.

Two checksum manifests are published beside the data, both the release-manifest v1 contract of
`src/pdeobs/download.py` (schema version 1, one entry per file with its SHA-256, byte size and
resolve URL). Each lists its shards **and** the `.manifest.json`, `.sha256` and `.quality.json`
sidecars, so a downloaded tree passes `api.load_dataset(..., verify=True)`:

| Manifest | Contents | Entries | Size |
|---|---|---:|---:|
| `release_manifest.json` | the paper-evaluated slice: one boundary and the `smooth_grf` construction per family, 14,000 records | 84 shards + 252 sidecars | 6.5 GB |
| `release_manifest_full.json` | the complete factorial design: 280 macrodomains, 560,000 records | 3,360 shards + 10,080 sidecars | 244 GB |

```bash
pdeobs download --tier full --output ./pdeobs-data \
  --manifest https://huggingface.co/datasets/ru1ch3n/PDE-OBS/resolve/main/release_manifest.json
```

The downloader validates the manifest's schema, status, tiers and digests before it fetches anything,
resumes interrupted files and verifies every file's SHA-256 on arrival; it has no default endpoint.
Every uploaded shard was checked against the `.sha256` sidecar written at generation time, and all
3,360 matched; one shard was then fetched through `pdeobs download` from its manifest entry and
accepted by `api.load_dataset(verify=True)`. The `.metadata.json` and `.metadata.csv` exports
(7.7 GB) are in the repository but not in the manifests: the loader does not need them, and every
record's metadata is also stored inside its shard.

## Checkpoints

The model repository holds the final checkpoint of every credited setting, laid out as
`models/<cluster-label>/<pde>/<method>/<train_view>/`, together with that attempt's training records
(`identity.json`, `completion.json`, `health.json`, `history.json`, `split_manifest.json`,
`factor_coverage.json`, `provenance.json`, `resolved.yaml`, the cohort protocol file, and
`checkpoints/training_config.json`). The three cluster labels stand for the machines the campaign ran
on and carry no institutional meaning.

Four properties matter when using them.

**Weights only.** Optimizer, gradient-scaler and RNG states were removed, about two thirds of each
file. The checkpoints support inference and rescoring, not exact continuation of training. The
removed keys are recorded per checkpoint in `models/models_manifest.<cluster-label>.json`.

**Anonymized metadata, unmodified parameters.** Every string naming a filesystem path, account, node,
host, scheduler job, partition, GPU identifier, interpreter or environment directory, run directory or
source-code revision was rewritten to a placeholder (`<ROOT_A>`, `<USER>`, `<NODE>`, `<JOB>`, `<PYTHON>`,
`<ENV>`, `<RUN>`, `<COMMIT-n>`), and raw scheduler records were reduced to their resource fields; model
parameters were not touched. Each manifest row records the SHA-256 of the released files only;
`results/public_deposits/release_map.json` in this repository binds each released checkpoint digest to
the original SHA-256 that the results index uses.

**Cohorts are not interchangeable.** Each row carries its training cohort, actual epochs and stop
reason. Any table that pools cohorts must say so per row, as `results/archived_campaign/results_long.csv`
does.

**Scoring provenance.** The metrics that accompany these checkpoints came from the campaign
evaluator, not from the strict scorer of [Scoring](scoring.md). Publishing the weights does not
convert an archived score into an independently rescored one.

## What is not published

The release manifests list all 441 released checkpoint files, and the release map in this
repository binds them to the original digests in the archive; all 441 bindings agree, including the
epoch-257 Heat/Transolver/random50 checkpoint absent from an older inventory.
The current prediction-verification preflight checks each actual stripped file
against its release digest before use; external availability remains distinct.

**No historical prediction arrays.** The campaign retained none. A separate
[prediction-verification exercise](prediction_verification.md) now exports new
arrays from the same checkpoints. A running verification is not a completed
all-model audit, and its artifacts do not silently replace the historical scores.

**Production consistency, not independent reference validation.** The
[hash-bound quality summary](../results/numerical_quality/README.md) reports
actual generation-time residual and consistency checks. No independent
reference-solution or grid-convergence study ships with it.

## Licences

Code is MIT ([LICENSE](../LICENSE)). The dataset repository declares its own terms on its card; the
code licence does not cover the data. The upstream licences recorded for the adapted baselines
describe those repositories as inspected, not this release ([Methods card](methods_card.md)).
