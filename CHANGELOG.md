# Changelog

## Unreleased (October 2026 revision; no tagged release yet)

- `pdeobs.revision`: the October 2026 revision modules.
  - V2 data specification and generator, reference solvers and numerical checks.
  - V2 observation layouts: U / C / L / G, MXR3 / MXR6 / MXR2, nested sets, grid jitter.
  - Classical references.
  - Covariance-weighted reconstruction (`revision.covariance`).
  - Raw / observed / hidden error split (`revision.dc`).
  - Paired parent bootstrap (`revision.stats`).
  - RPB Poisson adapter (`revision.external.rpb`).
- `configs/paper/october/`: frozen data, observation, classical, recipe and control configurations exported from the runs.
  The historical `configs/paper/` files are unchanged.
- `docs/october_revision.md`: what the revision adds and shows; README entry point and citation
  (arXiv:2609.36521).
- `docs/public_release.md` / README: both dataset manifests at `ru1ch3n/PDE-OBS` resolve every entry to the
  new address (checked 2026-10-16).
- `THIRD_PARTY_NOTICES.md`: RPB attribution (not redistributed).
- `results/revision_october/`: run-level results of the revision (parts A and B) with manifests and a column legend.
- `reproduce/october/`: table rebuild, weights-inference replay (path C) and training reproduction (path D) from the public
  package, with `--set KEY=VALUE` recipe overrides for controlled variants.
- `pdeobs.revision.fftcompat`: cuFFT-compatible inverse real FFT for CPU replays of the FNO checkpoints.
- `docs/installation.md`: October release-candidate install check.
- `reproduce/october/score_method.py`: score any predictor (or the built-in zero / nearest / covariance references) on the
  V2 views and on converted RPB data with the revision's protocol; `rebuild_tables.py --latex`.
- `results/revision_october/final_analyses/` (E1-E5 outputs as JSON) and `paper_assets.json` (every main table bound to
  its data, models, protocol, scoring and result files).
- README rewritten as a researcher entry (current results, four reproduction levels, own method); the September grid
  is the historical layer; `docs/public_release.md` and `docs/reproducing_the_paper.md` state their scope.
- Hugging Face dataset `october2026/results/part-A/update_20261021/` and `part-B/update_20261021/`: the per-record files of the 2026-10-21 exports (E6 / E6ev, N7 confirmation of the P4 U1024 specialists, E5), beside the unchanged 2026-10-17 files (2026-10-26).
- Public deposits: the package points to the `ru1ch3n/PDE-OBS` dataset and model repositories (`configs/revision/public_deposits.yaml`, `results/public_deposits/release_map.json`, release metadata); review-archive names relabelled; receipt `results/public_deposits/relabel_receipt_20261026.json`; `results/revision_v2/access_check.json` regenerated (2026-10-26).
- Hugging Face `ru1ch3n/PDE-OBS`, `october2026/`: weights of the 663 scored October runs (the 12 E6 runs added 2026-10-21) and of
  the ten completed path D retrains (`october2026/retrains/`), the V2 data the code reads,
  and the run-level / per-record results (part B without its run-level export); `reproduce/october/README.md` and `results/revision_october/README.md`
  point there.

## 0.2.1 (2026-09-26)

See `RELEASE_NOTES_v0.2.1.md`.
