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
- Public deposits: the package points to the `ru1ch3n/PDE-OBS` dataset and model repositories (`configs/revision/public_deposits.yaml`, `results/public_deposits/release_map.json`, release metadata); review-archive names relabelled; receipt `results/public_deposits/relabel_receipt_20261026.json`; `results/revision_v2/access_check.json` regenerated (2026-10-26).
- Hugging Face `ru1ch3n/PDE-OBS`, `october2026/`: weights of the 663 scored October runs (the 12 E6 runs added 2026-10-21) and of
  the ten completed path D retrains (`october2026/retrains/`), the V2 data the code reads,
  and the run-level / per-record results (part B without its run-level export); `reproduce/october/README.md` and `results/revision_october/README.md`
  point there.

## 0.2.1 (2026-09-26)

See `RELEASE_NOTES_v0.2.1.md`.
