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
- Hugging Face `ru1ch3n/PDE-OBS`, `october2026/`: weights of the 651 scored October runs, the V2 data the code reads,
  and the run-level / per-record results (part B without its run-level export); `reproduce/october/README.md` and `results/revision_october/README.md`
  point there.

## 0.2.1 (2026-09-26)

See `RELEASE_NOTES_v0.2.1.md`.
