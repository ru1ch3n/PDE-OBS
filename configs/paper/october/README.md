# October 2026 revision configurations

Frozen configurations of the October 2026 revision of PDE-OBS:

- V2 data
- V2 observation layouts
- classical references
- training recipes of every cohort
- single-factor controls
- the external RPB Poisson adapter

They are exported from the configurations that actually ran. The historical configurations in `configs/paper/` (original 441-row grid) are unchanged and keep their own meaning; nothing here re-interprets them.

| File | Content |
|---|---|
| `data_v2.yaml` | PDEs, slices, regimes, splits (incl. the sealed confirmation split), grids, exact counts |
| `observations_v2.yaml` | U / C / L / G layouts, mixtures MXR3 / MXR6 / MXR2, nested sets, grid perturbations, infeasible layouts |
| `classical.yaml` | gappy POD and covariance reconstruction, grids and validation-only selection |
| `recipes.yaml` | training recipe per (cohort, dataset, method) of the P4 / P5, V4, N3 / N5 / N6, RPB and E6 cohorts, with protocols and seeds (the P6 / P7, V4P6 / V4P7 and N8 re-checks reuse the P4 / V4C recipes with one change) |
| `controls.yaml` | N6, RPBn and E6 single-factor controls |
| `external_rpb.yaml` | RPB source, licence, authors' split, permissions, views |
