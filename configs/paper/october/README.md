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
| `recipes.yaml` | training recipe of every (cohort, dataset, method), with protocols and seeds |
| `controls.yaml` | N6, RPBn and E6 single-factor controls |
| `external_rpb.yaml` | RPB source, licence, authors' split, permissions, views |
