# October 2026 revision: run-level results

Every evaluation of the October 2026 revision gets one row per (run, evaluation suite, view block). Rows are exported from
the result files of the runs; nothing was re-scored or retyped. The tables of the revision can be rebuilt from these files
with:

```bash
python reproduce/october/rebuild_tables.py            # writes reproduce/october/out/tables.md and one CSV per table
```

## Files

| File | Content |
|---|---|
| `runs_long.part-A.csv` | 7,454 rows. Static Poisson / Darcy V2 (smooth, medium, rough); NS medium (P4, N5); N1-N8 except the NS rough parts; N7 static + NS medium; RPB in-distribution and OOD; E2 re-evaluations |
| `classical_long.part-A.csv` | 312 rows: classical references per (task, view, method) with the validation-selected settings |
| `manifest.part-A.json` | SHA-256 and row counts of the committed files and of the exported per-record files |
| `runs_long.part-B.csv` | 2,289 rows. NS smooth / medium / rough P4; NS rough P6 / P7, V4-C / V4-D / V4-E and the V4 P6 / P7 re-checks; N7 NS rough confirmation; N8 resolution runs of Poisson rough and NS rough |
| `classical_long.part-B.csv` | 282 rows: classical references of the NS slices and of the 64^2 / 256^2 resolution views. 168 further rows (Darcy / Poisson) were computed on both accounts, agree exactly with part A and are kept only there |
| `manifest.part-B.json` | SHA-256 and row counts as for part A; lists what was left out (2 voided attempts, 168 duplicate classical rows) |
| still to come | E5 (composition reference) and E6 (nearest-fill trainings and their evaluations), added when they finish |

## Columns of `runs_long`

| Column | Meaning |
|---|---|
| `run_id` | run or evaluation identity, `<cohort>__<pde>__<slice>__<method>__<protocol>[__variant]__s<seed>` (evaluation-only rows: `<suite family>__<suite>__<parent run>`) |
| `cohort` | P4 / P5 frozen V2 recipes; V4C / V4D / V4E method-tuned; N3GJ2, N5MXR6, N6, N8, RPB, RPBn, E6 trainings; N2ev, N3ev, N4ev, N5ev, N7ev, RPBood, E2ev evaluation-only suites; V2dev = recipe-screen runs that failed (screen results are not test results). Recipes per cohort: `configs/paper/october/recipes.yaml` |
| `source` | `native`, or `imported` = run trees trained in an earlier allocation and copied unchanged (checksum-verified) into the analysis tree |
| `pde`, `dataset_id`, `test_dataset_id`, `grid`, `method`, `protocol`, `seed`, `parent_run_id` | identity of the trained model and of the evaluated slice (`test_dataset_id` differs from `dataset_id` only for the N4 construction shift) |
| `suite`, `view` | evaluation suite and block label: `v2eval7` (seven V2 views, test split), `n2noise`, `v2noise14` (sensor noise), `n3geom` (grid shifts / jitter), `n4shift`, `n5nested` (nested K), `n7confirm` (sealed confirmation parents), `rpbood` (RPB OOD), `v2eval7_prefix_pass250` (P6 prefix checkpoint) |
| `n`, `mean`, `median`, `p90`, `max`, `valid` | strict-scorer block statistics of the per-record relative L2 (float64, full field, physical units); `valid` = the block passed the identity / shape / finiteness checks |
| `dc_raw`, `dc_obs`, `dc_hid`, `dc_hid_rel` | static noise-free blocks evaluated with release 33: error on the observed cells and in the hidden region with the common full-field denominator (`raw^2 = obs^2 + hid^2`; `hid` = error of the data-consistent prediction), and the hidden-region relative error (own denominator) |
| `run_status`, `final_train_loss`, `gpu_seconds` | final state of the training run (training pathologies are kept as `failed: ...` and never replaced), final training loss, GPU seconds attributed to the run |
| `v4_norm`, `lr`, `batch_size`, `epochs`, `fill_input`, `param_channel` | recipe fields of the trained model |

## Populations

- V2 test: 1,800 parents per slice (600 per regime).
- Confirmation: 600 sealed parents per slice.
- RPB: the authors' 256 in-distribution test records and 256 OOD records.

Classical references are fitted on the training split, with settings chosen on the validation split. Means over seeds
use the three main training seeds 1101-1103 unless a table says otherwise. The seed-extension runs 1104 / 1105 of P4 are
in the file but reported separately.

## Not in this directory

- **Per-record errors.** About 150 MB of compressed CSV for part A. Their SHA-256 are in `manifest.part-A.json`. They
  are published beside the data in the dataset `huggingface.co/datasets/ru1ch3n/PDE-OBS` under
  `october2026/results/part-A/`, with a copy of the manifest; part B's per-record files are under `part-B/` (its
  run-level export is not published there because two voided-attempt rows carry a machine path; `runs_long.part-B.csv`
  here is its run-level table).
- **Model weights** of every scored run are in the model repository `huggingface.co/ru1ch3n/PDE-OBS` under `october2026/`.
- **Raw predictions.**
- **Classical per-record errors and the classical confirmation, noise and rank studies.** Their summaries are in the
  revision report and in `docs/october_revision.md`.
