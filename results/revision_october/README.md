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
| `runs_long.part-A.csv` | 7,778 rows. Static Poisson / Darcy V2 (smooth, medium, rough); NS medium: the N5 MXR6 trainings and the nested-density evaluations of the P4 runs (the P4 NS runs themselves are in part B); N1-N7 and 21 of the 48 N8 resolution runs (mostly the 256^2 runs; the other 27 are in part B); N7 static + NS medium; RPB in-distribution and OOD; E2 re-evaluations; E6 nearest-fill trainings (12) and their grid-jitter / nested-density evaluation suites (added 2026-10-20); N7 confirmation of the six P4 Voronoi-CNN U1024 specialists of Poisson rough and Darcy rough (added 2026-10-21) |
| `classical_long.part-A.csv` | 312 rows: classical references per (task, view, method) with the validation-selected settings |
| `manifest.part-A.json` | SHA-256 and row counts of the committed files and of the exported per-record files; `update_20261020` records the E6 / E6ev rows added on 2026-10-20 and `update_20261021` the N7ev rows added on 2026-10-21; the per-record files of the 2026-10-21 export are on Hugging Face under `october2026/results/part-A/update_20261021/` |
| `runs_long.part-B.csv` | 2,316 rows. NS smooth / medium / rough P4; NS rough P6 / P7, V4-C / V4-D / V4-E and the V4 P6 / P7 re-checks; N7 NS rough confirmation; N8 resolution runs of Poisson rough and NS rough; E5 composition reference, NS rough (added 2026-10-21) |
| `classical_long.part-B.csv` | 282 rows: classical references of the NS slices and of the 64^2 / 256^2 resolution views. 168 further rows (Darcy / Poisson) were computed on both accounts, agree exactly with part A and are kept only there |
| `manifest.part-B.json` | SHA-256 and row counts as for part A; lists what was left out (2 voided attempts, 168 duplicate classical rows); `update_20261021` records the E5 rows added on 2026-10-21; the per-record files of that export are on Hugging Face under `october2026/results/part-B/update_20261021/` (without the run-level export) |
| updates | E6 trainings and E6ev suites added to part A (2026-10-20); E5 added to part B (2026-10-21); N7 confirmation of the P4 U1024 specialists added to part A (2026-10-21). Nothing pending. |

## Columns of `runs_long`

| Column | Meaning |
|---|---|
| `run_id` | run or evaluation identity, `<cohort>__<pde>__<slice>__<method>__<protocol>[__variant]__s<seed>` (evaluation-only rows: `<family>__[<suite>__]<parent run>`; N3GJ2 / N5MXR6 ids omit the protocol segment, V4E ids name the information cell, RPB ids use `rpb` as the slice) |
| `cohort` | P4 / P5 frozen V2 recipes; V4C / V4D / V4E method-tuned; N3GJ2, N5MXR6, N6, N8, RPB, RPBn, E6 trainings; E5compose (composition reference: covariance-reconstructed initial state -> the frozen full-state V4E FNO, NS rough; views `true_full`, `<layout>__cov`, `<layout>__nearest`); P6 / P7 (training-set size, noise, nearest fill and MXR6 re-checks with the frozen recipe) and V4P6 / V4P7 (the same with the V4 recipe); N2ev, N3ev, N4ev, N5ev, N7ev, RPBood, E2ev, E6ev evaluation-only suites and V4P7nev (noise evaluation of the V4C MXR3 parents); V2dev = recipe-screen runs that failed (screen results are not test results). Recipes per cohort: `configs/paper/october/recipes.yaml` |
| `source` | `native`, or `imported` = run trees trained in an earlier allocation and copied unchanged (checksum-verified) into the analysis tree |
| `pde`, `dataset_id`, `test_dataset_id`, `grid`, `method`, `protocol`, `seed`, `parent_run_id` | identity of the trained model and of the evaluated slice (`test_dataset_id` differs from `dataset_id` only for the N4 construction shift) |
| `suite`, `view` | evaluation suite and block label: `v2eval7` (seven V2 views, test split), `n2noise`, `v2noise14` (sensor noise), `n3geom` (grid shifts / jitter), `n4shift`, `n5nested` (nested K), `n7confirm` (sealed confirmation parents), `v2full1` (NS full initial state, U16384), `e5compose` (E5 forecasts), `rpbood` (RPB OOD), `v2eval7_prefix_pass250` (P6 prefix checkpoint) |
| `n`, `mean`, `median`, `p90`, `max`, `valid` | strict-scorer block statistics of the per-record relative L2 (float64, full field, physical units); `valid` = the block passed the identity / shape / finiteness checks |
| `dc_raw`, `dc_obs`, `dc_hid`, `dc_hid_rel` | filled for the E2 re-scoring rows (cohort E2ev; suites n3geom and rpbood): error on the observed cells and in the hidden region with the common full-field denominator (`raw^2 = obs^2 + hid^2`; `hid` = error of the data-consistent prediction), and the hidden-region relative error (own denominator) |
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

- **Per-record errors.** About 170 MB of compressed CSV for part A. Their SHA-256 are in `manifest.part-A.json`. They
  are published beside the data in the dataset `huggingface.co/datasets/ru1ch3n/PDE-OBS` under
  `october2026/results/part-A/`, with a copy of the manifest; part B's per-record files are under `part-B/` (its
  run-level export is not published there because two voided-attempt rows carry a machine path; `runs_long.part-B.csv`
  here is its run-level table). The 2026-10-21 exports are under `update_20261021/` in both part folders (part B again without the run-level export).
- **Model weights** of every scored run are in the model repository `huggingface.co/ru1ch3n/PDE-OBS` under `october2026/`.
- **Raw predictions.**
- **Classical per-record errors and the classical confirmation, noise and rank studies.** Their summaries are in the
  revision report and in `docs/october_revision.md`.
