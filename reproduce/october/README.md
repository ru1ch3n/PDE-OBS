# Rebuilding the October 2026 tables

```bash
python reproduce/october/rebuild_tables.py --results results/revision_october --out reproduce/october/out
```

The script uses the standard library only. It rebuilds nine tables from the run-level results:
1. static calibration
2. observation transfer (D3 / C3 per seed)
3. grid jitter, with the data-consistency split
4. NS medium sensor density
5. resolution
6. external RPB in-distribution and OOD
7. normalisation-only attribution
8. confirmation-set neural results
9. training failures

"Best neural" columns are descriptive envelopes over every trained model of the listed cohorts, not methods selected on
test data. The outputs are generated files; they are not committed.
