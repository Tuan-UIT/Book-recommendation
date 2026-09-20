# C03 completion evidence

Date: 21 September 2026

## Output

The split was generated from the verified C02 sample
`data/samples/c02_seed42_users1000/` with split seed `42`:

```powershell
python backend/data_scripts/prepare_c03_split.py `
  --input-dir data/samples/c02_seed42_users1000 `
  --output-dir data/samples/c03_seed42_users1000 `
  --seed 42
```

The private output is `data/samples/c03_seed42_users1000/`. Its
`split_audit.json` records:

| Cohort or split | Count |
| --- | ---: |
| C02 clean explicit ratings | 118,761 |
| C02 users with explicit ratings | 924 |
| Sampled users | 1,000 |
| Standard users (10+ works) | 782 |
| Sparse users (2--9 works) | 113 |
| One-rating, no-history users | 29 |
| Zero-history sampled users | 76 |
| Train rows | 95,659 |
| Validation rows | 11,480 |
| Test rows | 11,622 |

The output also includes training-only item and user statistics. The audit
records their hashes and states that validation/test rows are never used for
fit or selection.

## Verification

Focused C02 and C03 checks:

```text
8 passed
```

The C03 integrity checks are all `true`, including:

- every clean pair assigned exactly once;
- split pair IDs disjoint and user--work pairs still unique;
- standard, sparse, one-rating, and zero-history rules match the cohort file;
- training item/user statistics recompute from training rows only;
- held-out pair IDs are absent from training;
- the split seed is recorded and `source_row_number` is not treated as time.

This completes split preparation only. Model fitting, validation selection,
and the four-method comparison remain future tasks.
