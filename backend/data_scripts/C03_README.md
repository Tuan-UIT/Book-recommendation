# C03: reproducible train/validation/test splits

Run this after C02 has produced `clean_ratings.csv` and
`sample_user_ids.csv`:

```powershell
python backend/data_scripts/prepare_c03_split.py `
  --input-dir data/samples/c02_seed42_users1000 `
  --output-dir data/samples/c03_seed42_users1000 `
  --seed 42
```

The interaction source has no timestamp. `source_row_number` is retained as
provenance, but it is never treated as recency. Each user is shuffled with a
stable seed derived from the global seed and that research user ID.

Split policy:

- users with at least ten clean works: `floor(0.1*n)` validation rows,
  `floor(0.1*n)` test rows, and the remainder for training;
- users with 2--9 works: one test row, the remainder for training, and no
  validation row;
- users with one work: the work is test-only in the
  `one_rating_no_history` cohort;
- sampled users with no clean explicit rating remain in the
  `zero_history` cohort and have no split rows.

The output directory is private and ignored by Git. It contains the three
split files, `user_cohorts.csv`, training-only item/user statistics, and JSON
manifests with hashes and integrity checks. Validation and test rows are never
used to compute the training statistics or fit the models. Validation results
later select `k`, the common-item overlap threshold, `alpha`, and any other
tuned settings. Test data remain untouched until the configuration is frozen
for the final four-method comparison.

Run the focused checks with:

```powershell
python -m pytest backend/data_scripts/test_prepare_c03_split.py -q
```
