# C04-C06 recommenders

This package contains the first three methods required by the proposal:

- `evaluation.py`: fixed-denominator Precision@10, Recall@10 and binary
  NDCG@10;
- `catalog.py`: one training-only shared candidate catalog;
- `popularity.py`: explicit training-rating count baseline;
- `user_cf.py`: user-based Pearson CF over common items with positive valid
  neighbours, mean-centred prediction, and popularity fallback;
- `content.py`: description TF-IDF plus normalized author/genre blocks,
  positive-history or initial-genre profiles, cosine ranking, and fallback.

Run the focused checks from the project root:

```powershell
python -m pytest backend/tests/test_c04_c06_recommenders.py -q
```

Run the private validation experiment:

```powershell
python -m backend.data_scripts.run_c04_c06_validation
```

The runner deliberately has no test-split argument. It selects up to 5,000
works with at least three training ratings, evaluates every method on the same
validation population, and searches `k` in `{20, 40}` and minimum common-item
overlap in `{3, 5}`. It writes ignored private artifacts under
`artifacts/experiments/` and `artifacts/models/`.

Do not present these validation metrics as final test results. C07 adds the
Hybrid method and validation-selected `alpha`; C10 performs the one frozen
four-method test comparison.
