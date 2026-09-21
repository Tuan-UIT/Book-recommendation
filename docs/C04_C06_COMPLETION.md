# C04-C06 completion evidence

Date: 21 September 2026

## Reproducible configuration

The run used the verified C03 split and C02 metadata:

```powershell
python -m backend.data_scripts.run_c04_c06_validation
```

The shared catalog contains the top 5,000 works having at least three explicit
training ratings, ordered by training count descending and canonical item ID
ascending. Its observed rating-count range is 3--290. Validation and test
counts do not influence catalog membership.

Pearson CF searched the proposal's small starting grid:

- neighbour count `k`: 20, 40;
- minimum co-rated works: 3, 5;
- selection: maximum validation NDCG@10, then Precision@10, Recall@10, and the
  smaller overlap/`k` values for an exact tie.

The selected validation configuration is `k=20` and minimum overlap `5`.
Content settings were held fixed: at most 10,000 description TF-IDF features,
unigram/bigram terms with `min_df=2`, separately L2-normalized description,
author, and genre blocks, followed by overall L2 normalization.

## Shared validation population

| Count | Value |
| --- | ---: |
| Validation users | 782 |
| Users without a positive validation label | 25 |
| Users with positive labels only outside the training-selected catalog | 97 |
| Shared evaluated users with at least one relevant candidate | 660 |
| Relevant validation user-work pairs in the catalog | 3,566 |

Relevance is a validation rating of at least 4 for a work in the shared
catalog. Precision always divides by 10. Recall and binary NDCG use all
relevant candidate works for that user. All three methods use the same 660
users and exclude each user's known training works.

## Observed validation results

These are development/selection results, not the final untouched test result.

| Method | Precision@10 | Recall@10 | NDCG@10 | Full fallback users | Mean latency/user |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training popularity | 0.04273 | 0.09054 | 0.07953 | 0 | 0.019 ms |
| User Pearson CF (`k=20`, overlap 5) | 0.00364 | 0.00674 | 0.00609 | 34 | 9.456 ms |
| Content | 0.05167 | 0.14197 | 0.11619 | 1 | 40.745 ms |

Each method returned ten works for every evaluated user. Pearson CF produced
6,260 directly scored items and 340 popularity-fallback items. Content
produced 6,590 directly scored items and ten fallback items.

The weak Pearson result is retained as measured evidence. It likely reflects
limited co-rating overlap after the shared catalog restriction, but that is a
hypothesis to investigate rather than a reason to change the test set or hide
users. Hybrid improvement is not assumed.

## Method checks

### C04

- The hand-worked case with one relevant work at rank 2 produces
  Precision@10 = 0.1, Recall@10 = 1, and NDCG@10 = `1/log2(3)`.
- Duplicate recommendations are rejected.
- A user with no relevant candidate is counted but excluded from the macro
  metric population rather than assigned an invented score.
- Popularity uses training row counts only and stable ID tie-breaking.

### C05

- Pearson is centred over common rated works.
- Insufficient overlap, zero variance, non-positive similarity, and an empty
  denominator are unavailable estimates.
- Only positive valid neighbours enter the top-`k` set.
- Predictions use the target training mean plus similarity-weighted neighbour
  deviations from each neighbour's training mean.
- No valid prediction activates a declared popularity fallback.

### C06

- TF-IDF is fitted only on descriptions for the training-selected catalog.
- Author IDs and normalized genre labels are nonnegative sparse features;
  placeholder genre `unknown` is not treated as preference evidence.
- User profiles aggregate known training works rated at least 4.
- Known initial genres can form a new-user profile without reading held-out
  ratings.
- Blank descriptions retain author/genre evidence. A zero-feature item is
  unavailable, while an available cosine score of zero remains valid.

## Verification and private outputs

Focused and regression checks:

```text
13 passed
```

The ignored private run directory is
`artifacts/experiments/c04_c06_seed42_candidates5000/` and contains the shared
catalog, Pearson grid, per-user metrics, input hashes, timings, fallback counts,
and `validation_summary.json`. Reloadable model files are under
`artifacts/models/c04_c06_seed42_candidates5000/`.

The runner does not accept or read `test_ratings.csv`. The test split remains
reserved for the frozen C10 comparison after C07 selects the Hybrid weight.
