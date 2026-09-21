# C10 frozen test results

Date: 21 September 2026

The test runner verified the saved validation configuration and input hashes
before reading `test_ratings.csv`. It used the training-only fitted models,
the frozen 5,000-work candidate catalog and one shared population of 738 users
with at least one relevant test work in that catalog. The runner writes once
by default and refuses an accidental rerun.

| Method | Precision@10 | Recall@10 | NDCG@10 | Full fallback users | Mean latency per user |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training popularity | 0.03266 | 0.07412 | 0.06559 | 0 | 0.024 ms |
| User Pearson CF | 0.00312 | 0.01183 | 0.00731 | 109 | 10.124 ms |
| Content | 0.04499 | 0.12899 | 0.10266 | 15 | 63.054 ms |
| Hybrid, alpha 0.0 | 0.04499 | 0.12899 | 0.10266 | 15 | 109.115 ms |

Test cohort counts:

- 924 users had a test row.
- 61 had no positive test label.
- 125 had positive labels only outside the frozen catalog.
- 738 entered every method's macro-average, with 3,585 relevant candidate
  pairs.
- Among evaluated users, 10 had no training history, 35 had 1-4 training
  ratings, and 693 had at least 5.

The Hybrid tied content because validation selected zero weight for CF. The
Pearson method's weak result and 109 full fallbacks indicate limited useful
co-rating evidence in this sample and catalog. The random holdout protocol
measures recovery of hidden historical preferences, not future behavior or
quality for all Vietnamese readers.

Private evidence is saved in
`artifacts/experiments/c10_frozen_test_seed42_candidates5000/`. Aggregate
figures in `docs/figures/` are safe to include in the report.

**Later correction:** a validation-selected Pearson support rule and alpha 0.25
were evaluated in a separately recorded **second test access**. See
`CF_REVISION_RESULTS.md` for both result tables, the reason for the revision,
and the limitation created by having seen the original test result.
