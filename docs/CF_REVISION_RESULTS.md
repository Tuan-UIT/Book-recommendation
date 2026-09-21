# Pearson support revision and second test access

Date: 21 September 2026. This is a correction after the original C10 test
result had already been seen. The original model, validation artifacts and
first test summary remain unchanged. The new test is **access number two**;
it must not be described as the untouched first test.

## Reason and selection

The original CF validation NDCG@10 was 0.00609. Among 6,260 original CF top-ten
items before fallback, 6,053 had support from only one neighbour. Of 775,409
raw candidate predictions across the 660 shared validation users, 122,862
fell outside 1–5. The review tested a fixed small grid using validation labels
only: minimum candidate support {1,2,3}, shrinkage {0,2}, and optional clipping
to 1–5. The best validation NDCG used support 3, shrinkage 2 and clipping.
The score first clips the Pearson prediction to 1–5, then moves its deviation
from the target mean toward zero by `support / (support + 2)`.

On 660 validation users, this CF setting reached P@10 0.03197, R@10 0.06453
and NDCG@10 0.06155, with 60 full fallbacks. The unchanged content model
reached NDCG@10 0.11619. Rechecking the prespecified Hybrid alpha grid
{0, .25, .5, .75, 1} on validation selected alpha 0.25: P@10 0.06333,
R@10 0.16330, NDCG@10 0.14369. The models still use the same training-only
catalog and original work-level split. The audit and revised validation
artifacts are private under `artifacts/experiments/cf_support_validation_20260921/`
and `artifacts/experiments/c04_c07_cf_revision_20260921/`.

## Original frozen test and second test

Both rows use the same 738 eligible users, 5,000 candidate works and 3,585
relevant candidate pairs. 61 test users had no positive label; 125 had
positives only outside the catalog. Ten evaluated users had no training
history, 35 had 1–4 ratings, and 693 had at least 5.

| Method | Original P@10 | Original R@10 | Original NDCG@10 | Second P@10 | Second R@10 | Second NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training popularity | 0.03266 | 0.07412 | 0.06559 | 0.03266 | 0.07412 | 0.06559 |
| Pearson CF | 0.00312 | 0.01183 | 0.00731 | 0.02669 | 0.05733 | 0.05226 |
| Content | 0.04499 | 0.12899 | 0.10266 | 0.04499 | 0.12899 | 0.10266 |
| Hybrid | 0.04499 | 0.12899 | 0.10266 | 0.05840 | 0.15166 | 0.13094 |

Second-test mean per-user generation time in the same loaded-model procedure:
popularity 0.041 ms, CF 10.353 ms, content 88.727 ms and Hybrid 141.769 ms.
These are offline timings, not HTTP request times. CF full fallback rose from
109 to 139 users after the support rule; content and Hybrid each had 15 full
fallbacks. Do not infer that every user benefits from the revision.

The second test was triggered by diagnosing the weak first-test CF result,
although the exact support rule and alpha were chosen on validation. Its
improvement is therefore a transparent observed correction, not a new
independent generalization estimate. An additional untouched sample would be
needed to estimate that claim. Keep both test reports in the thesis.

The second summary records the prior summary hash, selection source, test
access index, input hashes, configuration, cohorts and per-user private metrics
at `artifacts/experiments/c10_second_test_cf_revision_20260921/test_summary.json`.
