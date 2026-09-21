# C07 Hybrid completion evidence

Date: 21 September 2026

The Hybrid recommender clips each Pearson prediction to 1-5, maps it to
`[0,1]`, and combines it with the content cosine score. A valid single
component is used directly. A missing component remains unavailable rather
than becoming numeric zero. Items with a valid component precede popularity
backfill items, and canonical item ID breaks score ties.

Validation searched `alpha` in `{0, 0.25, 0.5, 0.75, 1}` on the same 660
eligible users and the same 5,000-work catalog used by the other methods.
Maximum NDCG@10 selected `alpha = 0.0`. The measured Hybrid metrics therefore
equal the content metrics: Precision@10 0.05167, Recall@10 0.14197, and
NDCG@10 0.11619. This is a valid result, not evidence that combining the weak
Pearson component improved ranking.

The implementation retains component scores, valid-source status, support
counts and explanation evidence. Focused checks cover valid cosine zero,
unavailable fallback, re-rating 5 to 1, known-work exclusion and stable ties.
The corrected validation output is under the ignored private directory
`artifacts/experiments/c04_c07_seed42_candidates5000/`.

This section describes the original frozen configuration. A later Pearson
support audit selected a separate revision with alpha 0.25 on validation;
the first model and result were preserved. See `CF_REVISION_RESULTS.md`.
