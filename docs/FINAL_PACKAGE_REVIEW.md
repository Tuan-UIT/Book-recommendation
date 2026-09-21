# Final package review against R01-R12

Review date: 22 September 2026

This review uses the signed six-page proposal as the authority. It distinguishes
local, evidenced work from actions that require a hosting account, official
university formatting, supervisor review or a live rehearsal.

| Requirement | Status | Evidence and limitation |
| --- | --- | --- |
| R01 reproducible research data | Complete locally | `backend/data_scripts/prepare_sample.py`, `prepare_c03_split.py`, C02/C03 audit documents and frozen private manifests. Goodreads rows remain private. The 14 September metadata/duplicate audit was sampled as documented, not a full clean-data claim. |
| R02 design and identity separation | Complete | `R02_DESIGN.md`, PostgreSQL/SQLite schemas and separate website-account versus research-user identifiers. |
| R03 authenticated website flow | Complete locally | Flask registration/login/logout, password hashing, search, details and 1-5 rating upsert. Focused tests reject unauthenticated/invalid writes and prove two-account ownership isolation. |
| R04 training popularity | Complete | Deterministic training-only counts and fallback behavior in `backend/recommenders/popularity.py`. |
| R05 user Pearson CF | Complete with later revision | Original standalone CF did not clip; Hybrid clipped it as the proposal requires. A later validation-selected standalone revision clips, shrinks by support and requires three supporting neighbours. Both test accesses are disclosed in `CF_REVISION_RESULTS.md`. |
| R06 content filtering | Complete | Training-only TF-IDF description plus author/genre blocks, cosine profiles from ratings at least 4 or selected initial preferences, and missing-description handling. |
| R07 normalized Hybrid | Complete with later revision | CF is clipped/mapped to [0,1], combined with content when available, and distinguishes unavailable components from numeric zero. Original validation selected `alpha=0.0`; revised validation selected `alpha=0.25`. |
| R08 top-ten explanations | Complete locally | Unique work-level recommendations exclude rated works, return at most ten, and retain component/evidence text for explanations. |
| R09 refresh after ratings | Complete locally | Rating upsert invalidates the account cache and rebuilds scores/profile; tests cover a rating change from 5 to 1. |
| R10 controlled comparison | Complete with disclosed correction | Original frozen test and a separate second test access use the same 738 eligible users and 5,000-work catalog. Selection for the correction used validation; the first test had already been seen. See `CF_REVISION_RESULTS.md`. |
| R11 sparse/missing cases | Complete locally | Tests cover new/sparse accounts, missing descriptions, fewer than ten candidates, component fallback and account isolation. |
| R12 handover package | Drafts complete; external items pending | Source, setup, schema/design, results, revised report and slides, and demo script exist. A permitted public deployment URL, university template check, supervisor acceptance and a timed live rehearsal remain open. |

## Final measured result

On the frozen test population, content and Hybrid tied at Precision@10
`0.04499`, Recall@10 `0.12899` and NDCG@10 `0.10266` because validation selected
zero CF weight. Popularity reached NDCG@10 `0.06559`; Pearson CF reached
`0.00731`. These are reported measurements, not evidence that Hybrid must beat
every component.

A later second test access measured revised Hybrid P@10 `0.05840`, R@10
`0.15166`, NDCG@10 `0.13094`; the original test had already been seen. See
`CF_REVISION_RESULTS.md` for the full old/new comparison and its limitation.

## Verification record

- The focused/regression suite passed 26 tests after website integration.
- `THESIS_PRESENTATION_REVISED.pptx` passed package/layout validation; all 12
  slides were rendered and inspected, including revised validation, test and
  latency charts, with no visible overlap or clipping.
- `THESIS_REPORT_REVISED.docx` loaded successfully and contains 104 paragraphs,
  9 tables, 1 inline figure and 1 section. Word/PDF visual rendering
  was unavailable in this environment, so final pagination and the official
  university layout still require a manual Word review.
- The revised Flask website was inspected in an in-app browser at 375, 768 and
  1440 px; registration, login and asynchronous rating refresh worked locally.
  This is one browser, not a cross-browser claim.
- A private SQLite integration probe loaded all 5,000 model candidate IDs and
  verified account rating and exclusion. PostgreSQL still needs credentials
  and a live flow before it can be claimed as verified.

## Remaining external actions

1. Open `THESIS_REPORT_REVISED.docx` in Word, apply the university template and
   confirm pagination, captions, references and Vietnamese fonts.
2. Deploy only the self-created demo fixture to an approved host, then record
   the public URL. Never upload Goodreads research rows or dataset-user IDs.
3. Run the `DEMO_SCRIPT.md` rehearsal with a timer and record issues/fixes.
4. Obtain supervisor review/acceptance and keep the university milestones
   through 27 December 2026 in the schedule.
