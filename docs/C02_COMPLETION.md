# C02 completion evidence

Run completed on 21 September 2026 with the private UCSD Goodreads files,
seed `42`, and 1,000 users sampled independently from `user_id_map.csv`.

Measured outputs are in the ignored directory
`data/samples/c02_seed42_users1000/`.

| Check | Result |
| --- | ---: |
| Users sampled | 1,000 / 876,145 |
| Interaction rows scanned | 228,648,342 |
| Selected interaction rows | 263,844 |
| Explicit 1--5 rows before deduplication | 119,312 |
| Rating-0 rows kept separately | 144,532 |
| Clean rating rows | 118,761 |
| Edition-conflict groups | 541 |
| Edition-conflict rows removed | 551 |
| Canonical metadata items | 109,034 |
| Metadata records unmatched | 0 |
| Referenced author IDs / names found | 68,387 / 68,386 |
| Missing descriptions | 8,284 |
| Missing language codes | 32,790 |
| Unknown-genre items | 1,494 |

The one missing author name is a limitation of the source data. It is preserved
as `Unknown author` and recorded in the audit; no name was invented. Rating 0
never enters `clean_ratings.csv`.
Edition conflicts use the lowest numeric Goodreads source book ID, then source
row number. The representative display edition prefers a nonempty title and
description, then the lowest source book ID.

Verification:

```powershell
python -m pytest backend/data_scripts/test_prepare_sample.py -q
python backend/data_scripts/validate_c02_output.py `
  --output-dir data/samples/c02_seed42_users1000
```

The focused regression suite completed with `5 passed`. The validation report
returned no errors and confirms unique user--work pairs and pair IDs, valid
1--5 ratings, separated rating-0 rows, unique edition/source keys, and
successful rating-to-metadata joins. C03 (splitting) has not been started.
