# C02: reproducible sampled and cleaned data

Run `prepare_sample.py` from the project root after downloading the official
supplementary author file into the private raw-data directory:

```powershell
python backend/data_scripts/prepare_sample.py `
  --authors-metadata '..\..\00.REF\00.DATA\goodreads_book_authors.json.gz'
```

The default configuration samples 1,000 compact research users independently
of their ratings with seed 42, streams `goodreads_interactions.csv`, maps
compact book IDs through `book_id_map.csv`, and reads only the needed book
metadata. It does not split data or fit a model.

Outputs are private and ignored by Git under
`data/samples/c02_seed42_users1000/`:

- `sample_user_ids.csv` and `sample_interactions_mapped.csv`
- `clean_ratings.csv` for integer ratings 1--5
- `unrated_interactions.csv` for rating 0; these rows are never negative ratings
- `clean_books.csv` and `edition_map.csv` with work-level canonical IDs
- `authors.csv`, `book_authors.csv`, `genres.csv`, `book_genres.csv`, and
  `genre_mapping.csv`
- `audit_report.json`, `sample_manifest.json`, and `metadata_source.csv`

Edition conflicts use the lowest numeric Goodreads source book ID, then the
source row number. The representative display edition prefers a title and
description, then the lowest source book ID. Community shelves are normalized
with the versioned mapping `ucsd-popular-shelves-v1`; shelves such as `to-read`
and `owned` are excluded, and unknown genre remains `unknown`.

The author file is required by default. For an explicitly incomplete smoke run
only, pass `--allow-missing-author-names`; the audit records the missing join
and uses `Unknown author` as the display fallback.

Run the focused checks with:

```powershell
python -m pytest backend/data_scripts/test_prepare_sample.py -q
```
