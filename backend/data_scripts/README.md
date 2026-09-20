# Data and experiment scripts

Write the C01 mapping exercise here as `check_book_id_mapping.py`.

C02 preparation is implemented in `prepare_sample.py`. C03 splitting is
implemented in `prepare_c03_split.py`; see `C03_README.md` for the reproducible
command and the special sparse/zero-history cohorts.

Raw inputs remain at `../../../00.REF/00.DATA/`. Treat that directory as
read-only. Scripts should create derived files only under this project's
`data/` or `artifacts/` directories.
