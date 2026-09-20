# Book Recommendation System for Readers

Student implementation workspace for thesis 24540033.

The signed proposal and the project guide in `../02.FN/PROJECT_GUIDE/` control
the project scope. The large Goodreads source files stay in
`../00.REF/00.DATA/`; do not copy or commit them here.

## Layout

- `backend/app/`: Flask application code (later tasks C08-C09).
- `backend/data_scripts/`: student-written preparation and experiment scripts.
- `backend/tests/`: small verification examples and focused tests.
- `frontend/templates/`: server-rendered HTML templates.
- `frontend/static/`: CSS and browser-side JavaScript.
- `data/processed/`: private generated clean data; ignored by Git.
- `data/samples/`: private development samples; ignored by Git.
- `artifacts/`: private models and experiment outputs; ignored by Git.
- `config/`: non-secret configuration documentation.
- `docs/`: diagrams and thesis evidence created during the project.

## Local setup (PowerShell)

From this directory:

```powershell
& '..\..\.codex_work\venvs\book-recommendation-system-win\Scripts\Activate.ps1'
python --version
python -m pip --version
python -m pip install -r requirements.txt
```

The environment is kept outside this project directory because this OneDrive
location blocks tools from creating a virtual environment inside newly created
folders. It still belongs only to this project. The currently available native
Windows interpreter is Python 3.12.0rc3; replace it with a current stable
Python release before freezing the final experiment environment.

Copy `.env.example` to `.env` only when database configuration begins. Never
commit `.env` or real passwords.

## Current task

C03: create reproducible train/validation/test splits and verify that all
modeling choices use training data only. See
`backend/data_scripts/C03_README.md` for the command and split policy.

The C02 output retains the one source record with a missing author name as
`Unknown author`. This is a limitation of the source data, not an invented
name.
