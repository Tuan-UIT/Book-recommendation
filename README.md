# Book Recommendation System for Readers

Student implementation workspace for thesis 24540033.

The signed proposal under `../../02.FN/` and the project guide in
`../../PROJECT_GUIDE/` control the project scope. The large Goodreads source
files stay in `../../00.REF/00.DATA/`; do not copy or commit them here.

## Layout

- `backend/app/`: Flask application, database schemas and public-safe demo data.
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
folders. It still belongs only to this project. The frozen experiment used the
locally available Python 3.12.0rc3. Keep that fact in the report; use a stable
Python release for deployment and record any resulting environment change.

Copy `.env.example` to `.env` only when database configuration begins. Never
commit `.env` or real passwords.

## Current status

C01-C10 and the local demonstration flow have implementation and test
evidence. The original validation selected Pearson `k=20`, minimum overlap
`5`, and Hybrid `alpha=0.0`; its frozen comparison was run on 738 shared
eligible test users. A later validation-selected support correction and
Hybrid `alpha=0.25` have a separately disclosed second test access. See
`docs/CF_REVISION_RESULTS.md` and retain the original `docs/C10_RESULTS.md`.

The current website uses the visual direction of `../../00.REF/01.FE` while
keeping Flask and the server-side recommendation models. Store/cart, browser
password storage and browser-trained recommendations from that reference are
outside the signed proposal. The local demo uses only self-created fixtures.

Run all checks and the validation experiment with:

```powershell
python -m pytest -q
```

Do not rerun either saved test as a routine check. Use the installation guide
for demo configuration and the private research catalog import.

The C02 output retains the one source record with a missing author name as
`Unknown author`. This is a limitation of the source data, not an invented
name.

Local installation and demonstration steps are in `docs/INSTALLATION.md` and
`docs/DEMO_SCRIPT.md`. A public hosting URL and formal supervisor acceptance
remain external actions; they are not claimed as complete.
