# Installation and local demonstration

## Requirements

- A stable Python 3.12 release is recommended. The recorded frozen experiment
  used the locally available Python 3.12.0rc3 and must be described with that
  limitation.
- PostgreSQL is the target database. SQLite may be used only for the
  self-contained local demo and automated checks.
- Install the packages from `requirements.txt` in an isolated environment.

## PowerShell setup

```powershell
python -m venv .venv
& '.venv\Scripts\Activate.ps1'
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The example starts in demo mode with a local SQLite database and a self-created
fixture. Replace `SECRET_KEY` with a long random value before running. Never
commit `.env` or a real password. Initialize the schema and start Flask:

```powershell
$env:FLASK_APP = 'backend.app:create_app'
flask init-db
flask seed-demo
flask run
```

Open `http://127.0.0.1:5000`. For an internal PostgreSQL research installation,
set `DATABASE_URL` to a private connection string, set `DEMO_MODE=0`, and set
`MODEL_DIR` to `artifacts/models/c04_c07_cf_revision_20260921`. After `flask
init-db`, run `flask load-research-catalog`; it checks that all 5,000 work IDs
match the model and requires an empty database. A SQLite probe verified the
import and rating refresh; PostgreSQL remains unverified until a local
connection is provided. Do not use that configuration for a public demo.
Browser writes require CSRF tokens;
the templates and included JavaScript supply them automatically.

## Verification

```powershell
python -m pytest -q
python -m backend.data_scripts.run_c04_c07_validation
```

The frozen test has already been read once. Do not rerun it as a routine setup
check. Private Goodreads files and generated research artifacts must remain
outside the public repository.
