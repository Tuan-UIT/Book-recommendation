"""Initialize the local SQLite database with public-safe self-created data."""

from backend.app import create_app
from backend.app.db import get_db, init_db
from backend.app.demo_data import seed_demo_catalog


def main() -> None:
    app = create_app()
    with app.app_context():
        init_db()
        seed_demo_catalog(get_db())
    print("Initialized data/demo.sqlite3 with the self-created demo catalog.")


if __name__ == "__main__":
    main()
