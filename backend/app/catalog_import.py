"""Load only model-matched research catalog rows into a private database."""

from __future__ import annotations

import csv
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def load_research_catalog(db, *, metadata_dir: Path, candidate_path: Path,
                          model_candidate_ids: set[str]) -> dict[str, int]:
    if db.execute("SELECT COUNT(*) AS n FROM books").fetchone()["n"]:
        raise ValueError("load the research catalog into a fresh, empty database")
    candidates = {row["canonical_item_id"] for row in rows(candidate_path)}
    if not candidates or candidates != model_candidate_ids:
        raise ValueError("candidate catalog does not match the loaded model")

    books = []
    for row in rows(metadata_dir / "clean_books.csv"):
        if row["canonical_item_id"] in candidates:
            books.append((row["canonical_item_id"],
                          row["representative_goodreads_book_id"],
                          row["title"], row["description"], row["language_code"], False))
    if {row[0] for row in books} != candidates:
        raise ValueError("metadata is missing candidate work IDs")

    linked_authors = sorted({(row["canonical_item_id"], row["author_id"])
                             for row in rows(metadata_dir / "book_authors.csv")
                             if row["canonical_item_id"] in candidates})
    linked_genres = {(row["canonical_item_id"], row["genre_label"])
                     for row in rows(metadata_dir / "book_genres.csv")
                     if row["canonical_item_id"] in candidates and row["genre_label"]}
    needed_author_ids = {author_id for _, author_id in linked_authors}
    authors = [(row["author_id"], row["author_display_name"],
                row["author_name_missing"].strip().lower() in {"true", "1"})
               for row in rows(metadata_dir / "authors.csv")
               if row["author_id"] in needed_author_ids]
    if {row[0] for row in authors} != needed_author_ids:
        raise ValueError("author lookup is missing candidate author IDs")
    genres = [(label, label) for label in sorted({label for _, label in linked_genres})]

    try:
        db.executemany(
            "INSERT INTO books (canonical_item_id, representative_source_book_id, title, description, language_code, is_demo) "
            "VALUES (%s, %s, %s, %s, %s, %s)", books,
        )
        db.executemany(
            "INSERT INTO authors (author_id, display_name, name_missing) VALUES (%s, %s, %s)", authors,
        )
        db.executemany(
            "INSERT INTO genres (genre_id, label) VALUES (%s, %s)", genres,
        )
        db.executemany(
            "INSERT INTO book_authors (canonical_item_id, author_id) VALUES (%s, %s)", linked_authors,
        )
        db.executemany(
            "INSERT INTO book_genres (canonical_item_id, genre_id) VALUES (%s, %s)",
            sorted(linked_genres),
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"books": len(books), "authors": len(authors), "book_authors": len(linked_authors),
            "genres": len(genres), "book_genres": len(linked_genres)}
