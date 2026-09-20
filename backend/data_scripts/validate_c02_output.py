"""Validate and enrich the audit report produced by ``prepare_sample.py``."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate(output_dir: Path) -> dict:
    audit_path = output_dir / "audit_report.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    books = pd.read_csv(output_dir / "clean_books.csv", dtype="string", keep_default_na=False)
    ratings = pd.read_csv(output_dir / "clean_ratings.csv", dtype="string", keep_default_na=False)
    unrated = pd.read_csv(output_dir / "unrated_interactions.csv", dtype="string", keep_default_na=False)
    authors = pd.read_csv(output_dir / "authors.csv", dtype="string", keep_default_na=False)
    editions = pd.read_csv(output_dir / "edition_map.csv", dtype="string", keep_default_na=False)

    clean_pair_unique = not ratings.duplicated(["research_user_id", "canonical_item_id"]).any()
    pair_id_unique = ratings["pair_id"].is_unique
    rating_values_valid = ratings["rating"].astype(int).between(1, 5).all()
    zero_values_separate = unrated["rating"].astype(int).eq(0).all()
    edition_keys_unique = editions["goodreads_book_id"].is_unique
    work_keys_unique = books["canonical_item_id"].is_unique
    rating_editions_join = ratings["goodreads_book_id"].isin(
        set(editions["goodreads_book_id"])
    ).all()
    rating_works_join = ratings["canonical_item_id"].isin(
        set(books["canonical_item_id"])
    ).all()

    audit["missingness"] = {
        "clean_books": len(books),
        "missing_description_books": int(books["has_description"].eq("False").sum()),
        "missing_language_books": int(books["language_code"].eq("").sum()),
        "missing_author_books": int(books["has_author"].eq("False").sum()),
        "unknown_genre_books": int(books["has_genre"].eq("False").sum()),
        "authors_with_missing_names": int(
            authors["author_name_missing"].str.lower().eq("true").sum()
        ),
        "unrated_rows_kept_separately": len(unrated),
    }
    audit["integrity_checks"] = {
        "pair_id_unique": bool(pair_id_unique),
        "user_work_pairs_unique": bool(clean_pair_unique),
        "ratings_are_integer_1_to_5": bool(rating_values_valid),
        "rating_zero_is_separate": bool(zero_values_separate),
        "edition_map_source_ids_unique": bool(edition_keys_unique),
        "clean_books_work_ids_unique": bool(work_keys_unique),
        "clean_ratings_join_to_edition_map": bool(rating_editions_join),
        "clean_ratings_join_to_clean_books": bool(rating_works_join),
    }
    audit["validated_utc"] = now_utc()
    if not all(audit["integrity_checks"].values()):
        raise ValueError(f"C02 integrity checks failed: {audit['integrity_checks']}")
    write_json(audit_path, audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.output_dir)
    print(json.dumps({"missingness": result["missingness"], "integrity_checks": result["integrity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
