"""Focused C02 checks that do not require the private Goodreads downloads."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from prepare_sample import (
    METADATA_COLUMNS,
    build_metadata_outputs,
    load_metadata_for_books,
    map_shelf_to_genre,
    resolve_ratings,
    sample_users,
)


def test_user_sample_is_reproducible_without_ratings(tmp_path: Path) -> None:
    user_map = tmp_path / "user_id_map.csv"
    rows = ["user_id_csv,user_id"] + [f"{index},source-{index}" for index in range(20)]
    user_map.write_text("\n".join(rows) + "\n", encoding="utf-8")

    first, first_manifest = sample_users(user_map, sample_size=7, seed=42)
    second, second_manifest = sample_users(user_map, sample_size=7, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert first_manifest == second_manifest
    assert len(first) == 7


def test_shelf_mapping_does_not_treat_to_read_as_genre() -> None:
    assert map_shelf_to_genre("to-read") is None
    assert map_shelf_to_genre("historical-fiction") == "historical fiction"
    assert map_shelf_to_genre("science fiction") == "science fiction"


def test_metadata_join_and_work_conflict_resolution(tmp_path: Path) -> None:
    metadata_path = tmp_path / "books.json.gz"
    records = [
        {
            "book_id": "10",
            "work_id": "100",
            "title": "Short edition",
            "description": "",
            "language_code": "eng",
            "authors": [{"author_id": "1"}],
            "popular_shelves": [{"name": "to-read"}, {"name": "fantasy"}],
        },
        {
            "book_id": "11",
            "work_id": "100",
            "title": "Complete edition",
            "description": "A description",
            "language_code": "eng",
            "authors": [{"author_id": "1"}],
            "popular_shelves": [{"name": "fantasy"}],
        },
    ]
    with gzip.open(metadata_path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")

    metadata, details = load_metadata_for_books(
        metadata_path,
        {"10", "11"},
        cache_path=tmp_path / "cache.csv",
        progress_every=0,
    )
    assert details["metadata_missing"] == 0
    outputs = build_metadata_outputs(metadata, {"1": "Author One"}, tmp_path)
    assert outputs["clean_books"].iloc[0]["representative_goodreads_book_id"] == "11"
    assert set(outputs["edition_map"]["canonical_item_id"]) == {"work:100"}

    mapped = pd.DataFrame(
        [
            {"research_user_id": "7", "book_id_csv": "0", "goodreads_book_id": "10", "is_read": "1", "rating": "5", "is_reviewed": "0", "source_row_number": "1"},
            {"research_user_id": "7", "book_id_csv": "1", "goodreads_book_id": "11", "is_read": "1", "rating": "1", "is_reviewed": "0", "source_row_number": "2"},
            {"research_user_id": "7", "book_id_csv": "1", "goodreads_book_id": "11", "is_read": "1", "rating": "0", "is_reviewed": "0", "source_row_number": "3"},
        ]
    )
    clean, unrated, quarantined, audit = resolve_ratings(mapped, outputs["edition_map"])

    assert len(clean) == 1
    assert clean.iloc[0]["goodreads_book_id"] == "10"
    assert clean.iloc[0]["resolution_reason"] == "edition_conflict_lowest_source_book_id"
    assert len(unrated) == 1
    assert len(quarantined) == 0
    assert audit["edition_conflict_rows_removed"] == 1


def test_metadata_cache_is_not_treated_as_source_duplicate(tmp_path: Path) -> None:
    metadata_path = tmp_path / "books.json.gz"
    cache_path = tmp_path / "cache.csv"
    records = [
        {
            "book_id": "10",
            "work_id": "100",
            "title": "Cached book",
            "description": "From source",
            "language_code": "eng",
            "authors": [],
            "popular_shelves": [],
        },
        {
            "book_id": "11",
            "work_id": "110",
            "title": "New book",
            "description": "From source",
            "language_code": "eng",
            "authors": [],
            "popular_shelves": [],
        },
    ]
    with gzip.open(metadata_path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")

    pd.DataFrame(
        [
            {
                "goodreads_book_id": "10",
                "work_id": "100",
                "title": "Cached book",
                "description": "From cache",
                "language_code": "eng",
                "authors_json": "[]",
                "popular_shelves_json": "[]",
            }
        ],
        columns=METADATA_COLUMNS,
    ).to_csv(cache_path, index=False)

    metadata, details = load_metadata_for_books(
        metadata_path,
        {"10", "11"},
        cache_path=cache_path,
        progress_every=0,
    )

    assert set(metadata["goodreads_book_id"]) == {"10", "11"}
    assert details["metadata_cache_hits"] == 1
    assert details["metadata_missing"] == 0


def test_metadata_source_duplicate_is_rejected(tmp_path: Path) -> None:
    metadata_path = tmp_path / "books.json.gz"
    records = [
        {
            "book_id": "10",
            "work_id": "100",
            "title": "First record",
            "description": "",
            "language_code": "eng",
            "authors": [],
            "popular_shelves": [],
        },
        {
            "book_id": "10",
            "work_id": "100",
            "title": "Duplicate record",
            "description": "",
            "language_code": "eng",
            "authors": [],
            "popular_shelves": [],
        },
    ]
    with gzip.open(metadata_path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="duplicate book IDs: 10"):
        load_metadata_for_books(
            metadata_path,
            {"10"},
            cache_path=None,
            progress_every=0,
        )
