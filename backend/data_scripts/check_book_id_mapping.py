"""Validate compact Goodreads IDs and attach sample metadata."""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THESIS_ROOT = PROJECT_ROOT.parent.parent
RAW_DATA_DIR = THESIS_ROOT / "00.REF" / "00.DATA"

INTERACTIONS_PATH = RAW_DATA_DIR / "goodreads_interactions.csv"
BOOK_MAP_PATH = RAW_DATA_DIR / "book_id_map.csv"
BOOKS_METADATA_PATH = RAW_DATA_DIR / "goodreads_books.json.gz"
METADATA_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "book_metadata_cache.csv"


def load_book_mapping(path: Path) -> pd.DataFrame:
    """Load and validate the compact-ID-to-Goodreads-ID mapping."""

    book_map = pd.read_csv(
        path,
        usecols=["book_id_csv", "book_id"],
        dtype={"book_id_csv": "string", "book_id": "string"},
    )
    missing_compact_ids = int(book_map["book_id_csv"].isna().sum())
    missing_goodreads_ids = int(book_map["book_id"].isna().sum())

    if missing_compact_ids:
        raise ValueError(f"book_id_csv contains {missing_compact_ids} missing values")
    if missing_goodreads_ids:
        raise ValueError(f"book_id contains {missing_goodreads_ids} missing values")
    if not book_map["book_id_csv"].is_unique:
        raise ValueError("book_id_csv must be unique")
    if not book_map["book_id"].is_unique:
        raise ValueError("book_id must be unique")
    return book_map


def map_interactions(
    interactions_path: Path,
    book_map: pd.DataFrame,
    sample_size: int = 20,
) -> pd.DataFrame:
    """Map a small interaction sample to Goodreads IDs."""

    interactions = pd.read_csv(
        interactions_path,
        nrows=sample_size,
        dtype={"book_id": "string"},
    )
    if "book_id" not in interactions.columns:
        raise ValueError("interactions file does not contain a book_id column")

    interactions = interactions.rename(columns={"book_id": "book_id_csv"})
    mapped = interactions.merge(
        book_map,
        on="book_id_csv",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched = mapped.loc[mapped["_merge"] == "left_only", "book_id_csv"]
    if not unmatched.empty:
        missing_ids = sorted(unmatched.dropna().astype(str).unique())
        raise ValueError(
            f"{len(unmatched)} interaction rows have no book mapping: {missing_ids}"
        )
    if not mapped["_merge"].eq("both").all():
        raise ValueError("unexpected merge status while mapping interactions")

    return mapped.rename(columns={"book_id": "goodreads_book_id"}).drop(
        columns="_merge"
    )


def _write_metadata_cache(
    cache_path: Path | None,
    records: dict[str, dict[str, object]],
) -> None:
    """Persist metadata records atomically so interrupted scans remain useful."""

    if cache_path is None or not records:
        return

    columns = ["goodreads_book_id", "work_id", "title"]
    new_cache = pd.DataFrame(records.values(), columns=columns)
    if cache_path.exists():
        try:
            old_cache = pd.read_csv(cache_path, dtype={"goodreads_book_id": "string"})
            new_cache = pd.concat([old_cache, new_cache], ignore_index=True)
        except pd.errors.EmptyDataError:
            pass

    new_cache = new_cache.drop_duplicates("goodreads_book_id", keep="last")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    new_cache.to_csv(temporary_path, index=False)
    temporary_path.replace(cache_path)


def load_metadata(
    metadata_path: Path,
    goodreads_book_ids: set[str],
    progress_every: int | None = 250_000,
    cache_path: Path | None = METADATA_CACHE_PATH,
) -> pd.DataFrame:
    """Read only requested records from the line-delimited gzip metadata."""

    columns = ["goodreads_book_id", "work_id", "title"]
    matches: dict[str, dict[str, object]] = {}
    if cache_path and cache_path.exists():
        try:
            cached = pd.read_csv(
                cache_path,
                dtype={"goodreads_book_id": "string"},
            )
            cached = cached[cached["goodreads_book_id"].astype(str).isin(goodreads_book_ids)]
            matches = {
                str(record["goodreads_book_id"]): record
                for record in cached.to_dict(orient="records")
            }
        except (pd.errors.EmptyDataError, KeyError):
            matches = {}

    found_ids = set(matches)
    if found_ids:
        print(
            f"Metadata cache hits: {len(found_ids)}/{len(goodreads_book_ids)}",
            flush=True,
        )

    if found_ids != goodreads_book_ids:
        try:
            with gzip.open(metadata_path, mode="rt", encoding="utf-8") as metadata_file:
                for line_number, line in enumerate(metadata_file, start=1):
                    if progress_every and line_number % progress_every == 0:
                        print(f"Scanned metadata records: {line_number:,}", flush=True)
                    try:
                        book = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSON in metadata at line {line_number}"
                        ) from error

                    metadata_id = str(book.get("book_id", ""))
                    if metadata_id in goodreads_book_ids and metadata_id not in found_ids:
                        matches[metadata_id] = {
                            "goodreads_book_id": metadata_id,
                            "work_id": book.get("work_id"),
                            "title": book.get("title"),
                        }
                        found_ids.add(metadata_id)
                        if found_ids == goodreads_book_ids:
                            break
        finally:
            _write_metadata_cache(cache_path, matches)

    missing_ids = goodreads_book_ids - found_ids
    if missing_ids:
        raise ValueError(
            "Metadata records not found for Goodreads IDs: "
            f"{sorted(missing_ids)}"
        )
    return pd.DataFrame(
        [matches[metadata_id] for metadata_id in sorted(goodreads_book_ids)],
        columns=columns,
    )


def join_metadata(
    mapped: pd.DataFrame,
    metadata_path: Path,
    progress_every: int | None = 250_000,
    cache_path: Path | None = METADATA_CACHE_PATH,
) -> pd.DataFrame:
    """Join metadata to an already validated interaction mapping."""

    target_ids = set(mapped["goodreads_book_id"].astype(str))
    metadata = load_metadata(metadata_path, target_ids, progress_every, cache_path)
    joined = mapped.merge(
        metadata,
        on="goodreads_book_id",
        how="left",
        validate="many_to_one",
    )
    if joined[["work_id", "title"]].isna().all(axis=1).any():
        raise ValueError("some mapped interactions have no joined metadata")
    return joined


def build_sample(
    interactions_path: Path = INTERACTIONS_PATH,
    book_map_path: Path = BOOK_MAP_PATH,
    metadata_path: Path = BOOKS_METADATA_PATH,
    sample_size: int = 20,
    progress_every: int | None = 250_000,
    metadata_cache_path: Path | None = METADATA_CACHE_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the mapped interaction sample and its joined metadata."""

    book_map = load_book_mapping(book_map_path)
    mapped = map_interactions(interactions_path, book_map, sample_size)
    joined = join_metadata(mapped, metadata_path, progress_every, metadata_cache_path)
    return mapped, joined


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="number of interaction rows to validate (default: 20)",
    )
    parser.add_argument(
        "--with-metadata",
        action="store_true",
        help="scan the large gzip metadata file and join book details",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250_000,
        help="print metadata progress every N records",
    )
    args = parser.parse_args(argv)

    book_map = load_book_mapping(BOOK_MAP_PATH)
    mapped = map_interactions(INTERACTIONS_PATH, book_map, args.sample_size)
    print(f"Interaction rows checked: {len(mapped)}")
    print(f"Unique Goodreads IDs: {mapped['goodreads_book_id'].nunique()}")

    if not args.with_metadata:
        print("Metadata join skipped.")
        print("Use --with-metadata to scan goodreads_books.json.gz.")
        return

    try:
        joined = join_metadata(
            mapped,
            BOOKS_METADATA_PATH,
            args.progress_every,
            METADATA_CACHE_PATH,
        )
    except KeyboardInterrupt:
        raise SystemExit(
            "Metadata scan interrupted. Mapping validation succeeded; "
            "rerun with --with-metadata to continue the metadata scan."
        ) from None
    print(f"Metadata records joined: {joined['title'].notna().sum()}")
    print("\nJoined sample:")
    print(
        joined[["book_id_csv", "goodreads_book_id", "work_id", "title"]]
        .head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
