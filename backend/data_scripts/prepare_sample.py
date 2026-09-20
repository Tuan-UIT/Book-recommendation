"""Create a reproducible, work-level Goodreads development sample.

This is the C02 preparation step.  It deliberately does not split data or fit
any recommendation model.  The program samples compact research users first,
then streams the interaction file to collect their rows.  Rating ``0`` is
kept in a separate output and is never converted to a negative rating.

The raw UCSD Goodreads files are private inputs.  All generated files are
written below ``data/samples`` or ``data/processed`` in this project.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THESIS_ROOT = PROJECT_ROOT.parent.parent
RAW_DATA_DIR = THESIS_ROOT / "00.REF" / "00.DATA"

INTERACTIONS_PATH = RAW_DATA_DIR / "goodreads_interactions.csv"
BOOK_MAP_PATH = RAW_DATA_DIR / "book_id_map.csv"
BOOKS_METADATA_PATH = RAW_DATA_DIR / "goodreads_books.json.gz"
AUTHORS_METADATA_PATH = RAW_DATA_DIR / "goodreads_book_authors.json.gz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "samples" / "c02_seed42_users1000"
DEFAULT_METADATA_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "book_metadata_cache_v2.csv"

SCHEMA_VERSION = "c02-preparation-v1"
GENRE_MAPPING_VERSION = "ucsd-popular-shelves-v1"
DEFAULT_SEED = 42
DEFAULT_SAMPLE_USERS = 1_000
DEFAULT_CHUNK_SIZE = 250_000

MAPPED_COLUMNS = [
    "research_user_id",
    "book_id_csv",
    "goodreads_book_id",
    "is_read",
    "rating",
    "is_reviewed",
    "source_row_number",
]

METADATA_COLUMNS = [
    "goodreads_book_id",
    "work_id",
    "title",
    "description",
    "language_code",
    "authors_json",
    "popular_shelves_json",
]


# Community shelves are useful, imperfect genre evidence.  These are not
# user-preference scores and these shelves are intentionally excluded.
NON_GENRE_SHELVES = {
    "to-read",
    "toread",
    "currently-reading",
    "currently reading",
    "read",
    "owned",
    "own-it",
    "books-i-own",
    "favorites",
    "favourites",
    "library",
    "default",
    "books-about-books",
    "my-books",
    "collection",
    "kindle",
    "ebook",
    "ebooks",
    "audiobook",
    "audio-book",
    "audio-books",
    "did-not-finish",
    "dnf",
    "book-club",
    "bookclub",
    "read-2015",
    "read-2016",
    "read-2017",
    "read-2018",
    "read-2019",
    "read-2020",
    "read-2021",
    "read-2022",
    "read-2023",
    "read-2024",
    "read-2025",
    "read-2026",
}

# The order is deliberate: specific categories win over broad shelves such
# as "fiction".  The mapping is versioned and is written with every run.
GENRE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("science fiction", ("science fiction", "sci-fi", "scifi", "scifi")),
    ("historical fiction", ("historical fiction", "historical-fiction")),
    ("mystery", ("mystery", "mysteries")),
    ("thriller", ("thriller", "thrillers")),
    ("crime", ("crime", "crime fiction")),
    ("romance", ("romance", "romantic")),
    ("fantasy", ("fantasy", "fantasy fiction")),
    ("horror", ("horror", "horrors")),
    ("biography", ("biography", "biographies")),
    ("memoir", ("memoir", "memoirs")),
    ("history", ("history", "histories")),
    ("poetry", ("poetry", "poems", "poem")),
    ("young adult", ("young adult", "young-adult", "ya")),
    ("children", ("children", "childrens", "children's")),
    ("comics and graphic", ("comics", "comic", "graphic novels", "graphic-novels")),
    ("lgbtq+", ("lgbt", "lgbtq", "gay", "lesbian", "queer")),
    ("religion", ("religion", "christian", "christianity", "spirituality")),
    ("philosophy", ("philosophy", "philosophical")),
    ("psychology", ("psychology", "psychological")),
    ("self-help", ("self-help", "self help")),
    ("business", ("business", "economics", "finance")),
    ("education", ("education", "teaching")),
    ("cooking", ("cooking", "cookbooks", "food")),
    ("travel", ("travel", "travelogue")),
    ("art", ("art", "design")),
    ("non-fiction", ("non-fiction", "nonfiction", "non fiction")),
    ("literary fiction", ("literary fiction", "literary")),
    ("contemporary", ("contemporary", "contemporary fiction")),
    ("classics", ("classics", "classic")),
    ("fiction", ("fiction", "novels", "novel")),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    """Normalize Unicode and whitespace without inventing source content."""

    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def normalize_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = normalize_text(value)
    if not text:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def id_sort_key(value: str | None) -> tuple[int, int | str]:
    text = normalize_id(value)
    if text is None:
        return (1, "")
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def safe_json(value: Any) -> str:
    return json.dumps(value if value is not None else [], ensure_ascii=False, separators=(",", ":"))


def read_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object in {path} at line {line_number}")
            yield record


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def atomic_write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def file_fingerprint(path: Path, with_checksum: bool = False) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    if with_checksum:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def validate_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose another --output-dir or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def load_book_mapping(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["book_id_csv", "book_id"],
        dtype="string",
        keep_default_na=False,
    )
    frame["book_id_csv"] = frame["book_id_csv"].map(normalize_id)
    frame["book_id"] = frame["book_id"].map(normalize_id)
    if frame["book_id_csv"].isna().any() or frame["book_id"].isna().any():
        raise ValueError("book_id_map.csv contains a missing mapping key")
    if not frame["book_id_csv"].is_unique:
        raise ValueError("book_id_csv must be unique")
    if not frame["book_id"].is_unique:
        raise ValueError("Goodreads book_id must be unique in book_id_map.csv")
    return frame


def sample_users(
    user_map_path: Path,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reservoir-sample users without reading the interaction file."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    rng = random.Random(seed)
    reservoir: list[dict[str, str]] = []
    seen = 0
    seen_ids: set[str] = set()

    with user_map_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"user_id_csv", "user_id"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"user map must contain {sorted(required)}")
        for row in reader:
            compact_id = normalize_id(row.get("user_id_csv"))
            source_id = normalize_text(row.get("user_id"))
            if compact_id is None or not source_id:
                raise ValueError("user_id_map.csv contains a missing user ID")
            if compact_id in seen_ids:
                raise ValueError(f"duplicate compact user ID: {compact_id}")
            seen_ids.add(compact_id)
            seen += 1
            candidate = {"research_user_id": compact_id, "source_user_id": source_id}
            if len(reservoir) < sample_size:
                reservoir.append(candidate)
            else:
                replacement_index = rng.randrange(seen)
                if replacement_index < sample_size:
                    reservoir[replacement_index] = candidate

    reservoir.sort(key=lambda row: id_sort_key(row["research_user_id"]))
    selected = pd.DataFrame(reservoir, columns=["research_user_id", "source_user_id"])
    manifest = {
        "method": "reservoir_sampling",
        "seed": seed,
        "requested_users": sample_size,
        "available_users": seen,
        "selected_users": len(selected),
        "sampled_without_reading_ratings": True,
    }
    return selected, manifest


def stream_selected_interactions(
    interactions_path: Path,
    book_mapping: pd.DataFrame,
    selected_user_ids: set[str],
    output_path: Path,
    chunk_size: int,
    progress_every: int,
) -> dict[str, Any]:
    """Stream the huge interactions file and write only selected users."""

    if not selected_user_ids:
        raise ValueError("No selected users")
    mapping = book_mapping.set_index("book_id_csv")["book_id"]
    required = {"user_id", "book_id", "is_read", "rating", "is_reviewed"}
    counts = Counter()
    rows_seen = 0
    next_progress = progress_every if progress_every > 0 else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MAPPED_COLUMNS)
            writer.writeheader()
            chunks = pd.read_csv(
                interactions_path,
                dtype="string",
                keep_default_na=False,
                chunksize=chunk_size,
            )
            for chunk in chunks:
                if not required.issubset(chunk.columns):
                    raise ValueError(f"interactions file must contain {sorted(required)}")
                chunk_start = rows_seen
                rows_seen += len(chunk)
                counts["interaction_rows_scanned"] = rows_seen

                selected = chunk[chunk["user_id"].isin(selected_user_ids)].copy()
                if selected.empty:
                    if next_progress is not None and rows_seen >= next_progress:
                        print(f"Interaction rows scanned: {rows_seen:,}", flush=True)
                        next_progress += progress_every
                    continue

                counts["selected_interaction_rows"] += len(selected)
                selected["research_user_id"] = selected["user_id"].map(normalize_id)
                selected["book_id_csv"] = selected["book_id"].map(normalize_id)
                selected["goodreads_book_id"] = selected["book_id_csv"].map(mapping)
                selected["goodreads_book_id"] = selected["goodreads_book_id"].map(normalize_id)
                selected["source_row_number"] = [
                    chunk_start + int(position) + 1
                    for position in (selected.index - chunk.index[0])
                ]

                selected["rating_numeric"] = pd.to_numeric(selected["rating"], errors="coerce")
                valid_rating = (
                    selected["rating_numeric"].notna()
                    & (selected["rating_numeric"] % 1 == 0)
                    & selected["rating_numeric"].between(0, 5)
                )
                counts["selected_invalid_rating_rows"] += int((~valid_rating).sum())
                counts["selected_rating_zero_rows"] += int(
                    (valid_rating & selected["rating_numeric"].eq(0)).sum()
                )
                counts["selected_explicit_rating_rows"] += int(
                    (valid_rating & selected["rating_numeric"].between(1, 5)).sum()
                )
                counts["selected_book_mapping_missing_rows"] += int(
                    selected["goodreads_book_id"].isna().sum()
                )

                for row in selected[MAPPED_COLUMNS].itertuples(index=False, name=None):
                    writer.writerow(dict(zip(MAPPED_COLUMNS, row)))

                if next_progress is not None and rows_seen >= next_progress:
                    print(f"Interaction rows scanned: {rows_seen:,}", flush=True)
                    next_progress += progress_every
        temporary_path.replace(output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    counts["selected_users_with_rows"] = None
    return dict(counts)


def _metadata_record(book: dict[str, Any]) -> dict[str, Any] | None:
    book_id = normalize_id(book.get("book_id"))
    if book_id is None:
        return None
    return {
        "goodreads_book_id": book_id,
        "work_id": normalize_id(book.get("work_id")) or "",
        "title": normalize_text(book.get("title")),
        "description": normalize_text(book.get("description")),
        "language_code": normalize_text(book.get("language_code")).lower(),
        "authors_json": safe_json(book.get("authors", [])),
        "popular_shelves_json": safe_json(book.get("popular_shelves", [])),
    }


def load_metadata_for_books(
    metadata_path: Path,
    goodreads_book_ids: set[str],
    cache_path: Path | None,
    progress_every: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load requested metadata, using an extended local cache when present."""

    requested = {normalize_id(value) for value in goodreads_book_ids}
    requested.discard(None)
    if not requested:
        return pd.DataFrame(columns=METADATA_COLUMNS), {
            "metadata_requested": 0,
            "metadata_cache_hits": 0,
            "metadata_records_scanned": 0,
            "metadata_missing": 0,
        }
    if not metadata_path.exists():
        raise FileNotFoundError(f"Book metadata file not found: {metadata_path}")

    matches: dict[str, dict[str, Any]] = {}
    if cache_path and cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, dtype="string", keep_default_na=False)
            if set(METADATA_COLUMNS).issubset(cached.columns):
                cached = cached[cached["goodreads_book_id"].isin(requested)]
                matches = {
                    str(record["goodreads_book_id"]): record
                    for record in cached.to_dict(orient="records")
                }
        except pd.errors.EmptyDataError:
            pass

    found_before_scan = set(matches)
    missing_ids = requested - set(matches)
    scanned = 0
    duplicate_ids: set[str] = set()
    seen_new_ids: set[str] = set()
    if missing_ids:
        for book in read_json_lines(metadata_path):
            scanned += 1
            if progress_every and scanned % progress_every == 0:
                print(f"Metadata records scanned: {scanned:,}", flush=True)
            record = _metadata_record(book)
            if not record or record["goodreads_book_id"] not in missing_ids:
                continue
            book_id = record["goodreads_book_id"]
            if book_id in seen_new_ids:
                duplicate_ids.add(book_id)
                continue
            seen_new_ids.add(book_id)
            matches[book_id] = record
    if duplicate_ids:
        raise ValueError(
            "Requested metadata contains duplicate book IDs: "
            + ", ".join(sorted(duplicate_ids, key=id_sort_key)[:10])
        )

    if cache_path and matches:
        old = pd.DataFrame(columns=METADATA_COLUMNS)
        if cache_path.exists():
            try:
                candidate = pd.read_csv(cache_path, dtype="string", keep_default_na=False)
                if set(METADATA_COLUMNS).issubset(candidate.columns):
                    old = candidate[METADATA_COLUMNS]
            except pd.errors.EmptyDataError:
                pass
        new = pd.DataFrame(matches.values(), columns=METADATA_COLUMNS)
        combined = pd.concat([old, new], ignore_index=True).drop_duplicates(
            "goodreads_book_id", keep="last"
        )
        atomic_write_dataframe(combined, cache_path)

    missing = sorted(requested - set(matches), key=id_sort_key)
    details = {
        "metadata_requested": len(requested),
        "metadata_cache_hits": len(found_before_scan),
        "metadata_records_scanned": scanned,
        "metadata_missing": len(missing),
        "metadata_missing_ids_preview": missing[:20],
    }
    frame = pd.DataFrame(matches.values(), columns=METADATA_COLUMNS)
    return frame, details


def load_author_names(
    authors_path: Path | None,
    author_ids: set[str],
    allow_missing: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Read only the needed author names from the supplementary source."""

    requested = {normalize_id(value) for value in author_ids}
    requested.discard(None)
    if not requested:
        return {}, {"author_source": None, "author_ids_requested": 0, "author_names_found": 0}
    if authors_path is None or not authors_path.exists():
        if not allow_missing:
            expected = authors_path or AUTHORS_METADATA_PATH
            raise FileNotFoundError(
                "Author metadata is required for C02 but was not found at "
                f"{expected}. Download goodreads_book_authors.json.gz or pass "
                "--allow-missing-author-names to produce an explicitly incomplete audit."
            )
        return {}, {
            "author_source": str(authors_path or AUTHORS_METADATA_PATH),
            "author_source_available": False,
            "author_ids_requested": len(requested),
            "author_names_found": 0,
            "author_names_missing": len(requested),
        }

    names: dict[str, str] = {}
    for record in read_json_lines(authors_path):
        author_id = normalize_id(record.get("author_id"))
        if author_id not in requested:
            continue
        name = normalize_text(record.get("name"))
        if name:
            names[author_id] = name
    missing = requested - set(names)
    if missing and not allow_missing:
        preview = ", ".join(sorted(missing, key=id_sort_key)[:10])
        raise ValueError(
            f"Author metadata is missing {len(missing)} referenced names (for example: {preview}). "
            "Pass --allow-missing-author-names only if this limitation is intentional."
        )
    return names, {
        "author_source": str(authors_path),
        "author_source_available": True,
        "author_ids_requested": len(requested),
        "author_names_found": len(names),
        "author_names_missing": len(missing),
    }


def normalize_shelf(value: Any) -> str:
    text = normalize_text(value).lower().replace("_", "-")
    return re.sub(r"\s+", " ", text)


def map_shelf_to_genre(shelf: Any) -> str | None:
    normalized = normalize_shelf(shelf)
    if not normalized or normalized in NON_GENRE_SHELVES:
        return None
    for genre, tokens in GENRE_RULES:
        if normalized in tokens:
            return genre
        if any(token in normalized for token in tokens if len(token) >= 5):
            return genre
    return None


def parse_json_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def canonical_item_id(work_id: Any, goodreads_book_id: str) -> tuple[str, str]:
    normalized_work = normalize_id(work_id)
    if normalized_work and normalized_work != "0":
        return f"work:{normalized_work}", normalized_work
    return f"book:{goodreads_book_id}", ""


def build_metadata_outputs(
    metadata: pd.DataFrame,
    author_names: dict[str, str],
    output_dir: Path,
) -> dict[str, Any]:
    """Build canonical works, edition links, authors, genres and source tables."""

    if metadata.empty:
        empty_outputs = {
            "edition_map": pd.DataFrame(),
            "clean_books": pd.DataFrame(),
            "authors": pd.DataFrame(),
            "book_authors": pd.DataFrame(),
            "genres": pd.DataFrame(),
            "book_genres": pd.DataFrame(),
            "genre_mapping": pd.DataFrame(),
            "metadata_source": metadata,
        }
        return empty_outputs

    editions: list[dict[str, Any]] = []
    for record in metadata.to_dict(orient="records"):
        book_id = normalize_id(record["goodreads_book_id"])
        if book_id is None:
            continue
        item_id, normalized_work = canonical_item_id(record.get("work_id"), book_id)
        editions.append(
            {
                "goodreads_book_id": book_id,
                "canonical_item_id": item_id,
                "work_id": normalized_work,
                "title": normalize_text(record.get("title")),
                "description": normalize_text(record.get("description")),
                "language_code": normalize_text(record.get("language_code")).lower(),
                "authors": parse_json_array(record.get("authors_json")),
                "popular_shelves": parse_json_array(record.get("popular_shelves_json")),
            }
        )

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for edition in editions:
        grouped[edition["canonical_item_id"]].append(edition)

    representative_ids: dict[str, str] = {}
    clean_books: list[dict[str, Any]] = []
    edition_map: list[dict[str, Any]] = []
    author_rows: dict[str, dict[str, Any]] = {}
    book_author_rows: dict[tuple[str, str], dict[str, Any]] = {}
    book_genre_rows: dict[tuple[str, str], dict[str, Any]] = {}
    genre_mapping_rows: dict[tuple[str, str], dict[str, Any]] = {}

    for item_id, item_editions in sorted(grouped.items()):
        item_editions.sort(
            key=lambda edition: (
                not bool(edition["title"] and edition["description"]),
                not bool(edition["title"]),
                id_sort_key(edition["goodreads_book_id"]),
            )
        )
        representative = item_editions[0]
        representative_ids[item_id] = representative["goodreads_book_id"]
        authors_for_item: set[str] = set()
        genres_for_item: set[str] = set()

        for edition in item_editions:
            source_book_id = edition["goodreads_book_id"]
            edition_map.append(
                {
                    "goodreads_book_id": source_book_id,
                    "canonical_item_id": item_id,
                    "work_id": edition["work_id"],
                    "title": edition["title"],
                    "is_representative": source_book_id == representative["goodreads_book_id"],
                    "description_available": bool(edition["description"]),
                }
            )

            for author in edition["authors"]:
                if not isinstance(author, dict):
                    continue
                author_id = normalize_id(author.get("author_id"))
                if author_id is None:
                    continue
                authors_for_item.add(author_id)
                name = normalize_text(author_names.get(author_id) or author.get("name"))
                author_rows.setdefault(
                    author_id,
                    {
                        "author_id": author_id,
                        "author_name": name,
                        "author_display_name": name or "Unknown author",
                        "author_name_missing": not bool(name),
                    },
                )
                book_author_rows.setdefault(
                    (item_id, author_id),
                    {
                        "canonical_item_id": item_id,
                        "author_id": author_id,
                        "source_goodreads_book_id": source_book_id,
                    },
                )

            for shelf in edition["popular_shelves"]:
                raw_shelf = shelf.get("name") if isinstance(shelf, dict) else shelf
                normalized_shelf = normalize_shelf(raw_shelf)
                genre = map_shelf_to_genre(raw_shelf)
                mapping_key = (normalized_shelf, genre or "")
                genre_mapping_rows.setdefault(
                    mapping_key,
                    {
                        "source_shelf": normalize_text(raw_shelf),
                        "normalized_shelf": normalized_shelf,
                        "genre_label": genre or "",
                        "included_as_genre": bool(genre),
                        "mapping_version": GENRE_MAPPING_VERSION,
                    },
                )
                if genre:
                    genres_for_item.add(genre)
                    book_genre_rows.setdefault(
                        (item_id, genre),
                        {
                            "canonical_item_id": item_id,
                            "genre_label": genre,
                            "source_shelf": normalize_text(raw_shelf),
                            "mapping_version": GENRE_MAPPING_VERSION,
                        },
                    )

        if not genres_for_item:
            genres_for_item.add("unknown")
            book_genre_rows.setdefault(
                (item_id, "unknown"),
                {
                    "canonical_item_id": item_id,
                    "genre_label": "unknown",
                    "source_shelf": "",
                    "mapping_version": GENRE_MAPPING_VERSION,
                },
            )

        clean_books.append(
            {
                "canonical_item_id": item_id,
                "representative_goodreads_book_id": representative["goodreads_book_id"],
                "work_id": representative["work_id"],
                "title": representative["title"],
                "description": representative["description"],
                "language_code": representative["language_code"],
                "has_description": bool(representative["description"]),
                "has_author": bool(authors_for_item),
                "has_genre": genres_for_item != {"unknown"},
                "author_count": len(authors_for_item),
                "genre_count": len(genres_for_item - {"unknown"}),
                "edition_count": len(item_editions),
            }
        )

    genre_counts = Counter(row["genre_label"] for row in book_genre_rows.values())
    genres = [
        {
            "genre_label": label,
            "genre_id": re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "unknown",
            "book_count": count,
            "mapping_version": GENRE_MAPPING_VERSION,
        }
        for label, count in sorted(genre_counts.items())
    ]

    return {
        "edition_map": pd.DataFrame(edition_map),
        "clean_books": pd.DataFrame(clean_books),
        "authors": pd.DataFrame(list(author_rows.values())),
        "book_authors": pd.DataFrame(list(book_author_rows.values())),
        "genres": pd.DataFrame(genres),
        "book_genres": pd.DataFrame(list(book_genre_rows.values())),
        "genre_mapping": pd.DataFrame(list(genre_mapping_rows.values())),
        "metadata_source": metadata[METADATA_COLUMNS].copy(),
    }


def _empty_or_sorted(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return frame[columns].copy()


def resolve_ratings(
    mapped_interactions: pd.DataFrame,
    edition_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Quarantine invalid rows and resolve user-work duplicates deterministically."""

    if mapped_interactions.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, {
            "mapped_rows_loaded": 0,
            "explicit_rows_before_deduplication": 0,
            "clean_rating_rows": 0,
        }

    frame = mapped_interactions.copy()
    frame["rating_numeric"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame["goodreads_book_id"] = frame["goodreads_book_id"].map(normalize_id)
    frame["research_user_id"] = frame["research_user_id"].map(normalize_id)
    frame["source_row_number"] = pd.to_numeric(frame["source_row_number"], errors="coerce")
    valid_rating = (
        frame["rating_numeric"].notna()
        & (frame["rating_numeric"] % 1 == 0)
        & frame["rating_numeric"].between(0, 5)
    )
    frame["quarantine_reason"] = ""
    frame.loc[~valid_rating, "quarantine_reason"] = "invalid_rating"
    frame.loc[frame["goodreads_book_id"].isna(), "quarantine_reason"] = "book_mapping_missing"

    edition_lookup = edition_map[["goodreads_book_id", "canonical_item_id"]].drop_duplicates()
    frame = frame.merge(edition_lookup, on="goodreads_book_id", how="left", validate="many_to_one")
    frame.loc[
        frame["canonical_item_id"].isna() & frame["quarantine_reason"].eq(""),
        "quarantine_reason",
    ] = "metadata_unmatched"

    quarantined = frame[frame["quarantine_reason"].ne("")].copy()
    quarantine_columns = [
        "research_user_id",
        "book_id_csv",
        "goodreads_book_id",
        "canonical_item_id",
        "rating",
        "source_row_number",
        "quarantine_reason",
    ]
    quarantined = _empty_or_sorted(quarantined, quarantine_columns)

    usable = frame[frame["quarantine_reason"].eq("")].copy()
    usable["rating"] = usable["rating_numeric"].astype(int)
    unrated = usable[usable["rating"].eq(0)].copy()
    unrated_columns = [
        "research_user_id",
        "goodreads_book_id",
        "canonical_item_id",
        "source_row_number",
        "rating",
        "is_read",
        "is_reviewed",
    ]
    unrated = _empty_or_sorted(unrated, unrated_columns)

    explicit = usable[usable["rating"].between(1, 5)].copy()
    explicit_before = len(explicit)
    if explicit.empty:
        return pd.DataFrame(), unrated, quarantined, {
            "mapped_rows_loaded": len(mapped_interactions),
            "quarantined_rows": len(quarantined),
            "unrated_rows": len(unrated),
            "explicit_rows_before_deduplication": 0,
            "exact_duplicate_rows_removed": 0,
            "edition_conflict_rows_removed": 0,
            "clean_rating_rows": 0,
        }

    explicit["book_id_sort"] = explicit["goodreads_book_id"].map(id_sort_key)
    exact_order = explicit.sort_values(
        ["research_user_id", "goodreads_book_id", "source_row_number"],
        kind="mergesort",
    )
    exact_duplicate_mask = exact_order.duplicated(
        ["research_user_id", "goodreads_book_id"], keep="first"
    )
    exact_duplicate_rows_removed = int(exact_duplicate_mask.sum())
    no_exact_duplicates = exact_order.loc[~exact_duplicate_mask].copy()

    work_order = no_exact_duplicates.sort_values(
        ["research_user_id", "canonical_item_id", "book_id_sort", "source_row_number"],
        kind="mergesort",
    )
    work_group_sizes = work_order.groupby(
        ["research_user_id", "canonical_item_id"], dropna=False
    ).size()
    work_group_books = work_order.groupby(
        ["research_user_id", "canonical_item_id"], dropna=False
    )["goodreads_book_id"].nunique()
    edition_conflict_groups = int((work_group_books > 1).sum())
    edition_conflict_rows_removed = int((work_group_sizes[work_group_books > 1] - 1).sum())
    chosen = work_order.drop_duplicates(
        ["research_user_id", "canonical_item_id"], keep="first"
    ).copy()

    group_size_map = work_group_sizes.to_dict()
    book_count_map = work_group_books.to_dict()
    chosen["duplicate_group_size"] = [
        int(group_size_map[(row.research_user_id, row.canonical_item_id)])
        for row in chosen.itertuples()
    ]
    chosen["source_book_count"] = [
        int(book_count_map[(row.research_user_id, row.canonical_item_id)])
        for row in chosen.itertuples()
    ]
    chosen["resolution_reason"] = "unique"
    chosen.loc[chosen["duplicate_group_size"] > 1, "resolution_reason"] = (
        "edition_conflict_lowest_source_book_id"
    )
    chosen.loc[
        (chosen["duplicate_group_size"] > 1) & (chosen["source_book_count"] == 1),
        "resolution_reason",
    ] = "exact_user_book_duplicate_kept"
    chosen["pair_id"] = [
        hashlib.sha256(
            f"{row.research_user_id}|{row.canonical_item_id}".encode("utf-8")
        ).hexdigest()[:24]
        for row in chosen.itertuples()
    ]
    rating_columns = [
        "pair_id",
        "research_user_id",
        "canonical_item_id",
        "goodreads_book_id",
        "rating",
        "source_row_number",
        "duplicate_group_size",
        "source_book_count",
        "resolution_reason",
    ]
    clean = chosen[rating_columns].sort_values(
        ["research_user_id", "canonical_item_id"], kind="mergesort"
    )
    audit = {
        "mapped_rows_loaded": len(mapped_interactions),
        "quarantined_rows": len(quarantined),
        "unrated_rows": len(unrated),
        "explicit_rows_before_deduplication": explicit_before,
        "unique_user_book_rows_before_work_deduplication": len(no_exact_duplicates),
        "exact_duplicate_rows_removed": exact_duplicate_rows_removed,
        "edition_conflict_groups": edition_conflict_groups,
        "edition_conflict_rows_removed": edition_conflict_rows_removed,
        "clean_rating_rows": len(clean),
        "clean_unique_users": int(clean["research_user_id"].nunique()),
        "clean_unique_items": int(clean["canonical_item_id"].nunique()),
    }
    return clean, unrated, quarantined, audit


def write_output_tables(outputs: dict[str, Any], output_dir: Path) -> None:
    filenames = {
        "edition_map": "edition_map.csv",
        "clean_books": "clean_books.csv",
        "authors": "authors.csv",
        "book_authors": "book_authors.csv",
        "genres": "genres.csv",
        "book_genres": "book_genres.csv",
        "genre_mapping": "genre_mapping.csv",
        "metadata_source": "metadata_source.csv",
    }
    for key, filename in filenames.items():
        frame = outputs[key]
        atomic_write_dataframe(frame, output_dir / filename)


def build_source_manifest(
    author_path: Path | None,
    with_checksums: bool,
) -> dict[str, Any]:
    paths = {
        "interactions": INTERACTIONS_PATH,
        "book_id_map": BOOK_MAP_PATH,
        "user_id_map": RAW_DATA_DIR / "user_id_map.csv",
        "books_metadata": BOOKS_METADATA_PATH,
        "authors_metadata": author_path or AUTHORS_METADATA_PATH,
    }
    return {
        "source_page": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/UCSD%20Book%20Graph.html",
        "research_source": "UCSD Goodreads Book Graph, academic-use source",
        "raw_source_files": {
            name: file_fingerprint(path, with_checksums) for name, path in paths.items()
        },
        "checksums_computed": with_checksums,
        "public_data_not_copied_to_repository": True,
    }


def run_preparation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    validate_output_dir(output_dir, args.overwrite)

    required_paths = [INTERACTIONS_PATH, BOOK_MAP_PATH, args.user_map, BOOKS_METADATA_PATH]
    missing_inputs = [str(path) for path in required_paths if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing_inputs))

    selected_users, sampling = sample_users(args.user_map, args.sample_users, args.seed)
    atomic_write_dataframe(selected_users, output_dir / "sample_user_ids.csv")
    selected_ids = set(selected_users["research_user_id"].astype(str))

    print(
        f"Selected {len(selected_users):,} of {sampling['available_users']:,} users "
        f"with seed {args.seed}.",
        flush=True,
    )
    book_mapping = load_book_mapping(BOOK_MAP_PATH)
    mapped_path = output_dir / "sample_interactions_mapped.csv"
    interaction_counts = stream_selected_interactions(
        INTERACTIONS_PATH,
        book_mapping,
        selected_ids,
        mapped_path,
        args.chunk_size,
        args.progress_every,
    )
    mapped = pd.read_csv(mapped_path, dtype="string", keep_default_na=False)
    interaction_counts["selected_users_with_rows"] = int(mapped["research_user_id"].nunique())

    selected_goodreads_ids = {
        value
        for value in mapped["goodreads_book_id"].map(normalize_id)
        if value is not None
    }
    metadata, metadata_counts = load_metadata_for_books(
        BOOKS_METADATA_PATH,
        selected_goodreads_ids,
        args.metadata_cache,
        args.metadata_progress_every,
    )
    atomic_write_dataframe(metadata, output_dir / "metadata_joined.csv")

    referenced_author_ids: set[str] = set()
    for value in metadata["authors_json"].tolist():
        for author in parse_json_array(value):
            if isinstance(author, dict):
                author_id = normalize_id(author.get("author_id"))
                if author_id:
                    referenced_author_ids.add(author_id)
    author_names, author_counts = load_author_names(
        args.authors_metadata,
        referenced_author_ids,
        args.allow_missing_author_names,
    )
    outputs = build_metadata_outputs(metadata, author_names, output_dir)
    write_output_tables(outputs, output_dir)

    clean, unrated, quarantined, rating_audit = resolve_ratings(
        mapped, outputs["edition_map"]
    )
    atomic_write_dataframe(clean, output_dir / "clean_ratings.csv")
    atomic_write_dataframe(unrated, output_dir / "unrated_interactions.csv")
    atomic_write_dataframe(quarantined, output_dir / "quarantined_interactions.csv")

    audit = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "sampling": sampling,
        "interaction_stream": interaction_counts,
        "metadata": metadata_counts,
        "authors": author_counts,
        "ratings": rating_audit,
        "metadata_output": {
            "source_metadata_rows": len(metadata),
            "clean_books": len(outputs["clean_books"]),
            "edition_map_rows": len(outputs["edition_map"]),
            "authors": len(outputs["authors"]),
            "book_author_links": len(outputs["book_authors"]),
            "genres": len(outputs["genres"]),
            "book_genre_links": len(outputs["book_genres"]),
        },
        "density": (
            len(clean) / (len(selected_users) * max(len(outputs["clean_books"]), 1))
            if len(selected_users)
            else 0.0
        ),
        "rules": {
            "valid_rating": "integer 1-5 only",
            "rating_zero": "kept separately as unrated; never treated as negative",
            "canonical_item": "work:<work_id> when nonzero, else book:<goodreads_book_id>",
            "edition_conflict": "lowest numeric Goodreads source book ID, then source row number",
            "display_edition": "title and description available first, then lowest source book ID",
            "duplicate_pair_id": "sha256(research_user_id|canonical_item_id)[:24]",
            "genre_policy": "versioned mapping from community shelves; unknown remains unknown",
            "no_splitting_or_model_statistics": True,
        },
    }
    atomic_write_json(output_dir / "audit_report.json", audit)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "seed": args.seed,
        "sample_users": args.sample_users,
        "chunk_size": args.chunk_size,
        "output_dir": str(output_dir),
        "metadata_cache": str(args.metadata_cache) if args.metadata_cache else None,
        "authors_metadata": str(args.authors_metadata) if args.authors_metadata else None,
        "allow_missing_author_names": args.allow_missing_author_names,
        "genre_mapping_version": GENRE_MAPPING_VERSION,
        "software": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
        },
        "source_manifest": build_source_manifest(args.authors_metadata, args.checksums),
        "files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    atomic_write_json(output_dir / "sample_manifest.json", manifest)
    print(f"C02 preparation complete: {output_dir}", flush=True)
    print(f"Clean ratings: {len(clean):,}; canonical items: {len(outputs['clean_books']):,}", flush=True)
    return {"output_dir": output_dir, "audit": audit, "manifest": manifest}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-users", type=int, default=DEFAULT_SAMPLE_USERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    parser.add_argument("--metadata-progress-every", type=int, default=250_000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--user-map", type=Path, default=RAW_DATA_DIR / "user_id_map.csv")
    parser.add_argument("--authors-metadata", type=Path, default=None)
    parser.add_argument("--metadata-cache", type=Path, default=DEFAULT_METADATA_CACHE_PATH)
    parser.add_argument(
        "--allow-missing-author-names",
        action="store_true",
        help="write explicit Unknown author values and record the incomplete join",
    )
    parser.add_argument(
        "--checksums",
        action="store_true",
        help="compute SHA-256 hashes for the large raw input files",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_preparation(args)


if __name__ == "__main__":
    main()
