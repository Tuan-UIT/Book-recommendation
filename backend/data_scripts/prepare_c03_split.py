"""Create reproducible train/validation/test splits for the C02 output.

The interaction CSV has no timestamp.  ``source_row_number`` is therefore
provenance only; it must not be interpreted as recency.  Splits are assigned
with a deterministic per-user shuffle instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


COLS = [
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
NAMES = ("train", "validation", "test")
VERSION = "c03-split-v2"


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _sorted_user_ids(values: set[str] | list[str]) -> list[str]:
    """Sort numeric research IDs numerically while remaining string-safe."""

    return sorted(
        {str(value) for value in values},
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )


def load_ratings(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    missing = [column for column in COLS if column not in frame]
    if missing:
        raise ValueError(f"missing C02 columns: {missing}")
    frame = frame[COLS].copy()

    for column in ("research_user_id", "canonical_item_id", "pair_id"):
        if frame[column].eq("").any():
            raise ValueError(f"C02 column {column} contains an empty value")

    frame["rating"] = pd.to_numeric(frame["rating"], errors="raise")
    frame["source_row_number"] = pd.to_numeric(
        frame["source_row_number"], errors="raise"
    )
    if not frame["rating"].between(1, 5).all():
        raise ValueError("C03 accepts explicit ratings 1 through 5 only")
    if not (frame["rating"] % 1).eq(0).all():
        raise ValueError("ratings must be integer values")
    if frame["pair_id"].duplicated().any():
        raise ValueError("duplicate pair_id in C02 input")
    if frame.duplicated(["research_user_id", "canonical_item_id"]).any():
        raise ValueError("duplicate user--work pair in C02 input")
    if frame["source_row_number"].duplicated().any():
        raise ValueError("duplicate source row number in C02 input")
    return frame


def load_sample_users(path: Path) -> set[str]:
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    if "research_user_id" not in frame:
        raise ValueError("sample_user_ids.csv must contain research_user_id")
    users = frame["research_user_id"].astype("string")
    if users.eq("").any() or users.duplicated().any():
        raise ValueError("sample_user_ids.csv contains empty or duplicate user IDs")
    return set(users.tolist())


def _user_seed(seed: int, research_user_id: str) -> int:
    payload = f"{seed}\0{research_user_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def assign_splits(
    frame: pd.DataFrame, seed: int
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Assign every cleaned pair exactly once without using rating values."""

    ordered = frame.sort_values(
        ["research_user_id", "canonical_item_id", "pair_id"],
        kind="mergesort",
    )
    rows_by_split: dict[str, list[dict]] = {name: [] for name in NAMES}
    cohort_rows: list[dict] = []

    for user_id, group in ordered.groupby("research_user_id", sort=False):
        rows = group.to_dict("records")
        random.Random(_user_seed(seed, str(user_id))).shuffle(rows)
        count = len(rows)

        if count >= 10:
            validation_count = count // 10
            test_count = count // 10
            train_count = count - validation_count - test_count
            cohort = "standard_10plus"
            rows_by_split["train"].extend(rows[:train_count])
            rows_by_split["validation"].extend(
                rows[train_count : train_count + validation_count]
            )
            rows_by_split["test"].extend(rows[train_count + validation_count :])
        elif count >= 2:
            train_count = count - 1
            validation_count = 0
            test_count = 1
            cohort = "sparse_2_9"
            rows_by_split["train"].extend(rows[:train_count])
            rows_by_split["test"].extend(rows[train_count:])
        else:
            train_count = 0
            validation_count = 0
            test_count = 1
            cohort = "one_rating_no_history"
            rows_by_split["test"].extend(rows)

        cohort_rows.append(
            {
                "research_user_id": str(user_id),
                "clean_rating_count": count,
                "train_count": train_count,
                "validation_count": validation_count,
                "test_count": test_count,
                "cohort": cohort,
                "has_train_history": train_count > 0,
            }
        )

    parts = {
        name: pd.DataFrame(rows, columns=COLS).reset_index(drop=True)
        for name, rows in rows_by_split.items()
    }
    cohorts = pd.DataFrame(
        cohort_rows,
        columns=[
            "research_user_id",
            "clean_rating_count",
            "train_count",
            "validation_count",
            "test_count",
            "cohort",
            "has_train_history",
        ],
    )
    return parts, cohorts


def add_zero_history_users(cohorts: pd.DataFrame, sample_users: set[str]) -> pd.DataFrame:
    present = set(cohorts["research_user_id"].astype(str))
    missing = _sorted_user_ids(sample_users - present)
    if not missing:
        return cohorts
    zero_rows = pd.DataFrame(
        [
            {
                "research_user_id": user_id,
                "clean_rating_count": 0,
                "train_count": 0,
                "validation_count": 0,
                "test_count": 0,
                "cohort": "zero_history",
                "has_train_history": False,
            }
            for user_id in missing
        ],
        columns=cohorts.columns,
    )
    return pd.concat([cohorts, zero_rows], ignore_index=True)


def training_item_stats(train: pd.DataFrame) -> pd.DataFrame:
    columns = ["canonical_item_id", "train_rating_count", "train_rating_mean"]
    if train.empty:
        return pd.DataFrame(columns=columns)
    stats = train.groupby("canonical_item_id", as_index=False).agg(
        train_rating_count=("rating", "size"),
        train_rating_mean=("rating", "mean"),
    )
    stats["train_rating_count"] = stats["train_rating_count"].astype("int64")
    stats["train_rating_mean"] = stats["train_rating_mean"].round(6)
    return stats.sort_values("canonical_item_id", kind="mergesort").reset_index(
        drop=True
    )


def training_user_stats(train: pd.DataFrame) -> pd.DataFrame:
    columns = ["research_user_id", "train_rating_count", "train_rating_mean"]
    if train.empty:
        return pd.DataFrame(columns=columns)
    stats = train.groupby("research_user_id", as_index=False).agg(
        train_rating_count=("rating", "size"),
        train_rating_mean=("rating", "mean"),
    )
    stats["train_rating_count"] = stats["train_rating_count"].astype("int64")
    stats["train_rating_mean"] = stats["train_rating_mean"].round(6)
    return stats.sort_values("research_user_id", kind="mergesort").reset_index(
        drop=True
    )


def digest(frame: pd.DataFrame) -> str:
    data = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _split_counts(parts: dict[str, pd.DataFrame], user_id: str) -> dict[str, int]:
    return {
        name: int(frame["research_user_id"].astype(str).eq(user_id).sum())
        for name, frame in parts.items()
    }


def validate(
    source: pd.DataFrame,
    parts: dict[str, pd.DataFrame],
    item_stats: pd.DataFrame,
    user_stats: pd.DataFrame,
    cohorts: pd.DataFrame,
    sample_users: set[str],
    seed: int,
) -> dict:
    train, validation, test = (parts[name] for name in NAMES)
    all_rows = pd.concat([train, validation, test], ignore_index=True)
    source_pair_ids = set(source["pair_id"])
    all_pair_ids = list(all_rows["pair_id"])
    source_users = set(source["research_user_id"].astype(str))
    cohort_lookup = cohorts.set_index("research_user_id")

    split_rules_hold = True
    for user_id, row in cohort_lookup.iterrows():
        counts = _split_counts(parts, user_id)
        clean_count = int(row["clean_rating_count"])
        expected = (
            (0, 0, 0)
            if row["cohort"] == "zero_history"
            else (
                (clean_count - 2 * (clean_count // 10), clean_count // 10, clean_count // 10)
                if clean_count >= 10
                else ((clean_count - 1, 0, 1) if clean_count >= 2 else (0, 0, 1))
            )
        )
        actual = (counts["train"], counts["validation"], counts["test"])
        if actual != expected:
            split_rules_hold = False
            break

    expected_zero_history = sample_users - source_users
    actual_zero_history = set(
        cohorts.loc[cohorts["cohort"].eq("zero_history"), "research_user_id"].astype(str)
    )
    checks = {
        "every_row_assigned_once": (
            len(all_rows) == len(source)
            and set(all_pair_ids) == source_pair_ids
            and len(all_pair_ids) == len(set(all_pair_ids))
        ),
        "split_pair_ids_disjoint": len(set(all_pair_ids)) == len(all_rows),
        "user_work_pairs_remain_unique": not all_rows.duplicated(
            ["research_user_id", "canonical_item_id"]
        ).any(),
        "sample_users_cover_clean_users": source_users.issubset(sample_users),
        "zero_history_users_are_explicit": actual_zero_history == expected_zero_history,
        "split_rules_match_cohorts": split_rules_hold,
        "training_item_stats_recomputed_from_train_only": item_stats.equals(
            training_item_stats(train)
        ),
        "training_user_stats_recomputed_from_train_only": user_stats.equals(
            training_user_stats(train)
        ),
        "held_out_rows_absent_from_train": set(train["pair_id"]).isdisjoint(
            set(pd.concat([validation, test])["pair_id"])
        ),
        "seed_recorded": isinstance(seed, int),
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ValueError(f"C03 checks failed: {failed}")

    summary = {
        name: {
            "rows": int(len(frame)),
            "users": int(frame["research_user_id"].nunique()),
            "items": int(frame["canonical_item_id"].nunique()),
            "sha256": digest(frame),
        }
        for name, frame in parts.items()
    }
    cohort_counts = {
        str(key): int(value)
        for key, value in cohorts["cohort"].value_counts().sort_index().items()
    }
    return {
        "schema_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "input": {
            "rows": int(len(source)),
            "users_with_clean_ratings": int(source["research_user_id"].nunique()),
            "sample_users": int(len(sample_users)),
            "sha256": digest(source),
        },
        "split_policy": {
            "ordering": "canonical item ID and pair ID before seeded per-user shuffle",
            "randomness": "sha256(seed + research_user_id) initializes an independent deterministic shuffle",
            "standard_users": "users with at least 10 works: floor(0.1*n) validation and floor(0.1*n) test",
            "sparse_users": "users with 2-9 works: one random test work and the remainder in train; no validation",
            "one_rating_users": "one work held out for test with no training history",
            "zero_history_users": "sampled users absent from clean_ratings.csv remain an explicit functional cohort",
            "source_row_number": "provenance only; no timestamp or recency interpretation",
            "rating_zero": "excluded; C02 stores it separately and it is not a negative",
        },
        "cohorts": cohort_counts,
        "splits": summary,
        "training_only": {
            "item_stats_rows": int(len(item_stats)),
            "item_stats_sha256": digest(item_stats),
            "user_stats_rows": int(len(user_stats)),
            "user_stats_sha256": digest(user_stats),
            "fit_and_model_selection_input": "train_ratings.csv only",
            "validation_and_test_are_never_used_for_fit_or_selection": True,
        },
        "integrity_checks": checks,
    }


def create_split(input_dir: Path, output_dir: Path, seed: int = 42) -> dict:
    ratings_path = input_dir / "clean_ratings.csv"
    sample_users_path = input_dir / "sample_user_ids.csv"
    source = load_ratings(ratings_path)
    sample_users = load_sample_users(sample_users_path)
    parts, cohorts = assign_splits(source, seed)
    cohorts = add_zero_history_users(cohorts, sample_users)
    item_stats = training_item_stats(parts["train"])
    user_stats = training_user_stats(parts["train"])
    audit = validate(
        source,
        parts,
        item_stats,
        user_stats,
        cohorts,
        sample_users,
        seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in parts.items():
        write_csv(frame, output_dir / f"{name}_ratings.csv")
    write_csv(item_stats, output_dir / "training_item_stats.csv")
    write_csv(user_stats, output_dir / "training_user_stats.csv")
    write_csv(cohorts, output_dir / "user_cohorts.csv")
    write_json(output_dir / "split_audit.json", audit)

    source_manifest_path = input_dir / "sample_manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.exists()
        else {}
    )
    manifest = {
        "schema_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "source_schema_version": source_manifest.get("schema_version"),
        "source_sample_seed": source_manifest.get("seed"),
        "source_sample_users": source_manifest.get("sample_users"),
        "split_seed": seed,
        "files": [
            "split_audit.json",
            "split_manifest.json",
            "train_ratings.csv",
            "validation_ratings.csv",
            "test_ratings.csv",
            "training_item_stats.csv",
            "training_user_stats.csv",
            "user_cohorts.csv",
        ],
        "split_method": "deterministic seeded per-user shuffle with explicit sparse cohorts",
        "source_row_number_is_not_time": True,
    }
    write_json(output_dir / "split_manifest.json", manifest)
    return {"manifest": manifest, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = create_split(args.input_dir, args.output_dir, args.seed)
    print(
        json.dumps(
            {
                "manifest": result["manifest"],
                "integrity_checks": result["audit"]["integrity_checks"],
                "cohorts": result["audit"]["cohorts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
