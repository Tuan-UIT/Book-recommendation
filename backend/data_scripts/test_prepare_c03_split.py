from __future__ import annotations

import pandas as pd

from prepare_c03_split import (
    COLS,
    add_zero_history_users,
    assign_splits,
    training_item_stats,
    training_user_stats,
    validate,
)


def make_ratings(counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    source_row = 100
    for user_id, count in counts.items():
        for index in range(count):
            rows.append(
                {
                    "pair_id": f"pair-{user_id}-{index}",
                    "research_user_id": user_id,
                    "canonical_item_id": f"work:{user_id}-{index}",
                    "goodreads_book_id": str(source_row),
                    "rating": (index % 5) + 1,
                    "source_row_number": source_row,
                    "duplicate_group_size": 1,
                    "source_book_count": 1,
                    "resolution_reason": "unique",
                }
            )
            source_row += 1
    return pd.DataFrame(rows, columns=COLS)


def split_pair_map(parts: dict[str, pd.DataFrame]) -> dict[str, set[str]]:
    return {name: set(frame["pair_id"]) for name, frame in parts.items()}


def test_split_rules_and_special_cohorts() -> None:
    source = make_ratings({"1": 10, "2": 5, "3": 1})
    parts, cohorts = assign_splits(source, seed=42)
    cohorts = add_zero_history_users(cohorts, {"1", "2", "3", "4"})

    counts = cohorts.set_index("research_user_id")
    assert tuple(counts.loc["1", ["train_count", "validation_count", "test_count"]]) == (8, 1, 1)
    assert tuple(counts.loc["2", ["train_count", "validation_count", "test_count"]]) == (4, 0, 1)
    assert tuple(counts.loc["3", ["train_count", "validation_count", "test_count"]]) == (0, 0, 1)
    assert tuple(counts.loc["4", ["train_count", "validation_count", "test_count"]]) == (0, 0, 0)
    assert counts.loc["3", "cohort"] == "one_rating_no_history"
    assert counts.loc["4", "cohort"] == "zero_history"

    item_stats = training_item_stats(parts["train"])
    user_stats = training_user_stats(parts["train"])
    audit = validate(source, parts, item_stats, user_stats, cohorts, {"1", "2", "3", "4"}, 42)
    assert all(audit["integrity_checks"].values())


def test_same_seed_is_reproducible_and_source_row_is_not_recency() -> None:
    source = make_ratings({"1": 10, "2": 5})
    first_parts, _ = assign_splits(source, seed=42)
    second_parts, _ = assign_splits(source.sample(frac=1, random_state=7), seed=42)
    assert split_pair_map(first_parts) == split_pair_map(second_parts)

    changed_source_order = source.copy()
    changed_source_order["source_row_number"] = list(range(900, 900 + len(source)))
    changed_parts, _ = assign_splits(changed_source_order, seed=42)
    assert split_pair_map(first_parts) == split_pair_map(changed_parts)


def test_held_out_values_do_not_change_split_membership() -> None:
    source = make_ratings({"1": 10, "2": 5})
    original_parts, _ = assign_splits(source, seed=42)
    altered = source.copy()
    held_out = set(original_parts["validation"]["pair_id"]) | set(original_parts["test"]["pair_id"])
    altered.loc[altered["pair_id"].isin(held_out), "rating"] = 1
    altered_parts, _ = assign_splits(altered, seed=42)
    assert split_pair_map(original_parts) == split_pair_map(altered_parts)
