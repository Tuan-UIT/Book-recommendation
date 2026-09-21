"""Training-only candidate-catalog selection shared by every method."""

from __future__ import annotations

import pandas as pd

from .common import require_columns


CATALOG_COLUMNS = [
    "canonical_item_id",
    "train_rating_count",
    "train_rating_mean",
    "catalog_rank",
]


def select_candidate_catalog(
    training_item_stats: pd.DataFrame,
    *,
    max_items: int = 5_000,
    min_train_ratings: int = 3,
) -> pd.DataFrame:
    """Select works without consulting validation or test interactions.

    Works are ordered by training-rating count descending and canonical item ID
    ascending.  The stable ID tie-break makes the selection reproducible.
    """

    if max_items < 1:
        raise ValueError("max_items must be positive")
    if min_train_ratings < 1:
        raise ValueError("min_train_ratings must be positive")
    require_columns(
        training_item_stats,
        ("canonical_item_id", "train_rating_count", "train_rating_mean"),
        "training_item_stats",
    )
    frame = training_item_stats[
        ["canonical_item_id", "train_rating_count", "train_rating_mean"]
    ].copy()
    frame["canonical_item_id"] = frame["canonical_item_id"].astype(str)
    frame["train_rating_count"] = pd.to_numeric(
        frame["train_rating_count"], errors="raise"
    ).astype("int64")
    frame["train_rating_mean"] = pd.to_numeric(
        frame["train_rating_mean"], errors="raise"
    )
    if frame["canonical_item_id"].eq("").any():
        raise ValueError("training_item_stats contains an empty canonical item ID")
    if frame["canonical_item_id"].duplicated().any():
        raise ValueError("training_item_stats contains duplicate canonical item IDs")
    if (frame["train_rating_count"] < 1).any():
        raise ValueError("training rating counts must be positive")

    selected = frame.loc[frame["train_rating_count"] >= min_train_ratings]
    selected = selected.sort_values(
        ["train_rating_count", "canonical_item_id"],
        ascending=[False, True],
        kind="mergesort",
    ).head(max_items)
    if selected.empty:
        raise ValueError("candidate rules selected no training works")
    selected = selected.reset_index(drop=True)
    selected["catalog_rank"] = range(1, len(selected) + 1)
    return selected[CATALOG_COLUMNS]
