"""Shared validation and result types for recommendation methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class RankedItem:
    """One ranked work and the evidence source used to score it."""

    item_id: str
    score: float | None
    source: str
    support_count: int = 0
    cf_score: float | None = None
    content_score: float | None = None
    evidence_kind: str | None = None
    evidence_value: str | None = None
    evidence_ids: tuple[str, ...] = ()


def require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def validate_explicit_ratings(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return a normalized copy containing only valid explicit ratings."""

    require_columns(
        frame,
        ("research_user_id", "canonical_item_id", "rating"),
        name,
    )
    normalized = frame.copy()
    normalized["research_user_id"] = normalized["research_user_id"].astype(str)
    normalized["canonical_item_id"] = normalized["canonical_item_id"].astype(str)
    normalized["rating"] = pd.to_numeric(normalized["rating"], errors="raise")
    if normalized["research_user_id"].eq("").any():
        raise ValueError(f"{name} contains an empty research_user_id")
    if normalized["canonical_item_id"].eq("").any():
        raise ValueError(f"{name} contains an empty canonical_item_id")
    if not normalized["rating"].between(1, 5).all():
        raise ValueError(f"{name} must contain only explicit ratings 1 through 5")
    if normalized.duplicated(["research_user_id", "canonical_item_id"]).any():
        raise ValueError(f"{name} contains duplicate user--work pairs")
    return normalized


def user_history(frame: pd.DataFrame) -> dict[str, set[str]]:
    """Map each research user to the canonical works in known training history."""

    if frame.empty:
        return {}
    return {
        str(user_id): set(group["canonical_item_id"].astype(str))
        for user_id, group in frame.groupby("research_user_id", sort=False)
    }
