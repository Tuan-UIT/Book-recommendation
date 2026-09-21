"""Training-popularity baseline with deterministic exclusions and tie-breaking."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .common import RankedItem, user_history, validate_explicit_ratings


class PopularityRecommender:
    """Rank candidate works by explicit training-rating count only."""

    def fit(
        self, train_ratings: pd.DataFrame, candidate_items: Iterable[str]
    ) -> "PopularityRecommender":
        train = validate_explicit_ratings(train_ratings, "train_ratings")
        candidates = {str(item_id) for item_id in candidate_items}
        if not candidates:
            raise ValueError("candidate_items must not be empty")
        counts = (
            train.loc[train["canonical_item_id"].isin(candidates)]
            .groupby("canonical_item_id")
            .size()
            .to_dict()
        )
        missing = candidates - set(counts)
        if missing:
            raise ValueError("every candidate work must have a training rating")
        self.candidate_items = candidates
        self.counts = {str(item_id): int(count) for item_id, count in counts.items()}
        self.ranking = sorted(candidates, key=lambda item_id: (-counts[item_id], item_id))
        self.known_items = user_history(train)
        return self

    def recommend(
        self,
        research_user_id: str,
        *,
        limit: int = 10,
        additionally_exclude: Iterable[str] = (),
        source: str = "popularity",
    ) -> list[RankedItem]:
        if limit < 0:
            raise ValueError("limit must not be negative")
        user_id = str(research_user_id)
        excluded = set(self.known_items.get(user_id, set()))
        excluded.update(str(item_id) for item_id in additionally_exclude)
        results: list[RankedItem] = []
        for item_id in self.ranking:
            if item_id in excluded:
                continue
            results.append(
                RankedItem(
                    item_id=item_id,
                    score=float(self.counts[item_id]),
                    source=source,
                )
            )
            if len(results) == limit:
                break
        return results
