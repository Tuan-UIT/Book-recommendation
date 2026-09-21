"""Normalized weighted Hybrid with explicit component availability."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .common import RankedItem
from .content import ContentRecommender
from .popularity import PopularityRecommender
from .user_cf import UserPearsonCF


class HybridRecommender:
    """Combine CF and content scores, then backfill from training popularity."""

    def __init__(self, *, alpha: float):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = float(alpha)

    def fit(
        self,
        cf: UserPearsonCF,
        content: ContentRecommender,
        popularity: PopularityRecommender,
    ) -> "HybridRecommender":
        if cf.candidate_items != set(content.item_ids):
            raise ValueError("CF and content must use the same candidate catalog")
        if cf.candidate_items != popularity.candidate_items:
            raise ValueError("all methods must use the same candidate catalog")
        self.cf = cf
        self.content = content
        self.popularity = popularity
        self.candidate_items = set(cf.candidate_items)
        return self

    @staticmethod
    def _cf_to_unit_interval(score: float) -> float:
        return (float(np.clip(score, 1.0, 5.0)) - 1.0) / 4.0

    def recommend_from_ratings(
        self,
        ratings: dict[str, float | int],
        *,
        limit: int = 10,
        initial_genres: Iterable[str] = (),
        fallback_user_id: str = "website-account",
        exclude_neighbour_id: str | None = None,
    ) -> list[RankedItem]:
        if limit < 0:
            raise ValueError("limit must not be negative")
        known = {str(item_id) for item_id in ratings}
        cf_scores = self.cf.score_from_ratings(
            ratings, exclude_neighbour_id=exclude_neighbour_id
        )
        content_scores = self.content.score_from_ratings(
            ratings, initial_genres=initial_genres
        )
        return self.recommend_from_component_scores(
            known,
            cf_scores,
            content_scores,
            limit=limit,
            fallback_user_id=fallback_user_id,
        )

    def recommend_from_component_scores(
        self,
        known_items: Iterable[str],
        cf_scores: dict[str, RankedItem],
        content_scores: dict[str, RankedItem],
        *,
        limit: int = 10,
        fallback_user_id: str = "website-account",
    ) -> list[RankedItem]:
        """Combine already computed components for repeatable alpha searches."""

        if limit < 0:
            raise ValueError("limit must not be negative")
        known = {str(item_id) for item_id in known_items}
        scored: list[RankedItem] = []
        for item_id in sorted(self.candidate_items - known):
            cf_item = cf_scores.get(item_id)
            content_item = content_scores.get(item_id)
            if cf_item is None and content_item is None:
                continue
            cf01 = (
                self._cf_to_unit_interval(float(cf_item.score))
                if cf_item is not None and cf_item.score is not None
                else None
            )
            content_value = (
                float(content_item.score)
                if content_item is not None and content_item.score is not None
                else None
            )
            if cf01 is not None and content_value is not None:
                score = self.alpha * cf01 + (1.0 - self.alpha) * content_value
            elif cf01 is not None:
                score = cf01
            else:
                score = float(content_value)

            use_cf_evidence = cf_item is not None and (
                content_item is None
                or self.alpha * float(cf01) >= (1.0 - self.alpha) * float(content_value)
            )
            evidence = cf_item if use_cf_evidence else content_item
            scored.append(
                RankedItem(
                    item_id=item_id,
                    score=score,
                    source="hybrid",
                    support_count=evidence.support_count if evidence else 0,
                    cf_score=float(cf_item.score) if cf_item is not None else None,
                    content_score=(
                        float(content_item.score)
                        if content_item is not None
                        else None
                    ),
                    evidence_kind=evidence.evidence_kind if evidence else None,
                    evidence_value=evidence.evidence_value if evidence else None,
                    evidence_ids=evidence.evidence_ids if evidence else (),
                )
            )
        scored.sort(key=lambda result: (-float(result.score), result.item_id))
        selected = scored[:limit]
        if len(selected) < limit:
            selected_ids = known | {result.item_id for result in selected}
            fallback = self.popularity.recommend(
                fallback_user_id,
                limit=limit - len(selected),
                additionally_exclude=selected_ids,
                source="popularity_fallback",
            )
            selected.extend(
                RankedItem(
                    item_id=item.item_id,
                    score=item.score,
                    source=item.source,
                    evidence_kind="training_popularity",
                    evidence_value=str(int(item.score or 0)),
                )
                for item in fallback
            )
        return selected

    def recommend(
        self, research_user_id: str, *, limit: int = 10
    ) -> list[RankedItem]:
        user_id = str(research_user_id)
        ratings = self.cf.user_ratings.get(user_id, {})
        return self.recommend_from_ratings(
            ratings,
            limit=limit,
            fallback_user_id=user_id,
            exclude_neighbour_id=user_id,
        )
