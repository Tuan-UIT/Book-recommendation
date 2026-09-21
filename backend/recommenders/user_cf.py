"""User-based Pearson collaborative filtering required by the proposal."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

import pandas as pd

from .common import RankedItem, validate_explicit_ratings
from .popularity import PopularityRecommender


class UserPearsonCF:
    """Positive-neighbour, mean-centred user-based collaborative filtering."""

    def __init__(
        self, *, neighbour_count: int = 20, min_common_items: int = 3,
        minimum_candidate_support: int = 1, support_shrinkage: float = 0.0,
        clip_predictions: bool = False,
    ):
        if neighbour_count < 1:
            raise ValueError("neighbour_count must be positive")
        if min_common_items < 2:
            raise ValueError("min_common_items must be at least two")
        if minimum_candidate_support < 1 or support_shrinkage < 0:
            raise ValueError("candidate support and shrinkage must be nonnegative")
        self.neighbour_count = neighbour_count
        self.min_common_items = min_common_items
        self.minimum_candidate_support = minimum_candidate_support
        self.support_shrinkage = float(support_shrinkage)
        self.clip_predictions = bool(clip_predictions)

    def fit(
        self,
        train_ratings: pd.DataFrame,
        candidate_items: Iterable[str],
        popularity: PopularityRecommender,
    ) -> "UserPearsonCF":
        train = validate_explicit_ratings(train_ratings, "train_ratings")
        self.candidate_items = {str(item_id) for item_id in candidate_items}
        if not self.candidate_items:
            raise ValueError("candidate_items must not be empty")
        selected = train.loc[train["canonical_item_id"].isin(self.candidate_items)]
        self.user_ratings: dict[str, dict[str, float]] = {}
        for user_id, group in selected.groupby("research_user_id", sort=False):
            self.user_ratings[str(user_id)] = dict(
                zip(
                    group["canonical_item_id"].astype(str),
                    group["rating"].astype(float),
                    strict=True,
                )
            )
        self.user_means = {
            user_id: sum(ratings.values()) / len(ratings)
            for user_id, ratings in self.user_ratings.items()
        }
        item_raters: dict[str, set[str]] = defaultdict(set)
        for user_id, ratings in self.user_ratings.items():
            for item_id in ratings:
                item_raters[item_id].add(user_id)
        self.item_raters = dict(item_raters)
        self.popularity = popularity
        return self

    def pearson_similarity(self, first_user: str, second_user: str) -> float | None:
        """Return overlap-centred Pearson, or None when it is invalid."""

        first = self.user_ratings.get(str(first_user))
        second = self.user_ratings.get(str(second_user))
        if not first or not second:
            return None
        common = sorted(first.keys() & second.keys())
        if len(common) < self.min_common_items:
            return None
        first_mean = sum(first[item_id] for item_id in common) / len(common)
        second_mean = sum(second[item_id] for item_id in common) / len(common)
        first_deviations = [first[item_id] - first_mean for item_id in common]
        second_deviations = [second[item_id] - second_mean for item_id in common]
        first_ss = sum(value * value for value in first_deviations)
        second_ss = sum(value * value for value in second_deviations)
        if first_ss == 0.0 or second_ss == 0.0:
            return None
        numerator = sum(
            first_value * second_value
            for first_value, second_value in zip(
                first_deviations, second_deviations, strict=True
            )
        )
        similarity = numerator / math.sqrt(first_ss * second_ss)
        if not math.isfinite(similarity) or similarity <= 0.0:
            return None
        return similarity

    def neighbours(self, research_user_id: str) -> list[tuple[str, float]]:
        user_id = str(research_user_id)
        target = self.user_ratings.get(user_id)
        if not target:
            return []
        possible: set[str] = set()
        for item_id in target:
            possible.update(self.item_raters.get(item_id, set()))
        possible.discard(user_id)
        valid: list[tuple[str, float]] = []
        for neighbour_id in possible:
            similarity = self.pearson_similarity(user_id, neighbour_id)
            if similarity is not None:
                valid.append((neighbour_id, similarity))
        valid.sort(key=lambda value: (-value[1], value[0]))
        return valid[: self.neighbour_count]

    def _validated_target_ratings(
        self, ratings: dict[str, float | int]
    ) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for raw_item_id, raw_rating in ratings.items():
            item_id = str(raw_item_id)
            rating = float(raw_rating)
            if not math.isfinite(rating) or rating < 1 or rating > 5:
                raise ValueError("target ratings must be between 1 and 5")
            if item_id in self.candidate_items:
                normalized[item_id] = rating
        return normalized

    def neighbours_from_ratings(
        self,
        ratings: dict[str, float | int],
        *,
        exclude_neighbour_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Find research neighbours for an authenticated account history."""

        target = self._validated_target_ratings(ratings)
        if not target:
            return []
        possible: set[str] = set()
        for item_id in target:
            possible.update(self.item_raters.get(item_id, set()))
        if exclude_neighbour_id is not None:
            possible.discard(str(exclude_neighbour_id))
        valid: list[tuple[str, float]] = []
        for neighbour_id in possible:
            neighbour = self.user_ratings[neighbour_id]
            common = sorted(target.keys() & neighbour.keys())
            if len(common) < self.min_common_items:
                continue
            target_mean = sum(target[item_id] for item_id in common) / len(common)
            neighbour_mean = (
                sum(neighbour[item_id] for item_id in common) / len(common)
            )
            target_deviations = [target[item_id] - target_mean for item_id in common]
            neighbour_deviations = [
                neighbour[item_id] - neighbour_mean for item_id in common
            ]
            target_ss = sum(value * value for value in target_deviations)
            neighbour_ss = sum(value * value for value in neighbour_deviations)
            if target_ss == 0.0 or neighbour_ss == 0.0:
                continue
            numerator = sum(
                first * second
                for first, second in zip(
                    target_deviations, neighbour_deviations, strict=True
                )
            )
            similarity = numerator / math.sqrt(target_ss * neighbour_ss)
            if math.isfinite(similarity) and similarity > 0.0:
                valid.append((neighbour_id, similarity))
        valid.sort(key=lambda value: (-value[1], value[0]))
        return valid[: self.neighbour_count]

    def score_from_ratings(
        self,
        ratings: dict[str, float | int],
        *,
        exclude_neighbour_id: str | None = None,
    ) -> dict[str, RankedItem]:
        """Score unseen works without treating unavailable scores as zero."""

        target = self._validated_target_ratings(ratings)
        if not target:
            return {}
        neighbours = self.neighbours_from_ratings(
            target, exclude_neighbour_id=exclude_neighbour_id
        )
        if not neighbours:
            return {}
        numerators: dict[str, float] = defaultdict(float)
        denominators: dict[str, float] = defaultdict(float)
        support: dict[str, list[str]] = defaultdict(list)
        for neighbour_id, similarity in neighbours:
            neighbour_mean = self.user_means[neighbour_id]
            for item_id, rating in self.user_ratings[neighbour_id].items():
                if item_id in target or item_id not in self.candidate_items:
                    continue
                numerators[item_id] += similarity * (rating - neighbour_mean)
                denominators[item_id] += abs(similarity)
                support[item_id].append(neighbour_id)
        target_mean = sum(target.values()) / len(target)
        scored: dict[str, RankedItem] = {}
        for item_id, denominator in denominators.items():
            count = len(support[item_id])
            if denominator <= 0.0 or count < getattr(self, "minimum_candidate_support", 1):
                continue
            prediction = target_mean + numerators[item_id] / denominator
            if getattr(self, "clip_predictions", False):
                prediction = max(1.0, min(5.0, prediction))
            shrinkage = getattr(self, "support_shrinkage", 0.0)
            if shrinkage:
                prediction = target_mean + (prediction - target_mean) * count / (count + shrinkage)
            scored[item_id] = RankedItem(
                item_id=item_id,
                score=prediction,
                source="cf",
                support_count=count,
                cf_score=prediction,
                evidence_kind="similar_readers",
                evidence_value=str(count),
                evidence_ids=tuple(sorted(support[item_id])),
            )
        return scored

    def recommend_from_ratings(
        self,
        ratings: dict[str, float | int],
        *,
        limit: int = 10,
        fallback_user_id: str = "website-account",
        exclude_neighbour_id: str | None = None,
    ) -> list[RankedItem]:
        if limit < 0:
            raise ValueError("limit must not be negative")
        known = {str(item_id) for item_id in ratings}
        scored = list(
            self.score_from_ratings(
                ratings, exclude_neighbour_id=exclude_neighbour_id
            ).values()
        )
        scored.sort(
            key=lambda result: (
                -float(result.score) if result.score is not None else math.inf,
                result.item_id,
            )
        )
        selected = scored[:limit]
        if len(selected) < limit:
            selected_ids = known | {result.item_id for result in selected}
            selected.extend(
                self.popularity.recommend(
                    fallback_user_id,
                    limit=limit - len(selected),
                    additionally_exclude=selected_ids,
                    source="popularity_fallback",
                )
            )
        return selected

    def recommend(
        self, research_user_id: str, *, limit: int = 10
    ) -> list[RankedItem]:
        if limit < 0:
            raise ValueError("limit must not be negative")
        user_id = str(research_user_id)
        return self.recommend_from_ratings(
            self.user_ratings.get(user_id, {}),
            limit=limit,
            fallback_user_id=user_id,
            exclude_neighbour_id=user_id,
        )
