"""TF-IDF, author, and genre content-based recommendation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .common import RankedItem, require_columns, validate_explicit_ratings
from .popularity import PopularityRecommender


def _normalized_label(value: object) -> str:
    return str(value).strip().lower()


class ContentRecommender:
    """Rank unseen works by cosine similarity to a positive-history profile."""

    def __init__(self, *, max_tfidf_features: int = 10_000):
        if max_tfidf_features < 1:
            raise ValueError("max_tfidf_features must be positive")
        self.max_tfidf_features = max_tfidf_features

    def fit(
        self,
        train_ratings: pd.DataFrame,
        candidate_items: Iterable[str],
        books: pd.DataFrame,
        book_authors: pd.DataFrame,
        book_genres: pd.DataFrame,
        popularity: PopularityRecommender,
    ) -> "ContentRecommender":
        train = validate_explicit_ratings(train_ratings, "train_ratings")
        require_columns(books, ("canonical_item_id", "description"), "books")
        require_columns(
            book_authors,
            ("canonical_item_id", "author_id"),
            "book_authors",
        )
        require_columns(
            book_genres,
            ("canonical_item_id", "genre_label"),
            "book_genres",
        )
        self.item_ids = sorted({str(item_id) for item_id in candidate_items})
        if not self.item_ids:
            raise ValueError("candidate_items must not be empty")
        self.item_index = {
            item_id: index for index, item_id in enumerate(self.item_ids)
        }
        candidate_set = set(self.item_ids)

        book_rows = books.copy()
        book_rows["canonical_item_id"] = book_rows["canonical_item_id"].astype(str)
        if book_rows["canonical_item_id"].duplicated().any():
            raise ValueError("books contains duplicate canonical item IDs")
        descriptions = dict(
            zip(
                book_rows["canonical_item_id"],
                book_rows["description"].fillna("").astype(str),
                strict=True,
            )
        )
        description_values = [descriptions.get(item_id, "") for item_id in self.item_ids]
        self.tfidf: TfidfVectorizer | None
        if any(value.strip() for value in description_values):
            self.tfidf = TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_features=self.max_tfidf_features,
                dtype=np.float64,
            )
            try:
                description_features = self.tfidf.fit_transform(description_values)
            except ValueError as error:
                message = str(error).lower()
                if "empty vocabulary" not in message and "no terms remain" not in message:
                    raise
                self.tfidf = None
                description_features = sparse.csr_matrix((len(self.item_ids), 0))
        else:
            self.tfidf = None
            description_features = sparse.csr_matrix((len(self.item_ids), 0))

        authors_by_item: dict[str, set[str]] = defaultdict(set)
        for row in book_authors.itertuples(index=False):
            item_id = str(row.canonical_item_id)
            author_id = str(row.author_id).strip()
            if item_id in candidate_set and author_id:
                authors_by_item[item_id].add(author_id)
        author_dicts = [
            {author_id: 1.0 for author_id in sorted(authors_by_item[item_id])}
            for item_id in self.item_ids
        ]
        self.author_vectorizer = DictVectorizer(sparse=True, sort=True)
        author_features = self.author_vectorizer.fit_transform(author_dicts)

        genres_by_item: dict[str, set[str]] = defaultdict(set)
        for row in book_genres.itertuples(index=False):
            item_id = str(row.canonical_item_id)
            genre = _normalized_label(row.genre_label)
            if item_id in candidate_set and genre and genre != "unknown":
                genres_by_item[item_id].add(genre)
        genre_dicts = [
            {genre: 1.0 for genre in sorted(genres_by_item[item_id])}
            for item_id in self.item_ids
        ]
        self.genre_vectorizer = DictVectorizer(sparse=True, sort=True)
        genre_features = self.genre_vectorizer.fit_transform(genre_dicts)

        blocks = [
            normalize(block, norm="l2", copy=False)
            if block.shape[1]
            else block.astype(np.float64)
            for block in (description_features, author_features, genre_features)
        ]
        self.block_widths = tuple(block.shape[1] for block in blocks)
        combined = sparse.hstack(blocks, format="csr", dtype=np.float64)
        self.features = normalize(combined, norm="l2", copy=False)
        self.item_has_features = np.asarray(self.features.getnnz(axis=1) > 0).ravel()

        selected_train = train.loc[train["canonical_item_id"].isin(candidate_set)]
        self.known_items: dict[str, set[str]] = {}
        self.user_rating_values: dict[str, dict[str, float]] = {}
        self.positive_items: dict[str, list[str]] = {}
        for user_id, group in selected_train.groupby("research_user_id", sort=False):
            normalized_user_id = str(user_id)
            self.known_items[normalized_user_id] = set(
                group["canonical_item_id"].astype(str)
            )
            self.user_rating_values[normalized_user_id] = dict(
                zip(
                    group["canonical_item_id"].astype(str),
                    group["rating"].astype(float),
                    strict=True,
                )
            )
            self.positive_items[normalized_user_id] = sorted(
                group.loc[group["rating"] >= 4, "canonical_item_id"].astype(str)
            )
        self.popularity = popularity
        self.genres_by_item = dict(genres_by_item)
        self.authors_by_item = dict(authors_by_item)
        return self

    def _profile_from_indices(self, indices: list[int]) -> sparse.csr_matrix | None:
        if not indices:
            return None
        profile = sparse.csr_matrix(self.features[indices].sum(axis=0))
        if profile.nnz == 0:
            return None
        return normalize(profile, norm="l2", copy=False)

    def profile_for_user(self, research_user_id: str) -> sparse.csr_matrix | None:
        item_ids = self.positive_items.get(str(research_user_id), [])
        indices = [self.item_index[item_id] for item_id in item_ids]
        return self._profile_from_indices(indices)

    def profile_from_genres(
        self, genre_labels: Iterable[str]
    ) -> sparse.csr_matrix | None:
        """Build a new-user profile from normalized, known genre labels."""

        selected = {
            _normalized_label(label): 1.0
            for label in genre_labels
            if _normalized_label(label) and _normalized_label(label) != "unknown"
        }
        if not selected:
            return None
        genre_part = self.genre_vectorizer.transform([selected])
        if genre_part.nnz == 0:
            return None
        description_width, author_width, _ = self.block_widths
        profile = sparse.hstack(
            [
                sparse.csr_matrix((1, description_width)),
                sparse.csr_matrix((1, author_width)),
                normalize(genre_part, norm="l2", copy=False),
            ],
            format="csr",
        )
        return normalize(profile, norm="l2", copy=False)

    def profile_from_ratings(
        self, ratings: dict[str, float | int]
    ) -> tuple[sparse.csr_matrix | None, list[str]]:
        positive: list[str] = []
        for raw_item_id, raw_rating in ratings.items():
            item_id = str(raw_item_id)
            rating = float(raw_rating)
            if not np.isfinite(rating) or rating < 1 or rating > 5:
                raise ValueError("target ratings must be between 1 and 5")
            if rating >= 4 and item_id in self.item_index:
                positive.append(item_id)
        positive = sorted(set(positive))
        return (
            self._profile_from_indices([self.item_index[item] for item in positive]),
            positive,
        )

    def score_from_ratings(
        self,
        ratings: dict[str, float | int],
        *,
        initial_genres: Iterable[str] = (),
    ) -> dict[str, RankedItem]:
        """Score account candidates and retain concrete profile evidence."""

        profile, positive = self.profile_from_ratings(ratings)
        normalized_genres = sorted(
            {
                _normalized_label(label)
                for label in initial_genres
                if _normalized_label(label)
                and _normalized_label(label) != "unknown"
            }
        )
        if profile is None:
            profile = self.profile_from_genres(normalized_genres)
        if profile is None:
            return {}
        scores = cosine_similarity(profile, self.features, dense_output=True).ravel()
        known = {str(item_id) for item_id in ratings}
        results: dict[str, RankedItem] = {}
        for index, item_id in enumerate(self.item_ids):
            if item_id in known or not self.item_has_features[index]:
                continue
            matched_genres = sorted(
                self.genres_by_item.get(item_id, set()) & set(normalized_genres)
            )
            if matched_genres:
                evidence_kind = "matched_genre"
                evidence_value = matched_genres[0]
                evidence_ids = (matched_genres[0],)
            else:
                evidence_kind = "content_profile"
                evidence_value = str(len(positive))
                evidence_ids = tuple(positive)
            results[item_id] = RankedItem(
                item_id=item_id,
                score=float(scores[index]),
                source="content",
                support_count=len(positive) or len(normalized_genres),
                content_score=float(scores[index]),
                evidence_kind=evidence_kind,
                evidence_value=evidence_value,
                evidence_ids=evidence_ids,
            )
        return results

    def recommend_from_ratings(
        self,
        ratings: dict[str, float | int],
        *,
        limit: int = 10,
        initial_genres: Iterable[str] = (),
        fallback_user_id: str = "website-account",
    ) -> list[RankedItem]:
        if limit < 0:
            raise ValueError("limit must not be negative")
        known = {str(item_id) for item_id in ratings}
        scored = list(
            self.score_from_ratings(
                ratings, initial_genres=initial_genres
            ).values()
        )
        scored.sort(key=lambda result: (-float(result.score), result.item_id))
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
        self,
        research_user_id: str,
        *,
        limit: int = 10,
        initial_genres: Iterable[str] = (),
    ) -> list[RankedItem]:
        if limit < 0:
            raise ValueError("limit must not be negative")
        user_id = str(research_user_id)
        return self.recommend_from_ratings(
            self.user_rating_values.get(user_id, {}),
            limit=limit,
            initial_genres=initial_genres,
            fallback_user_id=user_id,
        )
