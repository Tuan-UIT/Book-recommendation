from __future__ import annotations

import math

import pandas as pd
import pytest

from backend.recommenders.catalog import select_candidate_catalog
from backend.recommenders.content import ContentRecommender
from backend.recommenders.hybrid import HybridRecommender
from backend.recommenders.evaluation import (
    binary_ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from backend.recommenders.popularity import PopularityRecommender
from backend.recommenders.user_cf import UserPearsonCF


def ratings(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["research_user_id", "canonical_item_id", "rating"],
    )


def test_hand_checked_metrics_and_fixed_precision_denominator() -> None:
    recommended = ["not-relevant", "relevant"]
    relevant = {"relevant"}
    assert precision_at_k(recommended, relevant, k=10) == pytest.approx(0.1)
    assert recall_at_k(recommended, relevant, k=10) == pytest.approx(1.0)
    assert binary_ndcg_at_k(recommended, relevant, k=10) == pytest.approx(
        1.0 / math.log2(3)
    )
    assert precision_at_k([], relevant, k=10) == 0.0
    with pytest.raises(ValueError, match="undefined"):
        recall_at_k(recommended, set(), k=10)
    with pytest.raises(ValueError, match="duplicate"):
        precision_at_k(["same", "same"], relevant, k=10)


def test_catalog_and_popularity_use_training_counts_with_stable_ties() -> None:
    stats = pd.DataFrame(
        {
            "canonical_item_id": ["c", "b", "a", "d"],
            "train_rating_count": [2, 3, 3, 1],
            "train_rating_mean": [4.0, 3.0, 5.0, 2.0],
        }
    )
    catalog = select_candidate_catalog(stats, max_items=3, min_train_ratings=2)
    assert catalog["canonical_item_id"].tolist() == ["a", "b", "c"]

    train = ratings(
        [
            ("u1", "a", 5),
            ("u1", "c", 2),
            ("u2", "a", 4),
            ("u2", "b", 3),
            ("u3", "b", 5),
        ]
    )
    model = PopularityRecommender().fit(train, ["a", "b", "c"])
    result = model.recommend("u1", limit=3)
    assert [item.item_id for item in result] == ["b"]
    assert result[0].score == 2.0


def test_pearson_cf_predicts_from_positive_valid_neighbour() -> None:
    train = ratings(
        [
            ("target", "a", 5),
            ("target", "b", 3),
            ("similar", "a", 4),
            ("similar", "b", 2),
            ("similar", "c", 5),
            ("opposite", "a", 1),
            ("opposite", "b", 5),
            ("opposite", "d", 5),
        ]
    )
    popularity = PopularityRecommender().fit(train, ["a", "b", "c", "d"])
    model = UserPearsonCF(neighbour_count=2, min_common_items=2).fit(
        train, ["a", "b", "c", "d"], popularity
    )
    assert model.pearson_similarity("target", "similar") == pytest.approx(1.0)
    assert model.pearson_similarity("target", "opposite") is None
    result = model.recommend("target", limit=2)
    assert result[0].item_id == "c"
    assert result[0].source == "cf"
    assert result[0].score == pytest.approx(4 + (5 - 11 / 3))
    assert result[1].item_id == "d"
    assert result[1].source == "popularity_fallback"


def test_pearson_zero_variance_and_no_history_use_fallback() -> None:
    train = ratings(
        [
            ("flat", "a", 3),
            ("flat", "b", 3),
            ("other", "a", 4),
            ("other", "b", 5),
            ("other", "c", 4),
        ]
    )
    popularity = PopularityRecommender().fit(train, ["a", "b", "c"])
    model = UserPearsonCF(neighbour_count=2, min_common_items=2).fit(
        train, ["a", "b", "c"], popularity
    )
    assert model.pearson_similarity("flat", "other") is None
    assert model.recommend("flat", limit=1)[0].source == "popularity_fallback"
    assert model.recommend("new-user", limit=1)[0].source == "popularity_fallback"


def test_content_uses_description_author_genre_and_empty_profile_fallback() -> None:
    items = ["fantasy-known", "fantasy-blank", "technology", "empty"]
    train = ratings(
        [
            ("reader", "fantasy-known", 5),
            ("reader", "technology", 2),
            ("other", "fantasy-blank", 4),
            ("other", "empty", 3),
        ]
    )
    books = pd.DataFrame(
        {
            "canonical_item_id": items,
            "description": ["dragon magic", "", "database systems", ""],
        }
    )
    book_authors = pd.DataFrame(
        {
            "canonical_item_id": ["fantasy-known", "fantasy-blank", "technology"],
            "author_id": ["author-1", "author-1", "author-2"],
        }
    )
    book_genres = pd.DataFrame(
        {
            "canonical_item_id": [
                "fantasy-known",
                "fantasy-blank",
                "technology",
                "empty",
            ],
            "genre_label": ["fantasy", "fantasy", "technology", "unknown"],
        }
    )
    popularity = PopularityRecommender().fit(train, items)
    model = ContentRecommender(max_tfidf_features=100).fit(
        train, items, books, book_authors, book_genres, popularity
    )

    result = model.recommend("reader", limit=2)
    assert result[0].item_id == "fantasy-blank"
    assert result[0].source == "content"
    assert result[0].score > 0
    assert result[1].source == "popularity_fallback"

    fallback = model.recommend("new-user", limit=1)
    assert fallback[0].source == "popularity_fallback"
    genre_result = model.recommend(
        "new-user", limit=1, initial_genres=["fantasy"]
    )
    assert genre_result[0].item_id == "fantasy-blank"
    assert genre_result[0].source == "content"


def test_hybrid_keeps_available_zero_above_unavailable_fallback() -> None:
    items = ["known-a", "known-b", "zero-content", "cf-item", "no-features"]
    train = ratings(
        [
            ("n1", "known-a", 5),
            ("n1", "known-b", 3),
            ("n1", "cf-item", 5),
            ("n2", "known-a", 4),
            ("n2", "known-b", 2),
            ("n2", "cf-item", 4),
            ("n3", "zero-content", 4),
            ("n3", "no-features", 3),
        ]
    )
    books = pd.DataFrame(
        {
            "canonical_item_id": items,
            "description": ["alpha shared", "beta shared", "unrelated words", "alpha shared", ""],
        }
    )
    authors = pd.DataFrame(
        {
            "canonical_item_id": ["known-a", "known-b", "zero-content", "cf-item"],
            "author_id": ["a", "b", "c", "a"],
        }
    )
    genres = pd.DataFrame(
        {
            "canonical_item_id": items,
            "genre_label": ["x", "y", "z", "x", "unknown"],
        }
    )
    popularity = PopularityRecommender().fit(train, items)
    cf = UserPearsonCF(neighbour_count=5, min_common_items=2).fit(
        train, items, popularity
    )
    content = ContentRecommender(max_tfidf_features=100).fit(
        train, items, books, authors, genres, popularity
    )
    hybrid = HybridRecommender(alpha=0.5).fit(cf, content, popularity)

    result = hybrid.recommend_from_ratings(
        {"known-a": 5, "known-b": 3}, limit=3
    )
    assert [item.item_id for item in result[:2]] == ["cf-item", "zero-content"]
    assert result[1].source == "hybrid"
    assert result[1].content_score == pytest.approx(0.0)
    assert result[2].item_id == "no-features"
    assert result[2].source == "popularity_fallback"


def test_rerating_removes_positive_item_from_online_content_profile() -> None:
    items = ["known", "similar", "popular"]
    train = ratings(
        [
            ("u1", "known", 5),
            ("u2", "similar", 4),
            ("u3", "popular", 3),
        ]
    )
    books = pd.DataFrame(
        {
            "canonical_item_id": items,
            "description": ["dragon castle", "dragon castle", "database"],
        }
    )
    authors = pd.DataFrame(
        {
            "canonical_item_id": items,
            "author_id": ["a", "a", "b"],
        }
    )
    genres = pd.DataFrame(
        {
            "canonical_item_id": items,
            "genre_label": ["fantasy", "fantasy", "technology"],
        }
    )
    popularity = PopularityRecommender().fit(train, items)
    content = ContentRecommender(max_tfidf_features=100).fit(
        train, items, books, authors, genres, popularity
    )
    before = content.recommend_from_ratings({"known": 5}, limit=1)
    after = content.recommend_from_ratings({"known": 1}, limit=1)
    assert before[0].item_id == "similar"
    assert before[0].source == "content"
    assert after[0].source == "popularity_fallback"
