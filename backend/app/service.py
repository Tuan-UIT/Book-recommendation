"""Online recommendation service using current account-owned data."""

from __future__ import annotations

from threading import Lock

from .db import get_db


class RecommendationService:
    METHODS = ("popularity", "cf", "content", "hybrid")

    def __init__(self, recommender):
        self.recommender = recommender
        self._cache: dict[tuple[int, str, int], list[dict]] = {}
        self._lock = Lock()

    def invalidate(self, account_id: int) -> None:
        with self._lock:
            for key in list(self._cache):
                if key[0] == int(account_id):
                    del self._cache[key]

    def recommend(self, account_id: int, *, method: str = "hybrid", limit: int = 10) -> list[dict]:
        if method not in self.METHODS:
            raise ValueError("unknown recommendation method")
        key = (int(account_id), method, limit)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return [dict(row) for row in cached]
        db = get_db()
        rating_rows = db.execute(
            "SELECT canonical_item_id, rating FROM account_ratings WHERE account_id = %s",
            (account_id,),
        ).fetchall()
        ratings = {str(row["canonical_item_id"]): int(row["rating"]) for row in rating_rows}
        genre_rows = db.execute(
            "SELECT g.label FROM account_genre_preferences p JOIN genres g ON g.genre_id = p.genre_id WHERE p.account_id = %s ORDER BY g.label",
            (account_id,),
        ).fetchall()
        genres = [str(row["label"]) for row in genre_rows]
        recommender = self.recommender
        fallback_id = f"account:{account_id}"
        if method == "popularity":
            ranked = recommender.popularity.recommend(
                fallback_id, limit=limit, additionally_exclude=ratings
            )
        elif method == "cf":
            ranked = recommender.cf.recommend_from_ratings(
                ratings, limit=limit, fallback_user_id=fallback_id
            )
        elif method == "content":
            ranked = recommender.content.recommend_from_ratings(
                ratings, limit=limit, initial_genres=genres,
                fallback_user_id=fallback_id,
            )
        else:
            ranked = recommender.recommend_from_ratings(
                ratings, limit=limit, initial_genres=genres,
                fallback_user_id=fallback_id,
            )
        if not ranked:
            return []
        item_ids = [item.item_id for item in ranked]
        evidence_ids = [item.evidence_value for item in ranked
                        if item.evidence_kind == "known_positive_item" and item.evidence_value]
        placeholders = ", ".join(["%s"] * (len(item_ids) + len(evidence_ids)))
        books = db.execute(
            f"SELECT canonical_item_id, title, description FROM books WHERE canonical_item_id IN ({placeholders})",
            item_ids + evidence_ids,
        ).fetchall()
        by_id = {str(book["canonical_item_id"]): book for book in books}
        item_placeholders = ", ".join(["%s"] * len(item_ids))
        author_rows = db.execute(
            "SELECT ba.canonical_item_id, a.display_name FROM book_authors ba "
            "JOIN authors a ON a.author_id = ba.author_id "
            f"WHERE ba.canonical_item_id IN ({item_placeholders}) ORDER BY a.display_name",
            item_ids,
        ).fetchall()
        genre_rows = db.execute(
            "SELECT bg.canonical_item_id, g.label FROM book_genres bg "
            "JOIN genres g ON g.genre_id = bg.genre_id "
            f"WHERE bg.canonical_item_id IN ({item_placeholders}) ORDER BY g.label",
            item_ids,
        ).fetchall()
        authors: dict[str, list[str]] = {}
        genres_by_item: dict[str, list[str]] = {}
        for row in author_rows:
            authors.setdefault(str(row["canonical_item_id"]), []).append(str(row["display_name"]))
        for row in genre_rows:
            genres_by_item.setdefault(str(row["canonical_item_id"]), []).append(str(row["label"]))
        results: list[dict] = []
        for item in ranked:
            book = by_id.get(item.item_id)
            if book is None:
                continue
            results.append(
                {
                    "canonical_item_id": str(book["canonical_item_id"]),
                    "title": str(book["title"]),
                    "description": str(book["description"]),
                    "authors": authors.get(item.item_id, []),
                    "genres": genres_by_item.get(item.item_id, []),
                    "score": item.score,
                    "source": item.source,
                    "reason": self._reason(item, by_id),
                }
            )
            if len(results) == limit:
                break
        with self._lock:
            self._cache[key] = [dict(row) for row in results]
        return results

    @staticmethod
    def _reason(item, books_by_id) -> str:
        if item.evidence_kind == "matched_genre":
            return f"Phù hợp thể loại {item.evidence_value} bạn đã chọn."
        if item.evidence_kind == "known_positive_item" and item.evidence_value:
            row = books_by_id.get(item.evidence_value)
            if row is not None:
                return f"Có nội dung gần với {row['title']}, sách bạn đã đánh giá cao."
        if item.evidence_kind == "similar_readers":
            return f"Có {item.support_count} người đọc có lịch sử đánh giá tương tự hỗ trợ gợi ý này."
        if item.evidence_kind == "training_popularity":
            return f"Phổ biến trong tập huấn luyện với {item.evidence_value} lượt đánh giá rõ ràng."
        if item.content_score == 0:
            return "Hồ sơ nội dung hợp lệ nhưng điểm tương đồng bằng 0; thứ tự được phá hòa bằng mã sách ổn định."
        if item.evidence_kind == "content_profile":
            return f"Đặc trưng nội dung phù hợp với hồ sơ từ {item.support_count} sách bạn đánh giá cao."
        return "Gợi ý dự phòng theo số lượt đánh giá trong tập huấn luyện."
