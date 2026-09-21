"""Clearly labelled self-created fixtures for the public-safe demonstration."""

from __future__ import annotations

import pandas as pd

from backend.recommenders.content import ContentRecommender
from backend.recommenders.hybrid import HybridRecommender
from backend.recommenders.popularity import PopularityRecommender
from backend.recommenders.user_cf import UserPearsonCF


DEMO_BOOKS = [
    ("demo:01", "Khu Rừng Sao", "Một hành trình giả tưởng qua khu rừng có phép thuật.", "fantasy", "An Lam"),
    ("demo:02", "Thành Phố Rồng", "Rồng và những người giữ thành phố cổ.", "fantasy", "An Lam"),
    ("demo:03", "Cánh Cửa Mùa Đông", "Cuộc phiêu lưu qua thế giới băng giá.", "fantasy", "Bình Minh"),
    ("demo:04", "Dòng Sông Ký Ức", "Tiểu thuyết về gia đình và ký ức.", "literary", "Chi Mai"),
    ("demo:05", "Những Lá Thư Không Gửi", "Câu chuyện đời thường qua các lá thư.", "literary", "Chi Mai"),
    ("demo:06", "Bên Kia Mùa Hạ", "Tình bạn và trưởng thành ở một thị trấn nhỏ.", "literary", "Duy Khang"),
    ("demo:07", "Mật Mã Phòng Kín", "Một vụ án trong căn phòng không lối ra.", "mystery", "Gia Hân"),
    ("demo:08", "Dấu Chân Trong Mưa", "Điều tra bí ẩn giữa thành phố mưa.", "mystery", "Gia Hân"),
    ("demo:09", "Chiếc Đồng Hồ Thứ Mười Ba", "Bí mật gia đình gắn với chiếc đồng hồ cũ.", "mystery", "Hải Nam"),
    ("demo:10", "Python Từ Đầu", "Nhập môn lập trình Python bằng ví dụ nhỏ.", "technology", "Khánh Linh"),
    ("demo:11", "Dữ Liệu Có Câu Chuyện", "Phân tích dữ liệu và trình bày kết quả rõ ràng.", "technology", "Khánh Linh"),
    ("demo:12", "Thiết Kế Hệ Thống Nhỏ", "Các nguyên tắc thiết kế dịch vụ web dễ hiểu.", "technology", "Long Vũ"),
    ("demo:13", "Bầu Trời Tháng Ba", "Thơ và tản văn về những ngày đầu năm.", "poetry", "Mai Phương"),
    ("demo:14", "Khoảng Lặng", "Tập thơ về thành phố và những khoảng lặng.", "poetry", "Mai Phương"),
    ("demo:15", "Đường Về Núi", "Ghi chép du hành qua miền núi.", "travel", "Nguyên Hà"),
    ("demo:16", "Biển Gọi", "Nhật ký một chuyến đi dọc bờ biển.", "travel", "Nguyên Hà"),
    ("demo:17", "Cuốn Sách Không Lời", "", "literary", "Tác giả chưa rõ"),
    ("demo:18", "Ngã Rẽ Xanh", "Những lựa chọn nhỏ cho lối sống bền vững.", "science", "Phúc An"),
]


DEMO_RATINGS = [
    ("r1", "demo:01", 5), ("r1", "demo:02", 5), ("r1", "demo:03", 4), ("r1", "demo:04", 2), ("r1", "demo:10", 1),
    ("r2", "demo:01", 4), ("r2", "demo:02", 5), ("r2", "demo:03", 5), ("r2", "demo:09", 3), ("r2", "demo:15", 2),
    ("r3", "demo:04", 5), ("r3", "demo:05", 5), ("r3", "demo:06", 4), ("r3", "demo:13", 4), ("r3", "demo:01", 2),
    ("r4", "demo:07", 5), ("r4", "demo:08", 5), ("r4", "demo:09", 4), ("r4", "demo:04", 3), ("r4", "demo:11", 2),
    ("r5", "demo:10", 5), ("r5", "demo:11", 5), ("r5", "demo:12", 4), ("r5", "demo:18", 4), ("r5", "demo:05", 2),
    ("r6", "demo:13", 5), ("r6", "demo:14", 5), ("r6", "demo:04", 4), ("r6", "demo:05", 4), ("r6", "demo:08", 2),
    ("r7", "demo:15", 5), ("r7", "demo:16", 5), ("r7", "demo:18", 4), ("r7", "demo:03", 3), ("r7", "demo:10", 2),
    ("r8", "demo:06", 5), ("r8", "demo:17", 4), ("r8", "demo:05", 4), ("r8", "demo:14", 3), ("r8", "demo:12", 2),
]


def seed_demo_catalog(db) -> None:
    genres = sorted({row[3] for row in DEMO_BOOKS})
    authors = sorted({row[4] for row in DEMO_BOOKS})
    db.executemany(
        "INSERT INTO genres (genre_id, label) VALUES (%s, %s) ON CONFLICT (genre_id) DO NOTHING",
        [(value, value) for value in genres],
    )
    db.executemany(
        "INSERT INTO authors (author_id, display_name, name_missing) VALUES (%s, %s, %s) ON CONFLICT (author_id) DO NOTHING",
        [(f"demo-author:{index}", name, name == "Tác giả chưa rõ") for index, name in enumerate(authors, 1)],
    )
    author_ids = {name: f"demo-author:{index}" for index, name in enumerate(authors, 1)}
    db.executemany(
        "INSERT INTO books (canonical_item_id, representative_source_book_id, title, description, language_code, is_demo) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (canonical_item_id) DO NOTHING",
        [(item, item, title, description, "vi", True) for item, title, description, _genre, _author in DEMO_BOOKS],
    )
    db.executemany(
        "INSERT INTO book_authors (canonical_item_id, author_id) VALUES (%s, %s) ON CONFLICT (canonical_item_id, author_id) DO NOTHING",
        [(item, author_ids[author]) for item, _title, _description, _genre, author in DEMO_BOOKS],
    )
    db.executemany(
        "INSERT INTO book_genres (canonical_item_id, genre_id) VALUES (%s, %s) ON CONFLICT (canonical_item_id, genre_id) DO NOTHING",
        [(item, genre) for item, _title, _description, genre, _author in DEMO_BOOKS],
    )
    db.commit()


def build_demo_recommender(alpha: float = 0.0) -> HybridRecommender:
    items = [row[0] for row in DEMO_BOOKS]
    train = pd.DataFrame(DEMO_RATINGS, columns=["research_user_id", "canonical_item_id", "rating"])
    books = pd.DataFrame(
        [(item, description) for item, _title, description, _genre, _author in DEMO_BOOKS],
        columns=["canonical_item_id", "description"],
    )
    author_names = sorted({row[4] for row in DEMO_BOOKS})
    author_ids = {name: f"demo-author:{index}" for index, name in enumerate(author_names, 1)}
    book_authors = pd.DataFrame(
        [(item, author_ids[author]) for item, _title, _description, _genre, author in DEMO_BOOKS],
        columns=["canonical_item_id", "author_id"],
    )
    book_genres = pd.DataFrame(
        [(item, genre) for item, _title, _description, genre, _author in DEMO_BOOKS],
        columns=["canonical_item_id", "genre_label"],
    )
    popularity = PopularityRecommender().fit(train, items)
    cf = UserPearsonCF(neighbour_count=5, min_common_items=2).fit(train, items, popularity)
    content = ContentRecommender(max_tfidf_features=500).fit(
        train, items, books, book_authors, book_genres, popularity
    )
    return HybridRecommender(alpha=alpha).fit(cf, content, popularity)
