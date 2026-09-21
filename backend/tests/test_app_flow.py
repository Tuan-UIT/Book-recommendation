from __future__ import annotations

import pytest
from werkzeug.security import check_password_hash

from backend.app import create_app
from backend.app.db import get_db, init_db
from backend.app.demo_data import build_demo_recommender, seed_demo_catalog


@pytest.fixture()
def app(tmp_path):
    database_path = tmp_path / "test.sqlite3"
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "DATABASE_URL": f"sqlite:///{database_path}",
            "RECOMMENDER": build_demo_recommender(alpha=0.0),
        }
    )
    with app.app_context():
        init_db()
        seed_demo_catalog(get_db())
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


def register(client, email="reader@example.test", password="password123"):
    return client.post(
        "/register", data={"email": email, "password": password}, follow_redirects=True
    )


def login(client, email="reader@example.test", password="password123"):
    return client.post(
        "/login", data={"email": email, "password": password}, follow_redirects=True
    )


def test_registration_hashes_password_and_login_works(app, client) -> None:
    response = register(client)
    assert "Đăng ký thành công" in response.get_data(as_text=True)
    with app.app_context():
        account = get_db().execute(
            "SELECT email, password_hash FROM accounts WHERE email = %s",
            ("reader@example.test",),
        ).fetchone()
        assert account["password_hash"] != "password123"
        assert check_password_hash(account["password_hash"], "password123")
    response = login(client)
    assert response.status_code == 200
    assert "Gợi ý dành cho bạn" in response.get_data(as_text=True)


def test_rating_requires_login_and_rejects_invalid_values(client) -> None:
    assert client.post("/ratings/demo:01", data={"rating": "5"}).status_code == 302
    register(client)
    login(client)
    assert client.post("/ratings/demo:01", data={"rating": "0"}).status_code == 400
    assert client.post("/ratings/demo:01", data={"rating": "6"}).status_code == 400


def test_two_accounts_own_separate_upserted_ratings(app, client) -> None:
    register(client, "one@example.test")
    login(client, "one@example.test")
    client.post("/ratings/demo:01", data={"rating": "5", "account_id": "999"})
    client.post("/ratings/demo:01", data={"rating": "1"})
    client.post("/logout")
    register(client, "two@example.test")
    login(client, "two@example.test")
    client.post("/ratings/demo:01", data={"rating": "3", "account_id": "1"})
    with app.app_context():
        rows = get_db().execute(
            "SELECT a.email, r.rating FROM account_ratings r JOIN accounts a ON a.account_id = r.account_id ORDER BY a.email"
        ).fetchall()
        assert [(row["email"], row["rating"]) for row in rows] == [
            ("one@example.test", 1),
            ("two@example.test", 3),
        ]


def test_recommendations_refresh_exclude_rated_work_and_are_unique(client) -> None:
    register(client)
    login(client)
    client.post("/preferences", data={"genres": ["fantasy"]})
    before = client.get("/api/recommendations").get_json()
    assert len(before) == 10
    assert len({row["canonical_item_id"] for row in before}) == 10
    rated_id = before[0]["canonical_item_id"]
    client.post(f"/api/ratings/{rated_id}", json={"rating": 5})
    after = client.get("/api/recommendations").get_json()
    assert rated_id not in {row["canonical_item_id"] for row in after}
    assert before != after


def test_rerating_updates_one_row_and_changes_positive_profile(app, client) -> None:
    register(client)
    login(client)
    client.post("/api/ratings/demo:01", json={"rating": 5})
    positive = client.get("/api/recommendations").get_json()
    client.post("/api/ratings/demo:01", json={"rating": 1})
    negative = client.get("/api/recommendations").get_json()
    with app.app_context():
        rows = get_db().execute(
            "SELECT rating FROM account_ratings WHERE canonical_item_id = %s",
            ("demo:01",),
        ).fetchall()
        assert [row["rating"] for row in rows] == [1]
    assert positive != negative


def test_missing_description_search_detail_and_short_list(client) -> None:
    response = client.get("/books?q=Cuốn+Sách+Không+Lời")
    assert "Cuốn Sách Không Lời" in response.get_data(as_text=True)
    response = client.get("/books/demo:17")
    assert "chưa có mô tả" in response.get_data(as_text=True).lower()
    register(client)
    login(client)
    for index in range(1, 13):
        client.post(f"/api/ratings/demo:{index:02d}", json={"rating": 3})
    recommendations = client.get("/api/recommendations").get_json()
    assert len(recommendations) == 6
