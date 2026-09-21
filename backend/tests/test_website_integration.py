"""Focused checks for the integrated website and browser-facing boundaries."""

from __future__ import annotations

import pytest

from backend.app import create_app
from backend.app.db import get_db, init_db
from backend.app.demo_data import seed_demo_catalog


@pytest.fixture()
def app(tmp_path):
    app = create_app({
        "TESTING": True,
        "SECRET_KEY": "test-only-secret",
        "DATABASE_URL": f"sqlite:///{tmp_path / 'website.sqlite3'}",
        "CSRF_ENABLED": True,
    })
    with app.app_context():
        init_db()
        seed_demo_catalog(get_db())
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


def token(client):
    client.get("/")
    with client.session_transaction() as session:
        return session["csrf_token"]


def register_and_login(client, email="reader@example.test"):
    csrf = token(client)
    assert client.post("/register", data={
        "email": email, "password": "password123", "csrf_token": csrf,
    }).status_code == 302
    assert client.post("/login", data={
        "email": email, "password": "password123", "csrf_token": csrf,
    }).status_code == 302


def test_csrf_protects_html_and_json_posts(client):
    csrf = token(client)
    assert client.post("/register", data={
        "email": "reader@example.test", "password": "password123",
    }).status_code == 400
    register_and_login(client)
    assert client.post("/api/ratings/demo:01", json={"rating": 5}).status_code == 400
    assert client.post("/api/ratings/demo:01", json={"rating": 5},
                       headers={"X-CSRF-Token": token(client)}).status_code == 200
    assert client.get("/api/recommendations").status_code == 200


def test_same_account_four_methods_and_rating_refresh(client):
    register_and_login(client)
    methods = ("popularity", "cf", "content", "hybrid")
    for method in methods:
        response = client.get(f"/api/recommendations?method={method}")
        assert response.status_code == 200
        items = response.get_json()
        assert len(items) == 10
        assert len({item["canonical_item_id"] for item in items}) == 10
        assert all("authors" in item and "genres" in item and item["reason"] for item in items)
    rated = client.get("/api/recommendations?method=popularity").get_json()[0]["canonical_item_id"]
    client.post(f"/api/ratings/{rated}", json={"rating": 5},
                headers={"X-CSRF-Token": token(client)})
    for method in methods:
        assert rated not in {
            row["canonical_item_id"]
            for row in client.get(f"/api/recommendations?method={method}").get_json()
        }


def test_rating_deletion_is_account_owned_and_search_escapes_wildcards(app, client):
    register_and_login(client, "first@example.test")
    csrf = token(client)
    client.post("/ratings/demo:01", data={"rating": "4", "csrf_token": csrf})
    assert client.get("/books?q=An+Lam").status_code == 200
    assert "Khu Rừng Sao" in client.get("/books?q=An+Lam").get_data(as_text=True)
    assert "Khu Rừng Sao" not in client.get("/books?q=%25").get_data(as_text=True)
    client.post("/logout", data={"csrf_token": csrf})
    register_and_login(client, "second@example.test")
    csrf = token(client)
    client.post("/ratings/demo:01/delete", data={"csrf_token": csrf})
    with app.app_context():
        rows = get_db().execute("SELECT rating FROM account_ratings").fetchall()
        assert [row["rating"] for row in rows] == [4]


def test_redirects_and_public_configuration(client):
    register_and_login(client)
    csrf = token(client)
    response = client.post("/ratings/demo:01", data={
        "rating": "4", "csrf_token": csrf, "next": "//attacker.example/",
    })
    assert response.headers["Location"].endswith("/books/demo:01")
    response = client.get("/")
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]
    assert "HttpOnly" in response.headers["Set-Cookie"]


def test_public_app_rejects_placeholder_secret(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "replace-with-a-local-random-value")
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        create_app()
