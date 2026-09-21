"""Flask application factory for the book recommendation website."""

from __future__ import annotations

import os
import secrets
from hmac import compare_digest
from functools import wraps
from pathlib import Path
from urllib.parse import urlsplit

import joblib
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from . import db as database
from .catalog_import import load_research_catalog
from .demo_data import build_demo_recommender, seed_demo_catalog
from .service import RecommendationService


def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if session.get("account_id") is None:
            return redirect(url_for("login", next=request.path))
        return view(**kwargs)

    return wrapped_view


def safe_next(value: str | None, fallback: str) -> str:
    if not value or not value.startswith("/") or value.startswith("//"):
        return fallback
    if "\\" in value or any(char in value for char in "\r\n"):
        return fallback
    parsed = urlsplit(value)
    return value if not parsed.scheme and not parsed.netloc else fallback


def create_app(test_config: dict | None = None) -> Flask:
    project_root = Path(__file__).resolve().parents[2]
    app = Flask(
        __name__,
        template_folder=str(project_root / "frontend" / "templates"),
        static_folder=str(project_root / "frontend" / "static"),
    )
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", ""),
        DATABASE_URL=os.environ.get(
            "DATABASE_URL", f"sqlite:///{project_root / 'data' / 'demo.sqlite3'}"
        ),
        MODEL_DIR=os.environ.get("MODEL_DIR", ""),
        DEMO_MODE=os.environ.get("DEMO_MODE", "1") == "1",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "0") == "1",
        MAX_CONTENT_LENGTH=64 * 1024,
    )
    if test_config:
        app.config.update(test_config)
    if app.testing:
        app.config.setdefault("CSRF_ENABLED", False)
    else:
        app.config["CSRF_ENABLED"] = True
        if not app.config["SECRET_KEY"] or app.config["SECRET_KEY"].startswith("replace-"):
            raise RuntimeError("Set a private, random SECRET_KEY before starting the website")
    if app.config["DEMO_MODE"] and app.config["MODEL_DIR"]:
        raise RuntimeError("DEMO_MODE cannot load a private research model")
    database.init_app(app)

    recommender = app.config.get("RECOMMENDER")
    if recommender is None:
        model_dir = Path(app.config["MODEL_DIR"]) if app.config["MODEL_DIR"] else None
        if not app.config["DEMO_MODE"] and model_dir and (model_dir / "hybrid.joblib").exists():
            recommender = joblib.load(model_dir / "hybrid.joblib")
        elif not app.config["DEMO_MODE"]:
            raise RuntimeError("Research mode requires MODEL_DIR/hybrid.joblib")
        else:
            recommender = build_demo_recommender(alpha=0.0)
    app.extensions["recommendations"] = RecommendationService(recommender)

    @app.before_request
    def load_logged_in_user() -> None:
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and app.config["CSRF_ENABLED"]:
            supplied = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token", "")
            expected = session.get("csrf_token", "")
            if not expected or not compare_digest(expected, supplied):
                abort(400, "Mã bảo vệ biểu mẫu không hợp lệ.")
        account_id = session.get("account_id")
        if account_id is None:
            request.account = None
        else:
            request.account = database.get_db().execute(
                "SELECT account_id, email FROM accounts WHERE account_id = %s",
                (account_id,),
            ).fetchone()

    @app.context_processor
    def template_context() -> dict:
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(32)
        return {
            "current_account": getattr(request, "account", None),
            "csrf_token": session["csrf_token"],
        }

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        )
        if request.is_secure and app.config["SESSION_COOKIE_SECURE"]:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.route("/register", methods=("GET", "POST"))
    def register():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            error = None
            if not email or "@" not in email:
                error = "Vui lòng nhập email hợp lệ."
            elif len(password) < 8:
                error = "Mật khẩu cần ít nhất 8 ký tự."
            if error is None:
                db = database.get_db()
                try:
                    db.execute(
                        "INSERT INTO accounts (email, password_hash) VALUES (%s, %s)",
                        (email, generate_password_hash(password)),
                    )
                    db.commit()
                except Exception as exc:
                    db.rollback()
                    if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
                        error = "Email này đã được đăng ký."
                    else:
                        raise
                else:
                    flash("Đăng ký thành công. Bạn có thể đăng nhập.", "success")
                    return redirect(url_for("login"))
            flash(error, "error")
        return render_template("register.html")

    @app.route("/login", methods=("GET", "POST"))
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            account = database.get_db().execute(
                "SELECT account_id, email, password_hash FROM accounts WHERE email = %s",
                (email,),
            ).fetchone()
            if account is None or not check_password_hash(account["password_hash"], password):
                flash("Email hoặc mật khẩu không đúng.", "error")
            else:
                next_path = safe_next(request.form.get("next"), url_for("recommendations"))
                session.clear()
                session["account_id"] = int(account["account_id"])
                return redirect(next_path)
        return render_template("login.html", next_path=safe_next(request.args.get("next"), ""))

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("index"))

    @app.get("/books")
    def books():
        query = request.args.get("q", "").strip()[:100]
        db = database.get_db()
        if query:
            escaped = query.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped}%"
            rows = db.execute(
                "SELECT b.canonical_item_id, b.title, b.description FROM books b "
                "WHERE LOWER(b.title) LIKE %s ESCAPE '\\' "
                "OR EXISTS (SELECT 1 FROM book_authors ba JOIN authors a ON a.author_id = ba.author_id "
                "WHERE ba.canonical_item_id = b.canonical_item_id AND LOWER(a.display_name) LIKE %s ESCAPE '\\') "
                "OR EXISTS (SELECT 1 FROM book_genres bg JOIN genres g ON g.genre_id = bg.genre_id "
                "WHERE bg.canonical_item_id = b.canonical_item_id AND LOWER(g.label) LIKE %s ESCAPE '\\') "
                "ORDER BY b.title, b.canonical_item_id LIMIT 50",
                (pattern, pattern, pattern),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT canonical_item_id, title, description FROM books ORDER BY title, canonical_item_id LIMIT 50"
            ).fetchall()
        book_authors: dict[str, list[str]] = {}
        if rows:
            ids = [str(row["canonical_item_id"]) for row in rows]
            placeholders = ", ".join(["%s"] * len(ids))
            author_rows = db.execute(
                "SELECT ba.canonical_item_id, a.display_name FROM book_authors ba "
                "JOIN authors a ON a.author_id = ba.author_id "
                f"WHERE ba.canonical_item_id IN ({placeholders}) ORDER BY a.display_name",
                ids,
            ).fetchall()
            for row in author_rows:
                book_authors.setdefault(str(row["canonical_item_id"]), []).append(
                    str(row["display_name"])
                )
        return render_template("books.html", books=rows, query=query, book_authors=book_authors)

    def fetch_book(item_id: str):
        db = database.get_db()
        book = db.execute(
            "SELECT canonical_item_id, title, description, language_code, is_demo FROM books WHERE canonical_item_id = %s",
            (item_id,),
        ).fetchone()
        if book is None:
            abort(404)
        authors = db.execute(
            "SELECT a.display_name FROM book_authors ba JOIN authors a ON a.author_id = ba.author_id WHERE ba.canonical_item_id = %s ORDER BY a.display_name",
            (item_id,),
        ).fetchall()
        genres = db.execute(
            "SELECT g.label FROM book_genres bg JOIN genres g ON g.genre_id = bg.genre_id WHERE bg.canonical_item_id = %s ORDER BY g.label",
            (item_id,),
        ).fetchall()
        rating = None
        if session.get("account_id") is not None:
            rating = db.execute(
                "SELECT rating FROM account_ratings WHERE account_id = %s AND canonical_item_id = %s",
                (session["account_id"], item_id),
            ).fetchone()
        return book, authors, genres, rating

    @app.get("/books/<path:item_id>")
    def book_detail(item_id: str):
        book, authors, genres, rating = fetch_book(item_id)
        return render_template(
            "book_detail.html",
            book=book,
            authors=authors,
            genres=genres,
            rating=rating,
        )

    def save_rating(item_id: str, raw_rating) -> int:
        if session.get("account_id") is None:
            abort(401)
        try:
            rating = int(raw_rating)
        except (TypeError, ValueError):
            abort(400, "Điểm đánh giá phải là số nguyên từ 1 đến 5.")
        if rating < 1 or rating > 5:
            abort(400, "Điểm đánh giá phải nằm trong khoảng 1 đến 5.")
        db = database.get_db()
        if db.execute(
            "SELECT 1 FROM books WHERE canonical_item_id = %s", (item_id,)
        ).fetchone() is None:
            abort(404)
        db.execute(
            "INSERT INTO account_ratings (account_id, canonical_item_id, rating) VALUES (%s, %s, %s) ON CONFLICT (account_id, canonical_item_id) DO UPDATE SET rating = excluded.rating, updated_at = CURRENT_TIMESTAMP",
            (session["account_id"], item_id, rating),
        )
        db.commit()
        app.extensions["recommendations"].invalidate(session["account_id"])
        return rating

    @app.post("/ratings/<path:item_id>")
    @login_required
    def rate_book(item_id: str):
        save_rating(item_id, request.form.get("rating"))
        flash("Đã lưu đánh giá và làm mới hồ sơ gợi ý.", "success")
        return redirect(safe_next(request.form.get("next"), url_for("book_detail", item_id=item_id)))

    @app.post("/ratings/<path:item_id>/delete")
    @login_required
    def delete_rating(item_id: str):
        database.get_db().execute(
            "DELETE FROM account_ratings WHERE account_id = %s AND canonical_item_id = %s",
            (session["account_id"], item_id),
        )
        database.get_db().commit()
        app.extensions["recommendations"].invalidate(session["account_id"])
        flash("Đã xóa đánh giá của bạn.", "success")
        return redirect(safe_next(request.form.get("next"), url_for("book_detail", item_id=item_id)))

    @app.route("/preferences", methods=("GET", "POST"))
    @login_required
    def preferences():
        db = database.get_db()
        if request.method == "POST":
            selected = sorted(set(request.form.getlist("genres")))
            valid = {
                str(row["genre_id"])
                for row in db.execute("SELECT genre_id FROM genres").fetchall()
            }
            if not set(selected) <= valid:
                abort(400, "Thể loại không hợp lệ.")
            db.execute(
                "DELETE FROM account_genre_preferences WHERE account_id = %s",
                (session["account_id"],),
            )
            db.executemany(
                "INSERT INTO account_genre_preferences (account_id, genre_id) VALUES (%s, %s)",
                [(session["account_id"], value) for value in selected],
            )
            db.commit()
            app.extensions["recommendations"].invalidate(session["account_id"])
            flash("Đã cập nhật thể loại yêu thích.", "success")
            return redirect(url_for("recommendations"))
        genres = db.execute("SELECT genre_id, label FROM genres ORDER BY label").fetchall()
        selected = {
            str(row["genre_id"])
            for row in db.execute(
                "SELECT genre_id FROM account_genre_preferences WHERE account_id = %s",
                (session["account_id"],),
            ).fetchall()
        }
        return render_template("preferences.html", genres=genres, selected=selected)

    @app.get("/profile")
    @login_required
    def profile():
        ratings = database.get_db().execute(
            "SELECT b.canonical_item_id, b.title, r.rating FROM account_ratings r "
            "JOIN books b ON b.canonical_item_id = r.canonical_item_id "
            "WHERE r.account_id = %s ORDER BY r.updated_at DESC, b.title",
            (session["account_id"],),
        ).fetchall()
        return render_template("profile.html", ratings=ratings)

    @app.get("/recommendations")
    @login_required
    def recommendations():
        method = request.args.get("method", "hybrid")
        if method not in RecommendationService.METHODS:
            abort(400)
        items = app.extensions["recommendations"].recommend(session["account_id"], method=method)
        return render_template(
            "recommendations.html", recommendations=items, method=method,
            hybrid_alpha=app.extensions["recommendations"].recommender.alpha,
        )

    @app.get("/api/recommendations")
    @login_required
    def recommendations_api():
        method = request.args.get("method", "hybrid")
        if method not in RecommendationService.METHODS:
            abort(400)
        return jsonify(
            app.extensions["recommendations"].recommend(session["account_id"], method=method)
        )

    @app.post("/api/ratings/<path:item_id>")
    @login_required
    def ratings_api(item_id: str):
        payload = request.get_json(silent=True) or {}
        rating = save_rating(item_id, payload.get("rating"))
        return jsonify({"canonical_item_id": item_id, "rating": rating})

    @app.cli.command("seed-demo")
    def seed_demo_command() -> None:
        if not app.config["DEMO_MODE"]:
            raise RuntimeError("seed-demo is available only in DEMO_MODE")
        seed_demo_catalog(database.get_db())
        print("Loaded self-created demo catalog.")

    @app.cli.command("load-research-catalog")
    def load_research_catalog_command() -> None:
        if app.config["DEMO_MODE"]:
            raise RuntimeError("research catalog loading is forbidden in DEMO_MODE")
        model = app.extensions["recommendations"].recommender
        counts = load_research_catalog(
            database.get_db(),
            metadata_dir=project_root / "data/samples/c02_seed42_users1000",
            candidate_path=project_root / "artifacts/experiments/c04_c07_cf_revision_20260921/candidate_catalog.csv",
            model_candidate_ids=model.candidate_items,
        )
        print(f"Loaded private model-matched catalog: {counts}")

    return app
