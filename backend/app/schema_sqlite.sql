CREATE TABLE IF NOT EXISTS accounts (
    account_id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS books (
    canonical_item_id TEXT PRIMARY KEY,
    representative_source_book_id TEXT,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    language_code TEXT NOT NULL DEFAULT '',
    is_demo INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS authors (
    author_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    name_missing INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS genres (
    genre_id TEXT PRIMARY KEY,
    label TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS book_authors (
    canonical_item_id TEXT NOT NULL REFERENCES books(canonical_item_id) ON DELETE CASCADE,
    author_id TEXT NOT NULL REFERENCES authors(author_id) ON DELETE RESTRICT,
    PRIMARY KEY (canonical_item_id, author_id)
);
CREATE TABLE IF NOT EXISTS book_genres (
    canonical_item_id TEXT NOT NULL REFERENCES books(canonical_item_id) ON DELETE CASCADE,
    genre_id TEXT NOT NULL REFERENCES genres(genre_id) ON DELETE RESTRICT,
    PRIMARY KEY (canonical_item_id, genre_id)
);
CREATE TABLE IF NOT EXISTS account_ratings (
    account_id INTEGER NOT NULL REFERENCES accounts(account_id) ON DELETE CASCADE,
    canonical_item_id TEXT NOT NULL REFERENCES books(canonical_item_id) ON DELETE CASCADE,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (account_id, canonical_item_id)
);
CREATE TABLE IF NOT EXISTS account_genre_preferences (
    account_id INTEGER NOT NULL REFERENCES accounts(account_id) ON DELETE CASCADE,
    genre_id TEXT NOT NULL REFERENCES genres(genre_id) ON DELETE CASCADE,
    selected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (account_id, genre_id)
);
CREATE INDEX IF NOT EXISTS books_title_lower_idx ON books (LOWER(title));
CREATE INDEX IF NOT EXISTS account_ratings_account_idx ON account_ratings (account_id);
