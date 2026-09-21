# C08 and C09 website completion evidence

Date: 21 September 2026

The Flask application now provides Vietnamese registration, login, logout,
catalog search, book details, initial genre preferences, 1-5 rating upsert,
HTML recommendations and JSON rating/recommendation endpoints. Passwords use
Werkzeug hashes. The server reads `account_id` only from the authenticated
session; a submitted owner field is ignored.

`backend/app/schema_postgresql.sql` implements the proposed PostgreSQL schema
and constraints. `schema_sqlite.sql` is a local test/demo adapter with the
same ownership and rating rules; it does not replace PostgreSQL as the target
database. The default local demo uses eighteen clearly labelled self-created
books and fictional aggregate model ratings. These fixtures demonstrate the
website only and are not research-quality evidence.

The online service calculates a profile from current account ratings or
selected genres. A rating or preference write invalidates the account cache.
The next request recomputes scores, excludes every rated canonical work and
returns ten unique works when possible. Reasons are tied to stored evidence:
matched genre, similar-reader support, content-profile support or
training-popularity fallback.

Automated checks cover password hashing, unauthenticated and invalid writes,
two-account isolation, upsert ownership, refresh after a rating change,
unique/rated-work exclusions, search/detail, a blank description and a valid
shorter list when fewer than ten works remain.
