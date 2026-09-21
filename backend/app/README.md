# Flask application

`create_app` implements the authenticated Vietnamese website. PostgreSQL is
the target database through `schema_postgresql.sql`; `schema_sqlite.sql` exists
for isolated tests and the self-created local demo. Run `flask init-db` before
the first request and `flask seed-demo` only for the public-safe fixture.

Rating ownership always comes from the authenticated session. The online
service invalidates the account cache after rating or preference changes and
uses the current rows to rebuild the target profile.
