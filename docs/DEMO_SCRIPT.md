# Demonstration script

Target duration: 6-8 minutes. Use the self-created demo catalog.

1. State that the public-safe fixture demonstrates behavior only. The metric
   table comes from the private Goodreads experiment.
2. Register the first account and sign in. Show that no research user ID is
   used as a website identity.
3. Select `fantasy` as an initial genre. Open the recommendation page and
   compare popularity, Pearson CF, content and Hybrid on the same account.
   Point out list length, fallback labels and one evidence-backed reason.
4. Search for `Khu Rừng Sao`, open its details and save a rating of 5. Refresh
   recommendations and show that the rated work is excluded.
5. Change that rating from 5 to 1. Explain that the cache is invalidated and
   the positive content profile is recomputed without retraining the offline
   model.
6. Open `Cuốn Sách Không Lời` to show the missing-description case. Author and
   genre features remain available.
7. Register a second account and rate the same work differently. Return to the
   first account to show that each account keeps its own row.
8. Present the original frozen table and the separate second-test correction.
   Explain that the original Hybrid tied content at alpha 0 and the revised
   validation selected alpha 0.25. Disclose that the first test had been seen
   before the later correction; do not call the second access untouched.
9. Close with the known limits: 1,000 sampled research users, historical
   Goodreads behavior, random holdout, imperfect genre metadata, and no claim
   about all Vietnamese readers.

Record defects, timing and audience comments after each rehearsal. A public
hosting URL remains pending until a permitted host and account are selected.
