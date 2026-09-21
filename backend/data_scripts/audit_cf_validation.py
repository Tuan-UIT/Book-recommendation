"""Compare small Pearson ranking adjustments using validation labels only.

This audit does not modify fitted models or read the frozen test split.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import pandas as pd

from backend.data_scripts.run_c04_c06_validation import relevant_validation_items, sha256_file
from backend.recommenders.evaluation import binary_ndcg_at_k, precision_at_k, recall_at_k


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "artifacts/models/c04_c07_seed42_candidates5000/user_pearson_cf.joblib"
VALIDATION = ROOT / "data/samples/c03_seed42_users1000/validation_ratings.csv"
OUTPUT = ROOT / "artifacts/experiments/cf_support_validation_20260921/audit.json"


def main() -> None:
    cf = joblib.load(MODEL)
    validation = pd.read_csv(VALIDATION, dtype="string", keep_default_na=False)
    validation["rating"] = pd.to_numeric(validation["rating"])
    relevant, cohorts = relevant_validation_items(validation, cf.candidate_items)
    users = sorted(relevant)
    grid = [(minimum, shrinkage, clip) for minimum in (1, 2, 3)
            for shrinkage in (0, 2) for clip in (False, True)]
    totals = {setting: defaultdict(float) for setting in grid}
    support_counts: Counter[int] = Counter()
    outside_scale = 0
    scored_count = 0

    for user in users:
        ratings = cf.user_ratings.get(user, {})
        raw_scores = cf.score_from_ratings(ratings, exclude_neighbour_id=user)
        target_mean = sum(ratings.values()) / len(ratings) if ratings else 0.0
        for setting in grid:
            minimum, shrinkage, clip = setting
            scored = []
            for item in raw_scores.values():
                if item.support_count < minimum:
                    continue
                value = float(item.score)
                if clip:
                    value = max(1.0, min(5.0, value))
                if shrinkage:
                    value = target_mean + (value - target_mean) * item.support_count / (item.support_count + shrinkage)
                scored.append((item.item_id, value))
            scored.sort(key=lambda pair: (-pair[1], pair[0]))
            selected = [item_id for item_id, _ in scored[:10]]
            if len(selected) < 10:
                selected.extend(item.item_id for item in cf.popularity.recommend(
                    user, limit=10 - len(selected),
                    additionally_exclude=set(ratings) | set(selected),
                ))
            relevant_items = relevant[user]
            totals[setting]["precision"] += precision_at_k(selected, relevant_items, k=10)
            totals[setting]["recall"] += recall_at_k(selected, relevant_items, k=10)
            totals[setting]["ndcg"] += binary_ndcg_at_k(selected, relevant_items, k=10)
            totals[setting]["full_fallback"] += int(not scored)
        for item in sorted(raw_scores.values(), key=lambda row: (-float(row.score), row.item_id))[:10]:
            support_counts[item.support_count] += 1
        scored_count += len(raw_scores)
        outside_scale += sum(float(item.score) < 1 or float(item.score) > 5 for item in raw_scores.values())

    rows = []
    for minimum, shrinkage, clip in grid:
        result = totals[(minimum, shrinkage, clip)]
        rows.append({
            "minimum_support": minimum,
            "shrinkage": shrinkage,
            "clip_to_1_5": clip,
            "precision_at_10": result["precision"] / len(users),
            "recall_at_10": result["recall"] / len(users),
            "ndcg_at_10": result["ndcg"] / len(users),
            "full_fallback_users": int(result["full_fallback"]),
        })
    rows.sort(key=lambda row: (-row["ndcg_at_10"], -row["precision_at_10"],
                               row["minimum_support"], row["shrinkage"], row["clip_to_1_5"]))
    output = {
        "policy": "validation only; existing frozen test untouched",
        "model_sha256": sha256_file(MODEL),
        "validation_sha256": sha256_file(VALIDATION),
        "evaluated_users": len(users),
        "cohorts": cohorts,
        "baseline_top_10_support_counts": dict(sorted(support_counts.items())),
        "raw_cf_scores": scored_count,
        "raw_cf_scores_outside_1_5": outside_scale,
        "grid": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
