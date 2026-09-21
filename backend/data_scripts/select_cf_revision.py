"""Freeze a validation-selected Pearson adjustment without touching old artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import pandas as pd

from backend.data_scripts.run_c04_c06_validation import (
    evaluate_method, relevant_validation_items, sha256_file, sorted_user_ids,
    write_csv, write_json,
)
from backend.recommenders.common import validate_explicit_ratings
from backend.recommenders.hybrid import HybridRecommender


ROOT = Path(__file__).resolve().parents[2]
OLD_VALIDATION = ROOT / "artifacts/experiments/c04_c07_seed42_candidates5000"
OLD_MODELS = ROOT / "artifacts/models/c04_c07_seed42_candidates5000"
AUDIT = ROOT / "artifacts/experiments/cf_support_validation_20260921/audit.json"
NEW_VALIDATION = ROOT / "artifacts/experiments/c04_c07_cf_revision_20260921"
NEW_MODELS = ROOT / "artifacts/models/c04_c07_cf_revision_20260921"


def main() -> None:
    if (NEW_VALIDATION / "validation_summary.json").exists():
        raise FileExistsError("CF revision validation already exists")
    previous = json.loads((OLD_VALIDATION / "validation_summary.json").read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit["model_sha256"] != sha256_file(OLD_MODELS / "user_pearson_cf.joblib"):
        raise ValueError("CF model changed after support audit")
    for details in previous["inputs"].values():
        if sha256_file(Path(details["path"])) != details["sha256"]:
            raise ValueError("validation input changed")
    selected = audit["grid"][0]
    if selected["minimum_support"] != 3 or selected["shrinkage"] != 2 or not selected["clip_to_1_5"]:
        raise ValueError("unexpected validation selection")

    popularity = joblib.load(OLD_MODELS / "popularity.joblib")
    cf = joblib.load(OLD_MODELS / "user_pearson_cf.joblib")
    content = joblib.load(OLD_MODELS / "content.joblib")
    cf.minimum_candidate_support = selected["minimum_support"]
    cf.support_shrinkage = selected["shrinkage"]
    cf.clip_predictions = selected["clip_to_1_5"]

    input_dir = ROOT / "data/samples/c03_seed42_users1000"
    validation = validate_explicit_ratings(pd.read_csv(
        input_dir / "validation_ratings.csv", dtype="string", keep_default_na=False
    ), "validation")
    train = validate_explicit_ratings(pd.read_csv(
        input_dir / "train_ratings.csv", dtype="string", keep_default_na=False
    ), "train")
    catalog = pd.read_csv(OLD_VALIDATION / "candidate_catalog.csv", dtype="string", keep_default_na=False)
    candidates = set(catalog["canonical_item_id"].astype(str))
    known = {str(user): set(group["canonical_item_id"].astype(str))
             for user, group in train.groupby("research_user_id", sort=False)}
    relevant, cohorts = relevant_validation_items(validation, candidates)
    users = sorted_user_ids(set(relevant))

    cf_rows, cf_summary = evaluate_method(
        "user_pearson_cf", lambda user: cf.recommend(user, limit=10),
        users, relevant, candidates, known, "cf",
    )
    content_scores = {user: content.score_from_ratings(content.user_rating_values.get(user, {}))
                      for user in users}
    cf_scores = {user: cf.score_from_ratings(cf.user_ratings.get(user, {}),
                                             exclude_neighbour_id=user) for user in users}
    alpha_rows = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        candidate = HybridRecommender(alpha=alpha).fit(cf, content, popularity)
        _, summary = evaluate_method(
            "hybrid",
            lambda user, model=candidate: model.recommend_from_component_scores(
                known.get(user, set()), cf_scores[user], content_scores[user],
                limit=10, fallback_user_id=user,
            ), users, relevant, candidates, known, "hybrid",
        )
        alpha_rows.append({"alpha": alpha, **summary})
    winner = max(alpha_rows, key=lambda row: (row["ndcg_at_10"],
                                              row["precision_at_10"],
                                              row["recall_at_10"], -row["alpha"]))
    hybrid = HybridRecommender(alpha=winner["alpha"]).fit(cf, content, popularity)

    NEW_VALIDATION.mkdir(parents=True, exist_ok=True)
    NEW_MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OLD_VALIDATION / "candidate_catalog.csv", NEW_VALIDATION / "candidate_catalog.csv")
    write_csv(pd.DataFrame(cf_rows), NEW_VALIDATION / "cf_validation_per_user.csv")
    write_csv(pd.DataFrame(alpha_rows), NEW_VALIDATION / "hybrid_validation_grid.csv")
    for name, model in (("popularity", popularity), ("user_pearson_cf", cf),
                        ("content", content), ("hybrid", hybrid)):
        joblib.dump(model, NEW_MODELS / f"{name}.joblib")
    summary = {
        "schema_version": "c04-c07-cf-revision-v1",
        "data_policy": {
            **previous["data_policy"],
            "prior_test_result_was_seen": True,
            "selection_for_revision": "validation labels only; audit triggered after weak original test CF result",
        },
        "configuration": {
            **previous["configuration"],
            "selected_alpha": winner["alpha"],
            "minimum_candidate_support": cf.minimum_candidate_support,
            "support_shrinkage": cf.support_shrinkage,
            "clip_predictions": cf.clip_predictions,
        },
        "inputs": previous["inputs"],
        "catalog": previous["catalog"],
        "validation_cohorts": cohorts,
        "selection_audit_sha256": sha256_file(AUDIT),
        "previous_validation_summary_sha256": sha256_file(OLD_VALIDATION / "validation_summary.json"),
        "methods": {"user_pearson_cf": cf_summary, "hybrid": winner},
        "validation_search": {"alpha_grid": alpha_rows},
    }
    write_json(NEW_VALIDATION / "validation_summary.json", summary)
    print(json.dumps({"cf": cf_summary, "alpha_selected": winner["alpha"],
                      "hybrid": winner}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
