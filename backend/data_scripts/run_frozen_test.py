"""Run the single frozen C10 comparison after validation selection is locked."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from backend.data_scripts.run_c04_c06_validation import (
    evaluate_method,
    sha256_file,
    sorted_user_ids,
    write_csv,
    write_json,
)
from backend.recommenders.common import validate_explicit_ratings


VERSION = "c10-frozen-test-v1"


def relevant_test_items(
    test: pd.DataFrame, candidate_items: set[str]
) -> tuple[dict[str, set[str]], dict[str, int]]:
    test_users = set(test["research_user_id"].astype(str))
    positive = test.loc[test["rating"] >= 4]
    positive_users = set(positive["research_user_id"].astype(str))
    eligible = positive.loc[positive["canonical_item_id"].isin(candidate_items)]
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in eligible.itertuples(index=False):
        relevant[str(row.research_user_id)].add(str(row.canonical_item_id))
    eligible_users = set(relevant)
    return dict(relevant), {
        "test_users": len(test_users),
        "users_without_positive_test_label": len(test_users - positive_users),
        "users_with_positive_test_label": len(positive_users),
        "users_with_positive_labels_only_outside_catalog": len(
            positive_users - eligible_users
        ),
        "eligible_users_with_relevant_candidate": len(eligible_users),
        "eligible_relevant_pairs": int(len(eligible)),
    }


def history_cohorts(
    test_users: set[str], eligible_users: set[str], known: dict[str, set[str]]
) -> dict[str, dict[str, int]]:
    def bucket(user_id: str) -> str:
        count = len(known.get(user_id, set()))
        if count == 0:
            return "0"
        if count <= 4:
            return "1-4"
        return "5+"

    result: dict[str, dict[str, int]] = {}
    for name, users in (("all_test_users", test_users), ("evaluated_users", eligible_users)):
        counts = {"0": 0, "1-4": 0, "5+": 0}
        for user_id in users:
            counts[bucket(user_id)] += 1
        result[name] = counts
    return result


def run(args: argparse.Namespace) -> dict:
    input_dir = Path(args.input_dir)
    validation_dir = Path(args.validation_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "test_summary.json"
    if summary_path.exists() and not args.allow_rerun:
        raise FileExistsError(
            f"{summary_path} already exists; refusing to rerun the frozen test"
        )

    validation_summary_path = validation_dir / "validation_summary.json"
    catalog_path = validation_dir / "candidate_catalog.csv"
    required_before_test = [
        validation_summary_path,
        catalog_path,
        input_dir / "train_ratings.csv",
        input_dir / "training_item_stats.csv",
        model_dir / "popularity.joblib",
        model_dir / "user_pearson_cf.joblib",
        model_dir / "content.joblib",
        model_dir / "hybrid.joblib",
    ]
    missing = [str(path) for path in required_before_test if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")

    validation_summary = json.loads(validation_summary_path.read_text(encoding="utf-8"))
    if validation_summary["data_policy"].get("test_ratings_read") is not False:
        raise ValueError("validation record does not prove test isolation")
    prior_test_path = Path(args.previous_test_summary) if args.previous_test_summary else None
    if validation_summary["data_policy"].get("prior_test_result_was_seen"):
        if not prior_test_path or not prior_test_path.exists() or not args.second_test_reason:
            raise ValueError("second test requires prior summary and documented reason")
        if prior_test_path.resolve() == summary_path.resolve():
            raise ValueError("second test must use a different output directory")
    for name, details in validation_summary["inputs"].items():
        recorded_path = Path(details["path"])
        if not recorded_path.exists():
            raise FileNotFoundError(recorded_path)
        recorded = details["sha256"]
        actual = sha256_file(recorded_path)
        if actual != recorded:
            raise ValueError(f"frozen {name} hash changed after validation")

    popularity = joblib.load(model_dir / "popularity.joblib")
    cf = joblib.load(model_dir / "user_pearson_cf.joblib")
    content = joblib.load(model_dir / "content.joblib")
    hybrid = joblib.load(model_dir / "hybrid.joblib")
    config = validation_summary["configuration"]
    if (
        cf.neighbour_count != config["selected_neighbour_count"]
        or cf.min_common_items != config["selected_min_common_items"]
        or hybrid.alpha != config["selected_alpha"]
        or getattr(cf, "minimum_candidate_support", 1) != config.get("minimum_candidate_support", 1)
        or getattr(cf, "support_shrinkage", 0.0) != config.get("support_shrinkage", 0.0)
        or getattr(cf, "clip_predictions", False) != config.get("clip_predictions", False)
    ):
        raise ValueError("loaded model parameters do not match validation selection")

    catalog = pd.read_csv(catalog_path, dtype="string", keep_default_na=False)
    candidate_items = set(catalog["canonical_item_id"].astype(str))
    if not candidate_items or candidate_items != popularity.candidate_items:
        raise ValueError("frozen candidate catalog does not match model")
    train = validate_explicit_ratings(
        pd.read_csv(input_dir / "train_ratings.csv", dtype="string", keep_default_na=False),
        "train_ratings",
    )
    known_by_user = {
        str(user_id): set(group["canonical_item_id"].astype(str))
        for user_id, group in train.groupby("research_user_id", sort=False)
    }

    # All configuration and artifact checks above finish before the test file is read.
    test_path = input_dir / "test_ratings.csv"
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    test = validate_explicit_ratings(
        pd.read_csv(test_path, dtype="string", keep_default_na=False),
        "test_ratings",
    )
    relevant_by_user, cohorts = relevant_test_items(test, candidate_items)
    evaluation_users = sorted_user_ids(set(relevant_by_user))

    methods = [
        (
            "popularity",
            lambda user_id: popularity.recommend(user_id, limit=10),
            "popularity",
        ),
        (
            "user_pearson_cf",
            lambda user_id: cf.recommend(user_id, limit=10),
            "cf",
        ),
        (
            "content",
            lambda user_id: content.recommend(user_id, limit=10),
            "content",
        ),
        (
            "hybrid",
            lambda user_id: hybrid.recommend(user_id, limit=10),
            "hybrid",
        ),
    ]
    all_rows: list[dict] = []
    method_summaries: dict[str, dict] = {}
    for name, recommend, valid_source in methods:
        rows, method_summary = evaluate_method(
            name,
            recommend,
            evaluation_users,
            relevant_by_user,
            candidate_items,
            known_by_user,
            valid_source,
        )
        all_rows.extend(rows)
        method_summaries[name] = method_summary

    output_dir.mkdir(parents=True, exist_ok=True)
    per_user_path = output_dir / "test_per_user.csv"
    write_csv(
        pd.DataFrame(all_rows).sort_values(
            ["method", "research_user_id"], kind="mergesort"
        ),
        per_user_path,
    )
    summary = {
        "schema_version": VERSION,
        "run_policy": {
            "selection_source": str(validation_summary_path.resolve()),
            "test_used_for_selection": False,
            "models_refit_after_validation": False,
            "shared_candidate_catalog": True,
            "shared_evaluation_users": True,
            "relevance": "test rating >= 4 within frozen candidate catalog",
            "precision_denominator": 10,
            "test_access_index": 2 if prior_test_path else 1,
            "prior_test_summary_sha256": sha256_file(prior_test_path) if prior_test_path else None,
            "second_test_reason": args.second_test_reason if prior_test_path else None,
            "prior_test_result_was_seen_before_this_selection": bool(prior_test_path),
        },
        "frozen_configuration": config,
        "input_hashes": {
            "validation_summary": sha256_file(validation_summary_path),
            "candidate_catalog": sha256_file(catalog_path),
            "train_ratings": sha256_file(input_dir / "train_ratings.csv"),
            "test_ratings": sha256_file(test_path),
        },
        "catalog_works": len(candidate_items),
        "test_cohorts": cohorts,
        "training_history_cohorts": history_cohorts(
            set(test["research_user_id"].astype(str)),
            set(evaluation_users),
            known_by_user,
        ),
        "methods": method_summaries,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "outputs": {"per_user_metrics": str(per_user_path.resolve())},
    }
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/samples/c03_seed42_users1000")
    parser.add_argument(
        "--validation-dir",
        default="artifacts/experiments/c04_c07_seed42_candidates5000",
    )
    parser.add_argument(
        "--model-dir", default="artifacts/models/c04_c07_seed42_candidates5000"
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/experiments/c10_frozen_test_seed42_candidates5000",
    )
    parser.add_argument(
        "--allow-rerun",
        action="store_true",
        help="Explicitly acknowledge that this is no longer the first test run",
    )
    parser.add_argument("--previous-test-summary", default="")
    parser.add_argument("--second-test-reason", default="")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
