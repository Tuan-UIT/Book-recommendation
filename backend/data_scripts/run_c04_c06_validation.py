"""Run the C04--C07 validation comparison without reading the test split.

The script selects one shared candidate catalog from training counts, fits the
popularity, Pearson CF, content, and Hybrid recommenders, and evaluates them
on the same validation users. Test labels are deliberately not an input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.recommenders.catalog import select_candidate_catalog
from backend.recommenders.common import RankedItem, validate_explicit_ratings
from backend.recommenders.content import ContentRecommender
from backend.recommenders.evaluation import (
    binary_ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from backend.recommenders.hybrid import HybridRecommender
from backend.recommenders.popularity import PopularityRecommender
from backend.recommenders.user_cf import UserPearsonCF


VERSION = "c04-c07-validation-v2"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sorted_user_ids(values: set[str]) -> list[str]:
    return sorted(
        values,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )


def relevant_validation_items(
    validation: pd.DataFrame, candidate_items: set[str]
) -> tuple[dict[str, set[str]], dict[str, int]]:
    validation_users = set(validation["research_user_id"].astype(str))
    positive = validation.loc[validation["rating"] >= 4]
    positive_users = set(positive["research_user_id"].astype(str))
    eligible = positive.loc[positive["canonical_item_id"].isin(candidate_items)]
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in eligible.itertuples(index=False):
        relevant[str(row.research_user_id)].add(str(row.canonical_item_id))
    eligible_users = set(relevant)
    cohorts = {
        "validation_users": len(validation_users),
        "users_without_positive_validation_label": len(
            validation_users - positive_users
        ),
        "users_with_positive_validation_label": len(positive_users),
        "users_with_positive_labels_only_outside_catalog": len(
            positive_users - eligible_users
        ),
        "eligible_users_with_relevant_candidate": len(eligible_users),
        "eligible_relevant_pairs": int(len(eligible)),
    }
    return dict(relevant), cohorts


def evaluate_method(
    name: str,
    recommend: Callable[[str], list[RankedItem]],
    user_ids: list[str],
    relevant_by_user: dict[str, set[str]],
    candidate_items: set[str],
    known_by_user: dict[str, set[str]],
    valid_source: str,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    source_counts: Counter[str] = Counter()
    full_fallback_users = 0
    partial_fallback_users = 0

    for user_id in user_ids:
        started = time.perf_counter()
        ranked = recommend(user_id)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        item_ids = [result.item_id for result in ranked]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError(f"{name} returned duplicate works for user {user_id}")
        invalid = set(item_ids) - candidate_items
        if invalid:
            raise ValueError(f"{name} returned works outside the shared catalog")
        already_known = set(item_ids) & known_by_user.get(user_id, set())
        if already_known:
            raise ValueError(f"{name} returned known works for user {user_id}")
        relevant = relevant_by_user[user_id]
        hits = len(set(item_ids[:10]) & relevant)
        sources = Counter(result.source for result in ranked)
        source_counts.update(sources)
        valid_items = sources.get(valid_source, 0)
        fallback_items = len(ranked) - valid_items if name != "popularity" else 0
        if name != "popularity":
            if valid_items == 0:
                full_fallback_users += 1
            elif fallback_items:
                partial_fallback_users += 1
        rows.append(
            {
                "research_user_id": user_id,
                "method": name,
                "relevant_candidate_count": len(relevant),
                "recommended_count": len(ranked),
                "hit_count": hits,
                "precision_at_10": precision_at_k(item_ids, relevant, k=10),
                "recall_at_10": recall_at_k(item_ids, relevant, k=10),
                "ndcg_at_10": binary_ndcg_at_k(item_ids, relevant, k=10),
                "valid_method_items": valid_items,
                "fallback_items": fallback_items,
                "latency_ms": elapsed_ms,
            }
        )

    frame = pd.DataFrame(rows)
    latencies = frame["latency_ms"].to_numpy(dtype=float)
    summary = {
        "evaluated_users": len(frame),
        "precision_at_10": float(frame["precision_at_10"].mean()),
        "recall_at_10": float(frame["recall_at_10"].mean()),
        "ndcg_at_10": float(frame["ndcg_at_10"].mean()),
        "mean_recommended_count": float(frame["recommended_count"].mean()),
        "full_fallback_users": full_fallback_users,
        "partial_fallback_users": partial_fallback_users,
        "source_item_counts": dict(sorted(source_counts.items())),
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "maximum": float(np.max(latencies)),
        },
    }
    return rows, summary


def run(args: argparse.Namespace) -> dict:
    input_dir = Path(args.input_dir)
    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    paths = {
        "train": input_dir / "train_ratings.csv",
        "validation": input_dir / "validation_ratings.csv",
        "item_stats": input_dir / "training_item_stats.csv",
        "books": metadata_dir / "clean_books.csv",
        "book_authors": metadata_dir / "book_authors.csv",
        "book_genres": metadata_dir / "book_genres.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required input files: {missing}")

    train = validate_explicit_ratings(
        pd.read_csv(paths["train"], dtype="string", keep_default_na=False),
        "train_ratings",
    )
    validation = validate_explicit_ratings(
        pd.read_csv(paths["validation"], dtype="string", keep_default_na=False),
        "validation_ratings",
    )
    item_stats = pd.read_csv(paths["item_stats"], dtype="string", keep_default_na=False)
    catalog = select_candidate_catalog(
        item_stats,
        max_items=args.max_candidates,
        min_train_ratings=args.min_candidate_ratings,
    )
    candidate_items = set(catalog["canonical_item_id"].astype(str))
    relevant_by_user, cohorts = relevant_validation_items(validation, candidate_items)
    evaluation_users = sorted_user_ids(set(relevant_by_user))
    known_by_user = {
        str(user_id): set(group["canonical_item_id"].astype(str))
        for user_id, group in train.groupby("research_user_id", sort=False)
    }

    fit_started = time.perf_counter()
    popularity = PopularityRecommender().fit(train, candidate_items)
    popularity_fit_seconds = time.perf_counter() - fit_started

    books = pd.read_csv(paths["books"], dtype="string", keep_default_na=False)
    book_authors = pd.read_csv(
        paths["book_authors"], dtype="string", keep_default_na=False
    )
    book_genres = pd.read_csv(
        paths["book_genres"], dtype="string", keep_default_na=False
    )
    fit_started = time.perf_counter()
    content = ContentRecommender(
        max_tfidf_features=args.max_tfidf_features
    ).fit(
        train,
        candidate_items,
        books,
        book_authors,
        book_genres,
        popularity,
    )
    content_fit_seconds = time.perf_counter() - fit_started

    cf_grid_rows: list[dict] = []
    cf_runs: list[tuple[UserPearsonCF, list[dict], dict, float]] = []
    cf_grid_started = time.perf_counter()
    for min_common_items in sorted(set(args.min_common_items_grid)):
        for neighbour_count in sorted(set(args.neighbour_count_grid)):
            fit_started = time.perf_counter()
            candidate_cf = UserPearsonCF(
                neighbour_count=neighbour_count,
                min_common_items=min_common_items,
            ).fit(train, candidate_items, popularity)
            fit_seconds = time.perf_counter() - fit_started
            rows, method_summary = evaluate_method(
                "user_pearson_cf",
                lambda user_id, model=candidate_cf: model.recommend(user_id, limit=10),
                evaluation_users,
                relevant_by_user,
                candidate_items,
                known_by_user,
                "cf",
            )
            cf_runs.append((candidate_cf, rows, method_summary, fit_seconds))
            cf_grid_rows.append(
                {
                    "neighbour_count": neighbour_count,
                    "min_common_items": min_common_items,
                    "precision_at_10": method_summary["precision_at_10"],
                    "recall_at_10": method_summary["recall_at_10"],
                    "ndcg_at_10": method_summary["ndcg_at_10"],
                    "full_fallback_users": method_summary["full_fallback_users"],
                    "partial_fallback_users": method_summary[
                        "partial_fallback_users"
                    ],
                    "mean_latency_ms": method_summary["latency_ms"]["mean"],
                    "fit_seconds": fit_seconds,
                }
            )
    cf_grid_seconds = time.perf_counter() - cf_grid_started
    best_cf_index = max(
        range(len(cf_runs)),
        key=lambda index: (
            cf_runs[index][2]["ndcg_at_10"],
            cf_runs[index][2]["precision_at_10"],
            cf_runs[index][2]["recall_at_10"],
            -cf_runs[index][0].min_common_items,
            -cf_runs[index][0].neighbour_count,
        ),
    )
    cf, cf_rows, cf_summary, cf_fit_seconds = cf_runs[best_cf_index]

    evaluations = [
        (
            "popularity",
            lambda user_id: popularity.recommend(user_id, limit=10),
            "popularity",
        ),
        ("content", lambda user_id: content.recommend(user_id, limit=10), "content"),
    ]
    all_rows: list[dict] = []
    all_rows.extend(cf_rows)
    method_summaries: dict[str, dict] = {"user_pearson_cf": cf_summary}
    for name, recommend, valid_source in evaluations:
        rows, summary = evaluate_method(
            name,
            recommend,
            evaluation_users,
            relevant_by_user,
            candidate_items,
            known_by_user,
            valid_source,
        )
        all_rows.extend(rows)
        method_summaries[name] = summary

    hybrid_grid_rows: list[dict] = []
    hybrid_runs: list[tuple[HybridRecommender, list[dict], dict]] = []
    cf_component_scores = {
        user_id: cf.score_from_ratings(
            cf.user_ratings.get(user_id, {}), exclude_neighbour_id=user_id
        )
        for user_id in evaluation_users
    }
    content_component_scores = {
        user_id: content.score_from_ratings(
            content.user_rating_values.get(user_id, {})
        )
        for user_id in evaluation_users
    }
    hybrid_grid_started = time.perf_counter()
    for alpha in sorted(set(args.alpha_grid)):
        hybrid_candidate = HybridRecommender(alpha=alpha).fit(
            cf, content, popularity
        )
        rows, method_summary = evaluate_method(
            "hybrid",
            lambda user_id, model=hybrid_candidate: model.recommend_from_component_scores(
                known_by_user.get(user_id, set()),
                cf_component_scores[user_id],
                content_component_scores[user_id],
                limit=10,
                fallback_user_id=user_id,
            ),
            evaluation_users,
            relevant_by_user,
            candidate_items,
            known_by_user,
            "hybrid",
        )
        hybrid_runs.append((hybrid_candidate, rows, method_summary))
        hybrid_grid_rows.append(
            {
                "alpha": alpha,
                "precision_at_10": method_summary["precision_at_10"],
                "recall_at_10": method_summary["recall_at_10"],
                "ndcg_at_10": method_summary["ndcg_at_10"],
                "full_fallback_users": method_summary["full_fallback_users"],
                "partial_fallback_users": method_summary["partial_fallback_users"],
                "mean_combination_latency_ms": method_summary["latency_ms"]["mean"],
            }
        )
    hybrid_grid_seconds = time.perf_counter() - hybrid_grid_started
    best_hybrid_index = max(
        range(len(hybrid_runs)),
        key=lambda index: (
            hybrid_runs[index][2]["ndcg_at_10"],
            hybrid_runs[index][2]["precision_at_10"],
            hybrid_runs[index][2]["recall_at_10"],
            -hybrid_runs[index][0].alpha,
        ),
    )
    hybrid = hybrid_runs[best_hybrid_index][0]
    hybrid_rows, hybrid_summary = evaluate_method(
        "hybrid",
        lambda user_id: hybrid.recommend(user_id, limit=10),
        evaluation_users,
        relevant_by_user,
        candidate_items,
        known_by_user,
        "hybrid",
    )
    all_rows.extend(hybrid_rows)
    method_summaries["hybrid"] = hybrid_summary

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    write_csv(catalog, output_dir / "candidate_catalog.csv")
    write_csv(pd.DataFrame(cf_grid_rows), output_dir / "cf_validation_grid.csv")
    write_csv(
        pd.DataFrame(hybrid_grid_rows), output_dir / "hybrid_validation_grid.csv"
    )
    per_user = pd.DataFrame(all_rows).sort_values(
        ["method", "research_user_id"], kind="mergesort"
    )
    write_csv(per_user, output_dir / "validation_per_user.csv")
    joblib.dump(popularity, model_dir / "popularity.joblib")
    joblib.dump(cf, model_dir / "user_pearson_cf.joblib")
    joblib.dump(content, model_dir / "content.joblib")
    joblib.dump(hybrid, model_dir / "hybrid.joblib")

    summary = {
        "schema_version": VERSION,
        "data_policy": {
            "fit_input": "train_ratings.csv only",
            "candidate_selection": (
                "training count descending, canonical item ID ascending"
            ),
            "validation_use": "metrics and later parameter selection only",
            "test_ratings_read": False,
            "relevance": "validation rating >= 4 within shared candidate catalog",
            "precision_denominator": 10,
        },
        "configuration": {
            "max_candidates": args.max_candidates,
            "min_candidate_ratings": args.min_candidate_ratings,
            "max_tfidf_features": args.max_tfidf_features,
            "selected_neighbour_count": cf.neighbour_count,
            "selected_min_common_items": cf.min_common_items,
            "selected_alpha": hybrid.alpha,
        },
        "validation_search": {
            "neighbour_count_grid": sorted(set(args.neighbour_count_grid)),
            "min_common_items_grid": sorted(set(args.min_common_items_grid)),
            "selection_metric": "maximum validation NDCG@10",
            "tie_break": (
                "Precision@10, Recall@10, then smaller common-item threshold and k"
            ),
            "grid_seconds": cf_grid_seconds,
            "rows": cf_grid_rows,
            "alpha_grid": sorted(set(args.alpha_grid)),
            "hybrid_selection_metric": "maximum validation NDCG@10",
            "hybrid_tie_break": (
                "Precision@10, Recall@10, then smaller alpha"
            ),
            "hybrid_grid_seconds": hybrid_grid_seconds,
            "hybrid_rows": hybrid_grid_rows,
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "catalog": {
            "works": len(catalog),
            "minimum_training_count": int(catalog["train_rating_count"].min()),
            "maximum_training_count": int(catalog["train_rating_count"].max()),
        },
        "validation_cohorts": cohorts,
        "fit_seconds": {
            "popularity": popularity_fit_seconds,
            "user_pearson_cf": cf_fit_seconds,
            "content": content_fit_seconds,
        },
        "methods": method_summaries,
        "outputs": {
            "candidate_catalog": str((output_dir / "candidate_catalog.csv").resolve()),
            "per_user_metrics": str((output_dir / "validation_per_user.csv").resolve()),
            "cf_validation_grid": str(
                (output_dir / "cf_validation_grid.csv").resolve()
            ),
            "hybrid_validation_grid": str(
                (output_dir / "hybrid_validation_grid.csv").resolve()
            ),
            "model_dir": str(model_dir.resolve()),
        },
    }
    write_json(output_dir / "validation_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="data/samples/c03_seed42_users1000",
        help="Private C03 split directory",
    )
    parser.add_argument(
        "--metadata-dir",
        default="data/samples/c02_seed42_users1000",
        help="Private C02 metadata directory",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/experiments/c04_c07_seed42_candidates5000",
    )
    parser.add_argument(
        "--model-dir",
        default="artifacts/models/c04_c07_seed42_candidates5000",
    )
    parser.add_argument("--max-candidates", type=int, default=5_000)
    parser.add_argument("--min-candidate-ratings", type=int, default=3)
    parser.add_argument(
        "--neighbour-count-grid", type=int, nargs="+", default=[20, 40]
    )
    parser.add_argument(
        "--min-common-items-grid", type=int, nargs="+", default=[3, 5]
    )
    parser.add_argument("--max-tfidf-features", type=int, default=10_000)
    parser.add_argument(
        "--alpha-grid", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1]
    )
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
