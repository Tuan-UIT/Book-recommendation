"""Hand-checkable binary top-k metrics used by all recommendation methods."""

from __future__ import annotations

import math
from collections.abc import Iterable


def _top_unique(recommended: Iterable[str], k: int) -> list[str]:
    if k < 1:
        raise ValueError("k must be positive")
    result: list[str] = []
    seen: set[str] = set()
    for raw_item_id in recommended:
        item_id = str(raw_item_id)
        if item_id in seen:
            raise ValueError("recommendation list contains duplicate canonical works")
        seen.add(item_id)
        if len(result) < k:
            result.append(item_id)
    return result


def precision_at_k(
    recommended: Iterable[str], relevant: Iterable[str], *, k: int = 10
) -> float:
    """Binary Precision@k with the proposal-required fixed denominator k."""

    top = _top_unique(recommended, k)
    relevant_set = {str(item_id) for item_id in relevant}
    hits = sum(item_id in relevant_set for item_id in top)
    return hits / k


def recall_at_k(
    recommended: Iterable[str], relevant: Iterable[str], *, k: int = 10
) -> float:
    """Binary Recall@k over all relevant held-out candidate works."""

    top = _top_unique(recommended, k)
    relevant_set = {str(item_id) for item_id in relevant}
    if not relevant_set:
        raise ValueError("Recall@k is undefined without a relevant candidate work")
    hits = sum(item_id in relevant_set for item_id in top)
    return hits / len(relevant_set)


def binary_ndcg_at_k(
    recommended: Iterable[str], relevant: Iterable[str], *, k: int = 10
) -> float:
    """Binary NDCG@k using min(k, relevant candidate count) for IDCG."""

    top = _top_unique(recommended, k)
    relevant_set = {str(item_id) for item_id in relevant}
    if not relevant_set:
        raise ValueError("NDCG@k is undefined without a relevant candidate work")
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item_id in enumerate(top, start=1)
        if item_id in relevant_set
    )
    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg
