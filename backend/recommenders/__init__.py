"""Recommendation methods required by the signed thesis proposal."""

from .catalog import select_candidate_catalog
from .content import ContentRecommender
from .hybrid import HybridRecommender
from .evaluation import binary_ndcg_at_k, precision_at_k, recall_at_k
from .popularity import PopularityRecommender
from .user_cf import UserPearsonCF

__all__ = [
    "ContentRecommender",
    "HybridRecommender",
    "PopularityRecommender",
    "UserPearsonCF",
    "binary_ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "select_candidate_catalog",
]
