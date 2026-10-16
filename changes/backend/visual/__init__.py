# ===========================================================================
# [NEW FILE - BACKUP FOR TEAMMATE & ANTIGRAVITY CONTEXT]
# Feature: Feature 1: Image Embeddings Package Init
# Relative Path: backend/visual/__init__.py
# Summary: Package exports for visual embedding, candidate proposal generation, instance matching, and visual retrieval service.
# Integration: Part of SecondMind Image Embeddings & Accurate Timestamps features.
# If your teammate's code fails or lacks this file, drop this directly into place.
# ===========================================================================

"""Visual perception, reference-image embeddings, candidate proposals, and feature matching."""

from backend.visual.encoder import (
    VisualEncoder,
    MobileNetVisualEncoder,
    FallbackColorGridEncoder,
    LocalFeatureMatcher,
    LocalVerificationResult,
    get_visual_encoder,
)
from backend.visual.proposals import BoundingBox, CandidateRegionProposalGenerator
from backend.visual.matcher import MatchStatus, CandidateMatch, VisualSearchResult, SimilarityMatcher
from backend.visual.service import VisualRetrievalService

__all__ = [
    "VisualEncoder",
    "MobileNetVisualEncoder",
    "FallbackColorGridEncoder",
    "LocalFeatureMatcher",
    "LocalVerificationResult",
    "get_visual_encoder",
    "BoundingBox",
    "CandidateRegionProposalGenerator",
    "MatchStatus",
    "CandidateMatch",
    "VisualSearchResult",
    "SimilarityMatcher",
    "VisualRetrievalService",
]
