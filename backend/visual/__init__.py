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
