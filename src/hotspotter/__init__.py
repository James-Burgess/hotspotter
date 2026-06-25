"""Reusable HotSpotter algorithm library."""

from hotspotter.config import (
    HotSpotterConfig,
    IdentificationConfig,
    MiewIdConfig,
    SiftConfig,
)
from hotspotter.data import AnnotatedImage, FeatureSet, Match, ScoredMatch
from hotspotter.name_scoring import (
    align_name_scores_with_annots,
    compute_csum_annot_scores,
    compute_fmech_score,
    compute_maxcsum_name_score,
    compute_sumamech_name_score,
    group_matches_by_name,
    score_matches_with_names,
)
from hotspotter.pipeline import identify
from hotspotter.spatial import make_sver_shortlist

__all__ = [
    # Config
    "IdentificationConfig",
    "HotSpotterConfig",
    "MiewIdConfig",
    "SiftConfig",
    # Data
    "FeatureSet",
    "AnnotatedImage",
    "Match",
    "ScoredMatch",
    # Pipeline
    "identify",
    # Name scoring
    "score_matches_with_names",
    "compute_fmech_score",
    "compute_csum_annot_scores",
    "compute_maxcsum_name_score",
    "compute_sumamech_name_score",
    "align_name_scores_with_annots",
    "group_matches_by_name",
    # Spatial
    "make_sver_shortlist",
]
