"""Pydantic models for algorithm configuration."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SiftConfig(BaseModel):
    scale: list[float] = Field(
        default_factory=lambda: [1.0, 4.0, 8.0],
        description="Scales for Hessian-affine detection",
    )
    ori_hist_bins: int = Field(default=36, ge=8, le=360)
    ori_hist_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class HotSpotterConfig(BaseModel):
    knn: int = Field(default=4, ge=1, description="Number of nearest neighbors")
    knorm: int = Field(
        default=1,
        ge=1,
        description="Number of normalizer neighbor columns. WBIA default: 1.",
    )
    kpad: int = Field(
        default=1, ge=0, description="Extra K columns for self-match buffer"
    )
    kpad_policy: Literal["fixed", "dynamic"] = Field(
        default="fixed",
        description="'fixed' uses kpad value; 'dynamic' computes from impossible annots",
    )
    score_method: Literal["csum", "nsum", "csum_wbia", "nsum_wbia", "sumamech"] = Field(
        default="nsum",
        description=(
            "Simple: 'csum' (per-annot sum), 'nsum' (per-annot avg). "
            "WBIA: 'nsum_wbia' (fmech), 'csum_wbia' (max-per-name), 'sumamech'"
        ),
    )
    prescore_method: Literal["csum", "nsum", "csum_wbia", "nsum_wbia", "sumamech"] = (
        Field(default="nsum")
    )
    normalizer_rule: Literal["last", "name"] = Field(
        default="last",
        description="'last' uses farthest neighbour; 'name' picks from different name ID",
    )
    can_match_samename: bool = Field(
        default=True,
        description="Allow matches to annotations sharing the query's name (WBIA default: True)",
    )
    can_match_sameimg: bool = Field(
        default=False,
        description="Allow matches to annotations in the same image as the query "
        "(WBIA default: False). When False, same-image ('contact') annotations "
        "are added to the impossible-daids filter.",
    )
    sqrd_dist_on: bool = Field(
        default=False,
        description="Keep distances in squared-norm space (no sqrt). WBIA default: False.",
    )
    normonly_on: bool = Field(
        default=False,
        description="Replace voting dists with normalizer dist (debug). WBIA default: False.",
    )
    rotation_invariance: bool = Field(
        default=False,
        description="Enable XY-dedup in fmech to prevent rotated duplicates from double-voting per name.",
    )
    minscale_thresh: float | None = Field(
        default=None,
        gt=0.0,
        description="Minimum keypoint scale for feature filtering before FLANN query.",
    )
    maxscale_thresh: float | None = Field(
        default=None,
        gt=0.0,
        description="Maximum keypoint scale for feature filtering before FLANN query.",
    )
    fgw_thresh: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum foreground weight for feature filtering before FLANN query.",
    )
    sv_on: bool = Field(default=True, description="Enable spatial verification")
    sv_n_name_shortlist: int = Field(default=40, ge=1)
    sv_n_annot_per_name: int = Field(
        default=999,
        ge=1,
        description="Max annotations per name in SV shortlist. WBIA's literal "
        "default is 3 (Config.py:288), but 999 (verify-all) is used here because "
        "cross-process FLANN noise makes prescore-based shortlist *selection* "
        "diverge between hotspotter and a WBIA oracle (SV pruning agreement drops "
        "to ~0.62 with 3 vs ~1.0 with 999). With verify-all the SV inlier test "
        "alone decides survival, and shortlisted-out annots fail SV regardless, so "
        "the surviving sets still agree. Set to 3 for bit-faithful WBIA behaviour "
        "in a single-process setting.",
    )
    sv_xy_thresh: float | None = Field(default=0.01, gt=0.0)
    sv_scale_thresh: float | None = Field(default=2.0, gt=0.0)
    sv_ori_thresh: float | None = Field(
        default=1.5707963267948966,
        gt=0.0,
        description="Max orientation delta in radians (WBIA default: TAU/4).",
    )
    sv_use_chip_extent: bool = Field(default=True)
    sv_weight_inliers: bool = Field(
        default=True,
        description="Bias RANSAC sampling toward high-FG features (WBIA weight_inliers). "
        "Does NOT multiply scores.",
    )
    sv_use_kp_affine_inliers: bool = Field(
        default=True,
        description="DEPRECATED — no longer used. Survival is gated by sver None return "
        "(affine ≥ 7); scoring always uses homography-refined inliers matching WBIA.",
    )
    sv_sver_output_weighting: bool = Field(
        default=False,
        description="Append per-inlier homography-error weight as a new fsv column "
        "and re-score (WBIA sver_output_weighting, default False).",
    )
    num_return: int = Field(default=10, ge=1)
    ratio_thresh: Optional[float] = Field(default=None, gt=0.0)
    lnbnn_ratio: float = Field(default=1.0, gt=0.0)
    fg_on: bool = Field(default=True)

    bar_l2_on: bool = Field(default=False)
    const_on: bool = Field(default=False)

    flann_algorithm: str = Field(default="kdtree")
    flann_trees: int = Field(default=8)
    flann_random_seed: int = Field(default=42)
    flann_checks: int = Field(default=800)
    flann_cores: int = Field(default=0)


class MiewIdConfig(BaseModel):
    enabled: bool = Field(default=False)


class IdentificationConfig(BaseModel):
    pipeline: Literal["HotSpotter", "MiewId", "CurvRank", "Deepsqueak"] = Field(
        default="HotSpotter"
    )
    hotspotter: HotSpotterConfig = Field(default_factory=HotSpotterConfig)
    miewid: MiewIdConfig = Field(default_factory=MiewIdConfig)
