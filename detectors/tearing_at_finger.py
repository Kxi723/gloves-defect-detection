"""
Tearing (fingertip) — a tear located at the end of a finger.

The problem
-----------
Two things make a tear hard to find by shape alone.

First, a natural finger valley is deep AND narrow — geometrically the same
as a tear notch — so depth and opening angle cannot separate them. They
are told apart by a different property: a valley's convex-hull chord spans
two *different* fingertips, whereas a tear notch begins and ends on the
same stretch of boundary. See :func:`gdd.features.is_finger_valley`.

Second, a crease that survives thresholding leaves a long thin sliver in
the mask that looks like a hole. Real tears are compact, so holes are
gated on elongation and extent.

The method
----------
1. Gather tear evidence anywhere on the glove:
   * interior holes in the segmentation mask (a through-tear exposes what
     is behind the glove), shape-gated as above;
   * deep, narrow convexity defects on the boundary, after finger valleys
     have been excluded.
2. Locate the fingertips from convex-hull extremes measured against the
   distance-transform palm centre, so no assumption is made about which
   way the glove points.
3. Keep only evidence lying within ``fingertip_radius_ratio`` palm radii
   of a fingertip.

Fingertips are worth their own defect class because they are where a glove
wears through first.

Measured on the 15-photo set: precision 100%, recall 20%. The recall is
the weak point — four of the five torn nitrile photos were taken with the
glove worn, where a fingertip tear barely changes the silhouette at all.
Photographing the glove flat and empty is what fixes this, not tuning.

Owner: Jason. Tunables live in ``PipelineConfig.tearing`` and
``PipelineConfig.fingertip``.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from gdd.config import PipelineConfig
from gdd.features import (
    BBox, DefectResult, angle_at, bbox_around, convexity_defect_list,
    find_holes, is_finger_valley, locate_fingertips, palm_center_and_radius,
)
from gdd.segmentation import SegmentationResult


def find_tear_evidence(segmentation: SegmentationResult,
                       config: PipelineConfig) -> List[Tuple[BBox, float, str]]:
    """All tear evidence on the glove as (bbox, confidence, note) triples.

    Shared with the plain ``tearing`` detector, which uses the same
    evidence without the fingertip filter.
    """
    cfg = config.tearing
    finger_cfg = config.fingertip
    _, palm_radius = palm_center_and_radius(segmentation.mask)
    evidence: List[Tuple[BBox, float, str]] = []

    # --- interior holes -------------------------------------------------- #
    for _, bbox, area in find_holes(
            segmentation,
            min_area=cfg.min_hole_area_fraction * segmentation.area,
            max_area=cfg.max_hole_area_fraction * segmentation.area,
            max_elongation=cfg.max_hole_elongation,
            min_extent=cfg.min_hole_extent):
        # Bigger hole relative to the glove means higher confidence.
        confidence = min(1.0, 0.5 + 5.0 * area / segmentation.area)
        evidence.append((bbox, confidence, f"hole {area / segmentation.area:.2%}"))

    if palm_radius <= 1:
        return evidence

    # --- deep narrow notches on the boundary ----------------------------- #
    tips = locate_fingertips(segmentation, finger_cfg.min_tip_distance_ratio,
                             finger_cfg.tip_merge_separation_ratio,
                             finger_cfg.expected_fingers)
    valley_tip_distance = cfg.valley_endpoint_tip_ratio * palm_radius
    for start, end, far, depth in convexity_defect_list(segmentation.contour):
        if is_finger_valley(start, end, tips, valley_tip_distance):
            continue  # anatomy, not damage
        depth_ratio = depth / palm_radius
        notch_angle = angle_at(far, start, end)
        if (depth_ratio >= cfg.min_defect_depth_ratio
                and notch_angle <= cfg.max_defect_angle_deg):
            half = max(8, int(0.15 * palm_radius))
            evidence.append((
                bbox_around(far, half, segmentation.mask.shape),
                min(1.0, 0.4 + depth_ratio / 2.0),
                f"notch depth={depth_ratio:.2f}R angle={notch_angle:.0f}deg",
            ))
    return evidence


def detect(image: np.ndarray, segmentation: SegmentationResult,
           config: PipelineConfig) -> DefectResult:
    """Detect tears located at a fingertip. See the module docstring."""
    cfg = config.tearing
    finger_cfg = config.fingertip
    _, palm_radius = palm_center_and_radius(segmentation.mask)
    tips = locate_fingertips(segmentation, finger_cfg.min_tip_distance_ratio,
                             finger_cfg.tip_merge_separation_ratio,
                             finger_cfg.expected_fingers)
    if not tips or palm_radius <= 1:
        return DefectResult(False, "tearing_at_finger",
                            details="fingertips could not be localised")

    evidence = find_tear_evidence(segmentation, config)
    max_tip_distance = cfg.fingertip_radius_ratio * palm_radius

    locations: List[BBox] = []
    notes: List[str] = []
    score = 0.0
    for (x, y, w, h), confidence, note in evidence:
        center = (x + w // 2, y + h // 2)
        if any(math.dist(center, tip) <= max_tip_distance for tip in tips):
            locations.append((x, y, w, h))
            notes.append(note)
            score = max(score, confidence)

    return DefectResult(
        defect_found=bool(locations),
        defect_type="tearing_at_finger",
        locations=locations,
        score=score,
        details=(
            f"{len(locations)} of {len(evidence)} tear finding(s) at "
            f"{len(tips)} localised fingertip(s): {'; '.join(notes)}"
            if locations else
            f"0 of {len(evidence)} tear finding(s) near "
            f"{len(tips)} localised fingertip(s)"
        ),
    )
