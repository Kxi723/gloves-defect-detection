"""
Tearing (fingertip) — a tear located at the end of a finger.

The problem
-----------
Three things make a tear hard to find by shape alone.

First, a natural finger valley is deep AND narrow — geometrically the same
as a tear notch — so depth and opening angle cannot separate them. They
are told apart by a different property: a valley's convex-hull chord spans
two *different* fingertips, whereas a tear notch begins and ends on the
same stretch of boundary. See :func:`gdd.features.is_finger_valley`.

Second, a crease that survives thresholding leaves a long thin sliver in
the mask that looks like a hole. Real tears are compact, so holes are
gated on elongation and extent.

Third, and this is what the shape channels cannot reach at all, a worn
glove has a hand inside it. The hand fills the opening, so the silhouette
barely moves and no hole forms in the mask. On the four worn nitrile
photos the hole and notch channels together produced zero candidates.

The method
----------
1. Gather tear evidence anywhere on the glove, from three channels that
   all express the same idea — part of the glove outline is not glove:
   * interior holes in the segmentation mask, where what is behind the
     glove is the background, shape-gated as above;
   * show-through patches, where what is behind the glove is the hand,
     found by chroma rather than by shape;
   * deep, narrow convexity defects on the boundary, after finger valleys
     have been excluded.
2. Locate the fingertips from convex-hull extremes measured against the
   distance-transform palm centre, so no assumption is made about which
   way the glove points.
3. Keep only evidence lying within ``fingertip_radius_ratio`` palm radii
   of a fingertip.

Fingertips are worth their own defect class because they are where a glove
wears through first.

Step 3 does more work than it looks. A worn glove shows a bare forearm at
the cuff, which is the same skin a tear reveals and is a far larger patch
of it, so the show-through channel finds it every time. It is rejected
because it is nowhere near a fingertip — which only holds now that the
forearm stump has stopped being reported AS a fingertip; see
:func:`gdd.features.locate_fingertips`.

Measured performance
--------------------
On the 27-photo set, 4 of 4 torn photos detected and two false positives,
so recall 100% and precision 67%. Every box drawn on a torn photo is on
the tear itself.

The show-through channel contributes no false positive. Both belong to
the HOLE channel (good_cotton_5 and dirty_latex_4) and both need two
independent faults to line up.

*The holes.* On a two-tone knit the segmentation threshold slices between
the two yarn colours, so patches of the darker yarn drop out of the mask
and read as interior holes — worst where the light falls off, which is
why they cluster on one side of good_cotton_5. They are demonstrably not
the backdrop showing through: measured in LAB, they sit 87 to 109 units
from the backdrop colour on cotton and 51 to 75 on latex, while the glove
itself is only 115 and 88 from it. Single-colour materials leak nothing
(0.0% of glove area on latex and nitrile against 4.5% on good_cotton_5).

*The fingertips.* Those holes only matter because a cuff corner is being
reported as a fingertip. On good_cotton_5 the three farthest hull points
are all on the cuff (3.14, 3.05, 2.96 palm radii) while the real fingers
sit at 2.20 and 1.84, because the glove is long-cuffed and lies
diagonally. Local WIDTH does not separate them — a cuff corner is a
corner, so the distance transform reads it as narrow, the same reason the
width test failed for the forearm stump.

The fix for the first fault is the symmetric counterpart of the backdrop
gates already in the show-through channel: a hole is only a through-tear
if it looks LIKE the backdrop, since that is what is behind the glove.
The two channels then partition cleanly — the hole channel takes patches
that match the backdrop, the show-through channel takes patches that do
not. It is not implemented yet because the photo set contains no tear
photographed flat on a backdrop, so there is no positive to calibrate the
threshold against, only negatives.

An earlier version of tear_nitrile_1 was unusable — a band of glare on
the wall crossed the fingertips, segmentation welded it onto the glove,
and the exposed skin matched the wall to 2.3 LAB units because the same
glare lit both. It was re-shot with the lamp moved and now detects on the
tear itself. Worth remembering as the failure mode that tuning cannot
reach: when the thing behind the tear and the backdrop are lit into the
same colour, no colour test can separate them, in principle.

Nothing here is a held-out measurement — all four torn photos were used
to place the thresholds, and they are all nitrile, all worn.

Owner: Jason. Tunables live in ``PipelineConfig.tearing`` and
``PipelineConfig.fingertip``.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np

from gdd.config import PipelineConfig
from gdd.features import (
    BBox, DefectResult, angle_at, bbox_around, components_as_boxes,
    convexity_defect_list, find_holes, fingertip_cross_section,
    is_finger_valley, locate_fingertips, palm_center_and_radius, robust_stats,
)
from gdd.segmentation import SegmentationResult, estimate_background_lab


def find_showthrough_patches(image: np.ndarray,
                             segmentation: SegmentationResult,
                             config: PipelineConfig,
                             tips: List[Tuple[int, int]]
                             ) -> List[Tuple[BBox, float, str]]:
    """Patches inside the glove outline that are not glove material.

    The hole channel only sees a tear when the backdrop shows through it.
    Put a hand in the glove and the opening is filled by the hand, so the
    mask stays solid and nothing is found — yet the tear is perfectly
    visible, because the hand is not the colour of the glove.

    Measured in CHROMA, deliberately. Highlights, shadows and creases all
    move a patch of glove along its own lightness axis without touching
    its hue, and those are the things a lightness test cannot help firing
    on. Whatever lies behind the glove has no reason to share the glove's
    hue. The reference is the glove's own robust median a/b, so nothing
    here assumes a glove colour or a skin colour.

    A patch that passes still has to clear two more things. It must be
    BEHIND the glove rather than in front of it, because segmentation can
    weld a piece of backdrop onto the silhouette; and it must be small
    enough to be an opening in a finger rather than a differently
    coloured piece of the glove. See ``TearingConfig``.
    """
    cfg = config.tearing
    center, palm_radius = palm_center_and_radius(segmentation.mask)

    margin = max(3, int(cfg.showthrough_margin_ratio * palm_radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * margin + 1, 2 * margin + 1))
    interior = cv2.erode(segmentation.mask, kernel) > 0
    if np.count_nonzero(interior) < 100:
        return []

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness, a_channel, b_channel = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    a_median, a_spread = robust_stats(a_channel[interior])
    b_median, b_spread = robust_stats(b_channel[interior])
    a_spread = max(a_spread, cfg.showthrough_mad_floor)
    b_spread = max(b_spread, cfg.showthrough_mad_floor)

    backdrop = estimate_background_lab(
        cv2.cvtColor(image, cv2.COLOR_BGR2LAB),
        config.segmentation.border_fraction).astype(np.float32)

    deviation = np.sqrt(((a_channel - a_median) / a_spread) ** 2
                        + ((b_channel - b_median) / b_spread) ** 2)
    candidate = ((deviation > cfg.showthrough_z_threshold)
                 & interior).astype(np.uint8) * 255
    # Opening first, so that speckle from a woven or grainy material does
    # not reach the connected-component stage at all.
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    max_area = cfg.max_showthrough_area_fraction * segmentation.area
    patches: List[Tuple[BBox, float, str]] = []
    for (x, y, w, h), area, member in components_as_boxes(
            candidate,
            min_area=cfg.min_showthrough_area_fraction * segmentation.area,
            min_extent=cfg.min_showthrough_extent):
        if area > max_area:
            continue  # segmentation failure, not a tear
        elongation = max(w, h) / max(min(w, h), 1)
        if elongation > cfg.max_showthrough_elongation:
            continue  # a sliver along a crease or the rim

        patch = np.array([np.median(lightness[member]),
                          np.median(a_channel[member]),
                          np.median(b_channel[member])], dtype=np.float32)
        if float(np.linalg.norm(patch - backdrop)) \
                < cfg.showthrough_min_backdrop_distance:
            continue  # backdrop, not something seen through the glove

        # An opening in a finger cannot be bigger than the finger. Only
        # applied when a fingertip is close enough for "the finger it is
        # in" to be defined; otherwise the area window above stands alone,
        # which is what the plain `tearing` detector wants anyway.
        nearest = min(tips, key=lambda p: math.dist((x + w // 2, y + h // 2), p),
                      default=None)
        if nearest is not None and math.dist((x + w // 2, y + h // 2), nearest) \
                <= cfg.fingertip_radius_ratio * palm_radius:
            cross_section = fingertip_cross_section(
                segmentation.mask, nearest, center, palm_radius)
            if cross_section > 1.0 and area / cross_section \
                    > cfg.max_showthrough_fingertip_fraction:
                continue  # a differently coloured fingertip, not a tear

        fraction = area / segmentation.area
        patches.append(((x, y, w, h),
                        min(1.0, 0.5 + 5.0 * fraction),
                        f"show-through {fraction:.2%}"))
    return patches


def find_tear_evidence(image: np.ndarray, segmentation: SegmentationResult,
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

    tips = locate_fingertips(segmentation, finger_cfg.min_tip_distance_ratio,
                             finger_cfg.tip_merge_separation_ratio,
                             finger_cfg.expected_fingers,
                             finger_cfg.tip_frame_cut_reach_ratio)

    # --- what is behind the glove, when the glove is worn ---------------- #
    # Needs the tips: a patch at a fingertip is sized against that
    # fingertip, not against the whole glove.
    evidence.extend(find_showthrough_patches(image, segmentation, config, tips))

    # --- deep narrow notches on the boundary ----------------------------- #
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
                             finger_cfg.expected_fingers,
                             finger_cfg.tip_frame_cut_reach_ratio)
    if not tips or palm_radius <= 1:
        return DefectResult(False, "tearing_at_finger",
                            details="fingertips could not be localised")

    evidence = find_tear_evidence(image, segmentation, config)
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
