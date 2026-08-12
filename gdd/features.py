"""
Shared building blocks for every detector in ``detectors/``.

Twelve defect modules, written by four people, all need the same handful of
measurements: where is the palm, how big is it, where are the fingertips,
how textured is this patch, is this colour an outlier. Those live here so
that each detector file contains only its own *decision* logic and none of
the plumbing.

Nothing in this module decides whether a glove is defective. It only
measures. Detectors import what they need and apply their own thresholds
from :mod:`gdd.config`.

Two conventions matter and are worth copying in any new detector:

* **Scale invariance.** Sizes are expressed as fractions of the palm radius
  (from :func:`palm_center_and_radius`) or of the glove area, never in raw
  pixels, so a threshold tuned on one photo holds at any resolution or
  camera distance.
* **Robust statistics.** Reference values come from
  :func:`robust_stats` (median and MAD) rather than mean and standard
  deviation, because the defect is part of the sample being measured; a
  few percent of stain pixels shifts a mean but leaves a median alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from gdd.segmentation import SegmentationResult

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class DefectResult:
    """Uniform output of every detector — the plugin contract.

    Every module in ``detectors/`` returns one of these from its
    ``detect(image, segmentation, config)`` function, which is what lets
    the GUI, the batch runner and the evaluator treat all twelve defects
    identically.
    """

    defect_found: bool
    defect_type: str
    locations: List[BBox] = field(default_factory=list)
    score: float = 0.0          # 0..1 confidence that the defect is real
    details: str = ""           # human-readable evidence, shown in reports


# --------------------------------------------------------------------------- #
# Geometry: where the glove is and how big it is
# --------------------------------------------------------------------------- #

def palm_center_and_radius(mask: np.ndarray) -> Tuple[Tuple[int, int], float]:
    """Locate the palm as the deepest point of the distance transform.

    The palm is the widest part of a glove, so the pixel farthest from any
    boundary (the distance-transform maximum) sits inside it, and that
    maximum distance is the palm's inscribed-circle radius. This is the
    scale reference every other threshold is expressed against.
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    return (int(max_loc[0]), int(max_loc[1])), float(max_val)


def glove_interior(seg: SegmentationResult, margin_ratio: float) -> np.ndarray:
    """The glove mask shrunk inwards by ``margin_ratio`` palm radii.

    Anything appearance based must ignore the rim: the segmentation
    boundary is never pixel exact, and the curved edge of a glove is always
    darker than its face, so an un-eroded mask reports a ring-shaped
    "defect" on every single photo.
    """
    _, palm_radius = palm_center_and_radius(seg.mask)
    margin = max(3, int(margin_ratio * palm_radius))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)
    )
    return cv2.erode(seg.mask, kernel)


def bbox_around(point: Tuple[int, int], half: int,
                shape: Tuple[int, ...]) -> BBox:
    """Small bounding box centred on a point, clamped to the image."""
    height, width = shape[:2]
    x = max(0, point[0] - half)
    y = max(0, point[1] - half)
    return (x, y, min(2 * half, width - x), min(2 * half, height - y))


# --------------------------------------------------------------------------- #
# Statistics and texture
# --------------------------------------------------------------------------- #

def robust_stats(values: np.ndarray) -> Tuple[float, float]:
    """Median and a robust standard deviation estimate (from the MAD).

    The median absolute deviation is scaled by 1.4826 so that, for normally
    distributed data, the result matches the ordinary standard deviation.
    """
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, max(1.4826 * mad, 1e-6)


def local_texture_energy(image: np.ndarray, window: int) -> np.ndarray:
    """Local standard deviation of lightness — a cheap texture measure.

    High on woven or crinkled material, near zero on a smooth surface.
    Var(X) = E[X^2] - E[X]^2, clamped because rounding can push it below
    zero on flat regions.
    """
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    mean = cv2.blur(lightness, (window, window))
    mean_of_squares = cv2.blur(lightness * lightness, (window, window))
    return np.sqrt(np.maximum(mean_of_squares - mean * mean, 0.0))


# --------------------------------------------------------------------------- #
# Contour analysis: convexity defects, holes, fingertips
# --------------------------------------------------------------------------- #

def convexity_defect_list(
    contour: np.ndarray,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], float]]:
    """Convexity defects as (start, end, far, depth_px) tuples.

    A convexity defect is a region where the contour caves in from its
    convex hull; ``far`` is the deepest point of the cavity. Finger valleys
    and boundary tears both appear here, and are told apart by the caller.
    """
    if contour is None or len(contour) < 5:
        return []
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return []
    # convexityDefects requires monotonic hull indices.
    hull_idx = np.sort(hull_idx.flatten()).reshape(-1, 1)
    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except cv2.error:
        return []  # self-intersecting contour; no usable defects
    if defects is None:
        return []

    result = []
    for start_i, end_i, far_i, depth_fixed in defects.reshape(-1, 4):
        start = tuple(int(v) for v in contour[start_i][0])
        end = tuple(int(v) for v in contour[end_i][0])
        far = tuple(int(v) for v in contour[far_i][0])
        result.append((start, end, far, depth_fixed / 256.0))  # fixed-point
    return result


def angle_at(far: Tuple[int, int], a: Tuple[int, int],
             b: Tuple[int, int]) -> float:
    """Opening angle in degrees at ``far`` between the rays to ``a`` and ``b``.

    Small angle means a narrow notch (tear-like); larger means a wide valley.
    """
    v1 = np.array(a, dtype=np.float64) - np.array(far, dtype=np.float64)
    v2 = np.array(b, dtype=np.float64) - np.array(far, dtype=np.float64)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denominator < 1e-9:
        return 180.0
    cosine = float(np.clip(np.dot(v1, v2) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def hole_shape_metrics(contour: np.ndarray) -> Tuple[float, float]:
    """(elongation, extent) of a hole contour.

    ``elongation`` is the major/minor axis ratio of the fitted ellipse;
    ``extent`` is contour area over bounding-box area. Both dimensionless,
    so thresholds hold at any resolution.
    """
    area = cv2.contourArea(contour)
    x, y, width, height = cv2.boundingRect(contour)
    extent = area / max(float(width * height), 1.0)

    if len(contour) >= 5:
        (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        major, minor = max(axis_a, axis_b), max(min(axis_a, axis_b), 1e-6)
        elongation = major / minor
    else:  # too few points to fit an ellipse; fall back to the bbox
        elongation = max(width, height) / max(min(width, height), 1)
    return float(elongation), float(extent)


def find_holes(seg: SegmentationResult, min_area: float, max_area: float,
               max_elongation: float = 4.5, min_extent: float = 0.35
               ) -> List[Tuple[np.ndarray, BBox, float]]:
    """Interior holes of the glove as (contour, bbox, area) tuples.

    A through-tear exposes what is behind the glove, so it survives
    segmentation as a hole in ``mask_raw``. Area limits reject residual
    noise and wholesale segmentation failures; the shape gate rejects
    crease artefacts (see :func:`hole_shape_metrics`).
    """
    contours, _ = cv2.findContours(seg.holes_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue
        elongation, extent = hole_shape_metrics(contour)
        if elongation > max_elongation or extent < min_extent:
            continue  # crease sliver, not a tear
        holes.append((contour, cv2.boundingRect(contour), float(area)))
    return holes


def locate_fingertips(seg: SegmentationResult, min_tip_distance_ratio: float,
                      merge_separation_ratio: float,
                      max_tips: int = 5) -> List[Tuple[int, int]]:
    """Fingertip points from convex-hull extremes. No orientation assumed.

    1. Palm centre and radius from the distance transform.
    2. Hull points further than ``min_tip_distance_ratio`` palm radii from
       the centre are fingertip candidates, since fingertips are the
       extremities of the silhouette.
    3. Hull vertices cluster densely around a curved tip, so candidates
       closer together than the merge separation collapse into one, the
       farthest winning.

    Because everything is measured from the palm centre outwards, this
    works whether the glove points up, sideways or diagonally — which
    matters for hand-held phone photos.
    """
    center, palm_radius = palm_center_and_radius(seg.mask)
    if palm_radius <= 1:
        return []

    hull = cv2.convexHull(seg.contour)
    candidates: List[Tuple[float, Tuple[int, int]]] = []
    for point in hull[:, 0]:
        candidate = (int(point[0]), int(point[1]))
        distance = math.dist(candidate, center)
        if distance >= min_tip_distance_ratio * palm_radius:
            candidates.append((distance, candidate))
    candidates.sort(reverse=True)  # farthest first, so real tips win merges

    merged: List[Tuple[int, int]] = []
    min_separation = merge_separation_ratio * palm_radius
    for _, candidate in candidates:
        if all(math.dist(candidate, kept) >= min_separation for kept in merged):
            merged.append(candidate)
    return merged[:max_tips]


def is_finger_valley(start: Tuple[int, int], end: Tuple[int, int],
                     tips: List[Tuple[int, int]],
                     max_tip_distance: float) -> bool:
    """Is this convexity defect the natural valley between two fingers?

    A valley between adjacent fingers is deep AND narrow — exactly like a
    tear — so depth and angle alone cannot separate them. The reliable
    difference is that a valley's hull chord spans two *different*
    fingertips, while a tear notch starts and ends on the same stretch of
    boundary.
    """
    def nearest_tip(point: Tuple[int, int]) -> int:
        best_index, best_distance = -1, float("inf")
        for index, tip in enumerate(tips):
            distance = math.dist(point, tip)
            if distance < best_distance:
                best_index, best_distance = index, distance
        return best_index if best_distance <= max_tip_distance else -1

    tip_a, tip_b = nearest_tip(start), nearest_tip(end)
    return tip_a >= 0 and tip_b >= 0 and tip_a != tip_b


def components_as_boxes(mask: np.ndarray, min_area: float,
                        min_extent: float = 0.0
                        ) -> List[Tuple[BBox, float, np.ndarray]]:
    """Connected components as (bbox, area, member-pixel mask) triples.

    The third element lets a caller measure something else inside the
    component — texture, colour, whatever its defect needs.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    results: List[Tuple[BBox, float, np.ndarray]] = []
    for i in range(1, count):
        x, y, width, height, area = (
            int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]),
            float(stats[i, cv2.CC_STAT_AREA]),
        )
        if area < min_area:
            continue
        if min_extent > 0.0 and area / max(float(width * height), 1.0) < min_extent:
            continue
        results.append(((x, y, width, height), area, labels == i))
    return results
