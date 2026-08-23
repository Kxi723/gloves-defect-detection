
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .segmentation import SegmentationResult

BBox = Tuple[int, int, int, int]


# detector result
@dataclass
class DefectResult:

    defect_found: bool
    defect_type: str
    locations: List[BBox] = field(default_factory=list)
    score: float = 0.0
    details: str = ""


    mask: Optional[np.ndarray] = None
    methods: List[str] = field(default_factory=list)
    metrics: Dict[str, object] = field(default_factory=dict)
    parameters: Dict[str, object] = field(default_factory=dict)


# glove size
def palm_center_and_radius(mask: np.ndarray) -> Tuple[Tuple[int, int], float]:
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    return (int(max_loc[0]), int(max_loc[1])), float(max_val)


def glove_interior(seg: SegmentationResult, margin_ratio: float) -> np.ndarray:
    _, palm_radius = palm_center_and_radius(seg.mask)
    margin = max(3, int(margin_ratio * palm_radius))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)
    )
    return cv2.erode(seg.mask, kernel)


def bbox_around(point: Tuple[int, int], half: int,
                shape: Tuple[int, ...]) -> BBox:
    height, width = shape[:2]
    x = max(0, point[0] - half)
    y = max(0, point[1] - half)
    return (x, y, min(2 * half, width - x), min(2 * half, height - y))


# stable median and spread
def robust_stats(values: np.ndarray) -> Tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, max(1.4826 * mad, 1e-6)


def local_texture_energy(image: np.ndarray, window: int) -> np.ndarray:
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    mean = cv2.blur(lightness, (window, window))
    mean_of_squares = cv2.blur(lightness * lightness, (window, window))
    return np.sqrt(np.maximum(mean_of_squares - mean * mean, 0.0))


def lab_chroma(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[:, :, 1:] - 128.0


def median_chroma(chroma: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    values = chroma[mask]
    return float(np.median(values[:, 0])), float(np.median(values[:, 1]))


def off_hue_distance(region: Tuple[float, float],
                     reference: Tuple[float, float]) -> float:
    ref_length = math.hypot(*reference)
    if ref_length < 1e-6:
        return math.hypot(*region)
    unit_a, unit_b = reference[0] / ref_length, reference[1] / ref_length
    projection = region[0] * unit_a + region[1] * unit_b
    if projection <= 0.0:
        return math.hypot(*region)
    return abs(region[0] * unit_b - region[1] * unit_a)


def convexity_defect_list(
    contour: np.ndarray,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], float]]:
    if contour is None or len(contour) < 5:
        return []
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return []

    hull_idx = np.sort(hull_idx.flatten()).reshape(-1, 1)
    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except cv2.error:
        return []
    if defects is None:
        return []

    result = []
    for start_i, end_i, far_i, depth_fixed in defects.reshape(-1, 4):
        start = tuple(int(v) for v in contour[start_i][0])
        end = tuple(int(v) for v in contour[end_i][0])
        far = tuple(int(v) for v in contour[far_i][0])
        result.append((start, end, far, depth_fixed / 256.0))
    return result


def angle_at(far: Tuple[int, int], a: Tuple[int, int],
             b: Tuple[int, int]) -> float:
    v1 = np.array(a, dtype=np.float64) - np.array(far, dtype=np.float64)
    v2 = np.array(b, dtype=np.float64) - np.array(far, dtype=np.float64)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denominator < 1e-9:
        return 180.0
    cosine = float(np.clip(np.dot(v1, v2) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def hole_shape_metrics(contour: np.ndarray) -> Tuple[float, float]:
    area = cv2.contourArea(contour)
    x, y, width, height = cv2.boundingRect(contour)
    extent = area / max(float(width * height), 1.0)

    if len(contour) >= 5:
        (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        major, minor = max(axis_a, axis_b), max(min(axis_a, axis_b), 1e-6)
        elongation = major / minor
    else:
        elongation = max(width, height) / max(min(width, height), 1)
    return float(elongation), float(extent)


# find inner holes
def find_holes(seg: SegmentationResult, min_area: float, max_area: float,
               max_elongation: float = 4.5, min_extent: float = 0.35
               ) -> List[Tuple[np.ndarray, BBox, float]]:
    contours, _ = cv2.findContours(seg.holes_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue
        elongation, extent = hole_shape_metrics(contour)
        if elongation > max_elongation or extent < min_extent:
            continue
        holes.append((contour, cv2.boundingRect(contour), float(area)))
    return holes


def cut_by_frame(mask: np.ndarray, point: Tuple[int, int],
                 palm_radius: float, reach_ratio: float) -> bool:
    if reach_ratio <= 0:
        return False
    height, width = mask.shape[:2]
    reach = max(1, int(reach_ratio * palm_radius))
    x0, x1 = max(0, point[0] - reach), min(width, point[0] + reach + 1)
    y0, y1 = max(0, point[1] - reach), min(height, point[1] + reach + 1)

    edges = []
    if y0 == 0:
        edges.append(mask[0, x0:x1])
    if y1 == height:
        edges.append(mask[height - 1, x0:x1])
    if x0 == 0:
        edges.append(mask[y0:y1, 0])
    if x1 == width:
        edges.append(mask[y0:y1, width - 1])
    return any(bool(np.any(edge)) for edge in edges)


# find finger tips
def locate_fingertips(seg: SegmentationResult, min_tip_distance_ratio: float,
                      merge_separation_ratio: float,
                      max_tips: int = 5,
                      frame_cut_reach_ratio: float = 0.5
                      ) -> List[Tuple[int, int]]:
    center, palm_radius = palm_center_and_radius(seg.mask)
    if palm_radius <= 1:
        return []

    hull = cv2.convexHull(seg.contour)
    candidates: List[Tuple[float, Tuple[int, int]]] = []
    for point in hull[:, 0]:
        candidate = (int(point[0]), int(point[1]))
        distance = math.dist(candidate, center)
        if distance < min_tip_distance_ratio * palm_radius:
            continue
        if cut_by_frame(seg.mask, candidate, palm_radius, frame_cut_reach_ratio):
            continue
        candidates.append((distance, candidate))
    candidates.sort(reverse=True)

    merged: List[Tuple[int, int]] = []
    min_separation = merge_separation_ratio * palm_radius
    for _, candidate in candidates:
        if all(math.dist(candidate, kept) >= min_separation for kept in merged):
            merged.append(candidate)
    return merged[:max_tips]


def fingertip_cross_section(mask: np.ndarray, tip: Tuple[int, int],
                            center: Tuple[int, int], palm_radius: float,
                            backoff_ratio: float = 0.30) -> float:
    vector_x, vector_y = center[0] - tip[0], center[1] - tip[1]
    norm = math.hypot(vector_x, vector_y) or 1.0
    height, width = mask.shape[:2]
    sample_x = int(round(tip[0] + vector_x / norm * backoff_ratio * palm_radius))
    sample_y = int(round(tip[1] + vector_y / norm * backoff_ratio * palm_radius))
    sample_x = max(0, min(width - 1, sample_x))
    sample_y = max(0, min(height - 1, sample_y))

    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    half_width = float(distance[sample_y, sample_x])
    return math.pi * half_width * half_width


def is_finger_valley(start: Tuple[int, int], end: Tuple[int, int],
                     tips: List[Tuple[int, int]],
                     max_tip_distance: float) -> bool:
    def nearest_tip(point: Tuple[int, int]) -> int:
        best_index, best_distance = -1, float("inf")
        for index, tip in enumerate(tips):
            distance = math.dist(point, tip)
            if distance < best_distance:
                best_index, best_distance = index, distance
        return best_index if best_distance <= max_tip_distance else -1

    tip_a, tip_b = nearest_tip(start), nearest_tip(end)
    return tip_a >= 0 and tip_b >= 0 and tip_a != tip_b


# connected regions
def components_as_boxes(mask: np.ndarray, min_area: float,
                        min_extent: float = 0.0
                        ) -> List[Tuple[BBox, float, np.ndarray]]:
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
