from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np

@dataclass
class PreprocessConfig:
    max_dimension: int = 1024
    bilateral_diameter: int = 7
    bilateral_sigma_color: float = 50
    bilateral_sigma_space: float = 50

@dataclass
class SegmentationConfig:
    border_fraction: float = 0.04
    min_area_fraction: float = 0.12
    max_area_fraction: float = 0.90
    open_kernel: int = 5
    close_kernel: int = 9
    min_hole_area_fraction: float = 0.0004
    texture_window: int = 9
    illumination_border_fraction: float = 0.06
    background_mad_floor: float = 4.0
    background_z_clip: float = 8.0

@dataclass
class TearingConfig:
    min_hole_area_fraction: float = 0.0008
    max_hole_area_fraction: float = 0.25
    max_hole_elongation: float = 4.5
    min_hole_extent: float = 0.35
    min_defect_depth_ratio: float = 0.35
    max_defect_angle_deg: float = 60.0
    valley_endpoint_tip_ratio: float = 0.7
    fingertip_radius_ratio: float = 0.55
    showthrough_margin_ratio: float = 0.05
    showthrough_z_threshold: float = 3.0
    showthrough_mad_floor: float = 1.5
    min_showthrough_area_fraction: float = 0.0015
    max_showthrough_area_fraction: float = 0.05
    max_showthrough_fingertip_fraction: float = 1.5
    min_showthrough_extent: float = 0.30
    max_showthrough_elongation: float = 4.5
    showthrough_min_backdrop_distance: float = 8.0
    min_score: float = 0.0

@dataclass
class FingertipConfig:
    expected_fingers: int = 5
    min_tip_distance_ratio: float = 1.35
    tip_merge_separation_ratio: float = 0.45
    tip_frame_cut_reach_ratio: float = 0.5

@dataclass
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    tearing: TearingConfig = field(default_factory=TearingConfig)
    fingertip: FingertipConfig = field(default_factory=FingertipConfig)

@dataclass
class SegmentationResult:
    mask: np.ndarray
    mask_raw: np.ndarray
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    area: float
    cue: str = "unknown"

    @property
    def holes_mask(self) -> np.ndarray:
        return cv2.subtract(self.mask, self.mask_raw)

BBox = Tuple[int, int, int, int]

@dataclass
class DefectResult:
    defect_found: bool
    defect_type: str
    locations: List[BBox] = field(default_factory=list)
    score: float = 0.0
    details: str = ""

def resize_to_limit(image: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return image
    scale = max_dimension / longest
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    channel_means = img.reshape(-1, 3).mean(axis=0)
    gray_mean = float(channel_means.mean())
    gains = gray_mean / np.maximum(channel_means, 1e-6)
    balanced = img * gains.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)

def preprocess(image: np.ndarray, config: Optional[PreprocessConfig] = None) -> np.ndarray:
    cfg = config or PreprocessConfig()
    if image is None or image.size == 0:
        raise ValueError("preprocess() received an empty image")
    result = resize_to_limit(image, cfg.max_dimension)
    result = gray_world_white_balance(result)
    return cv2.bilateralFilter(result, d=cfg.bilateral_diameter, sigmaColor=cfg.bilateral_sigma_color, sigmaSpace=cfg.bilateral_sigma_space)

def estimate_background_lab(lab: np.ndarray, border_fraction: float) -> np.ndarray:
    h, w = lab.shape[:2]
    b = max(2, int(round(min(h, w) * border_fraction)))
    strip = np.concatenate([lab[:b, :].reshape(-1, 3), lab[-b:, :].reshape(-1, 3), lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    return np.median(strip, axis=0).astype(np.float32)

def _background_distance_mask(lab: np.ndarray, border_fraction: float) -> np.ndarray:
    background = estimate_background_lab(lab, border_fraction)
    distance = np.linalg.norm(lab.astype(np.float32) - background, axis=2)
    distance_u8 = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def _flatten_illumination(channel: np.ndarray, border_fraction: float) -> np.ndarray:
    h, w = channel.shape[:2]
    b = max(2, int(round(min(h, w) * border_fraction)))
    selected = np.zeros((h, w), dtype=bool)
    selected[:b, :] = True
    selected[-b:, :] = True
    selected[:, :b] = True
    selected[:, -b:] = True
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def design(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], axis=1)
    
    coefficients, *_ = np.linalg.lstsq(design(xx[selected], yy[selected]), channel[selected].astype(np.float32), rcond=None)
    model = (design(xx.ravel(), yy.ravel()) @ coefficients).reshape(h, w)
    model = np.maximum(model, 1.0)
    flattened = channel.astype(np.float32) / model * float(np.median(model))
    return np.clip(flattened, 0, 255).astype(np.uint8)

def _background_model_mask(image: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    border = _flatten_illumination
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = border(lab[:, :, 0], cfg.illumination_border_fraction)
    h, w = image.shape[:2]
    b = max(2, int(round(min(h, w) * cfg.illumination_border_fraction)))
    selected = np.zeros((h, w), dtype=bool)
    selected[:b, :] = True
    selected[-b:, :] = True
    selected[:, :b] = True
    selected[:, -b:] = True
    squared = np.zeros((h, w), dtype=np.float32)
    for channel in range(3):
        values = lab[:, :, channel]
        median = float(np.median(values[selected]))
        spread = float(np.median(np.abs(values[selected] - median))) * 1.4826
        spread = max(spread, cfg.background_mad_floor)
        z = np.minimum(np.abs(values - median) / spread, cfg.background_z_clip)
        squared += z ** 2
    distance = np.sqrt(squared)
    scaled = np.clip(distance / (cfg.background_z_clip * np.sqrt(3.0)) * 255.0, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def _channel_otsu_masks(hsv: np.ndarray) -> List[np.ndarray]:
    masks: List[np.ndarray] = []
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    _, s_mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(s_mask)
    _, v_mask = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(v_mask)
    masks.append(cv2.bitwise_not(v_mask))
    return masks

def _texture_energy_mask(image: np.ndarray, window: int) -> np.ndarray:
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    mean = cv2.blur(lightness, (window, window))
    mean_of_squares = cv2.blur(lightness * lightness, (window, window))
    # Var(X) = E[X^2] - E[X]^2; clamped because rounding can make it < 0.
    std = np.sqrt(np.maximum(mean_of_squares - mean * mean, 0.0))
    std = cv2.GaussianBlur(std, (0, 0), window / 2.0)
    std_u8 = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(std_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def _morphological_cleanup(mask: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_k)
    return cleaned

def _keep_largest_component(mask: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(mask), None
    largest = max(contours, key=cv2.contourArea)
    solid = np.zeros_like(mask)
    cv2.drawContours(solid, [largest], -1, 255, thickness=cv2.FILLED)
    return solid, largest

def _score_candidate(mask: np.ndarray, cfg: SegmentationConfig) -> float:
    image_area = float(mask.shape[0] * mask.shape[1])
    cleaned = _morphological_cleanup(mask, cfg)
    count, labels = cv2.connectedComponents(cleaned)
    if count <= 1:
        return -1.0
    largest_label = 1 + int(np.argmax([np.count_nonzero(labels == i) for i in range(1, count)]))
    component = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    solid, contour = _keep_largest_component(component)
    if contour is None:
        return -1.0
    solid_area = float(np.count_nonzero(solid))
    area_fraction = solid_area / image_area
    if not (cfg.min_area_fraction <= area_fraction <= cfg.max_area_fraction):
        return -1.0
    border = np.concatenate([solid[0, :], solid[-1, :], solid[:, 0], solid[:, -1]])
    border_occupancy = float(np.count_nonzero(border)) / float(border.size)
    shred = np.count_nonzero(cv2.subtract(solid, component)) / max(solid_area, 1.0)
    hull_perimeter = max(cv2.arcLength(cv2.convexHull(contour), True), 1.0)
    raggedness = cv2.arcLength(contour, True) / hull_perimeter
    return ((1.0 - border_occupancy) * 2.0 + (1.0 - abs(area_fraction - 0.30)) - 4.0 * min(shred, 0.5) - 0.5 * max(0.0, raggedness - 1.6))

def _fill_noise_holes(mask_raw: np.ndarray, solid: np.ndarray, cfg: SegmentationConfig, glove_area: float) -> np.ndarray:
    holes = cv2.subtract(solid, mask_raw)
    hole_contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = cfg.min_hole_area_fraction * glove_area
    result = mask_raw.copy()
    for hc in hole_contours:
        if cv2.contourArea(hc) < min_area:
            cv2.drawContours(result, [hc], -1, 255, thickness=cv2.FILLED)
    return result

def segment_glove(image: np.ndarray, config: Optional[SegmentationConfig] = None) -> Optional[SegmentationResult]:
    cfg = config or SegmentationConfig()
    img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    candidates: List[Tuple[str, np.ndarray]] = [("bg_distance", _background_distance_mask(img, 0.04))]
    otsu_masks = _channel_otsu_masks(hsv)
    candidates.append(("saturation", otsu_masks[0]))
    candidates.append(("value", otsu_masks[1]))
    candidates.append(("value_inverted", otsu_masks[2]))
    candidates.append(("texture", _texture_energy_mask(image, cfg.texture_window)))
    primary = _background_model_mask(image, cfg)
    if _score_candidate(primary, cfg) >= 0:
        best_cue, best_mask = "bg_model", primary
    else:
        scored = [(_score_candidate(mask, cfg), name, mask) for name, mask in candidates]
        best_score, best_cue, best_mask = max(scored, key=lambda triple: triple[0])
        if best_score < 0:
            return None
    cleaned = _morphological_cleanup(best_mask, cfg)
    num, labels = cv2.connectedComponents(cleaned)
    if num <= 1:
        return None
    largest_label = 1 + int(np.argmax([np.count_nonzero(labels == i) for i in range(1, num)]))
    component = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    solid, contour = _keep_largest_component(component)
    if contour is None:
        return None
    area = cv2.contourArea(contour)
    if area < cfg.min_area_fraction * image.shape[0] * image.shape[1]:
        return None
    mask_raw = _fill_noise_holes(component, solid, cfg, area)
    return SegmentationResult(mask=solid, mask_raw=mask_raw, contour=contour, bbox=cv2.boundingRect(contour), area=float(area), cue=best_cue,)

def palm_center_and_radius(mask: np.ndarray) -> Tuple[Tuple[int, int], float]:
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    return (int(max_loc[0]), int(max_loc[1])), float(max_val)

def glove_interior(seg: SegmentationResult, margin_ratio: float) -> np.ndarray:
    _, palm_radius = palm_center_and_radius(seg.mask)
    margin = max(3, int(margin_ratio * palm_radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    return cv2.erode(seg.mask, kernel)

def robust_stats(values: np.ndarray) -> Tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, max(1.4826 * mad, 1e-6)

def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))

def _colour_margin(z_median: float, z_gate: float) -> float:
    return _clip01(z_median / max(z_gate, 1e-6) - 1.0)

def _backdrop_margin(patch: np.ndarray, backdrop: np.ndarray, glove: np.ndarray) -> float:
    span = float(np.linalg.norm(glove - backdrop))
    if span < 1e-6:
        return 0.0
    return _clip01(1.0 - float(np.linalg.norm(patch - backdrop)) / span)

def _opening_margin(area: float, cross_section: float) -> float:
    if cross_section <= 1.0:
        return 0.0
    return _clip01(area / cross_section)

def _position_margin(center: Tuple[int, int], tips: List[Tuple[int, int]], max_tip_distance: float) -> float:
    if not tips or max_tip_distance <= 0:
        return 0.0
    return _clip01(1.0 - min(math.dist(center, tip) for tip in tips) / max_tip_distance)

def _combine_margins(*margins: float) -> float:
    values = [max(float(m), 1e-3) for m in margins]
    return float(np.exp(sum(math.log(v) for v in values) / len(values)))

def _mask_elongation(member: np.ndarray) -> float:
    moments = cv2.moments(member.astype(np.uint8), binaryImage=True)
    if moments["m00"] <= 0:
        return 1.0
    mu20 = moments["mu20"] / moments["m00"]
    mu02 = moments["mu02"] / moments["m00"]
    mu11 = moments["mu11"] / moments["m00"]
    common = math.sqrt(max((mu20 - mu02) ** 2 + 4.0 * mu11 * mu11, 0.0))
    major = math.sqrt(max(2.0 * (mu20 + mu02 + common), 0.0))
    minor = math.sqrt(max(2.0 * (mu20 + mu02 - common), 0.0))
    return float(major / max(minor, 1e-6))

def bbox_around(point: Tuple[int, int], half: int, shape: Tuple[int, ...]) -> BBox:
    height, width = shape[:2]
    x = max(0, point[0] - half)
    y = max(0, point[1] - half)
    return (x, y, min(2 * half, width - x), min(2 * half, height - y))

def convexity_defect_list(contour: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], float]]:
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

def angle_at(far: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int]) -> float:
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


def find_holes(seg: SegmentationResult, min_area: float, max_area: float, max_elongation: float = 4.5, min_extent: float = 0.35) -> List[Tuple[np.ndarray, BBox, float]]:
    contours, _ = cv2.findContours(seg.holes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

def cut_by_frame(mask: np.ndarray, point: Tuple[int, int], palm_radius: float, reach_ratio: float) -> bool:
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

def locate_fingertips(seg: SegmentationResult, min_tip_distance_ratio: float, merge_separation_ratio: float, max_tips: int = 5, frame_cut_reach_ratio: float = 0.5) -> List[Tuple[int, int]]:
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


def fingertip_cross_section(mask: np.ndarray, tip: Tuple[int, int], center: Tuple[int, int], palm_radius: float, backoff_ratio: float = 0.30, distance: Optional[np.ndarray] = None) -> float:
    vector_x, vector_y = center[0] - tip[0], center[1] - tip[1]
    norm = math.hypot(vector_x, vector_y) or 1.0
    height, width = mask.shape[:2]
    sample_x = int(round(tip[0] + vector_x / norm * backoff_ratio * palm_radius))
    sample_y = int(round(tip[1] + vector_y / norm * backoff_ratio * palm_radius))
    sample_x = max(0, min(width - 1, sample_x))
    sample_y = max(0, min(height - 1, sample_y))
    if distance is None:
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    half_width = float(distance[sample_y, sample_x])
    return math.pi * half_width * half_width

def is_finger_valley(start: Tuple[int, int], end: Tuple[int, int], tips: List[Tuple[int, int]], max_tip_distance: float) -> bool:
    def nearest_tip(point: Tuple[int, int]) -> int:
        best_index, best_distance = -1, float("inf")
        for index, tip in enumerate(tips):
            distance = math.dist(point, tip)
            if distance < best_distance:
                best_index, best_distance = index, distance
        return best_index if best_distance <= max_tip_distance else -1
    tip_a, tip_b = nearest_tip(start), nearest_tip(end)
    return tip_a >= 0 and tip_b >= 0 and tip_a != tip_b


def components_as_boxes(mask: np.ndarray, min_area: float, min_extent: float = 0.0) -> List[Tuple[BBox, float, np.ndarray]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    results: List[Tuple[BBox, float, np.ndarray]] = []
    for i in range(1, count):
        x, y, width, height, area = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]), int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]), float(stats[i, cv2.CC_STAT_AREA]))
        if area < min_area:
            continue
        if min_extent > 0.0 and area / max(float(width * height), 1.0) < min_extent:
            continue
        results.append(((x, y, width, height), area, labels == i))
    return results

def find_showthrough_patches(image: np.ndarray, segmentation: SegmentationResult, config: Config, tips: List[Tuple[int, int]]) -> List[Tuple[BBox, float, str]]:
    cfg = config.tearing
    center, palm_radius = palm_center_and_radius(segmentation.mask)
    margin = max(3, int(cfg.showthrough_margin_ratio * palm_radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    interior = cv2.erode(segmentation.mask, kernel) > 0
    if np.count_nonzero(interior) < 100:
        return []
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness, a_channel, b_channel = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    a_median, a_spread = robust_stats(a_channel[interior])
    b_median, b_spread = robust_stats(b_channel[interior])
    a_spread = max(a_spread, cfg.showthrough_mad_floor)
    b_spread = max(b_spread, cfg.showthrough_mad_floor)
    backdrop = estimate_background_lab(cv2.cvtColor(image, cv2.COLOR_BGR2LAB), config.segmentation.border_fraction).astype(np.float32)
    deviation = np.sqrt(((a_channel - a_median) / a_spread) ** 2 + ((b_channel - b_median) / b_spread) ** 2)
    candidate = ((deviation > cfg.showthrough_z_threshold) & interior).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    distance = cv2.distanceTransform(segmentation.mask, cv2.DIST_L2, 5)
    max_tip_distance = cfg.fingertip_radius_ratio * palm_radius
    max_area = cfg.max_showthrough_area_fraction * segmentation.area
    patches: List[Tuple[BBox, float, str]] = []
    for (x, y, w, h), area, member in components_as_boxes(candidate, min_area=cfg.min_showthrough_area_fraction * segmentation.area, min_extent=cfg.min_showthrough_extent):
        if area > max_area:
            continue
        if _mask_elongation(member) > cfg.max_showthrough_elongation:
            continue
        patch = np.array([np.median(lightness[member]), np.median(a_channel[member]), np.median(b_channel[member])], dtype=np.float32)
        if float(np.linalg.norm(patch - backdrop)) < cfg.showthrough_min_backdrop_distance:
            continue
        spot = (x + w // 2, y + h // 2)
        nearest = min(tips, key=lambda point: math.dist(spot, point), default=None)
        cross_section = 0.0
        if nearest is not None and math.dist(spot, nearest) <= max_tip_distance:
            cross_section = fingertip_cross_section(segmentation.mask, nearest, center, palm_radius, distance=distance)
            if cross_section > 1.0 and area / cross_section > cfg.max_showthrough_fingertip_fraction:
                continue
        confidence = _combine_margins(
            _colour_margin(float(np.median(deviation[member])), cfg.showthrough_z_threshold),
            _opening_margin(area, cross_section),
            _position_margin(spot, tips, max_tip_distance))
        patches.append(((x, y, w, h), confidence, "show-through {:.2%}".format(area / segmentation.area)))
    return patches

def find_tear_evidence(image: np.ndarray, segmentation: SegmentationResult, config: Config) -> List[Tuple[BBox, float, str]]:
    cfg = config.tearing
    finger_cfg = config.fingertip
    center, palm_radius = palm_center_and_radius(segmentation.mask)
    evidence: List[Tuple[BBox, float, str]] = []
    tips: List[Tuple[int, int]] = []
    if palm_radius > 1:
        tips = locate_fingertips(segmentation, finger_cfg.min_tip_distance_ratio, finger_cfg.tip_merge_separation_ratio, finger_cfg.expected_fingers, finger_cfg.tip_frame_cut_reach_ratio)
    max_tip_distance = cfg.fingertip_radius_ratio * palm_radius
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    inside = segmentation.mask > 0
    glove_lab = np.array([np.median(lab[:, :, c][inside]) for c in range(3)], dtype=np.float32)
    backdrop = estimate_background_lab(cv2.cvtColor(image, cv2.COLOR_BGR2LAB), config.segmentation.border_fraction).astype(np.float32)
    distance = cv2.distanceTransform(segmentation.mask, cv2.DIST_L2, 5)
    for contour, bbox, area in find_holes(segmentation, min_area=cfg.min_hole_area_fraction * segmentation.area, max_area=cfg.max_hole_area_fraction * segmentation.area, max_elongation=cfg.max_hole_elongation, min_extent=cfg.min_hole_extent):
        x, y, w, h = bbox
        spot = (x + w // 2, y + h // 2)
        filled = np.zeros(segmentation.mask.shape, np.uint8)
        cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
        member = filled > 0
        hole_lab = np.array([np.median(lab[:, :, c][member]) for c in range(3)], dtype=np.float32)
        nearest = min(tips, key=lambda point: math.dist(spot, point), default=None)
        cross_section = 0.0
        if nearest is not None and math.dist(spot, nearest) <= max_tip_distance:
            cross_section = fingertip_cross_section(segmentation.mask, nearest, center, palm_radius, distance=distance)
        confidence = _combine_margins(
            _backdrop_margin(hole_lab, backdrop, glove_lab),
            _opening_margin(area, cross_section),
            _position_margin(spot, tips, max_tip_distance))
        evidence.append((bbox, confidence, "hole {:.2%}".format(area / segmentation.area)))
    if palm_radius <= 1:
        return evidence
    evidence.extend(find_showthrough_patches(image, segmentation, config, tips))
    valley_tip_distance = cfg.valley_endpoint_tip_ratio * palm_radius
    for start, end, far, depth in convexity_defect_list(segmentation.contour):
        if is_finger_valley(start, end, tips, valley_tip_distance):
            continue
        depth_ratio = depth / palm_radius
        notch_angle = angle_at(far, start, end)
        if (depth_ratio >= cfg.min_defect_depth_ratio and notch_angle <= cfg.max_defect_angle_deg):
            half = max(8, int(0.15 * palm_radius))
            confidence = _combine_margins(
                _clip01(depth_ratio / max(cfg.min_defect_depth_ratio, 1e-6) - 1.0),
                _clip01(1.0 - notch_angle / max(cfg.max_defect_angle_deg, 1e-6)),
                _position_margin(far, tips, max_tip_distance))
            evidence.append((bbox_around(far, half, segmentation.mask.shape), confidence, "notch depth={:.2f}R angle={:.0f}deg".format(depth_ratio, notch_angle)))
    return evidence

def _analyse(image: np.ndarray, segmentation: SegmentationResult, config: Config) -> DefectResult:
    cfg = config.tearing
    finger_cfg = config.fingertip
    _, palm_radius = palm_center_and_radius(segmentation.mask)
    tips = locate_fingertips(segmentation, finger_cfg.min_tip_distance_ratio, finger_cfg.tip_merge_separation_ratio, finger_cfg.expected_fingers, finger_cfg.tip_frame_cut_reach_ratio)
    if not tips or palm_radius <= 1:
        return DefectResult(False, "tearing_at_finger", details="fingertips could not be localised")

    evidence = find_tear_evidence(image, segmentation, config)
    max_tip_distance = cfg.fingertip_radius_ratio * palm_radius
    locations: List[BBox] = []
    notes: List[str] = []
    score = 0.0
    for (x, y, w, h), confidence, note in evidence:
        spot = (x + w // 2, y + h // 2)
        if any(math.dist(spot, tip) <= max_tip_distance for tip in tips):
            if confidence < cfg.min_score:
                continue
            locations.append((x, y, w, h))
            notes.append("{} conf={:.2f}".format(note, confidence))
            score = max(score, confidence)
    joined = "; ".join(notes)
    return DefectResult(defect_found=bool(locations), defect_type="tearing_at_finger", locations=locations,
        score=score,
        details=("{} of {} tear finding(s) at {} localised fingertip(s): {}".format(len(locations), len(evidence), len(tips), joined) if locations else "0 of {} tear finding(s) near {} localised fingertip(s)".format(len(evidence), len(tips))))

def detect(image: np.ndarray, segmentation: Optional[SegmentationResult] = None, config: Optional[Config] = None) -> DefectResult:
    cfg = config or Config()
    if segmentation is None:
        image = preprocess(image, cfg.preprocess)
        segmentation = segment_glove(image, cfg.segmentation)
        if segmentation is None:
            return DefectResult(False, "tearing_at_finger", details="the glove could not be separated from the background")
    return _analyse(image, segmentation, cfg)
