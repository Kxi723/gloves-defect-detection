from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np

@dataclass
class PreprocessConfig:
    max_dimension: int = 1024
    white_balance: bool = True
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
class FoldConfig:
    interior_margin_ratio: float = 0.12
    clahe_clip_limit: float = 2.5
    clahe_tile_grid: int = 8
    fine_sigma_ratio: float = 0.020
    coarse_sigma_ratio: float = 0.120
    z_threshold: float = 2.2
    min_length_ratio: float = 0.55
    min_elongation: float = 3.0
    min_crease_count: int = 1
    palm_radius_ratio: float = 1.4
    ridge_bridge_ratio: float = 0.18
    ridge_bridge_angles: int = 12
    bridge_min_elongation: float = 2.5
    bridged_min_length_ratio: float = 0.80
    group_angle_degrees: float = 22.0
    group_collinear_degrees: float = 25.0
    group_max_gap: float = 0.55
    max_lightness_delta: float = -8.0
    stripe_tensor_sigma_ratio: float = 0.035
    stripe_deviation_degrees: float = 25.0
    use_chroma_residual: bool = True
    material_edge_z: float = 8.0
    material_edge_sigma_ratio: float = 0.02
    material_edge_dilate_ratio: float = 0.03

@dataclass
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    fold: FoldConfig = field(default_factory=FoldConfig)

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
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    result = resize_to_limit(image, cfg.max_dimension)
    if cfg.white_balance:
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
    model = np.maximum(model, 1.0)      # a near-zero fit would explode below
    flattened = channel.astype(np.float32) / model * float(np.median(model))
    return np.clip(flattened, 0, 255).astype(np.uint8)

def _background_model_mask(image: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    border = _flatten_illumination  # (naming the dependency for readers)
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
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    candidates: List[Tuple[str, np.ndarray]] = [("bg_distance", _background_distance_mask(lab, cfg.border_fraction))]
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
    return SegmentationResult(mask=solid, mask_raw=mask_raw, contour=contour, bbox=cv2.boundingRect(contour), area=float(area), cue=best_cue)

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

def normalize_illumination(image: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_equalized = clahe.apply(l_channel)
    lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))
    return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)

def _band_pass(channel: np.ndarray, palm_radius: float, cfg: FoldConfig) -> np.ndarray:
    fine = max(1.0, cfg.fine_sigma_ratio * palm_radius)
    coarse = max(fine + 1.0, cfg.coarse_sigma_ratio * palm_radius)
    return (cv2.GaussianBlur(channel, (0, 0), fine) - cv2.GaussianBlur(channel, (0, 0), coarse))

def fold_ridge_response(image: np.ndarray, interior: np.ndarray, palm_radius: float, cfg: FoldConfig) -> np.ndarray:
    equalized = normalize_illumination(image, cfg.clahe_clip_limit, cfg.clahe_tile_grid)
    lightness = cv2.cvtColor(equalized, cv2.COLOR_BGR2LAB)[:, :, 0]
    response = _band_pass(lightness.astype(np.float32), palm_radius, cfg)
    response[interior == 0] = 0.0
    return response

def stripe_deviation(image: np.ndarray, interior: np.ndarray, palm_radius: float, cfg: FoldConfig) -> np.ndarray:
    equalized = normalize_illumination(image, cfg.clahe_clip_limit, cfg.clahe_tile_grid)
    gray = cv2.cvtColor(equalized, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sigma = max(1.0, cfg.stripe_tensor_sigma_ratio * palm_radius)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    cos2 = jxx - jyy
    sin2 = 2.0 * jxy
    magnitude = np.maximum(np.hypot(cos2, sin2), 1e-6)
    inside = interior > 0
    if not inside.any():
        return np.zeros(gray.shape, np.float32)
    total_c = float(cos2[inside].sum())
    total_s = float(sin2[inside].sum())
    total = max(float(np.hypot(total_c, total_s)), 1e-6)
    total_c, total_s = total_c / total, total_s / total
    aligned = np.clip((cos2 / magnitude) * total_c + (sin2 / magnitude) * total_s, -1.0, 1.0)
    deviation = np.degrees(np.arccos(aligned)) / 2.0
    deviation[~inside] = 0.0
    return deviation

def chroma_residual(image: np.ndarray, interior: np.ndarray, palm_radius: float, cfg: FoldConfig) -> np.ndarray:
    equalized = normalize_illumination(image, cfg.clahe_clip_limit, cfg.clahe_tile_grid)
    lab = cv2.cvtColor(equalized, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = np.abs(_band_pass(lab[:, :, 0], palm_radius, cfg))
    chroma = np.hypot(_band_pass(lab[:, :, 1], palm_radius, cfg), _band_pass(lab[:, :, 2], palm_radius, cfg))
    inside = interior > 0
    residual = np.zeros(lightness.shape, np.float32)
    if np.count_nonzero(inside) < 100:
        return residual
    design = np.stack([chroma[inside], np.ones(int(inside.sum()), np.float32)], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, lightness[inside], rcond=None)
    residual[inside] = lightness[inside] - design @ coefficients
    return residual

def _clean(binary: np.ndarray) -> np.ndarray:
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

def _line_kernel(length: int, angle_degrees: float) -> np.ndarray:
    kernel = np.zeros((length, length), np.uint8)
    centre = length // 2
    radians = np.deg2rad(angle_degrees)
    for step in np.linspace(-centre, centre, 2 * length):
        x = int(round(centre + step * np.cos(radians)))
        y = int(round(centre + step * np.sin(radians)))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    return kernel

def _line_like(binary: np.ndarray, min_elongation: float) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros_like(binary)
    for index in range(1, count):
        width = stats[index, cv2.CC_STAT_WIDTH]
        height = stats[index, cv2.CC_STAT_HEIGHT]
        area = stats[index, cv2.CC_STAT_AREA]
        if area < 12:
            continue
        span = float(np.hypot(width, height))
        thickness = area / max(span, 1.0)
        if span / max(thickness, 1e-6) >= min_elongation:
            keep[labels == index] = 255
    return keep

def _bridged_variants(binary: np.ndarray, palm_radius: float, cfg: FoldConfig) -> List[np.ndarray]:
    seeds = _line_like(binary, cfg.bridge_min_elongation)
    length = max(5, int(cfg.ridge_bridge_ratio * palm_radius)) | 1
    variants = [binary]
    for index in range(cfg.ridge_bridge_angles):
        angle = 180.0 * index / cfg.ridge_bridge_angles
        variants.append(cv2.bitwise_or(binary, cv2.morphologyEx(seeds, cv2.MORPH_CLOSE, _line_kernel(length, angle))))
    return variants


def _shaped_creases(binary: np.ndarray, palm_region: np.ndarray, palm_radius: float, cfg: FoldConfig, bridge: bool = False) -> List[Tuple[np.ndarray, float]]:
    cleaned = _clean(binary)
    variants = (_bridged_variants(cleaned, palm_radius, cfg) if bridge else [cleaned])
    out: List[Tuple[np.ndarray, float]] = []
    for position, variant in enumerate(variants):
        minimum = palm_radius * (cfg.min_length_ratio if position == 0 else cfg.bridged_min_length_ratio)
        contours, _ = cv2.findContours(variant, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 5:
                continue
            (cx, cy), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
            major = max(axis_a, axis_b)
            minor = max(min(axis_a, axis_b), 1e-6)
            if major < minimum or major / minor < cfg.min_elongation:
                continue
            row, col = int(round(cy)), int(round(cx))
            if not (0 <= row < palm_region.shape[0] and 0 <= col < palm_region.shape[1] and palm_region[row, col]):
                continue
            out.append((contour, float(major)))
    return out

def _fragment_pool(binary: np.ndarray, cfg: FoldConfig) -> List[np.ndarray]:
    contours, _ = cv2.findContours(_clean(binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pool: List[np.ndarray] = []
    for contour in contours:
        if len(contour) < 5 or cv2.contourArea(contour) < 30:
            continue
        (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        major, minor = max(axis_a, axis_b), max(min(axis_a, axis_b), 1e-6)
        if major / minor >= cfg.bridge_min_elongation:
            pool.append(contour)
    return pool

def _axis_of(contour: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    radians = np.deg2rad(angle)
    direction = np.array([-np.sin(radians), np.cos(radians)], np.float32)
    return (np.array([cx, cy], np.float32), direction, float(max(axis_a, axis_b)))

def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(first, second))), 0.0, 1.0))))

def _extend_along_line(contour: np.ndarray, pool: List[np.ndarray], palm_radius: float, cfg: FoldConfig) -> Tuple[BBox, float]:
    centre, direction, length = _axis_of(contour)
    members = [contour]
    for other in pool:
        if other is contour or len(other) < 5:
            continue
        other_centre, other_direction, other_length = _axis_of(other)
        if _angle_between(direction, other_direction) > cfg.group_angle_degrees:
            continue
        link = other_centre - centre
        distance = float(np.linalg.norm(link))
        if distance < 1e-6:
            continue
        if _angle_between(link / distance, direction) > cfg.group_collinear_degrees:
            continue
        if distance - (length + other_length) / 2.0 > cfg.group_max_gap * palm_radius:
            continue
        members.append(other)
    points = np.vstack([m.reshape(-1, 2) for m in members]).astype(np.float32)
    along = (points - points.mean(axis=0)) @ direction
    extent = float(along.max() - along.min())
    return cv2.boundingRect(points.astype(np.int32)), max(extent, length)

def _is_shadow(contour: np.ndarray, lightness: np.ndarray, glove_median: float, cfg: FoldConfig) -> Tuple[bool, float]:
    stencil = np.zeros(lightness.shape, np.uint8)
    cv2.drawContours(stencil, [contour], -1, 255, thickness=cv2.FILLED)
    inside = stencil > 0
    if not inside.any():
        return False, 0.0
    delta = float(np.median(lightness[inside])) - glove_median
    return delta <= cfg.max_lightness_delta, delta

def material_boundary(image: np.ndarray, interior: np.ndarray, palm_radius: float, cfg: FoldConfig) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    sigma = max(1.0, cfg.material_edge_sigma_ratio * palm_radius)
    edge = np.zeros(image.shape[:2], np.float32)
    for channel in (1, 2):
        blurred = cv2.GaussianBlur(lab[:, :, channel], (0, 0), sigma)
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        edge += np.hypot(gx, gy)
    inside = interior > 0
    if not inside.any():
        return np.zeros(image.shape[:2], np.uint8)
    median, spread = robust_stats(edge[inside])
    band = ((edge > median + cfg.material_edge_z * spread) & inside).astype(np.uint8) * 255
    size = max(3, int(cfg.material_edge_dilate_ratio * palm_radius)) | 1
    return cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))


def _analyse(image: np.ndarray, segmentation: SegmentationResult, config: Config) -> DefectResult:
    cfg = config.fold
    interior = glove_interior(segmentation, cfg.interior_margin_ratio)
    palm_center, palm_radius = palm_center_and_radius(segmentation.mask)
    if np.count_nonzero(interior) < 100 or palm_radius < 10:
        return DefectResult(False, "damage_by_fold", details="glove interior too small to analyse")

    palm_disc = np.zeros_like(interior)
    cv2.circle(palm_disc, palm_center, int(cfg.palm_radius_ratio * palm_radius), 255, cv2.FILLED)
    palm_region = cv2.bitwise_and(interior, palm_disc)
    if np.count_nonzero(palm_region) < 100:
        return DefectResult(False, "damage_by_fold", details="palm region too small to analyse")
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    glove_median = float(np.median(lightness[interior > 0]))
    candidates: List[Tuple[np.ndarray, float, str]] = []
    pools: dict = {}
    response = fold_ridge_response(image, interior, palm_radius, cfg)
    _, spread = robust_stats(response[interior > 0])
    ridge = ((np.abs(response) > cfg.z_threshold * spread) & (palm_region > 0)).astype(np.uint8) * 255
    ridge = cv2.bitwise_and(ridge, cv2.bitwise_not(
        material_boundary(image, interior, palm_radius, cfg)))
    pools["shading"] = _fragment_pool(ridge, cfg)
    for contour, major in _shaped_creases(ridge, palm_region, palm_radius, cfg, bridge=True):
        candidates.append((contour, major, "shading"))
    deviation = stripe_deviation(image, interior, palm_radius, cfg)
    bent = ((deviation > cfg.stripe_deviation_degrees) & (palm_region > 0)).astype(np.uint8) * 255
    bent = cv2.morphologyEx(bent, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bent = cv2.morphologyEx(bent, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    pools["weave"] = _fragment_pool(bent, cfg)
    for contour, major in _shaped_creases(bent, palm_region, palm_radius, cfg):
        candidates.append((contour, major, "weave"))
    if cfg.use_chroma_residual:
        residual = chroma_residual(image, interior, palm_radius, cfg)
        _, residual_spread = robust_stats(residual[interior > 0])
        weak = ((residual > cfg.z_threshold * residual_spread) & (palm_region > 0)).astype(np.uint8) * 255
        pools["residual"] = _fragment_pool(weak, cfg)
        for contour, major in _shaped_creases(weak, palm_region, palm_radius, cfg):
            candidates.append((contour, major, "residual"))
    creases: List[BBox] = []
    lengths: List[float] = []
    channels: List[str] = []
    rejected_bright = 0
    claimed = np.zeros(interior.shape, np.uint8)
    for contour, major, channel in sorted(candidates, key=lambda c: -c[1]):
        shadow, _ = _is_shadow(contour, lightness, glove_median, cfg)
        if not shadow:
            rejected_bright += 1
            continue
        stencil = np.zeros(interior.shape, np.uint8)
        cv2.drawContours(stencil, [contour], -1, 255, thickness=cv2.FILLED)
        overlap = np.count_nonzero(cv2.bitwise_and(stencil, claimed))
        if overlap > 0.4 * max(np.count_nonzero(stencil), 1):
            continue
        claimed = cv2.bitwise_or(claimed, stencil)
        box, extent = _extend_along_line(contour, pools.get(channel, []), palm_radius, cfg)
        creases.append(box)
        lengths.append(max(major, extent))
        channels.append(channel)
    found = len(creases) >= cfg.min_crease_count
    longest = max(lengths) / palm_radius if lengths else 0.0
    if creases:
        seen = ", ".join(sorted(set(channels)))
        detail = (f"{len(creases)} palm crease(s), longest {longest:.2f}R, via {seen}")
    elif rejected_bright:
        detail = (f"{rejected_bright} bright ridge(s) rejected as glare rather than a crease")
    else:
        detail = (f"no palm crease longer than {cfg.min_length_ratio:g}R (searched {cfg.palm_radius_ratio:g}R around the palm centre)")
    return DefectResult(defect_found=found, defect_type="damage_by_fold", locations=creases if found else [], score=min(1.0, longest / 1.5) if found else 0.0, details=detail)

def detect(image: np.ndarray, segmentation: Optional[SegmentationResult] = None, config: Optional[Config] = None) -> DefectResult:
    cfg = config or Config()
    if segmentation is None:
        image = preprocess(image, cfg.preprocess)
        segmentation = segment_glove(image, cfg.segmentation)
        if segmentation is None:
            return DefectResult(False, "damage_by_fold", details="the glove could not be separated from the background")
    return _analyse(image, segmentation, cfg)

