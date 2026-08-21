from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
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
class DirtConfig:
    interior_margin_ratio: float = 0.10
    z_threshold: float = 2.0
    min_area_fraction: float = 0.010
    min_extent: float = 0.40
    max_texture_ratio: float = 0.25
    texture_window: int = 9
    min_glove_chroma: float = 12.0
    min_off_hue_distance: float = 15.0

@dataclass
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    dirt: DirtConfig = field(default_factory=DirtConfig)

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
    if image.ndim == 2:  # tolerate grayscale input
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
    coefficients, *_ = np.linalg.lstsq(design(xx[selected], yy[selected]), channel[selected].astype(np.float32), rcond=None,)
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
    candidates: List[Tuple[str, np.ndarray]] = [("bg_distance", _background_distance_mask(lab, cfg.border_fraction)),]
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

def off_hue_distance(region: Tuple[float, float], reference: Tuple[float, float]) -> float:
    ref_length = math.hypot(*reference)
    if ref_length < 1e-6:
        return math.hypot(*region)
    unit_a, unit_b = reference[0] / ref_length, reference[1] / ref_length
    projection = region[0] * unit_a + region[1] * unit_b
    if projection <= 0.0:
        return math.hypot(*region)
    return abs(region[0] * unit_b - region[1] * unit_a)

def components_as_boxes(mask: np.ndarray, min_area: float, min_extent: float = 0.0) -> List[Tuple[BBox, float, np.ndarray]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    results: List[Tuple[BBox, float, np.ndarray]] = []
    for i in range(1, count):
        x, y, width, height, area = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]), int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]), float(stats[i, cv2.CC_STAT_AREA]),)
        if area < min_area:
            continue
        if min_extent > 0.0 and area / max(float(width * height), 1.0) < min_extent:
            continue
        results.append(((x, y, width, height), area, labels == i))
    return results

def _analyse(image: np.ndarray, segmentation: SegmentationResult, config: Config) -> DefectResult:
    cfg = config.dirt
    interior = glove_interior(segmentation, cfg.interior_margin_ratio)
    if np.count_nonzero(interior) < 100:
        return DefectResult(False, "dirty", details="glove interior too small to analyse")
    selection = interior > 0
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    median, spread = robust_stats(lightness[selection])
    z_score = np.abs(lightness - median) / spread
    texture = local_texture_energy(image, cfg.texture_window)
    glove_texture = max(float(np.median(texture[selection])), 1e-6)
    chroma = lab_chroma(image)
    glove_chroma = median_chroma(chroma, selection)
    hue_route_available = math.hypot(*glove_chroma) > cfg.min_glove_chroma
    candidate = ((z_score > cfg.z_threshold) & selection).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    locations: List[BBox] = []
    stained_area = 0.0
    evidence: List[str] = []
    for bbox, area, members in components_as_boxes(candidate, min_area=cfg.min_area_fraction * segmentation.area, min_extent=cfg.min_extent):
        texture_ratio = float(np.median(texture[members])) / glove_texture
        off_hue = off_hue_distance(median_chroma(chroma, members), glove_chroma)
        covered = texture_ratio <= cfg.max_texture_ratio
        foreign_colour = hue_route_available and off_hue > cfg.min_off_hue_distance
        if not (covered or foreign_colour):
            continue
        locations.append(bbox)
        stained_area += area
        evidence.append(f"texture {texture_ratio:.2f}x" if covered else f"hue {off_hue:.0f} off the glove's own")
    stained_fraction = stained_area / segmentation.area
    return DefectResult(
        defect_found=bool(locations),
        defect_type="dirty",
        locations=locations,
        score=min(1.0, stained_fraction / 0.05) if locations else 0.0,
        details=(f"{len(locations)} dirty region(s), {stained_fraction:.2%} of the glove ({'; '.join(evidence)})" if locations else f"no region that is off-colour (z>{cfg.z_threshold:g}) and either texture-free (<{cfg.max_texture_ratio:g}x) or off-hue" + ("" if hue_route_available else "; glove too neutral for the hue test")))

def detect(image: np.ndarray, segmentation: Optional[SegmentationResult] = None, config: Optional[Config] = None) -> DefectResult:
    cfg = config or Config()
    if segmentation is None:
        image = preprocess(image, cfg.preprocess)
        segmentation = segment_glove(image, cfg.segmentation)
        if segmentation is None:
            return DefectResult(False, "dirty", details="the glove could not be separated from the background")
    return _analyse(image, segmentation, cfg)
