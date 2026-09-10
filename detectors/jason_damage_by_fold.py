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
    edge_weight: float = 1.5
    field_z: float = 5.0
    field_trim: float = 0.75
    field_subsample: int = 3
    field_exclude_ratio: float = 0.12
    refine_max_dimension: int = 480
    refine_iterations: int = 3
    refine_core_ratio: float = 0.20
    refine_reach_ratio: float = 0.30

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
    min_span_ratio: float = 1.33
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

def _ellipse(radius: float) -> np.ndarray:
    r = max(int(radius), 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

def _equivalent_radius(mask: np.ndarray) -> float:
    return float(np.sqrt(max(np.count_nonzero(mask), 1) / np.pi))

def _largest_blob(mask: np.ndarray, cfg: SegmentationConfig) -> Optional[np.ndarray]:
    cleaned = _morphological_cleanup(mask, cfg)
    count, labels = cv2.connectedComponents(cleaned)
    if count <= 1:
        return None
    largest = 1 + int(np.argmax([np.count_nonzero(labels == i) for i in range(1, count)]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)

def _gradient_field(image: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.4)
    return cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

def _edge_support(gradient: np.ndarray, solid: np.ndarray) -> float:
    """Mean image gradient along the mask boundary, over the gradient scale of the
    frame. An outline that follows the glove sits on real edges; one that wanders
    into flat backdrop scores near zero."""
    ring = cv2.subtract(solid, cv2.erode(solid, np.ones((3, 3), np.uint8)))
    if np.count_nonzero(ring) < 20:
        return 0.0
    scale = max(float(np.percentile(gradient, 90)), 1e-6)
    return float(np.mean(gradient[ring > 0])) / scale

def _surface_terms(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(x), x, y, x * x, y * y, x * y, x * x * y, x * y * y], axis=1)

def _backdrop_residual_mask(image: np.ndarray, seed: np.ndarray, cfg: SegmentationConfig) -> Optional[np.ndarray]:
    """Foreground as the residual from a smooth surface fitted to the backdrop.

    Sampling only the frame border treats the cloth as one flat colour. A lamp
    puts a bright halo around the glove and lets the corners fall away, so the
    middle of the cloth ends up far from the border median and reads as glove.
    Fitting a low order surface to the pixels that are actually backdrop predicts
    that falloff instead, and only real material is left over."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = lab.shape[:2]
    grown = cv2.dilate(seed, _ellipse(cfg.field_exclude_ratio * _equivalent_radius(seed)))
    sample = np.zeros((height, width), dtype=bool)
    sample[::cfg.field_subsample, ::cfg.field_subsample] = True
    sample &= grown == 0
    flat = sample.ravel()
    if int(np.count_nonzero(flat)) < 200:
        return None

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx = xx / max(width - 1, 1) * 2.0 - 1.0
    yy = yy / max(height - 1, 1) * 2.0 - 1.0
    terms = _surface_terms(xx.ravel(), yy.ravel())
    rows = terms[flat]

    squared = np.zeros((height, width), dtype=np.float32)
    for channel in range(3):
        target = lab[:, :, channel].ravel()[flat]
        coefficients, *_ = np.linalg.lstsq(rows, target, rcond=None)
        residual = np.abs(rows @ coefficients - target)
        keep = residual <= np.quantile(residual, cfg.field_trim)
        if int(np.count_nonzero(keep)) > 100:
            coefficients, *_ = np.linalg.lstsq(rows[keep], target[keep], rcond=None)
            residual = np.abs(rows @ coefficients - target)
        spread = max(float(np.median(residual)) * 1.4826, 2.0)
        surface = (terms @ coefficients).reshape(height, width)
        squared += ((lab[:, :, channel] - surface) / spread) ** 2
    distance = np.sqrt(squared / 3.0)
    return (distance > cfg.field_z).astype(np.uint8) * 255

def _grabcut_refine(image: np.ndarray, coarse: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """Re-cut the boundary with GrabCut, at reduced resolution.

    The seed says roughly where the glove is; the graph cut then decides the edge
    from the colour statistics of both sides, which is what drops a sleeve or an
    arm that the thresholds had joined onto the glove."""
    height, width = coarse.shape[:2]
    scale = min(1.0, cfg.refine_max_dimension / max(height, width))
    size = (max(int(round(width * scale)), 16), max(int(round(height * scale)), 16))
    small_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    small = cv2.resize(coarse, size, interpolation=cv2.INTER_NEAREST)
    if np.count_nonzero(small) < 40:
        return coarse

    radius = _equivalent_radius(small)
    core = cv2.erode(small, _ellipse(cfg.refine_core_ratio * radius))
    if np.count_nonzero(core) < 20:
        return coarse
    reach = cv2.dilate(small, _ellipse(cfg.refine_reach_ratio * radius))

    state = np.full(small.shape, cv2.GC_BGD, np.uint8)
    state[reach > 0] = cv2.GC_PR_BGD
    state[small > 0] = cv2.GC_PR_FGD
    state[core > 0] = cv2.GC_FGD
    band = max(2, int(round(min(size) * cfg.border_fraction)))
    edge = np.zeros(small.shape, dtype=bool)
    edge[:band, :] = True
    edge[-band:, :] = True
    edge[:, :band] = True
    edge[:, -band:] = True
    state[edge & (small == 0)] = cv2.GC_BGD

    try:
        cv2.grabCut(small_image, state, None, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), cfg.refine_iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return coarse
    refined = np.where((state == cv2.GC_FGD) | (state == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if np.count_nonzero(refined) < 0.25 * np.count_nonzero(small):
        return coarse  # the cut collapsed, keep the seed
    return cv2.resize(refined, (width, height), interpolation=cv2.INTER_NEAREST)

def segment_glove(image: np.ndarray, config: Optional[SegmentationConfig] = None) -> Optional[SegmentationResult]:
    cfg = config or SegmentationConfig()
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    otsu_masks = _channel_otsu_masks(hsv)
    candidates: List[Tuple[str, np.ndarray]] = [
        ("bg_model", _background_model_mask(image, cfg)),
        ("bg_distance", _background_distance_mask(lab, cfg.border_fraction)),
        ("saturation", otsu_masks[0]),
        ("value", otsu_masks[1]),
        ("value_inverted", otsu_masks[2]),
        ("texture", _texture_energy_mask(image, cfg.texture_window)),
    ]

    # every cue is scored on area, compactness and how well its outline lands on
    # real image edges, and the best one wins
    gradient = _gradient_field(image)
    best_score, best_cue, best_mask = 0.0, "", None
    for name, mask in candidates:
        score = _score_candidate(mask, cfg)
        if score < 0:
            continue
        blob = _largest_blob(mask, cfg)
        if blob is not None:
            solid, contour = _keep_largest_component(blob)
            if contour is not None:
                score += cfg.edge_weight * _edge_support(gradient, solid)
        if best_mask is None or score > best_score:
            best_score, best_cue, best_mask = score, name, mask
    if best_mask is None:
        return None

    component = _largest_blob(best_mask, cfg)
    if component is None:
        return None

    # second pass, now that the glove is roughly located the backdrop can be modelled
    residual = _backdrop_residual_mask(image, component, cfg)
    if residual is not None and _score_candidate(residual, cfg) > -1.0:
        refined = _largest_blob(residual, cfg)
        if refined is not None:
            component = refined

    cut = _largest_blob(_grabcut_refine(image, component, cfg), cfg)
    if cut is not None:
        component = cut

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

def robust_stats(values: np.ndarray) -> Tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, max(1.4826 * mad, 1e-6)

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

def _clean(binary: np.ndarray) -> np.ndarray:
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

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

def _is_shadow(contour: np.ndarray, lightness: np.ndarray, glove_median: float, cfg: FoldConfig) -> Tuple[bool, float]:
    stencil = np.zeros(lightness.shape, np.uint8)
    cv2.drawContours(stencil, [contour], -1, 255, thickness=cv2.FILLED)
    inside = stencil > 0
    if not inside.any():
        return False, 0.0
    delta = float(np.median(lightness[inside])) - glove_median
    return delta <= cfg.max_lightness_delta, delta

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
    ridge = cv2.bitwise_and(ridge, cv2.bitwise_not(material_boundary(image, interior, palm_radius, cfg)))
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
    # a fold is pressed across the palm, a natural wrinkle or a knit stripe is a short line
    spanning = [i for i, length in enumerate(lengths) if length >= cfg.min_span_ratio * palm_radius]
    found = len(spanning) >= cfg.min_crease_count
    longest = max(lengths) / palm_radius if lengths else 0.0
    if spanning:
        seen = ", ".join(sorted({channels[i] for i in spanning}))
        detail = (f"{len(spanning)} palm crease(s), longest {longest:.2f}R, via {seen}")
    elif creases:
        detail = (f"{len(creases)} dark line(s) in the palm, the longest runs {longest:.2f}R but a fold spans at least {cfg.min_span_ratio:g}R")
    elif rejected_bright:
        detail = (f"{rejected_bright} bright ridge(s) rejected as glare rather than a crease")
    else:
        detail = (f"no palm crease longer than {cfg.min_length_ratio:g}R (searched {cfg.palm_radius_ratio:g}R around the palm centre)")
    return DefectResult(defect_found=found, defect_type="damage_by_fold", locations=[creases[i] for i in spanning] if found else [], score=min(1.0, longest / 1.5) if found else 0.0, details=detail)

def detect(image: np.ndarray, segmentation: Optional[SegmentationResult] = None, config: Optional[Config] = None) -> DefectResult:
    cfg = config or Config()
    if segmentation is None:
        image = preprocess(image, cfg.preprocess)
        segmentation = segment_glove(image, cfg.segmentation)
        if segmentation is None:
            return DefectResult(False, "damage_by_fold", details="the glove could not be separated from the background")
    return _analyse(image, segmentation, cfg)
