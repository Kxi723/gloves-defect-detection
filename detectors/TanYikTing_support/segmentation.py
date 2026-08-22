
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import SegmentationConfig


# segmentation result
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


# estimate background
def estimate_background_lab(lab: np.ndarray, border_fraction: float) -> np.ndarray:
    h, w = lab.shape[:2]
    b = max(2, int(round(min(h, w) * border_fraction)))
    strip = np.concatenate(
        [
            lab[:b, :].reshape(-1, 3),
            lab[-b:, :].reshape(-1, 3),
            lab[:, :b].reshape(-1, 3),
            lab[:, -b:].reshape(-1, 3),
        ]
    )
    return np.median(strip, axis=0).astype(np.float32)


def _background_distance_mask(lab: np.ndarray, border_fraction: float) -> np.ndarray:
    background = estimate_background_lab(lab, border_fraction)
    distance = np.linalg.norm(lab.astype(np.float32) - background, axis=2)
    distance_u8 = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    _, mask = cv2.threshold(distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _flatten_illumination(channel: np.ndarray,
                          border_fraction: float) -> np.ndarray:
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

    coefficients, *_ = np.linalg.lstsq(
        design(xx[selected], yy[selected]),
        channel[selected].astype(np.float32),
        rcond=None,
    )
    model = (design(xx.ravel(), yy.ravel()) @ coefficients).reshape(h, w)
    model = np.maximum(model, 1.0)
    flattened = channel.astype(np.float32) / model * float(np.median(model))
    return np.clip(flattened, 0, 255).astype(np.uint8)


def _background_model_mask(image: np.ndarray,
                           cfg: SegmentationConfig) -> np.ndarray:
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
    scaled = np.clip(
        distance / (cfg.background_z_clip * np.sqrt(3.0)) * 255.0, 0, 255
    ).astype(np.uint8)
    _, mask = cv2.threshold(scaled, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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


# texture mask
def _texture_energy_mask(image: np.ndarray, window: int) -> np.ndarray:
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    mean = cv2.blur(lightness, (window, window))
    mean_of_squares = cv2.blur(lightness * lightness, (window, window))

    std = np.sqrt(np.maximum(mean_of_squares - mean * mean, 0.0))
    std = cv2.GaussianBlur(std, (0, 0), window / 2.0)
    std_u8 = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(std_u8, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


# score mask
def _score_candidate(mask: np.ndarray, cfg: SegmentationConfig) -> float:
    image_area = float(mask.shape[0] * mask.shape[1])
    cleaned = _morphological_cleanup(mask, cfg)
    count, labels = cv2.connectedComponents(cleaned)
    if count <= 1:
        return -1.0
    largest_label = 1 + int(
        np.argmax([np.count_nonzero(labels == i) for i in range(1, count)])
    )
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

    return ((1.0 - border_occupancy) * 2.0
            + (1.0 - abs(area_fraction - 0.30))
            - 4.0 * min(shred, 0.5)
            - 0.5 * max(0.0, raggedness - 1.6))


# clean mask
def _morphological_cleanup(mask: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    open_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel)
    )
    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel)
    )
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


def _fill_noise_holes(mask_raw: np.ndarray, solid: np.ndarray,
                      cfg: SegmentationConfig, glove_area: float) -> np.ndarray:
    holes = cv2.subtract(solid, mask_raw)
    hole_contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    min_area = cfg.min_hole_area_fraction * glove_area
    result = mask_raw.copy()
    for hc in hole_contours:
        if cv2.contourArea(hc) < min_area:
            cv2.drawContours(result, [hc], -1, 255, thickness=cv2.FILLED)
    return result


# build glove mask
def segment_glove(
    image: np.ndarray, config: Optional[SegmentationConfig] = None
) -> Optional[SegmentationResult]:
    cfg = config or SegmentationConfig()
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    candidates: List[Tuple[str, np.ndarray]] = [
        ("bg_distance", _background_distance_mask(lab, cfg.border_fraction)),
    ]
    otsu_masks = _channel_otsu_masks(hsv)
    candidates.append(("saturation", otsu_masks[0]))
    candidates.append(("value", otsu_masks[1]))
    candidates.append(("value_inverted", otsu_masks[2]))
    candidates.append(
        ("texture", _texture_energy_mask(image, cfg.texture_window))
    )


    primary = _background_model_mask(image, cfg)
    if _score_candidate(primary, cfg) >= 0:
        best_cue, best_mask = "bg_model", primary
    else:
        scored = [(_score_candidate(mask, cfg), name, mask)
                  for name, mask in candidates]
        best_score, best_cue, best_mask = max(scored,
                                              key=lambda triple: triple[0])
        if best_score < 0:
            return None


    cleaned = _morphological_cleanup(best_mask, cfg)


    num, labels = cv2.connectedComponents(cleaned)
    if num <= 1:
        return None
    largest_label = 1 + int(
        np.argmax([np.count_nonzero(labels == i) for i in range(1, num)])
    )
    component = np.where(labels == largest_label, 255, 0).astype(np.uint8)


    solid, contour = _keep_largest_component(component)
    if contour is None:
        return None
    area = cv2.contourArea(contour)
    if area < cfg.min_area_fraction * image.shape[0] * image.shape[1]:
        return None


    mask_raw = _fill_noise_holes(component, solid, cfg, area)

    return SegmentationResult(
        mask=solid,
        mask_raw=mask_raw,
        contour=contour,
        bbox=cv2.boundingRect(contour),
        area=float(area),
        cue=best_cue,
    )
