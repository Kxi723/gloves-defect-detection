from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import SegmentationConfig

@dataclass
class SegmentationResult:
    mask: np.ndarray
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    area: float
    cue: str = "unknown"
    source_image: Optional[np.ndarray] = None

def hsi_saturation_and_intensity(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    channel_sum = rgb.sum(axis=2)
    safe_sum = np.maximum(channel_sum, 1.0)
    intensity = channel_sum / 3.0
    saturation = 1.0 - (3.0 * np.min(rgb, axis=2) / safe_sum)
    saturation[channel_sum <= 0.0] = 0.0
    return np.clip(saturation, 0.0, 1.0), intensity

def basic_global_threshold(
    channel: np.ndarray,
    *,
    invert: bool = False,
    tolerance: float = 1.0,
    maximum_iterations: int = 30,
) -> np.ndarray:

    values = channel.astype(np.float32)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)

    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    if maximum - minimum < 1e-6:
        return np.zeros(values.shape, dtype=np.uint8)

    threshold = (minimum + maximum) / 2.0
    for _ in range(maximum_iterations):
        lower_group = finite_values[finite_values <= threshold]
        upper_group = finite_values[finite_values > threshold]
        if lower_group.size == 0 or upper_group.size == 0:
            break
        new_threshold = (
            float(np.mean(lower_group)) + float(np.mean(upper_group))
        ) / 2.0
        if abs(new_threshold - threshold) < tolerance:
            threshold = new_threshold
            break
        threshold = new_threshold

    selected = values <= threshold if invert else values > threshold
    return np.where(selected, 255, 0).astype(np.uint8)

def _hsi_channel_masks(
    saturation: np.ndarray,
    intensity: np.ndarray,
) -> List[np.ndarray]:

    return [
        basic_global_threshold(saturation, tolerance=1.0 / 255.0),
        basic_global_threshold(intensity),
        basic_global_threshold(intensity, invert=True),
    ]

def _texture_energy_mask(intensity: np.ndarray, window: int) -> np.ndarray:
    mean = cv2.blur(intensity, (window, window))
    mean_of_squares = cv2.blur(intensity * intensity, (window, window))

    std = np.sqrt(np.maximum(mean_of_squares - mean * mean, 0.0))
    std = cv2.GaussianBlur(std, (0, 0), window / 2.0)
    return basic_global_threshold(std)

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

def segment_glove(
    image: np.ndarray, config: Optional[SegmentationConfig] = None
) -> Optional[SegmentationResult]:


    cfg = config or SegmentationConfig()
    saturation, intensity = hsi_saturation_and_intensity(image)


    hsi_masks = _hsi_channel_masks(saturation, intensity)
    candidates: List[Tuple[str, np.ndarray]] = []
    candidates.append(("hsi_saturation", hsi_masks[0]))
    candidates.append(("hsi_intensity", hsi_masks[1]))
    candidates.append(("hsi_intensity_inverted", hsi_masks[2]))
    candidates.append(
        ("hsi_intensity_texture", _texture_energy_mask(intensity, cfg.texture_window))
    )

    scored = [(_score_candidate(mask, cfg), name, mask)
              for name, mask in candidates]
    best_score, best_cue, best_mask = max(scored, key=lambda triple: triple[0])
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

    return SegmentationResult(
        mask=solid,
        contour=contour,
        bbox=cv2.boundingRect(contour),
        area=float(area),
        cue=best_cue,
    )
