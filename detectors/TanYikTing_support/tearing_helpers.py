from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np

from .config import PipelineConfig
from .features import BBox, components_as_boxes, palm_center_and_radius, robust_stats
from .segmentation import SegmentationResult, estimate_background_lab


# find colour changes inside glove
def find_showthrough_patches(image: np.ndarray,
                             segmentation: SegmentationResult,
                             config: PipelineConfig
                             ) -> List[Tuple[BBox, float, str]]:
    cfg = config.tearing
    _, palm_radius = palm_center_and_radius(segmentation.mask)
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
            continue
        elongation = max(w, h) / max(min(w, h), 1)
        if elongation > cfg.max_showthrough_elongation:
            continue

        patch = np.array([np.median(lightness[member]),
                          np.median(a_channel[member]),
                          np.median(b_channel[member])], dtype=np.float32)
        if float(np.linalg.norm(patch - backdrop)) < cfg.showthrough_min_backdrop_distance:
            continue

        fraction = area / segmentation.area
        patches.append(((x, y, w, h),
                        min(1.0, 0.5 + 5.0 * fraction),
                        f"show-through {fraction:.2%}"))
    return patches
