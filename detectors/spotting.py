from __future__ import annotations
from typing import List
import math
import cv2
import numpy as np

from .TanYikTing_support.config import PipelineConfig
from .TanYikTing_support.preprocessing import preprocess
from .TanYikTing_support.features import BBox, DefectResult, glove_interior, palm_center_and_radius
from .TanYikTing_support.segmentation import SegmentationResult, segment_glove


# runner setup
Config = PipelineConfig
def detect(image: np.ndarray, segmentation: SegmentationResult,
           config: PipelineConfig) -> DefectResult:
    cfg = config.spotting
    interior = glove_interior(segmentation, cfg.interior_margin_ratio)
    selection = interior > 0
    if np.count_nonzero(selection) < 100:
        return DefectResult(False, "spotting", details="glove interior too small")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    median_v = float(np.median(val[selection]))
    median_s = float(np.median(sat[selection]))

    k = max(1, int(cfg.blur_kernel)) | 1
    val_s = cv2.GaussianBlur(val, (k, k), 0)
    sat_s = cv2.GaussianBlur(sat, (k, k), 0)

    if median_v >= cfg.light_glove_value_cutoff:


        dark_limit = min(median_v - cfg.light_dark_delta,
                         cfg.light_absolute_value_max)
        raw = (val_s < dark_limit) & selection
        branch = "light glove: dark-spot threshold"
    else:


        bright_neutral = ((val_s > median_v + cfg.dark_bright_delta) &
                          (sat_s <= cfg.dark_low_saturation_max))
        chromatic = ((sat_s >= cfg.dark_high_saturation_min) &
                     (val_s >= median_v + cfg.dark_colour_value_delta))
        raw = (bright_neutral | chromatic) & selection
        branch = "dark glove: bright-neutral or saturated-colour threshold"

    candidate = raw.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (cfg.open_kernel, cfg.open_kernel))
    # remove small noise
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (candidate > 0).astype(np.uint8), connectivity=8)

    locations: List[BBox] = []
    accepted_mask = np.zeros_like(candidate)
    accepted_centres: List[tuple] = []
    total_area = 0.0

    # filter spot regions
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        area_fraction = area / max(segmentation.area, 1.0)
        if not (cfg.min_area_fraction <= area_fraction <= cfg.max_area_fraction):
            continue
        extent = area / max(float(w * h), 1.0)
        elongation = max(w, h) / max(min(w, h), 1)
        if extent < cfg.min_extent or elongation > cfg.max_elongation:
            continue

        member_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(member_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        contour_area = cv2.contourArea(contour)
        circularity = (4.0 * math.pi * contour_area / (perimeter * perimeter)
                       if perimeter > 1e-6 else 0.0)
        if circularity < cfg.min_circularity:
            continue

        locations.append((x, y, w, h))
        accepted_mask[labels == label] = 255
        accepted_centres.append(tuple(map(float, centroids[label])))
        total_area += area

    _, palm_radius = palm_center_and_radius(segmentation.mask)
    if len(accepted_centres) >= 2 and palm_radius > 1:
        centres = np.asarray(accepted_centres, np.float32)
        spread_pixels = float(np.linalg.norm(centres.max(axis=0) - centres.min(axis=0)))
        spread_ratio = spread_pixels / palm_radius
    else:
        spread_ratio = 0.0

    area_fraction_total = total_area / max(segmentation.area, 1.0)
    found = (len(locations) >= cfg.min_spot_count and
             spread_ratio >= cfg.min_spread_ratio and
             area_fraction_total >= cfg.min_total_area_fraction)
    if found:
        count_score = min(1.0, len(locations) / max(cfg.full_confidence_count, 1))
        spread_score = min(1.0, spread_ratio / max(cfg.full_confidence_spread_ratio, 1e-6))
        score = min(1.0, 0.55 * count_score + 0.45 * spread_score)
    else:
        score = 0.0
        accepted_mask[:] = 0
        locations = []

    detail = (f"{len(accepted_centres)} compact spot candidate(s); "
              f"spread={spread_ratio:.2f}R; need >= {cfg.min_spot_count} spots "
              f"and >= {cfg.min_spread_ratio:.1f}R spread")
    if found:
        detail = (f"{len(accepted_centres)} compact isolated spot(s), "
                  f"spread={spread_ratio:.2f}R, total "
                  f"{total_area / max(segmentation.area, 1):.2%} of glove")

    return DefectResult(
        found, "spotting", locations, score, detail,
        mask=accepted_mask,
        methods=[
            "HSV glove brightness/material estimation",
            branch,
            "morphological opening",
            "connected-component area/shape filtering",
            "component circularity filtering",
            "spatial distribution / centroid-spread validation",
        ],
        metrics={
            "glove_median_value": median_v,
            "glove_median_saturation": median_s,
            "compact_spot_count": len(accepted_centres),
            "spot_spread_ratio_to_palm": spread_ratio,
            "spot_area_fraction": total_area / max(segmentation.area, 1.0),
        },
        parameters={
            "light_glove_value_cutoff": cfg.light_glove_value_cutoff,
            "light_dark_delta": cfg.light_dark_delta,
            "light_absolute_value_max": cfg.light_absolute_value_max,
            "dark_bright_delta": cfg.dark_bright_delta,
            "min_spot_count": cfg.min_spot_count,
            "min_spread_ratio": cfg.min_spread_ratio,
            "min_total_area_fraction": cfg.min_total_area_fraction,
            "max_area_fraction": cfg.max_area_fraction,
            "min_circularity": cfg.min_circularity,
        },
    )
