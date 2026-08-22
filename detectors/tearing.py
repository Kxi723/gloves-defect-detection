from __future__ import annotations
from typing import Dict, List, Tuple
import cv2
import numpy as np

from .TanYikTing_support.config import PipelineConfig
from .TanYikTing_support.preprocessing import preprocess
from .TanYikTing_support.features import BBox, DefectResult, palm_center_and_radius, robust_stats
from .TanYikTing_support.segmentation import SegmentationResult, segment_glove
from .TanYikTing_support.tearing_helpers import find_showthrough_patches


# runner setup
Config = PipelineConfig
def _contact_side(mask: np.ndarray) -> str:
    counts: Dict[str, int] = {
        "top": int(np.count_nonzero(mask[0, :])),
        "bottom": int(np.count_nonzero(mask[-1, :])),
        "left": int(np.count_nonzero(mask[:, 0])),
        "right": int(np.count_nonzero(mask[:, -1])),
    }
    return max(counts, key=counts.get)


def _inside_cuff_exclusion(x: int, y: int, w: int, h: int,
                           image_w: int, image_h: int,
                           side: str, fraction: float) -> bool:
    if side == "bottom":
        return y + h >= (1.0 - fraction) * image_h
    if side == "top":
        return y <= fraction * image_h
    if side == "right":
        return x + w >= (1.0 - fraction) * image_w
    return x <= fraction * image_w


def _light_glove_tears(image: np.ndarray, segmentation: SegmentationResult,
                       config: PipelineConfig) -> DefectResult:
    cfg = config.general_tearing
    # light glove method
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    sat_f = sat.astype(np.float32)
    local_sat = cv2.GaussianBlur(
        sat_f, (0, 0), sigmaX=cfg.light_local_sigma,
        sigmaY=cfg.light_local_sigma)
    residual = sat_f - local_sat

    # find skin like slit areas
    candidate = ((hue <= cfg.light_skin_hue_max) &
                 (sat >= cfg.light_skin_saturation_min) &
                 (val >= cfg.light_skin_value_min) &
                 (residual >= cfg.light_saturation_residual_min))
    candidate = candidate.astype(np.uint8) * 255
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # check each region
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (candidate > 0).astype(np.uint8), connectivity=8)
    image_h, image_w = candidate.shape
    cuff_side = _contact_side(segmentation.mask)

    locations: List[BBox] = []
    accepted_mask = np.zeros_like(candidate)
    strengths: List[float] = []
    notes: List[str] = []

    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if x <= 4 or y <= 4 or x + w >= image_w - 4 or y + h >= image_h - 4:
            continue
        if _inside_cuff_exclusion(
                x, y, w, h, image_w, image_h, cuff_side,
                cfg.light_cuff_exclusion_fraction):
            continue

        area_fraction = area / max(segmentation.area, 1.0)
        if not (cfg.light_min_component_area_fraction <= area_fraction <=
                cfg.light_max_component_area_fraction):
            continue
        if min(w, h) < cfg.light_min_short_side_pixels:
            continue

        member = labels == label
        strength = float(np.median(residual[member]))
        locations.append((x, y, w, h))
        accepted_mask[member] = 255
        strengths.append(strength)
        notes.append(
            f"local skin slit area={area_fraction:.2%}, "
            f"sat-contrast={strength:.1f}")

    if not locations:
        return DefectResult(
            False, "tearing", score=0.0,
            details=("no isolated skin-like slit passed the pale-glove "
                     "area/cuff filters"),
            mask=accepted_mask,
            methods=[
                "HSV skin-like colour threshold",
                "local saturation contrast",
                "morphological opening/closing",
                "connected-component area/shape filtering",
                "cuff-side exclusion",
            ],
            metrics={"candidate_components": max(count - 1, 0),
                     "accepted_regions": 0,
                     "cuff_side": cuff_side},
            parameters={
                "light_saturation_residual_min": cfg.light_saturation_residual_min,
                "light_min_component_area_fraction": cfg.light_min_component_area_fraction,
                "light_cuff_exclusion_fraction": cfg.light_cuff_exclusion_fraction,
            },
        )

    best_strength = max(strengths)
    area_score = min(1.0, max(
        (w * h) / max(segmentation.area, 1.0) for _, _, w, h in locations) / 0.005)
    contrast_score = min(1.0, best_strength / 40.0)
    score = min(1.0, 0.45 + 0.30 * area_score + 0.25 * contrast_score)
    return DefectResult(
        True, "tearing", locations, score,
        f"{len(locations)} pale-glove tear slit(s): " + "; ".join(notes),
        mask=accepted_mask,
        methods=[
            "HSV skin-like colour threshold",
            "local saturation contrast",
            "morphological opening/closing",
            "connected-component area/shape filtering",
            "cuff-side exclusion",
        ],
        metrics={
            "accepted_regions": len(locations),
            "max_local_saturation_contrast": best_strength,
            "cuff_side": cuff_side,
        },
        parameters={
            "light_saturation_residual_min": cfg.light_saturation_residual_min,
            "light_min_component_area_fraction": cfg.light_min_component_area_fraction,
            "light_cuff_exclusion_fraction": cfg.light_cuff_exclusion_fraction,
        },
    )


def _dark_glove_tears(image: np.ndarray, segmentation: SegmentationResult,
                      config: PipelineConfig) -> DefectResult:
    cfg = config.general_tearing
    base = config.tearing
    # dark glove method
    candidates = find_showthrough_patches(image, segmentation, config)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)


    _, palm_radius = palm_center_and_radius(segmentation.mask)
    margin = max(3, int(base.showthrough_margin_ratio * max(palm_radius, 1.0)))
    interior = cv2.erode(
        segmentation.mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))) > 0
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    _, ach, bch = cv2.split(lab)
    am, asp = robust_stats(ach[interior])
    bm, bsp = robust_stats(bch[interior])
    asp = max(asp, base.showthrough_mad_floor)
    bsp = max(bsp, base.showthrough_mad_floor)
    deviation = np.sqrt(((ach - am) / asp) ** 2 + ((bch - bm) / bsp) ** 2)
    raw = ((deviation > base.showthrough_z_threshold) & interior).astype(np.uint8) * 255
    raw = cv2.morphologyEx(
        raw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    locations: List[BBox] = []
    accepted_mask = np.zeros_like(raw)
    notes: List[str] = []
    best = 0.0

    for bbox, raw_conf, note in candidates:
        x, y, w, h = bbox
        area_fraction = (w * h) / max(segmentation.area, 1.0)
        if area_fraction < cfg.dark_min_bbox_area_fraction:
            continue

        member = raw[y:y+h, x:x+w] > 0
        if not np.any(member):
            continue
        hh = float(np.median(hue[y:y+h, x:x+w][member]))
        ss = float(np.median(sat[y:y+h, x:x+w][member]))
        vv = float(np.median(val[y:y+h, x:x+w][member]))
        if hh > cfg.dark_skin_hue_max:
            continue
        if not (cfg.dark_skin_saturation_min <= ss <= cfg.dark_skin_saturation_max):
            continue
        if vv < cfg.dark_skin_value_min:
            continue

        locations.append(bbox)
        accepted_mask[y:y+h, x:x+w][member] = 255
        confidence = min(
            1.0,
            0.55 + 0.25 * min(area_fraction / 0.025, 1.0)
            + 0.20 * min((vv - cfg.dark_skin_value_min) / 70.0, 1.0))
        best = max(best, confidence, raw_conf)
        notes.append(
            f"{note}; area={area_fraction:.2%}, H={hh:.0f}, "
            f"S={ss:.0f}, V={vv:.0f}")

    return DefectResult(
        bool(locations), "tearing", locations,
        best if locations else 0.0,
        (f"{len(locations)} dark-glove skin show-through tear(s): "
         + "; ".join(notes) if locations else
         f"0 accepted from {len(candidates)} colour-outlier candidate(s)"),
        mask=accepted_mask,
        methods=[
            "LAB chroma outlier / show-through detection",
            "morphological opening",
            "connected-component area filtering",
            "HSV skin-like appearance validation",
        ],
        metrics={"raw_candidates": len(candidates),
                 "accepted_regions": len(locations)},
        parameters={
            "showthrough_z_threshold": base.showthrough_z_threshold,
            "dark_min_bbox_area_fraction": cfg.dark_min_bbox_area_fraction,
            "dark_skin_value_min": cfg.dark_skin_value_min,
            "dark_skin_saturation_max": cfg.dark_skin_saturation_max,
        },
    )


def detect(image: np.ndarray, segmentation: SegmentationResult,
           config: PipelineConfig) -> DefectResult:
    interior = segmentation.mask > 0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    median_value = float(np.median(hsv[:, :, 2][interior])) if np.any(interior) else 0.0
    if median_value >= config.general_tearing.light_glove_value_cutoff:
        result = _light_glove_tears(image, segmentation, config)
        result.metrics["glove_median_value"] = median_value
        return result
    result = _dark_glove_tears(image, segmentation, config)
    result.metrics["glove_median_value"] = median_value
    return result
