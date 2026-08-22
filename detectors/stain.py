from __future__ import annotations

import cv2
import numpy as np

from detectors.gdd_lyc.config import PipelineConfig
from detectors.gdd_lyc.features import (BBox, DefectResult, components, interior_mask,
                                        masked_local_mean)
from detectors.gdd_lyc.preprocessing import preprocess
from detectors.gdd_lyc.segmentation import segment_glove

NAME = "stain"


def detect(image: np.ndarray, config: PipelineConfig | None = None) -> DefectResult:
    cfg = config or PipelineConfig()
    pre = preprocess(image, cfg.preprocess)
    seg = segment_glove(pre.lab, cfg.segmentation)

    analysis_mask = seg.mask if seg and hasattr(seg, "mask") else None

    if not seg.ok:
        return DefectResult(
            name=NAME,
            found=False,
            score=0.0,
            assessed=False,
            note=seg.note or "segmentation unreliable; not judged",
            analysis_mask=analysis_mask,
        )

    long_edge = max(pre.bgr.shape[:2])

    interior = interior_mask(seg.mask, cfg.features.interior_erode_frac, long_edge)
    inside = interior > 0
    if inside.sum() < 500:
        return DefectResult(
            name=NAME,
            found=False,
            score=0.0,
            assessed=False,
            note="glove interior too small to measure",
            analysis_mask=analysis_mask,
        )

    lightness = pre.lab[:, :, 0].astype(np.float32)
    window = max(3, int(cfg.stain.local_window_frac * long_edge) | 1)
    baseline = masked_local_mean(lightness, interior, window)
    residual = np.abs(lightness - baseline)

    spread = float(residual[inside].std())
    if spread < 1e-6:
        return DefectResult(
            name=NAME,
            found=False,
            score=0.0,
            assessed=False,
            note="glove is perfectly flat; nothing to measure",
            analysis_mask=analysis_mask,
        )

    cut = cfg.stain.residual_sigma * spread
    candidates = ((residual > cut) & inside).astype(np.uint8) * 255
    candidates = cv2.morphologyEx(
        candidates, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    area = float(inside.sum())
    kept = [c for c in components(candidates, area,
                                  cfg.stain.min_component_area_frac)
            if c.elongation <= cfg.stain.max_elongation
            and c.fill >= cfg.stain.min_fill]

    score = sum(c.area_frac for c in kept)
    glove_median = float(np.median(lightness[inside]))
    bright_region = ((lightness > glove_median + 50.0) & inside).astype(np.uint8) * 255
    bright_region = cv2.morphologyEx(
        bright_region, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    bright_components = [c for c in components(bright_region, area,
                                               cfg.stain.min_component_area_frac)
                         if c.fill >= cfg.stain.min_bright_fill]
    largest_bright_component = max(
        bright_components, key=lambda component: component.area, default=None)
    largest_bright = (largest_bright_component.area_frac
                      if largest_bright_component is not None else 0.0)
    large_stain = largest_bright >= 0.05
    if large_stain:
        score = max(score, largest_bright)
    found = score >= cfg.stain.min_stain_area_frac
    boxes = [c.box for c in kept] if found else []
    if found and large_stain and largest_bright_component is not None:
        boxes.append(largest_bright_component.box)

    return DefectResult(
        name=NAME,
        found=found,
        score=score,
        threshold=cfg.stain.min_stain_area_frac,
        boxes=boxes,
        measurements={
            "stain_area_frac": score,
            "verdict_threshold": cfg.stain.min_stain_area_frac,
            "regions_kept": float(len(kept)),
            "residual_sigma_px": spread,
            "residual_cut": cut,
            "largest_bright_region_frac": largest_bright,
            "interior_area_frac": area / inside.size,
        },
        note=("compact lightness anomalies exceed the measured threshold" if found
              else "no compact lightness anomaly above the measured threshold"),
        analysis_mask=analysis_mask,
    )
