from __future__ import annotations

import numpy as np

from detectors.gdd_lyc.config import PipelineConfig
from detectors.gdd_lyc.features import (DefectResult, gradient_magnitude, interior_mask,
                                        tile_surface)
from detectors.gdd_lyc.preprocessing import preprocess
from detectors.gdd_lyc.reference import split_scene
from detectors.gdd_lyc.segmentation import segment_glove

NAME = "wrinkle"


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

    parts = split_scene(pre, cfg.segmentation, require_ruler=False)
    region = parts.glove if (parts.ok and (parts.glove > 0).sum() > 500) else seg.mask

    interior = interior_mask(region, cfg.features.interior_erode_frac, long_edge)
    inside = interior > 0

    if not inside.any():
        return DefectResult(
            name=NAME,
            found=False,
            score=0.0,
            assessed=False,
            note="glove interior empty after erosion",
            analysis_mask=analysis_mask,
        )

    lightness = pre.lab[..., 0].astype(np.float32)
    mean_l = float(lightness[inside].mean())
    if mean_l < 1.0:
        return DefectResult(
            name=NAME,
            found=False,
            score=0.0,
            assessed=False,
            note="glove too dark to measure contrast",
            analysis_mask=analysis_mask,
        )

    glove_wide_weber = float(gradient_magnitude(lightness)[inside].mean() / mean_l)

    tile_px = max(8, int(cfg.wrinkle.tile_frac * long_edge))
    tiles = tile_surface(lightness, interior, tile_px, cfg.wrinkle.tile_min_cover)
    if not tiles:
        return DefectResult(
            name=NAME,
            found=False,
            score=0.0,
            assessed=False,
            note="glove too small to tile at this resolution",
            analysis_mask=analysis_mask,
        )

    creased = [t for t in tiles if t.weber >= cfg.wrinkle.tile_weber_threshold]
    creased_fraction = len(creased) / len(tiles)
    found = creased_fraction >= cfg.wrinkle.wrinkled_tile_frac

    boxes = [t.box for t in creased] if found else []

    note = (
        ""
        if found
        else (
            "below threshold -- detection validated on nitrile (6 positives) and "
            "latex (2 positives), both directions. Cotton is NOT separable "
            "by surface texture and needs a shape measure. See WrinkleConfig"
        )
    )

    return DefectResult(
        name=NAME,
        found=found,
        score=creased_fraction,
        threshold=cfg.wrinkle.wrinkled_tile_frac,
        boxes=boxes,
        measurements={
            "creased_tile_frac": creased_fraction,
            "threshold": cfg.wrinkle.wrinkled_tile_frac,
            "tiles_total": float(len(tiles)),
            "tiles_creased": float(len(creased)),
            "glove_wide_weber": glove_wide_weber,
            "mean_lightness": mean_l,
            "interior_area_frac": float(inside.sum()) / inside.size,
        },
        note=note,
        analysis_mask=analysis_mask,
    )
