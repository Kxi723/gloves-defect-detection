"""
Dirty — foreign matter sitting on the glove surface.

The problem
-----------
The obvious test, "find pixels that are much darker than the glove", does
not work. Every shadow is darker than the glove, and on a blue-and-white
knitted cotton glove the two yarn colours already span most of the
lightness range, so a colour-outlier test is simultaneously insensitive
(the stain is barely two robust sigma from the median) and full of false
alarms. Measured directly, that approach scored 0% precision and 0%
recall.

The method
----------
The decisive observation is about texture, not colour. Dirt lies ON TOP of
the material and hides its weave; a shadow dims the material without
hiding anything. So each candidate region is measured twice:

  * how far its lightness sits from the glove's own median (robust
    z-score, so the stain cannot bias its own reference), and
  * how much of the glove's normal texture survives inside it.

On the 15-photo set the two populations were nearly an order of magnitude
apart — real stains retained 0.06 to 0.12 of the surrounding texture,
shadows on undamaged gloves retained 0.92 to 1.54. The texture ratio is
what carries this detector; the lightness test only nominates candidates.

Note this runs on the CLAHE-free image. Equalising local contrast flattened
the stain from 2.75 robust sigma down to 1.2, below any usable threshold.

Measured on the 15-photo set: precision 83%, recall 100%.

Owner: Jason. Tunables live in ``PipelineConfig.dirt``.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from gdd.config import PipelineConfig
from gdd.features import (
    BBox, DefectResult, components_as_boxes, glove_interior,
    local_texture_energy, robust_stats,
)
from gdd.segmentation import SegmentationResult


def detect(image: np.ndarray, segmentation: SegmentationResult,
           config: PipelineConfig) -> DefectResult:
    """Detect dirt as a region that is off-colour AND has lost its texture."""
    cfg = config.dirt
    interior = glove_interior(segmentation, cfg.interior_margin_ratio)
    if np.count_nonzero(interior) < 100:
        return DefectResult(False, "dirty",
                            details="glove interior too small to analyse")

    selection = interior > 0
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    median, spread = robust_stats(lightness[selection])
    # Two-sided: dirt can be darker than a pale glove or lighter than a
    # dark one. The texture test below rejects the shadows this lets in.
    z_score = np.abs(lightness - median) / spread

    texture = local_texture_energy(image, cfg.texture_window)
    glove_texture = max(float(np.median(texture[selection])), 1e-6)

    candidate = ((z_score > cfg.z_threshold) & selection).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)

    locations: List[BBox] = []
    stained_area = 0.0
    ratios: List[float] = []
    for bbox, area, members in components_as_boxes(
            candidate, min_area=cfg.min_area_fraction * segmentation.area,
            min_extent=cfg.min_extent):
        texture_ratio = float(np.median(texture[members])) / glove_texture
        if texture_ratio > cfg.max_texture_ratio:
            continue  # texture survived, so the material is not covered
        locations.append(bbox)
        stained_area += area
        ratios.append(texture_ratio)

    stained_fraction = stained_area / segmentation.area
    return DefectResult(
        defect_found=bool(locations),
        defect_type="dirty",
        locations=locations,
        # 5% of the glove area is treated as full confidence.
        score=min(1.0, stained_fraction / 0.05) if locations else 0.0,
        details=(
            f"{len(locations)} dirty region(s), {stained_fraction:.2%} of the "
            f"glove, texture retained {min(ratios):.2f}x"
            if locations else
            f"no region that is both off-colour (z>{cfg.z_threshold:g}) and "
            f"texture-free (<{cfg.max_texture_ratio:g}x)"
        ),
    )
