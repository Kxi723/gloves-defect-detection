"""
Damage by fold — a permanent crease pressed across the palm.

The problem
-----------
A work glove is covered in detail at two very different scales. The
manufactured grip coating on latex is a dense, fine, high-frequency
crinkle. A fold is a broad ridge an order of magnitude wider. On top of
both sits a slow illumination gradient across the photo. Thresholding the
image directly finds all three at once and cannot tell them apart.

The method
----------
1. Equalise local contrast (CLAHE), so a faint crease in a dim corner is
   as visible as a bright one.
2. Band-pass the lightness with a difference of Gaussians. The fine sigma
   blurs the grip texture away, the coarse sigma removes the illumination
   gradient, and what survives in between is fold-sized by construction.
   Both sigmas are fractions of the palm radius, so the filter tracks
   glove size and image resolution.
3. Keep the strongest ridges by robust z-score, measured over the WHOLE
   glove — ridge strength has to be judged against this glove's normal
   texture.
4. Keep only components that are long and elongated: a crease is a line,
   a patch of shading is a blob.
5. Count only creases within ``palm_radius_ratio`` of the palm centre.

Step 5 is the one that makes this defect separable at all. Every handled
glove carries creases at the cuff and over the finger joints, so searching
the whole glove flagged all five undamaged nitrile photos. Restricting to
the palm cut false positives from 6 to 2 with no loss of true detections.

Measured on the 15-photo set: precision 67%, recall 80%.

Owner: Jason. Tunables live in ``PipelineConfig.fold``.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from gdd.config import FoldConfig, PipelineConfig
from gdd.features import (
    BBox, DefectResult, glove_interior, palm_center_and_radius, robust_stats,
)
from gdd.preprocessing import normalize_illumination
from gdd.segmentation import SegmentationResult


def fold_ridge_response(image: np.ndarray, interior: np.ndarray,
                        palm_radius: float, cfg: FoldConfig) -> np.ndarray:
    """Band-pass filtered lightness, isolating fold-scale structure.

    CLAHE is applied here and nowhere else in the pipeline; see
    ``FoldConfig.clahe_clip_limit`` for why this detector is the exception.
    """
    image = normalize_illumination(image, cfg.clahe_clip_limit, cfg.clahe_tile_grid)
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)

    fine_sigma = max(1.0, cfg.fine_sigma_ratio * palm_radius)
    coarse_sigma = max(fine_sigma + 1.0, cfg.coarse_sigma_ratio * palm_radius)
    response = (cv2.GaussianBlur(lightness, (0, 0), fine_sigma)
                - cv2.GaussianBlur(lightness, (0, 0), coarse_sigma))
    response[interior == 0] = 0.0
    return response


def detect(image: np.ndarray, segmentation: SegmentationResult,
           config: PipelineConfig) -> DefectResult:
    """Detect a fold crease across the palm. See the module docstring."""
    cfg = config.fold
    interior = glove_interior(segmentation, cfg.interior_margin_ratio)
    palm_center, palm_radius = palm_center_and_radius(segmentation.mask)
    if np.count_nonzero(interior) < 100 or palm_radius < 10:
        return DefectResult(False, "damage_by_fold",
                            details="glove interior too small to analyse")

    # Search only the palm; measure the reference over the whole glove.
    palm_disc = np.zeros_like(interior)
    cv2.circle(palm_disc, palm_center,
               int(cfg.palm_radius_ratio * palm_radius), 255, cv2.FILLED)
    palm_region = cv2.bitwise_and(interior, palm_disc)
    if np.count_nonzero(palm_region) < 100:
        return DefectResult(False, "damage_by_fold",
                            details="palm region too small to analyse")

    response = fold_ridge_response(image, interior, palm_radius, cfg)

    # Ridges AND valleys: a fold shows as a bright crest beside a dark
    # trough, and which one dominates depends on where the light is.
    _, spread = robust_stats(response[interior > 0])
    strong = ((np.abs(response) > cfg.z_threshold * spread)
              & (palm_region > 0)).astype(np.uint8) * 255

    # Close along the ridge so a crease broken by speckle becomes one
    # component, then drop isolated specks.
    strong = cv2.morphologyEx(strong, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    min_length = cfg.min_length_ratio * palm_radius
    creases: List[BBox] = []
    lengths: List[float] = []
    contours, _ = cv2.findContours(strong, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour) < 5:
            continue
        # A fitted ellipse gives length (major axis) and elongation directly.
        (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        major, minor = max(axis_a, axis_b), max(min(axis_a, axis_b), 1e-6)
        if major < min_length or major / minor < cfg.min_elongation:
            continue
        creases.append(cv2.boundingRect(contour))
        lengths.append(float(major))

    found = len(creases) >= cfg.min_crease_count
    longest = max(lengths) / palm_radius if lengths else 0.0
    return DefectResult(
        defect_found=found,
        defect_type="damage_by_fold",
        locations=creases if found else [],
        # A crease spanning 1.5 palm radii is treated as full confidence.
        score=min(1.0, longest / 1.5) if found else 0.0,
        details=(
            f"{len(creases)} palm crease(s), longest {longest:.2f}R, "
            f"ridge z>{cfg.z_threshold:g}"
            if creases else
            f"no palm crease longer than {cfg.min_length_ratio:g}R "
            f"(searched {cfg.palm_radius_ratio:g}R around the palm centre)"
        ),
    )
