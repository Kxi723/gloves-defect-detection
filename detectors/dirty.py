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
The lightness test only nominates candidates. What decides is whether the
region's appearance can be explained by the light falling on the glove, or
whether something foreign is lying there. Two independent signatures say
"foreign", and a region is accepted on either one.

**Texture collapse.** Fine or greasy dirt lies ON TOP of the material and
hides its weave; a shadow dims the material without hiding anything. On
the original 15-photo set the two populations were nearly an order of
magnitude apart — real stains retained 0.06 to 0.12 of the surrounding
texture, shadows on undamaged gloves retained 0.92 to 1.54.

**Off-hue colour.** Texture collapse assumes the dirt is fine enough to
fill the weave, and a coarse powder does the opposite: it heaps up, the
glove still shows between the grains, and the region measures as textured
as the glove around it. The cream coffee powder on ``dirty_latex_1``
retained 0.87, so the texture test alone rejected it. Hue still separates
it, because lighting can only slide a colour along the ray from neutral
grey through the surface's own hue — a highlight washes it toward neutral,
a shadow deepens it, and neither moves it sideways. Measured over the
candidate regions on the blue latex glove, the powder sat 34.7 off that
ray while the two specular highlights on undamaged gloves sat 5.1 and 6.4.
The ray has no direction on a near-neutral glove, so this route is gated
on the glove having some colour of its own; see
:func:`gdd.features.off_hue_distance`.

Note this runs on the CLAHE-free image. Equalising local contrast flattened
the stain from 2.75 robust sigma down to 1.2, below any usable threshold.

Measured on the 15-photo set: precision 83%, recall 100%.

Owner: Jason. Tunables live in ``PipelineConfig.dirt``.
"""

from __future__ import annotations

import math
from typing import List

import cv2
import numpy as np

from gdd.config import PipelineConfig
from gdd.features import (
    BBox, DefectResult, components_as_boxes, glove_interior, lab_chroma,
    local_texture_energy, median_chroma, off_hue_distance, robust_stats,
)
from gdd.segmentation import SegmentationResult


def detect(image: np.ndarray, segmentation: SegmentationResult,
           config: PipelineConfig) -> DefectResult:
    """Detect dirt as an off-colour region the lighting cannot account for.

    Nominated on lightness, then accepted on either of two signatures of
    foreign matter: the glove's texture has been covered over, or the
    region's hue has left the glove's own (see the module docstring).
    """
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

    chroma = lab_chroma(image)
    glove_chroma = median_chroma(chroma, selection)
    # A near-neutral glove gives the hue ray no direction, so the off-hue
    # route is unavailable and the texture route has to carry the photo.
    hue_route_available = math.hypot(*glove_chroma) > cfg.min_glove_chroma

    candidate = ((z_score > cfg.z_threshold) & selection).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)

    locations: List[BBox] = []
    stained_area = 0.0
    evidence: List[str] = []
    for bbox, area, members in components_as_boxes(
            candidate, min_area=cfg.min_area_fraction * segmentation.area,
            min_extent=cfg.min_extent):
        texture_ratio = float(np.median(texture[members])) / glove_texture
        off_hue = off_hue_distance(median_chroma(chroma, members), glove_chroma)

        covered = texture_ratio <= cfg.max_texture_ratio
        foreign_colour = hue_route_available and off_hue > cfg.min_off_hue_distance
        if not (covered or foreign_colour):
            continue  # texture survived AND the colour is just the light

        locations.append(bbox)
        stained_area += area
        evidence.append(f"texture {texture_ratio:.2f}x" if covered
                        else f"hue {off_hue:.0f} off the glove's own")

    stained_fraction = stained_area / segmentation.area
    return DefectResult(
        defect_found=bool(locations),
        defect_type="dirty",
        locations=locations,
        # 5% of the glove area is treated as full confidence.
        score=min(1.0, stained_fraction / 0.05) if locations else 0.0,
        details=(
            f"{len(locations)} dirty region(s), {stained_fraction:.2%} of the "
            f"glove ({'; '.join(evidence)})"
            if locations else
            f"no region that is off-colour (z>{cfg.z_threshold:g}) and either "
            f"texture-free (<{cfg.max_texture_ratio:g}x) or off-hue"
            + ("" if hue_route_available else "; glove too neutral for the hue test")
        ),
    )
