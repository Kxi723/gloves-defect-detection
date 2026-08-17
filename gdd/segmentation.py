"""
Glove segmentation: separate the glove from an arbitrary background.

Robustness strategy
-------------------
We cannot assume a fixed glove or background colour (photos come from
different phones, rooms and desks), so instead of one hard-coded threshold
the module builds SEVERAL candidate masks with independent classical cues
and picks the most plausible one:

    A. Background-distance mask - sample the image border to estimate the
       background colour in LAB, threshold the per-pixel colour distance
       with Otsu. Works for any glove colour that differs from its backdrop.
    B. Saturation Otsu mask     - coloured gloves (rubber/latex) are more
                                  saturated than typical gray/white desks.
    C. Value Otsu masks         - light glove on dark background and the
                                  inverse, from the HSV value channel.

Each candidate is scored on (1) plausible area fraction and (2) how little
it touches the image border — a real glove is a compact object roughly in
frame, while a failed threshold selects the background itself.

The winner is cleaned with morphological opening/closing, the largest
connected contour is kept, and interior holes are filled to produce a solid
silhouette. Crucially the *unfilled* mask is preserved too: holes in it are
exactly what the tearing detectors look for.

Public API:
    segment_glove(image, config) -> SegmentationResult
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from gdd.config import SegmentationConfig


@dataclass
class SegmentationResult:
    """Everything downstream detectors need about the segmented glove."""

    mask: np.ndarray          # solid silhouette, holes filled (uint8, 0/255)
    mask_raw: np.ndarray      # cleaned mask BEFORE hole filling (uint8)
    contour: np.ndarray       # largest external contour, OpenCV point array
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) around the glove
    area: float               # contour area in pixels
    cue: str = "unknown"      # which candidate cue won (shown in reports)

    @property
    def holes_mask(self) -> np.ndarray:
        """Interior holes only: pixels solid in `mask` but empty in raw.

        A tear that goes through the glove exposes the background, so it
        appears as a hole here. Computed lazily because only the tearing
        detectors need it.
        """
        return cv2.subtract(self.mask, self.mask_raw)


# --------------------------------------------------------------------------- #
# Candidate mask generation
# --------------------------------------------------------------------------- #

def estimate_background_lab(lab: np.ndarray, border_fraction: float) -> np.ndarray:
    """Median LAB colour of the image border strip.

    The glove is photographed roughly centred, so the border is dominated
    by background. Median (not mean) keeps the estimate stable even when a
    finger crosses one edge.

    Public because detectors need it too: knowing what the backdrop looks
    like is how ``tearing_at_finger`` tells a view through a tear from a
    piece of backdrop that segmentation mistakenly swallowed.
    """
    h, w = lab.shape[:2]
    b = max(2, int(round(min(h, w) * border_fraction)))
    strip = np.concatenate(
        [
            lab[:b, :].reshape(-1, 3),    # top
            lab[-b:, :].reshape(-1, 3),   # bottom
            lab[:, :b].reshape(-1, 3),    # left
            lab[:, -b:].reshape(-1, 3),   # right
        ]
    )
    return np.median(strip, axis=0).astype(np.float32)


def _background_distance_mask(lab: np.ndarray, border_fraction: float) -> np.ndarray:
    """Threshold the LAB distance-to-background map with Otsu.

    LAB is perceptually uniform, so Euclidean distance approximates how
    different a pixel *looks* from the background — this handles any
    glove/background colour pair without per-colour tuning.
    """
    background = estimate_background_lab(lab, border_fraction)
    distance = np.linalg.norm(lab.astype(np.float32) - background, axis=2)
    distance_u8 = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    _, mask = cv2.threshold(distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _flatten_illumination(channel: np.ndarray,
                          border_fraction: float) -> np.ndarray:
    """Divide out a quadric illumination model fitted to the border strip.

    Otsu picks ONE global threshold, which is only meaningful when a given
    material has the same brightness everywhere in frame. Under a lamp or a
    phone's vignette it does not, so the threshold slices through the
    background itself and welds a ragged wedge of desk onto the silhouette.
    Measured on good_latex_4 that wedge inflated the palm radius from 176
    to 232 px, +32%, and since every detector threshold is a ratio of the
    palm radius it silently rescaled the whole pipeline.

    The border strip is background by construction (the assumption
    ``_estimate_background_lab`` already makes), so a surface fitted to it
    models the lighting without ever sampling the glove. A quadric rather
    than a plane because vignetting falls off in both axes at once.
    """
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
    model = np.maximum(model, 1.0)      # a near-zero fit would explode below
    flattened = channel.astype(np.float32) / model * float(np.median(model))
    return np.clip(flattened, 0, 255).astype(np.uint8)


def _background_model_mask(image: np.ndarray,
                           cfg: SegmentationConfig) -> np.ndarray:
    """Keep whatever is far from the background's own colour distribution.

    Every other cue asks "does the glove look like itself?", which fails on
    a latex-coated glove: it is a smooth dark blue coating AND a bright grey
    knit, and no single threshold holds both. This cue inverts the question.
    The DESK is the uniform thing, so model it from the border strip and
    keep what is unlike it. Both materials are unlike the desk, and neither
    has to resemble the other.

    Each LAB channel is normalised by its own robust spread, with a floor.
    The floor is not cosmetic: a grey desk has almost no chroma variation,
    so the raw MAD of a/b comes out near 1.0 and a blue pixel then scores 27
    sigma. That tail dragged Otsu up to a distance of 17 on good_latex_2
    while the knit sat at 8.6 and the desk at 3.6, cutting the whole glove
    away. Differences under a few LAB units are below perceptual noise
    anyway, so flooring costs nothing real.
    """
    border = _flatten_illumination  # (naming the dependency for readers)
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
    """Otsu masks from the HSV saturation and value channels.

    Saturation separates coloured gloves from neutral backgrounds; value
    (both polarities) separates light-on-dark and dark-on-light scenes.
    """
    masks: List[np.ndarray] = []
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    _, s_mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(s_mask)
    _, v_mask = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(v_mask)
    masks.append(cv2.bitwise_not(v_mask))
    return masks


# --------------------------------------------------------------------------- #
# Texture cue
#
# The four cues above are all COLOUR cues and share one blind spot: a glove
# whose own colours straddle the background defeats every one of them. A
# blue-and-white knitted cotton glove on a mid-grey desk is exactly that
# case, the white yarn lighter than the desk and the blue yarn darker, so
# any single threshold cuts through the middle of the glove. This cue asks
# whether the surface is TEXTURED instead, so it keeps working when no
# contrasting background is available.
# --------------------------------------------------------------------------- #

def _texture_energy_mask(image: np.ndarray, window: int) -> np.ndarray:
    """Otsu on the local standard deviation of lightness.

    Local standard deviation over a small window is high on woven or
    crinkled material and near zero on a smooth surface, so thresholding it
    separates fabric from furniture regardless of their colours. The energy
    map is blurred before thresholding so the response forms solid regions
    rather than a stipple that morphology would erase.
    """
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    mean = cv2.blur(lightness, (window, window))
    mean_of_squares = cv2.blur(lightness * lightness, (window, window))
    # Var(X) = E[X^2] - E[X]^2; clamped because rounding can make it < 0.
    std = np.sqrt(np.maximum(mean_of_squares - mean * mean, 0.0))
    std = cv2.GaussianBlur(std, (0, 0), window / 2.0)
    std_u8 = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(std_u8, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


# --------------------------------------------------------------------------- #
# Candidate scoring
# --------------------------------------------------------------------------- #

def _score_candidate(mask: np.ndarray, cfg: SegmentationConfig) -> float:
    """How plausible is this candidate as a glove silhouette? Higher is better.

    Scores the mask AFTER cleanup and largest-component selection, not the
    raw threshold output, because those can disagree sharply: on the cotton
    photos a saturation mask covering a plausible 16% of the frame
    shattered during cleanup into a largest piece of only 4.7%, yet judged
    raw it outscored a cue whose cleaned silhouette was a correct 62%.

    Four terms: reward low border contact and a moderate size, penalise a
    silhouette punched full of holes and one with a ragged outline.
    """
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
    # Fraction of the silhouette punched out by noise; a shredded mask is a
    # failed threshold even when its outline looks right.
    shred = np.count_nonzero(cv2.subtract(solid, component)) / max(solid_area, 1.0)
    hull_perimeter = max(cv2.arcLength(cv2.convexHull(contour), True), 1.0)
    raggedness = cv2.arcLength(contour, True) / hull_perimeter

    return ((1.0 - border_occupancy) * 2.0
            + (1.0 - abs(area_fraction - 0.30))
            - 4.0 * min(shred, 0.5)
            - 0.5 * max(0.0, raggedness - 1.6))


# --------------------------------------------------------------------------- #
# Mask cleanup
# --------------------------------------------------------------------------- #

def _morphological_cleanup(mask: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """Opening (remove speckle) followed by closing (bridge small gaps)."""
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
    """Keep only the largest external contour (the glove); drop clutter.

    Returns (solid filled mask, contour). Filling the external contour also
    closes interior holes, giving us the solid silhouette.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(mask), None
    largest = max(contours, key=cv2.contourArea)
    solid = np.zeros_like(mask)
    cv2.drawContours(solid, [largest], -1, 255, thickness=cv2.FILLED)
    return solid, largest


def _fill_noise_holes(mask_raw: np.ndarray, solid: np.ndarray,
                      cfg: SegmentationConfig, glove_area: float) -> np.ndarray:
    """Fill only *tiny* holes in the raw mask (segmentation noise).

    Genuine tears must survive into ``mask_raw`` for the defect detectors,
    but single-pixel salt noise inside the silhouette should not.
    """
    holes = cv2.subtract(solid, mask_raw)
    hole_contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    min_area = cfg.min_hole_area_fraction * glove_area
    result = mask_raw.copy()
    for hc in hole_contours:
        if cv2.contourArea(hc) < min_area:
            cv2.drawContours(result, [hc], -1, 255, thickness=cv2.FILLED)
    return result


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def segment_glove(
    image: np.ndarray, config: Optional[SegmentationConfig] = None
) -> Optional[SegmentationResult]:
    """Segment the glove in a *preprocessed* BGR image.

    Args:
        image:  Output of :func:`gdd.preprocessing.preprocess`.
        config: Tunables; defaults to :class:`SegmentationConfig` defaults.

    Returns:
        A :class:`SegmentationResult`, or ``None`` when no plausible glove
        region was found (callers should report a segmentation failure
        rather than run detectors on garbage).
    """
    cfg = config or SegmentationConfig()
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. Build candidate masks from independent cues and pick the best.
    #    The winning cue name is carried on the result so a report can show
    #    which one actually did the work on each photo.
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

    # The background model is tried FIRST rather than entered into the vote.
    # The scorer cannot be trusted to pick it: its size term peaks at 0.30
    # of the frame, so on good_latex_2 a mask that had lost the blue
    # fingertips (32.3%) outscored the complete silhouette (35.5%), and its
    # raggedness term penalises a contour for the long perimeter that
    # correctly resolving five finger gaps necessarily produces. Both terms
    # prefer a mask that has quietly dropped part of the glove. The vote is
    # kept as the fallback for photos where the background model itself
    # fails the plausibility check.
    primary = _background_model_mask(image, cfg)
    if _score_candidate(primary, cfg) >= 0:
        best_cue, best_mask = "bg_model", primary
    else:
        scored = [(_score_candidate(mask, cfg), name, mask)
                  for name, mask in candidates]
        best_score, best_cue, best_mask = max(scored,
                                              key=lambda triple: triple[0])
        if best_score < 0:
            return None  # every cue failed the plausibility checks

    # 2. Morphological cleanup, then isolate the glove blob.
    cleaned = _morphological_cleanup(best_mask, cfg)
    # Restrict to the largest connected component so desk clutter that
    # passed thresholding does not survive into the raw mask.
    num, labels = cv2.connectedComponents(cleaned)
    if num <= 1:
        return None
    largest_label = 1 + int(
        np.argmax([np.count_nonzero(labels == i) for i in range(1, num)])
    )
    component = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # 3. Solid silhouette + external contour.
    solid, contour = _keep_largest_component(component)
    if contour is None:
        return None
    area = cv2.contourArea(contour)
    if area < cfg.min_area_fraction * image.shape[0] * image.shape[1]:
        return None

    # 4. Keep real holes (potential tears), fill only noise-sized ones.
    mask_raw = _fill_noise_holes(component, solid, cfg, area)

    return SegmentationResult(
        mask=solid,
        mask_raw=mask_raw,
        contour=contour,
        bbox=cv2.boundingRect(contour),
        area=float(area),
        cue=best_cue,
    )
