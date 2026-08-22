"""Build the glove silhouette used by TS's three detectors.

Robustness strategy
-------------------
Instead of relying on one glove colour, the module builds candidate masks
from image cues taught in the module and selects the most plausible silhouette:

    A. HSI saturation mask - separates a coloured glove from a neutral desk.
    B. HSI intensity masks - handles both light-on-dark and dark-on-light.
    C. Texture mask        - local HSI-intensity variation for knitted or
                             textured glove surfaces.

Every candidate is thresholded by the iterative Basic Global Thresholding
method from the segmentation slides.

Each candidate is scored on (1) plausible area fraction and (2) how little
it touches the image border — a real glove is a compact object roughly in
frame, while a failed threshold selects the background itself.

The winner is cleaned with morphological opening/closing and reduced to its
largest connected contour, which is filled to produce a solid silhouette.

Public API:
    segment_glove(image, config) -> SegmentationResult
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import SegmentationConfig


@dataclass
class SegmentationResult:
    """Everything downstream detectors need about the segmented glove."""

    mask: np.ndarray          # solid silhouette (uint8, 0/255)
    contour: np.ndarray       # largest external contour, OpenCV point array
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) around the glove
    area: float               # contour area in pixels
    cue: str = "unknown"      # which candidate cue won (shown in reports)
    source_image: Optional[np.ndarray] = None  # resized image before preprocessing


# --------------------------------------------------------------------------- #
# Candidate mask generation
# --------------------------------------------------------------------------- #

def hsi_saturation_and_intensity(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the HSI saturation and intensity formulas from the slides."""
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
    """Threshold one channel by the iterative two-group mean method."""
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
    """Basic-global masks from HSI saturation and intensity."""
    return [
        basic_global_threshold(saturation, tolerance=1.0 / 255.0),
        basic_global_threshold(intensity),
        basic_global_threshold(intensity, invert=True),
    ]


# --------------------------------------------------------------------------- #
# Texture cue
#
# The HSI cues above share one blind spot: a glove
# whose own colours straddle the background defeats every one of them. A
# blue-and-white knitted cotton glove on a mid-grey desk is exactly that
# case, the white yarn lighter than the desk and the blue yarn darker, so
# any single threshold cuts through the middle of the glove. This cue asks
# whether the surface is TEXTURED instead, so it keeps working when no
# contrasting background is available.
# --------------------------------------------------------------------------- #

def _texture_energy_mask(intensity: np.ndarray, window: int) -> np.ndarray:
    """Basic-global threshold of local HSI-intensity variation.

    Local standard deviation over a small window is high on woven or
    crinkled material and near zero on a smooth surface, so thresholding it
    separates fabric from furniture regardless of their colours. The energy
    map is blurred before thresholding so the response forms solid regions
    rather than a stipple that morphology would erase.
    """
    mean = cv2.blur(intensity, (window, window))
    mean_of_squares = cv2.blur(intensity * intensity, (window, window))
    # Var(X) = E[X^2] - E[X]^2; clamped because rounding can make it < 0.
    std = np.sqrt(np.maximum(mean_of_squares - mean * mean, 0.0))
    std = cv2.GaussianBlur(std, (0, 0), window / 2.0)
    return basic_global_threshold(std)


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


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def segment_glove(
    image: np.ndarray, config: Optional[SegmentationConfig] = None
) -> Optional[SegmentationResult]:
    """Segment the glove in a *preprocessed* BGR image.

    Args:
        image:  Output of :func:`ts_support.preprocessing.preprocess`.
        config: Tunables; defaults to :class:`SegmentationConfig` defaults.

    Returns:
        A :class:`SegmentationResult`, or ``None`` when no plausible glove
        region was found (callers should report a segmentation failure
        rather than run detectors on garbage).
    """
    cfg = config or SegmentationConfig()
    saturation, intensity = hsi_saturation_and_intensity(image)

    # 1. Build candidate masks from independent cues and pick the best.
    #    The winning cue name is carried on the result so a report can show
    #    which one actually did the work on each photo.
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

    return SegmentationResult(
        mask=solid,
        contour=contour,
        bbox=cv2.boundingRect(contour),
        area=float(area),
        cue=best_cue,
    )
