"""
Preprocessing: make phone photos comparable before segmentation.

Phone photos of gloves vary wildly in resolution, colour cast and lighting.
This module normalises all three so the downstream stages can use stable
thresholds:

    1. Resize        - bound the longest side (speed + scale-stable params).
    2. White balance - gray-world assumption removes lighting colour casts.
    3. Denoise       - bilateral filter (edge-preserving, unlike Gaussian).

CLAHE is available as :func:`normalize_illumination` but is NOT in the
chain; see :func:`preprocess` for why, and `detectors/damage_by_fold.py`
for the one place that opts into it.

Public API:
    preprocess(image, config)                 -> normalized BGR (uint8)
    normalize_illumination(image, clip, tile) -> CLAHE'd BGR (uint8)
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from gdd.config import PreprocessConfig


def resize_to_limit(image: np.ndarray, max_dimension: int) -> np.ndarray:
    """Downscale so the longest side is at most ``max_dimension`` pixels.

    Never upscales (that would only interpolate noise). INTER_AREA is the
    recommended interpolation for shrinking as it averages source pixels.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return image
    scale = max_dimension / longest
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    """Gray-world white balance.

    Assumes the average colour of a scene is neutral gray; any deviation is
    attributed to the illuminant and divided out per channel. This is a
    classical technique that counters e.g. the yellow cast of indoor bulbs,
    which would otherwise shift the glove's hue between photos.
    """
    img = image.astype(np.float32)
    channel_means = img.reshape(-1, 3).mean(axis=0)          # B, G, R means
    gray_mean = float(channel_means.mean())
    # Guard against a degenerate all-black channel (division by ~zero).
    gains = gray_mean / np.maximum(channel_means, 1e-6)
    balanced = img * gains.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)


def normalize_illumination(
    image: np.ndarray, clip_limit: float, tile_grid: int
) -> np.ndarray:
    """Apply CLAHE to the L channel of LAB colour space.

    LAB separates lightness (L) from colour (A, B), so equalising only L
    flattens shadows/highlights while leaving glove colour — which the
    segmentation stage relies on — untouched. CLAHE works on local tiles
    with a clip limit, avoiding the noise blow-up of global equalisation.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid)
    )
    l_equalized = clahe.apply(l_channel)
    lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))
    return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)


def preprocess(
    image: np.ndarray, config: Optional[PreprocessConfig] = None
) -> np.ndarray:
    """Normalise a photo: resize, white balance, denoise.

    CLAHE is deliberately NOT part of this chain. Local contrast
    equalisation is exactly the wrong operation for everything downstream,
    and it broke three stages before that was understood: it lifted crease
    highlights on a wrinkled glove to near-background brightness, punching
    16-24 spurious holes into masks that otherwise segmented with 0-1, and
    it flattened a black stain on knitted cotton from 2.75 robust sigma
    down to 1.2, below any usable threshold.

    The one detector that genuinely wants equalised contrast — fold, which
    measures relative brightness ridges — calls
    :func:`normalize_illumination` itself. Each detector declares the
    normalisation it needs rather than sharing one compromise.

    Args:
        image:  Input photo as a BGR ``uint8`` array (as read by
                ``cv2.imread``).
        config: Tunables; defaults to :class:`PreprocessConfig` defaults.

    Returns:
        Normalised BGR ``uint8`` image, possibly downscaled.
    """
    cfg = config or PreprocessConfig()
    if image is None or image.size == 0:
        raise ValueError("preprocess() received an empty image")
    if image.ndim == 2:  # tolerate grayscale input
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    result = resize_to_limit(image, cfg.max_dimension)
    if cfg.white_balance:
        result = gray_world_white_balance(result)
    return cv2.bilateralFilter(
        result,
        d=cfg.bilateral_diameter,
        sigmaColor=cfg.bilateral_sigma_color,
        sigmaSpace=cfg.bilateral_sigma_space,
    )
