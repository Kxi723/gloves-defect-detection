"""PPT-supported preprocessing used by TS's three detector workflows.

The input is resized for bounded processing cost and denoised with a small
median filter. The detector then measures colours relative to the current
image instead of applying separate global colour correction.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import PreprocessConfig


def resize_to_limit(image: np.ndarray, max_dimension: int) -> np.ndarray:
    """Downscale the longest side to ``max_dimension`` without upscaling."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return image
    scale = max_dimension / longest
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def preprocess(
    image: np.ndarray, config: Optional[PreprocessConfig] = None
) -> np.ndarray:
    """Resize and median-filter one BGR image."""
    cfg = config or PreprocessConfig()
    if image is None or image.size == 0:
        raise ValueError("preprocess() received an empty image")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    result = resize_to_limit(image, cfg.max_dimension)
    kernel = max(3, int(cfg.median_kernel))
    if kernel % 2 == 0:
        kernel += 1
    return cv2.medianBlur(result, kernel)
