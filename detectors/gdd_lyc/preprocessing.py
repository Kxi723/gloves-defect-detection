from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from detectors.gdd_lyc.config import PreprocessConfig


@dataclass
class Preprocessed:

    bgr: np.ndarray
    lab: np.ndarray
    scale: float


def resize_to_working(image: np.ndarray, long_edge: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = long_edge / float(max(h, w))
    if scale >= 1.0:
        return image.copy(), 1.0
    size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def normalize_illumination(bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=(cfg.clahe_tile_grid, cfg.clahe_tile_grid),
    )
    return cv2.merge([clahe.apply(l), a, b])


def preprocess(image: np.ndarray, cfg: PreprocessConfig) -> Preprocessed:
    working, scale = resize_to_working(image, cfg.working_long_edge)
    denoised = cv2.medianBlur(working, cfg.median_ksize)
    lab = normalize_illumination(denoised, cfg)
    return Preprocessed(bgr=denoised, lab=lab, scale=scale)
