from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from detectors.gdd_lyc.config import SegmentationConfig


@dataclass
class SegmentationResult:
    mask: np.ndarray
    cue: str
    separability: float
    threshold: int
    background_lab: np.ndarray
    area_frac: float
    ok: bool
    note: str = ""


def otsu_split(gray: np.ndarray) -> tuple[int, float]:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1.0)
    levels = np.arange(256, dtype=np.float64)

    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    cum_mean = np.cumsum(p * levels)
    total_mean = cum_mean[-1]

    valid = (w0 > 1e-9) & (w1 > 1e-9)
    sigma_b = np.zeros(256, dtype=np.float64)
    sigma_b[valid] = (
        (total_mean * w0[valid] - cum_mean[valid]) ** 2 / (w0[valid] * w1[valid])
    )

    t = int(np.argmax(sigma_b))
    total_var = float((p * (levels - total_mean) ** 2).sum())
    eta = float(sigma_b[t] / total_var) if total_var > 1e-9 else 0.0
    return t, eta


def _to_u8(f: np.ndarray) -> np.ndarray:
    return cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def estimate_background_lab(lab: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    h, w = lab.shape[:2]
    b = max(4, int(cfg.border_band_frac * min(h, w)))
    band = np.concatenate([
        lab[:b].reshape(-1, 3),
        lab[-b:].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3),
        lab[:, -b:].reshape(-1, 3),
    ])
    return np.median(band, axis=0)


def candidate_cues(lab: np.ndarray, bg: np.ndarray) -> dict[str, np.ndarray]:
    lab_f = lab.astype(np.float32)
    l, a, b = lab_f[..., 0], lab_f[..., 1], lab_f[..., 2]

    delta_e = np.sqrt((l - bg[0]) ** 2 + (a - bg[1]) ** 2 + (b - bg[2]) ** 2)
    chroma = np.sqrt((a - bg[1]) ** 2 + (b - bg[2]) ** 2)
    lightness = np.abs(l - bg[0])

    return {"lab_distance": delta_e, "chroma": chroma, "lightness": lightness}


def _ellipse(frac: float, long_edge: int) -> np.ndarray:
    k = max(3, int(frac * long_edge) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    scratch = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, scratch, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, float]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask), 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    winner = 1 + int(np.argmax(areas))
    keep = np.where(labels == winner, 255, 0).astype(np.uint8)
    return keep, float(areas.max()) / float(mask.size)


def segment_glove(lab: np.ndarray, cfg: SegmentationConfig) -> SegmentationResult:
    long_edge = max(lab.shape[:2])
    bg = estimate_background_lab(lab, cfg)

    best_name, best_mask, best_eta, best_t = None, None, -1.0, 0
    for name, cue in candidate_cues(lab, bg).items():
        gray = _to_u8(cue)
        t, eta = otsu_split(gray)
        if eta > best_eta:
            _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
            best_name, best_mask, best_eta, best_t = name, binary, eta, t

    opened = cv2.morphologyEx(
        best_mask, cv2.MORPH_OPEN, _ellipse(cfg.open_ksize_frac, long_edge))
    closed = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, _ellipse(cfg.close_ksize_frac, long_edge))

    glove, area_frac = _largest_component(closed)
    glove = fill_holes(glove)

    ragged = 0.0
    outline, _ = cv2.findContours(glove, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if outline:
        biggest = max(outline, key=cv2.contourArea)
        hull = max(cv2.arcLength(cv2.convexHull(biggest), True), 1.0)
        ragged = cv2.arcLength(biggest, True) / hull

    note = ""
    if area_frac < cfg.min_glove_area_frac:
        note = f"glove covers only {area_frac * 100:.1f}% of the frame"
    elif best_eta < cfg.min_separability:
        note = f"no cue separated cleanly (best eta {best_eta:.2f})"
    elif ragged > cfg.max_contour_raggedness:
        note = (f"outline is not one solid object (raggedness {ragged:.1f}, limit "
                f"{cfg.max_contour_raggedness:.1f}): the glove did not separate from "
                f"this background. Use a backdrop far in colour from the glove.")
    ok = not note
    return SegmentationResult(
        mask=glove,
        cue=best_name or "none",
        separability=best_eta,
        threshold=best_t,
        background_lab=bg,
        area_frac=area_frac,
        ok=ok,
        note=note,
    )
