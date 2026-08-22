from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from detectors.gdd_lyc.config import SegmentationConfig
from detectors.gdd_lyc.preprocessing import Preprocessed
from detectors.gdd_lyc.segmentation import (_to_u8, candidate_cues, estimate_background_lab,
                                            otsu_split)


@dataclass
class SceneParts:

    glove: np.ndarray
    skin: np.ndarray
    ruler: np.ndarray
    ruler_width: float
    ok: bool
    note: str = ""


def _foreground(pre: Preprocessed, cfg: SegmentationConfig) -> np.ndarray:
    bg = estimate_background_lab(pre.lab, cfg)
    best_eta, best = -1.0, None
    for cue in candidate_cues(pre.lab, bg).values():
        gray = _to_u8(cue)
        t, eta = otsu_split(gray)
        if eta > best_eta:
            best_eta, best = eta, cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)[1]
    return cv2.morphologyEx(best, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def _drop_table_edge(fg: np.ndarray) -> np.ndarray:
    height, width = fg.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    out = np.zeros_like(fg)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < 0.004 * fg.size:
            continue
        if (stats[i, cv2.CC_STAT_TOP] <= 2
                and stats[i, cv2.CC_STAT_WIDTH] > 0.9 * width):
            continue
        out[labels == i] = 255
    return out


def find_ruler(fg: np.ndarray) -> tuple[float, np.ndarray] | None:
    height, width = fg.shape
    rows, widths, centres = [], [], []
    for y in range(int(0.05 * height), int(0.98 * height)):
        xs = np.nonzero(fg[y])[0]
        if xs.size == 0:
            continue
        end = int(xs[-1])
        if end >= width - 2:
            continue
        present = set(xs.tolist())
        start = end
        while start - 1 in present:
            start -= 1
        rows.append(y)
        widths.append(end - start + 1)
        centres.append((start + end) / 2.0)

    if len(rows) < 50:
        return None
    rows_a = np.array(rows, float)
    widths_a = np.array(widths, float)
    centres_a = np.array(centres, float)
    ruler_width = float(np.median(widths_a))
    clean = np.abs(widths_a - ruler_width) <= 0.25 * ruler_width
    if clean.sum() < 40:
        return None

    if ruler_width > 0.25 * width:
        return None
    if clean.mean() < 0.45:
        return None
    span = (rows_a[clean].max() - rows_a[clean].min()) / float(height)
    if span < 0.35:
        return None

    slope, intercept = np.polyfit(rows_a[clean], centres_a[clean], 1)
    band = np.zeros_like(fg)
    for y in range(height):
        cx = slope * y + intercept
        x0 = int(round(cx - ruler_width / 2 - 2))
        x1 = int(round(cx + ruler_width / 2 + 2))
        band[y, max(0, x0):min(width, x1 + 1)] = 255
    return ruler_width, cv2.bitwise_and(band, fg)


def split_scene(pre: Preprocessed, cfg: SegmentationConfig,
                require_ruler: bool = True) -> SceneParts:
    fg = _drop_table_edge(_foreground(pre, cfg))
    found = find_ruler(fg)
    if found is None:
        if require_ruler:
            empty = np.zeros_like(fg)
            return SceneParts(fg, empty, empty, 0.0, False,
                              "no reference ruler found")
        ruler_width, ruler = 0.0, np.zeros_like(fg)
        note = "no ruler; cuff-to-wrist fit ratio is still measurable"
    else:
        ruler_width, ruler = found
        note = ""

    rest = cv2.bitwise_and(fg, cv2.bitwise_not(ruler))
    rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(rest, 8)
    if count <= 1:
        empty = np.zeros_like(fg)
        return SceneParts(empty, empty, ruler, ruler_width, False, "no hand found")
    hand = np.where(labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])),
                    255, 0).astype(np.uint8)

    inside = hand > 0
    ys, _ = np.nonzero(inside)
    y0, y1 = int(ys.min()), int(ys.max())
    top = inside.copy()
    top[y0 + int(0.30 * (y1 - y0 + 1)):] = False
    lab = pre.lab.astype(np.float32)
    ref = np.array([np.median(lab[..., c][top]) for c in range(3)])
    dist = np.sqrt(((lab - ref) ** 2).sum(axis=2))
    scaled = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thr, _ = cv2.threshold(scaled[inside].reshape(-1, 1), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    far = ((scaled > thr) & inside).astype(np.uint8) * 255
    far = cv2.morphologyEx(far, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(far, 8)
    skin = np.zeros_like(fg)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < 0.01 * fg.size:
            continue
        if np.where(labels == i, 1, 0)[-1].any():
            skin[labels == i] = 255

    glove = cv2.bitwise_and(hand, cv2.bitwise_not(skin))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(glove, 8)
    if count > 1:
        glove = np.where(
            labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])),
            255, 0).astype(np.uint8)
    return SceneParts(glove, skin, ruler, ruler_width, True, note)


def measure(parts: SceneParts) -> dict[str, float]:
    contour = max(cv2.findContours(parts.glove, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)[0],
                  key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    unit = parts.ruler_width
    return {
        "ruler_width_px": unit,
        "length": max(w, h) / unit,
        "width": min(w, h) / unit,
        "area": float((parts.glove > 0).sum()) / (unit * unit),
    }
