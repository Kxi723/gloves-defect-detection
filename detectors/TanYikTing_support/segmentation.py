from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import SegmentationConfig


@dataclass
class SegmentationResult:
    mask: np.ndarray
    mask_raw: np.ndarray
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    area: float
    cue: str = "unknown"

    @property
    def holes_mask(self) -> np.ndarray:
        return cv2.subtract(self.mask, self.mask_raw)


def estimate_background_lab(lab: np.ndarray, border_fraction: float) -> np.ndarray:
    h, w = lab.shape[:2]
    b = max(2, int(round(min(h, w) * border_fraction)))
    strip = np.concatenate([
        lab[:b, :].reshape(-1, 3),
        lab[-b:, :].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3),
        lab[:, -b:].reshape(-1, 3),
    ])
    return np.median(strip, axis=0).astype(np.float32)


def _background_surface(image: np.ndarray, border_fraction: float):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    b = max(3, int(round(min(h, w) * border_fraction)))

    border = np.zeros((h, w), bool)
    border[:b, :] = True
    border[-b:, :] = True
    border[:, :b] = True
    border[:, -b:] = True

    ys, xs = np.where(border)
    values = lab[border]
    median = np.median(values, axis=0)
    chroma = np.linalg.norm(values[:, 1:] - median[1:], axis=1)
    light = np.abs(values[:, 0] - median[0])
    keep = (chroma <= np.percentile(chroma, 70)) & (light <= np.percentile(light, 75))
    if np.count_nonzero(keep) < 300:
        keep[:] = True

    x = xs[keep].astype(np.float32) / max(w - 1, 1)
    y = ys[keep].astype(np.float32) / max(h - 1, 1)
    design = np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], axis=1)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xx / max(w - 1, 1)
    yn = yy / max(h - 1, 1)
    full = np.stack([np.ones_like(xn), xn, yn, xn * xn, yn * yn, xn * yn], axis=-1)

    model = np.zeros_like(lab)
    for ch in range(3):
        coef, *_ = np.linalg.lstsq(design, values[keep, ch], rcond=None)
        model[:, :, ch] = np.tensordot(full, coef, axes=([2], [0]))
    return lab, model


def _texture(image: np.ndarray, window: int) -> np.ndarray:
    l = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    mean = cv2.blur(l, (window, window))
    mean2 = cv2.blur(l * l, (window, window))
    return np.sqrt(np.maximum(mean2 - mean * mean, 0.0))


def _cleanup(mask: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    op = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel))
    cl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, op)
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, cl)


def _largest(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    solid = np.zeros_like(mask)
    cv2.drawContours(solid, [contour], -1, 255, cv2.FILLED)
    return solid, contour


def _candidate(mask: np.ndarray, cue: str, cfg: SegmentationConfig):
    cleaned = _cleanup(mask, cfg)
    count, labels, stats, centres = cv2.connectedComponentsWithStats(cleaned)
    if count <= 1:
        return None

    h, w = cleaned.shape
    best = None
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 0.03 * h * w:
            continue
        cx, cy = centres[label]
        centre_penalty = ((cx - w / 2) / w) ** 2 + ((cy - h / 2) / h) ** 2
        score = area / float(h * w) - 0.20 * centre_penalty
        if best is None or score > best[0]:
            best = (score, label)
    if best is None:
        return None

    raw = np.where(labels == best[1], 255, 0).astype(np.uint8)
    solid, contour = _largest(raw)
    if contour is None:
        return None
    area = float(cv2.contourArea(contour))
    frac = area / float(h * w)
    if not (cfg.min_area_fraction <= frac <= cfg.max_area_fraction):
        return None
    return SegmentationResult(solid, raw, contour, cv2.boundingRect(contour), area, cue)


def _tip_count(mask: np.ndarray, contour: np.ndarray) -> int:
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, radius, _, centre = cv2.minMaxLoc(dist)
    if radius <= 1:
        return 0

    hull = cv2.convexHull(contour)
    points = []
    for p in hull[:, 0]:
        x, y = int(p[0]), int(p[1])
        d = float(np.hypot(x - centre[0], y - centre[1]))
        if d >= 1.35 * radius:
            points.append((d, (x, y)))
    points.sort(reverse=True)

    kept = []
    gap = 0.45 * radius
    for _, p in points:
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= gap for q in kept):
            kept.append(p)
        if len(kept) >= 5:
            break
    return len(kept)


def _quality(image: np.ndarray, seg, residual_l: np.ndarray,
             residual_c: np.ndarray, texture: np.ndarray) -> float:
    if seg is None:
        return -999.0

    h, w = image.shape[:2]
    active = seg.mask > 0
    if not np.any(active):
        return -999.0

    shadow = (residual_l < -18) & (residual_c < 3.5)
    smooth_bg = (residual_c < 2.8) & (texture < np.percentile(texture, 45))
    shadow_frac = float(np.mean(shadow[active]))
    bg_frac = float(np.mean(smooth_bg[active]))
    border_count = (np.count_nonzero(active[0, :]) + np.count_nonzero(active[-1, :])
                    + np.count_nonzero(active[:, 0]) + np.count_nonzero(active[:, -1]))
    border_frac = border_count / max(2.0 * (h + w), 1.0)

    x, y, bw, bh = seg.bbox
    span = 0.5 * (bw / w + bh / h)
    tips = _tip_count(seg.mask, seg.contour)
    area_frac = seg.area / float(h * w)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 40, 120), np.ones((3, 3), np.uint8))
    boundary = np.zeros((h, w), np.uint8)
    cv2.drawContours(boundary, [seg.contour], -1, 255, 2)
    edge_support = np.count_nonzero((boundary > 0) & (edges > 0)) / max(np.count_nonzero(boundary), 1)

    return (
        1.2 * span
        + 0.13 * min(tips, 5)
        + 0.8 * edge_support
        - 3.0 * shadow_frac
        - 2.0 * bg_frac
        - 5.0 * border_frac
        - 1.2 * abs(area_frac - 0.28)
    )


def _skin_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    hue, sat, val = cv2.split(hsv)
    _, cr, cb = cv2.split(ycc)

    mask = (
        (hue <= 25)
        & (sat >= 25)
        & (sat <= 180)
        & (val >= 75)
        & (cr >= 133)
        & (cr <= 180)
        & (cb >= 70)
        & (cb <= 135)
    ).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))


def find_forearm_component(image: np.ndarray, glove_mask: np.ndarray):
    skin = _skin_mask(image)
    h, w = glove_mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skin)

    pad = max(9, int(round(0.035 * min(h, w))))
    near = cv2.dilate(
        glove_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1)),
    ) > 0

    best = None
    for label in range(1, count):
        x, y, bw, bh, area = map(int, stats[label])
        if area < 0.003 * h * w:
            continue
        comp = labels == label
        close = int(np.count_nonzero(comp & near))
        if close < max(20, int(0.002 * h * w)):
            continue

        contacts = {
            "top": int(np.count_nonzero(comp[0, :])),
            "bottom": int(np.count_nonzero(comp[-1, :])),
            "left": int(np.count_nonzero(comp[:, 0])),
            "right": int(np.count_nonzero(comp[:, -1])),
        }
        side = max(contacts, key=contacts.get)
        border = contacts[side]
        if border <= 0:
            continue

        score = close + 0.25 * area + 20.0 * border
        if best is None or score > best[0]:
            best = (score, side, comp, (x, y, bw, bh), contacts)

    if best is not None:
        _, side, comp, bbox, contacts = best
        return side, comp.astype(np.uint8) * 255, bbox, contacts

    contours, _ = cv2.findContours(glove_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "bottom", None, None, {"top": 0, "bottom": 0, "left": 0, "right": 0}

    x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    gaps = {"top": y, "bottom": h - y - bh, "left": x, "right": w - x - bw}
    side = min(gaps, key=gaps.get)
    return side, None, None, {"top": 0, "bottom": 0, "left": 0, "right": 0}


def _trim_arm(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    side, component, _, _ = find_forearm_component(image, mask)
    if component is None:
        return mask

    h, w = mask.shape
    out = mask.copy()
    skin = component > 0

    if side in ("top", "bottom"):
        ratios = np.zeros(h, np.float32)
        for y in range(h):
            row = mask[y] > 0
            ratios[y] = np.count_nonzero(skin[y] & row) / max(np.count_nonzero(row), 1)
        order = range(h - 1, -1, -1) if side == "bottom" else range(h)
    else:
        ratios = np.zeros(w, np.float32)
        for x in range(w):
            col = mask[:, x] > 0
            ratios[x] = np.count_nonzero(skin[:, x] & col) / max(np.count_nonzero(col), 1)
        order = range(w - 1, -1, -1) if side == "right" else range(w)

    seen = False
    streak = 0
    cut = None
    for pos in order:
        ratio = float(ratios[pos])
        if ratio > 0.45:
            seen = True
            streak = 0
        elif seen and ratio < 0.20:
            streak += 1
            if streak >= 8:
                cut = pos
                break
        else:
            streak = 0

    if cut is None:
        return out

    margin = max(2, int(round(0.012 * (h if side in ("top", "bottom") else w))))
    if side == "bottom":
        out[min(h, cut + 8 + margin):, :] = 0
    elif side == "top":
        out[:max(0, cut - 8 - margin), :] = 0
    elif side == "right":
        out[:, min(w, cut + 8 + margin):] = 0
    else:
        out[:, :max(0, cut - 8 - margin)] = 0
    return out



def _grow_light_mask(base: np.ndarray, residual_l: np.ndarray,
                     residual_c: np.ndarray) -> np.ndarray:
    allowed = (((residual_c > 2.2) & (residual_l > -15.0))
               | ((residual_l > 20.0) & (residual_c > 1.7)))
    allowed &= ~((residual_l < -12.0) & (residual_c < 3.2))
    allowed = allowed.astype(np.uint8) * 255
    allowed = cv2.morphologyEx(
        allowed, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    out = base.copy()
    near_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for _ in range(3):
        near = cv2.dilate(out, near_kernel) > 0
        out = np.where((out > 0) | (near & (allowed > 0)), 255, 0).astype(np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, close_kernel)
    return out

def _edge_repair(image: np.ndarray, base: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 25, 80)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = base.shape
    repaired = np.zeros_like(base)
    near = cv2.dilate(base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))

    for contour in contours:
        area = cv2.contourArea(contour)
        if not (0.01 * h * w <= area <= 0.45 * h * w):
            continue
        temp = np.zeros_like(base)
        cv2.drawContours(temp, [contour], -1, 255, cv2.FILLED)
        if np.any((temp > 0) & (near > 0)):
            repaired = cv2.bitwise_or(repaired, temp)

    merged = cv2.bitwise_or(base, repaired)
    return cv2.morphologyEx(merged, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))


def _fill_small_holes(raw: np.ndarray, solid: np.ndarray,
                      cfg: SegmentationConfig, area: float) -> np.ndarray:
    holes = cv2.subtract(solid, raw)
    contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = raw.copy()
    limit = cfg.min_hole_area_fraction * area
    for contour in contours:
        if cv2.contourArea(contour) < limit:
            cv2.drawContours(out, [contour], -1, 255, cv2.FILLED)
    return out


def _finalise(mask: np.ndarray, cue: str, cfg: SegmentationConfig):
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels, stats, centres = cv2.connectedComponentsWithStats(cleaned)
    if count <= 1:
        return None

    h, w = cleaned.shape
    best = None
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 0.02 * h * w:
            continue
        cx, cy = centres[label]
        centre_penalty = ((cx - w / 2) / w) ** 2 + ((cy - h / 2) / h) ** 2
        score = area / float(h * w) - 0.20 * centre_penalty
        if best is None or score > best[0]:
            best = (score, label)
    if best is None:
        return None

    raw = np.where(labels == best[1], 255, 0).astype(np.uint8)
    solid, contour = _largest(raw)
    if contour is None:
        return None

    area = float(cv2.contourArea(contour))
    if area < cfg.min_area_fraction * h * w:
        return None

    raw = _fill_small_holes(raw, solid, cfg, area)
    return SegmentationResult(solid, raw, contour, cv2.boundingRect(contour), area, cue)


def segment_glove(image: np.ndarray, config: Optional[SegmentationConfig] = None,
                  mode: str = "default") -> Optional[SegmentationResult]:
    cfg = config or SegmentationConfig()
    lab, model = _background_surface(image, cfg.illumination_border_fraction)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    texture = _texture(image, cfg.texture_window)

    residual_l = lab[:, :, 0] - model[:, :, 0]
    residual_c = np.linalg.norm(lab[:, :, 1:] - model[:, :, 1:], axis=2)

    centre = hsv[
        image.shape[0] // 3:2 * image.shape[0] // 3,
        image.shape[1] // 3:2 * image.shape[1] // 3,
        2,
    ]
    light_glove = float(np.median(centre)) > 105.0

    abs_score = np.sqrt((residual_l / 8.0) ** 2 + (residual_c / 4.0) ** 2)
    abs_u8 = cv2.normalize(abs_score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, abs_mask = cv2.threshold(abs_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if light_glove:
        signed = (((residual_l > 12.0) & (residual_c > 2.5))
                  | ((residual_c > 4.5) & (residual_l > -12.0)))
        signed &= ~((residual_c < 3.0) & (texture < np.percentile(texture, 50)))
    else:
        signed = ((residual_l < -28.0) | ((residual_c > 5.0) & (residual_l < -8.0)))
    signed = signed.astype(np.uint8) * 255

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    _, s_mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, v_mask = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    masks = [
        ("signed", signed),
        ("background", abs_mask),
        ("saturation", s_mask),
        ("value", v_mask),
        ("value_inverted", cv2.bitwise_not(v_mask)),
    ]

    candidates = []
    for name, mask in masks:
        seg = _candidate(mask, name, cfg)
        if seg is not None:
            candidates.append((name, seg, _quality(image, seg, residual_l, residual_c, texture)))
    if not candidates:
        return None

    candidates.sort(key=lambda x: x[2], reverse=True)
    name, selected, _ = candidates[0]

    raw = selected.mask_raw.copy()
    area_fraction = selected.area / float(image.shape[0] * image.shape[1])
    if not light_glove:
        raw = _trim_arm(raw, image)
    elif mode == "beading":
        raw = _edge_repair(image, raw)
    elif area_fraction < 0.18:
        raw = _grow_light_mask(raw, residual_l, residual_c)

    result = _finalise(raw, name, cfg)
    return result if result is not None else selected
