from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass
class DefectResult:

    name: str
    found: bool
    score: float
    boxes: list[BBox] = field(default_factory=list)
    measurements: dict[str, float] = field(default_factory=dict)
    note: str = ""

    assessed: bool = True

    threshold: float | None = None

    analysis_mask: np.ndarray | None = None
    debug_mask: np.ndarray | None = None

    raw_score: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.raw_score = self.score
        if self.threshold is not None and self.threshold > 0:
            raw = max(0.0, self.raw_score)
            self.score = raw / (raw + self.threshold)

    @property
    def defect_found(self) -> bool:
        return self.found

    @property
    def defect_type(self) -> str:
        return self.name

    @property
    def locations(self) -> list[tuple[int, int, int, int]]:
        return [b.as_tuple() if hasattr(b, "as_tuple") else b for b in self.boxes]

    @property
    def details(self) -> str:
        if self.note:
            return self.note
        if self.measurements:
            meas_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in self.measurements.items())
            return f"score={self.score:.2f} ({meas_str})"
        return f"score={self.score:.2f}"



@dataclass
class Component:

    area: int
    box: BBox
    area_frac: float
    elongation: float
    fill: float


def interior_mask(mask: np.ndarray, erode_frac: float, long_edge: int) -> np.ndarray:
    k = max(3, int(erode_frac * long_edge) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask, se)


def masked_local_mean(channel: np.ndarray, mask: np.ndarray, ksize: int) -> np.ndarray:
    m = (mask > 0).astype(np.float32)
    ch = channel.astype(np.float32)
    k = max(3, int(ksize) | 1)
    num = cv2.blur(ch * m, (k, k))
    den = cv2.blur(m, (k, k))
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), ch)


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def components(binary: np.ndarray, region_area: float,
               min_area_frac: float = 0.0) -> list[Component]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out: list[Component] = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        frac = area / max(region_area, 1.0)
        if frac < min_area_frac:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        long_side, short_side = max(w, h), max(1, min(w, h))
        out.append(Component(
            area=area,
            box=BBox(x, y, w, h),
            area_frac=frac,
            elongation=long_side / short_side,
            fill=area / float(max(1, w * h)),
        ))
    return out


@dataclass
class Tile:

    box: BBox
    weber: float


def tile_surface(lightness: np.ndarray, interior: np.ndarray, tile_px: int,
                 min_cover: float) -> list[Tile]:
    grad = gradient_magnitude(lightness)
    inside = interior > 0
    height, width = lightness.shape
    tiles: list[Tile] = []

    for y in range(0, height - tile_px + 1, tile_px):
        for x in range(0, width - tile_px + 1, tile_px):
            patch_mask = inside[y:y + tile_px, x:x + tile_px]
            if patch_mask.mean() < min_cover:
                continue
            patch_light = lightness[y:y + tile_px, x:x + tile_px][patch_mask]
            patch_grad = grad[y:y + tile_px, x:x + tile_px][patch_mask]
            mean_light = float(patch_light.mean())
            if mean_light < 1.0:
                continue
            tiles.append(Tile(box=BBox(x, y, tile_px, tile_px),
                              weber=float(patch_grad.mean() / mean_light)))
    return tiles
