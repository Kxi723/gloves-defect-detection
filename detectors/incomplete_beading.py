from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np

from .TanYikTing_support.config import PipelineConfig
from .TanYikTing_support.preprocessing import preprocess
from .TanYikTing_support.features import BBox, DefectResult, palm_center_and_radius
from .TanYikTing_support.segmentation import (
    SegmentationResult,
    segment_glove as _segment_glove,
    find_forearm_component,
)


Config = PipelineConfig


def segment_glove(image: np.ndarray, config):
    return _segment_glove(image, config, mode="beading")


def _inner_profile(component: np.ndarray, side: str, target: int) -> np.ndarray:
    ys, xs = np.where(component > 0)
    if len(xs) < 10:
        return np.empty((0, 2), np.float32)

    pairs: List[Tuple[float, float]] = []
    if side in ("top", "bottom"):
        step = max(2, int((xs.max() - xs.min() + 1) / max(target, 1)))
        for x in range(int(xs.min()), int(xs.max()) + 1, step):
            yy = ys[xs == x]
            if len(yy):
                boundary = yy.max() if side == "top" else yy.min()
                pairs.append((float(x), float(boundary)))
    else:
        step = max(2, int((ys.max() - ys.min() + 1) / max(target, 1)))
        for y in range(int(ys.min()), int(ys.max()) + 1, step):
            xx = xs[ys == y]
            if len(xx):
                boundary = xx.max() if side == "left" else xx.min()
                pairs.append((float(y), float(boundary)))
    return np.asarray(pairs, np.float32)


def detect(image: np.ndarray, segmentation: SegmentationResult,
           config: PipelineConfig) -> DefectResult:
    cfg = config.beading
    h, w = segmentation.mask.shape

    side, component, bbox, contacts = find_forearm_component(
        image, segmentation.mask)
    if component is None or bbox is None:
        return DefectResult(
            False, "incomplete_beading",
            details="no forearm found near cuff",
            methods=[
                "HSV forearm segmentation",
                "cuff side selection",
                "cuff boundary curve fitting",
            ],
            metrics={"cuff_side": side, "frame_contacts": contacts},
            parameters={
                "skin_saturation_min": cfg.skin_saturation_min,
                "min_roughness_ratio": cfg.min_roughness_ratio,
            },
        )

    x, y, bw, bh = bbox
    depth_fraction = (bh / h) if side in ("top", "bottom") else (bw / w)
    profile = _inner_profile(component, side, cfg.profile_sample_target)
    _, palm_radius = palm_center_and_radius(segmentation.mask)

    if len(profile) < cfg.min_profile_points or palm_radius <= 1:
        return DefectResult(
            False, "incomplete_beading",
            details=f"only {len(profile)} cuff profile points",
            methods=[
                "HSV forearm segmentation",
                "cuff side selection",
                "cuff boundary curve fitting",
            ],
            metrics={
                "cuff_side": side,
                "profile_points": len(profile),
                "frame_contacts": contacts,
            },
            parameters={
                "skin_saturation_min": cfg.skin_saturation_min,
                "min_roughness_ratio": cfg.min_roughness_ratio,
            },
        )

    axis = profile[:, 0].astype(np.float64)
    boundary = profile[:, 1].astype(np.float64)
    coefficients = np.polyfit(axis, boundary, 2)
    fitted = np.polyval(coefficients, axis)
    residual = boundary - fitted

    roughness_pixels = float(np.std(residual))
    roughness_ratio = roughness_pixels / palm_radius
    p90_pixels = float(np.percentile(np.abs(residual), 90))

    depth_ok = cfg.min_cuff_depth_fraction <= depth_fraction <= cfg.max_cuff_depth_fraction
    rough_ok = roughness_ratio >= cfg.min_roughness_ratio
    found = depth_ok and rough_ok

    dmask = np.zeros((h, w), np.uint8)
    pts = []
    for a, b in profile:
        px, py = ((int(round(a)), int(round(b))) if side in ("top", "bottom")
                  else (int(round(b)), int(round(a))))
        pts.append((px, py))
        cv2.circle(dmask, (px, py), cfg.boundary_thickness, 255, -1)

    if found and pts:
        arr = np.asarray(pts)
        pad = 10
        x1 = max(0, int(arr[:, 0].min()) - pad)
        y1 = max(0, int(arr[:, 1].min()) - pad)
        x2 = min(w, int(arr[:, 0].max()) + pad)
        y2 = min(h, int(arr[:, 1].max()) + pad)
        locations: List[BBox] = [(x1, y1, x2 - x1, y2 - y1)]

        rough_strength = min(
            1.0, roughness_ratio / cfg.full_confidence_roughness_ratio)
        depth_mid = (
            cfg.min_cuff_depth_fraction + cfg.max_cuff_depth_fraction) / 2.0
        depth_half = (
            cfg.max_cuff_depth_fraction - cfg.min_cuff_depth_fraction) / 2.0
        depth_strength = max(
            0.0,
            1.0 - abs(depth_fraction - depth_mid) / max(depth_half, 1e-6),
        )
        score = min(
            1.0, 0.55 + 0.35 * rough_strength + 0.10 * depth_strength)
    else:
        locations = []
        score = 0.0
        dmask[:] = 0

    return DefectResult(
        found, "incomplete_beading", locations, score,
        (f"cuff side={side}; forearm depth={depth_fraction:.1%}; "
         f"profile roughness={roughness_ratio:.3f}R "
         f"(std={roughness_pixels:.1f}px, p90={p90_pixels:.1f}px)"),
        mask=dmask,
        methods=[
            "HSV forearm segmentation",
            "cuff side selection",
            "inner cuff boundary extraction",
            "quadratic cuff profile fitting",
            "normalised roughness measurement",
        ],
        metrics={
            "cuff_side": side,
            "forearm_depth_fraction": depth_fraction,
            "profile_points": len(profile),
            "roughness_pixels_std": roughness_pixels,
            "roughness_ratio_to_palm": roughness_ratio,
            "profile_p90_deviation_pixels": p90_pixels,
            "frame_contacts": contacts,
        },
        parameters={
            "skin_saturation_min": cfg.skin_saturation_min,
            "min_cuff_depth_fraction": cfg.min_cuff_depth_fraction,
            "max_cuff_depth_fraction": cfg.max_cuff_depth_fraction,
            "min_roughness_ratio": cfg.min_roughness_ratio,
        },
    )
