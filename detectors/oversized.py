from __future__ import annotations

import cv2
import numpy as np

from detectors.gdd_lyc.config import PipelineConfig
from detectors.gdd_lyc.features import BBox, DefectResult
from detectors.gdd_lyc.preprocessing import preprocess
from detectors.gdd_lyc.reference import split_scene
from detectors.gdd_lyc.segmentation import segment_glove

NAME = "oversized"


def _span(mask: np.ndarray, y: int) -> int:
    xs = np.nonzero(mask[y])[0]
    return int(xs[-1] - xs[0] + 1) if xs.size >= 2 else 0


def detect(image: np.ndarray, config: PipelineConfig | None = None) -> DefectResult:
    cfg = config or PipelineConfig()
    pre = preprocess(image, cfg.preprocess)
    seg = segment_glove(pre.lab, cfg.segmentation)

    analysis_mask = seg.mask if seg and hasattr(seg, "mask") else None

    if not seg.ok:
        return DefectResult(
            name=NAME,
            found=False,
            score=0.0,
            assessed=False,
            note=seg.note or "segmentation unreliable; not judged",
            analysis_mask=analysis_mask,
        )

    parts = split_scene(pre, cfg.segmentation, require_ruler=False)
    skin_y, _ = np.nonzero(parts.skin)
    if not parts.ok or skin_y.size < 12:
        return DefectResult(
            name=NAME,
            found=False,
            score=0.0,
            assessed=False,
            note="not assessed: cuff and bare wrist must both be visible",
            analysis_mask=analysis_mask,
        )

    cuff_y = int(skin_y.min())
    skin_end = int(skin_y.max())
    band = max(6, int(cfg.oversize.cuff_band_frac * max(pre.bgr.shape[:2])))

    hand = cv2.bitwise_or(parts.glove, parts.skin)
    cuff_rows = range(max(0, cuff_y - band), min(hand.shape[0], cuff_y + 2))
    cuff_spans = [_span(hand, y) for y in cuff_rows]
    cuff_spans = [v for v in cuff_spans if v > 0]
    if not cuff_spans:
        return DefectResult(
            name=NAME,
            found=False,
            score=0.0,
            assessed=False,
            note="not assessed: cuff boundary not measurable",
            analysis_mask=analysis_mask,
        )
    cuff_width = float(np.percentile(cuff_spans, 90))

    wrist_rows = range(min(skin_end, cuff_y + band),
                       min(skin_end + 1, cuff_y + max(2 * band, 18)))
    wrist_widths = [int(np.count_nonzero(parts.skin[y])) for y in wrist_rows]
    wrist_widths = [v for v in wrist_widths if v > 0]
    if not wrist_widths:
        return DefectResult(
            name=NAME,
            found=False,
            score=0.0,
            assessed=False,
            note="not assessed: visible wrist is too short",
            analysis_mask=analysis_mask,
        )
    wrist_width = float(np.median(wrist_widths))
    ratio = cuff_width / wrist_width
    found = ratio >= cfg.oversize.min_cuff_to_wrist_ratio

    ys = list(cuff_rows)
    xs = np.nonzero(hand[min(ys):max(ys) + 1])[1]
    boxes = []
    if found and xs.size:
        boxes = [BBox(int(xs.min()), min(ys), int(xs.max() - xs.min() + 1),
                      int(max(ys) - min(ys) + 1))]

    return DefectResult(
        name=NAME,
        found=found,
        score=ratio,
        threshold=cfg.oversize.min_cuff_to_wrist_ratio,
        boxes=boxes,
        measurements={
            "cuff_outer_width_px": cuff_width,
            "wrist_width_px": wrist_width,
            "cuff_to_wrist_ratio": ratio,
            "verdict_threshold": cfg.oversize.min_cuff_to_wrist_ratio,
        },
        note=("cuff opening is wider than the permitted wrist ratio" if found
              else "cuff is proportionate to the visible wrist"),
        analysis_mask=analysis_mask,
    )
