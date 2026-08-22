from __future__ import annotations
from typing import List
import textwrap
import cv2
import numpy as np


def _locations(result):
    return list(getattr(result, "locations", []) or [])


# build defect mask
def result_mask(shape, result) -> np.ndarray:
    h, w = shape[:2]
    detector_mask = getattr(result, "mask", None)
    if detector_mask is None:
        detector_mask = getattr(result, "debug_mask", None)
    if isinstance(detector_mask, np.ndarray) and detector_mask.shape[:2] == (h, w):
        if detector_mask.ndim == 3:
            detector_mask = cv2.cvtColor(detector_mask, cv2.COLOR_BGR2GRAY)
        return np.where(detector_mask > 0, 255, 0).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    for x, y, bw, bh in _locations(result):
        cv2.rectangle(mask, (int(x), int(y)),
                      (int(x + bw), int(y + bh)), 255, cv2.FILLED)
    return mask


def _format_value(value) -> str:
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if 0 <= value <= 1:
            return f"{value:.4f} ({value:.1%})"
        return f"{value:.4f}"
    return str(value)


# build explain text
def analysis_lines(report, result, detector_key: str, label: str) -> List[str]:
    boxes = ", ".join(str(tuple(map(int, b))) for b in _locations(result)) or "none"
    found = bool(getattr(result, "defect_found", False))
    details = str(getattr(result, "details", "") or "none")
    score = float(getattr(result, "score", 0.0) or 0.0)

    lines = [
        "FINAL LABEL", label if found else f"No {label}", "",
        "DETECTOR FILE", f"detectors/{detector_key}.py", "",
        "LOCATION / BOXES", boxes, "", "METHODS USED",
    ]
    methods = list(getattr(result, "methods", []) or [])
    if methods:
        lines.extend([f"- {method}" for method in methods])
    else:
        lines.append("- detector-specific classical image-processing rules")

    lines.extend(["", "MEASURED FEATURES"])
    seg = getattr(report, "segmentation", None)
    if seg is not None:
        cue = getattr(seg, "cue", "unknown")
        area = getattr(seg, "area", None)
        bbox = getattr(seg, "bbox", None)
        lines.append(f"segmentation_cue: {cue}")
        if area is not None:
            lines.append(f"glove_area_pixels: {int(area)}")
        if bbox is not None:
            lines.append(f"glove_bbox: {tuple(map(int, bbox))}")

    metrics = getattr(result, "metrics", None) or getattr(result, "measurements", None) or {}
    for key, value in metrics.items():
        lines.append(f"{key}: {_format_value(value)}")
    lines.append(f"detector_score: {score:.3f} ({score:.1%})")
    elapsed = float(getattr(report, "elapsed_seconds", 0.0) or 0.0)
    lines.append(f"elapsed_seconds: {elapsed:.3f}")

    lines.extend(["", "PARAMETERS THAT AFFECT THE DECISION"])
    parameters = getattr(result, "parameters", None) or {}
    if parameters:
        for key, value in parameters.items():
            lines.append(f"{key}: {_format_value(value)}")
    else:
        threshold = getattr(result, "threshold", None)
        if threshold is not None:
            lines.append(f"decision_threshold: {_format_value(threshold)}")
        else:
            lines.append("see the detector configuration for its thresholds")

    lines.extend(["", "DECISION EVIDENCE", details, "", "CAPTURE WARNINGS"])
    warnings = list(getattr(report, "warnings", []) or [])
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- none"])
    lines.extend(["", "FINAL CONFIDENCE SCORE", f"{score:.1%}"])
    return lines


def _fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 248, np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


# draw explain view
def render_explainable_report(original: np.ndarray, report, result,
                              detector_key: str, label: str,
                              annotated: np.ndarray) -> np.ndarray:
    panel_w, panel_h = 500, 350
    gap, title_h = 18, 34
    left_w = panel_w * 2 + gap
    right_w = 760
    top_h = 56
    total_h = max(1040, top_h + (panel_h + title_h) * 2 + gap + 24)
    canvas = np.full((total_h, left_w + gap + right_w + 36, 3), 255, np.uint8)
    cv2.putText(canvas, "2. Processing stages", (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (45, 45, 45), 2, cv2.LINE_AA)
    cv2.putText(canvas, "3. Explainable result", (left_w + gap + 18, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (45, 45, 45), 2, cv2.LINE_AA)

    target_h, target_w = annotated.shape[:2]
    orig = cv2.resize(original, (target_w, target_h), interpolation=cv2.INTER_AREA)
    seg = getattr(report, "segmentation", None)
    glove_mask = getattr(seg, "mask_raw", None) if seg is not None else None
    if glove_mask is None and seg is not None:
        glove_mask = getattr(seg, "mask", None)
    if glove_mask is None:
        glove_mask = np.zeros((target_h, target_w), np.uint8)
    dmask = result_mask((target_h, target_w), result)
    panels = [("Original", orig), ("Glove mask", glove_mask),
              ("Defect mask", dmask), ("Annotated result", annotated)]

    for i, (title, image) in enumerate(panels):
        row, col = divmod(i, 2)
        x = 18 + col * (panel_w + gap)
        y = top_h + row * (panel_h + title_h + gap)
        cv2.putText(canvas, title, (x, y + 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (55, 55, 55), 1, cv2.LINE_AA)
        fitted = _fit_panel(image, panel_w, panel_h)
        canvas[y + title_h:y + title_h + panel_h, x:x + panel_w] = fitted
        cv2.rectangle(canvas, (x, y + title_h),
                      (x + panel_w, y + title_h + panel_h), (220, 220, 220), 1)

    tx, ty = left_w + gap + 18, top_h + 12
    for raw in analysis_lines(report, result, detector_key, label):
        wrapped = [""] if raw == "" else (textwrap.wrap(raw, width=74,
                   subsequent_indent="  ") or [""])
        for line in wrapped:
            if ty > total_h - 20:
                break
            cv2.putText(canvas, line, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, (35, 35, 35), 1, cv2.LINE_AA)
            ty += 22
    return canvas
