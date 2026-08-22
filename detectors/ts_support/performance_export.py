

from __future__ import annotations

import cv2
import numpy as np


def _display_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _panel(image: np.ndarray, title: str, width: int, height: int) -> np.ndarray:

    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    header = max(40, round(height * 0.05))
    panel = np.full((header + height, width, 3), 255, np.uint8)
    panel[header:] = image
    cv2.putText(
        panel,
        title,
        (10, round(header * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.55, min(0.85, width / 650)),
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        panel,
        (0, header),
        (width - 1, header + height - 1),
        (210, 210, 210),
        1,
    )
    return panel


def build_performance_image(
    original: np.ndarray,
    glove_mask: np.ndarray,
    defect_mask: np.ndarray,
    detected: np.ndarray,
    image_name: str,
    status: str,
    score: float,
) -> np.ndarray:

    height, width = detected.shape[:2]
    stages = [
        _panel(original, "1. Original", width, height),
        _panel(
            _display_mask(glove_mask, width, height),
            "2. Glove mask",
            width,
            height,
        ),
        _panel(
            _display_mask(defect_mask, width, height),
            "3. Defect evidence mask",
            width,
            height,
        ),
        _panel(detected, "4. Detected result", width, height),
    ]

    gap = max(44, round(width * 0.07))
    footer = max(48, round(height * 0.06))
    stage_height = stages[0].shape[0]
    canvas_width = width * len(stages) + gap * (len(stages) - 1)
    canvas = np.full((stage_height + footer, canvas_width, 3), 255, np.uint8)

    x = 0
    arrow_y = stage_height // 2
    for index, stage in enumerate(stages):
        canvas[:stage_height, x:x + width] = stage
        x += width
        if index < len(stages) - 1:
            cv2.arrowedLine(
                canvas,
                (x + 8, arrow_y),
                (x + gap - 8, arrow_y),
                (230, 95, 30),
                2,
                cv2.LINE_AA,
                tipLength=0.28,
            )
            x += gap

    cv2.putText(
        canvas,
        f"{image_name}   |   {status}   |   score {score:.3f}",
        (10, stage_height + round(footer * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.55, min(0.8, canvas_width / 3600)),
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )
    return canvas
