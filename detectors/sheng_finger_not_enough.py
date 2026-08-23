from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .ts_support.config import PipelineConfig, get_config
from .ts_support.features import BBox, DefectResult
from .ts_support.preprocessing import preprocess, resize_to_limit
from .ts_support.segmentation import segment_glove

def _odd_kernel_size(value: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(value)))
    return size if size % 2 == 1 else size + 1


def detect(
    image: np.ndarray,
    config: PipelineConfig | None = None,
) -> DefectResult:
    
    config = config or get_config()
    source_image = resize_to_limit(image, config.preprocess.max_dimension)
    image = preprocess(image, config.preprocess)
    cfg = config.finger_not_enough
    skin_cfg = config.skin_colour
    independent_segmentation = segment_glove(image, config.segmentation)
    if independent_segmentation is None:
        return DefectResult(
            False,
            "finger_not_enough",
            details="independent Finger Not Enough glove mask failed",
            debug_mask=np.zeros(image.shape[:2], dtype=np.uint8),
            analysis_mask=np.zeros(image.shape[:2], dtype=np.uint8),
        )
    independent_segmentation.source_image = source_image
    segmentation = independent_segmentation
    rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB).astype(np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    channel_range = (
        np.maximum.reduce((red, green, blue))
        - np.minimum.reduce((red, green, blue))
    )

    skin_pixels = (
        (red > skin_cfg.red_min)
        & (green > skin_cfg.green_min)
        & (blue > skin_cfg.blue_min)
        & (red > green)
        & (green > blue)
        & ((red - green) > skin_cfg.red_green_difference_min)
        & ((red - green) < skin_cfg.red_green_difference_max)
        & ((red - blue) > skin_cfg.red_blue_difference_min)
        & (channel_range > skin_cfg.channel_range_min)
        & (
            ((red + green + blue) / 3)
            < skin_cfg.mean_brightness_max
        )
    )
    skin_mask = np.where(skin_pixels, 255, 0).astype(np.uint8)
    skin_mask = cv2.medianBlur(skin_mask, 5)
    clean_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, clean_element)
    skin_mask = cv2.morphologyEx(
        skin_mask, cv2.MORPH_CLOSE, clean_element, iterations=2
    )

    glove_x, glove_y, glove_width, glove_height = segmentation.bbox
    cuff_start = glove_y + round(glove_height * cfg.cuff_start_fraction)
    skin_mask[max(cuff_start, 0):, :] = 0

    contact_size = _odd_kernel_size(
        min(source_image.shape[:2]) * cfg.contact_kernel_fraction
    )
    contact_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (contact_size, contact_size)
    )
    near_glove = cv2.dilate(segmentation.mask, contact_element)
    skin_mask[near_glove == 0] = 0

    count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        skin_mask, connectivity=8
    )
    minimum_component_area = max(
        cfg.min_component_pixels,
        round(segmentation.area * cfg.min_component_fraction),
    )
    locations: List[BBox] = []
    kept_mask = np.zeros_like(skin_mask)
    exposed_area = 0.0
    for label in range(1, count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if area < minimum_component_area:
            continue
        component_x = int(statistics[label, cv2.CC_STAT_LEFT])
        component_y = int(statistics[label, cv2.CC_STAT_TOP])
        component_width = int(statistics[label, cv2.CC_STAT_WIDTH])
        component_height = int(statistics[label, cv2.CC_STAT_HEIGHT])
        component_fraction = area / max(segmentation.area, 1.0)
        relative_x = (component_x - glove_x) / max(glove_width, 1)
        relative_y = (component_y - glove_y) / max(glove_height, 1)
        relative_width = component_width / max(glove_width, 1)
        relative_height = component_height / max(glove_height, 1)
        component_aspect = component_width / max(component_height, 1)
        at_fingertip = (
            relative_y <= cfg.fingertip_top_fraction
            or relative_x <= cfg.fingertip_side_fraction
            or relative_x + relative_width
            >= 1.0 - cfg.fingertip_side_fraction
        )
        plausible_shape = (
            component_fraction <= cfg.max_component_fraction
            and cfg.min_component_width_fraction
            <= relative_width
            <= cfg.max_component_width_fraction
            and relative_height <= cfg.max_component_height_fraction
            and cfg.min_component_aspect
            <= component_aspect
            <= cfg.max_component_aspect
        )
        if not (at_fingertip and plausible_shape):
            continue
        kept_mask[labels == label] = 255
        locations.append((
            component_x,
            component_y,
            component_width,
            component_height,
        ))
        exposed_area += area

    exposed_fraction = exposed_area / max(segmentation.area, 1.0)

    found = exposed_fraction >= cfg.min_exposed_area_fraction
    if not found:
        locations = []
    return DefectResult(
        defect_found=found,
        defect_type="finger_not_enough",
        locations=locations,
        score=min(1.0, exposed_fraction / cfg.area_score_scale)
        if found else 0.0,
        details=(
            f"connected exposed-skin area {exposed_fraction:.2%} of glove "
            f"(threshold {cfg.min_exposed_area_fraction:.2%}); "
            f"{len(locations)} fingertip-shaped component(s)"
        ),
        debug_mask=kept_mask,
        analysis_mask=segmentation.mask,
    )

detect.owns_pipeline = True
