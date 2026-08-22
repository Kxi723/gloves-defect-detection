"""Finger Not Enough detection for the assigned glove photographs.

Defect definition
-----------------
In this prototype, Finger Not Enough is represented by a glove finger that
ends before the wearer's finger, leaving a connected area of fingertip skin
visible beyond the glove material.

Detection problem
-----------------
Skin-coloured background objects and small colour variations can resemble an
exposed fingertip. Wrist skin is also normally visible below the cuff. The
detector therefore cannot accept every skin-coloured pixel as a defect.

Method
------
1. Build an independent glove mask from the raw photograph.
2. Use an RGB skin-colour threshold to form exposed-skin candidates.
3. Apply median filtering and morphological opening/closing to remove noise.
4. Exclude the cuff region and retain only candidates in contact with the
   glove.
5. Measure connected components and their exposed area. For a candidate just
   below the main area threshold, use a morphological skeleton only as
   supporting evidence that the skin continues a glove-finger centre line.

Decision rule
-------------
The primary decision is made from connected exposed-skin area relative to the
glove. Skeleton evidence is used only for the marginal area range; it is not
the main detector. Accepted components are returned as the defect locations.

Known limitations
-----------------
The implementation detects the visible-skin representation used in the test
images. It cannot confirm a shortened but closed glove finger when no skin is
visible. Skin colour, lighting and contact between the skin and glove can also
affect the result.

Owner: TS. Key decision thresholds and score scales live in
``PipelineConfig.finger_not_enough`` and ``PipelineConfig.skin_colour``.
"""


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


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """Reduce a binary hand silhouette to one-pixel centre lines."""
    work = np.where(mask > 0, 255, 0).astype(np.uint8)
    skeleton = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(
            skeleton, cv2.subtract(work, opened)
        )
        work = cv2.erode(work, element)
    return skeleton


def _skin_branch_fraction(
    glove_mask: np.ndarray,
    skin_mask: np.ndarray,
    glove_box: BBox,
) -> float:
    """Measure whether exposed skin continues a finger centre line.

    The calculation is performed on a small crop so the iterative
    morphological skeleton remains fast.  A genuine shortened glove finger
    produces a centre line that passes from the glove into the connected skin
    region; isolated skin-coloured noise does not.
    """
    glove_x, glove_y, glove_width, glove_height = glove_box
    hand_mask = cv2.bitwise_or(glove_mask, skin_mask)
    hand_crop = hand_mask[
        glove_y:glove_y + glove_height,
        glove_x:glove_x + glove_width,
    ]
    skin_crop = skin_mask[
        glove_y:glove_y + glove_height,
        glove_x:glove_x + glove_width,
    ]
    if hand_crop.size == 0 or cv2.countNonZero(skin_crop) == 0:
        return 0.0

    maximum_side = 256
    scale = min(
        1.0,
        maximum_side / max(glove_width, glove_height, 1),
    )
    small_size = (
        max(1, round(glove_width * scale)),
        max(1, round(glove_height * scale)),
    )
    hand_small = cv2.resize(
        hand_crop, small_size, interpolation=cv2.INTER_NEAREST
    )
    skin_small = cv2.resize(
        skin_crop, small_size, interpolation=cv2.INTER_NEAREST
    )
    skeleton = _morphological_skeleton(hand_small)
    skin_neighbourhood = cv2.dilate(
        skin_small,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    branch_pixels = cv2.countNonZero(
        cv2.bitwise_and(skeleton, skin_neighbourhood)
    )
    return branch_pixels / max(small_size[1], 1)


def detect(
    image: np.ndarray,
    config: PipelineConfig | None = None,
) -> DefectResult:
    """Run the complete Finger Not Enough workflow on one raw image."""
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
        kept_mask[labels == label] = 255
        locations.append((
            int(statistics[label, cv2.CC_STAT_LEFT]),
            int(statistics[label, cv2.CC_STAT_TOP]),
            int(statistics[label, cv2.CC_STAT_WIDTH]),
            int(statistics[label, cv2.CC_STAT_HEIGHT]),
        ))
        exposed_area += area

    exposed_fraction = exposed_area / max(segmentation.area, 1.0)

    # Area remains the primary decision.  Skeleton evidence is used only for
    # a marginal region (80-100% of the area threshold), which recovers a
    # narrow but genuine exposed fingertip without accepting tiny colour noise.
    skeleton_branch_fraction = 0.0
    primary_area_pass = (
        exposed_fraction >= cfg.min_exposed_area_fraction
    )
    marginal_area_threshold = (
        cfg.marginal_area_ratio * cfg.min_exposed_area_fraction
    )
    if not primary_area_pass and exposed_fraction >= marginal_area_threshold:
        skeleton_branch_fraction = _skin_branch_fraction(
            segmentation.mask, kept_mask, segmentation.bbox
        )
    skeleton_pass = (
        exposed_fraction >= marginal_area_threshold
        and skeleton_branch_fraction >= cfg.min_skeleton_branch_fraction
    )
    found = primary_area_pass or skeleton_pass
    if not found:
        locations = []
    return DefectResult(
        defect_found=found,
        defect_type="finger_not_enough",
        locations=locations,
        score=min(
            1.0,
            max(
                exposed_fraction / cfg.area_score_scale,
                skeleton_branch_fraction / cfg.skeleton_score_scale,
            ),
        ) if found else 0.0,
        details=(
            f"connected exposed-skin area {exposed_fraction:.2%} of glove "
            f"(threshold {cfg.min_exposed_area_fraction:.2%}); "
            f"skin-linked skeleton branch {skeleton_branch_fraction:.2%} "
            f"of glove height (marginal-area support threshold "
            f"{cfg.min_skeleton_branch_fraction:.2%})"
        ),
        debug_mask=kept_mask,
        analysis_mask=segmentation.mask,
    )

detect.owns_pipeline = True
