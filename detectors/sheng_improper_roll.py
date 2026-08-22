"""Improper Roll detection on Cotton and black Nitrile gloves.

Defect definition
-----------------
Improper Roll is represented by a glove cuff that has been folded or rolled
upward, shortening the visible cuff and leaving folded glove material near its
lower edge.

Detection problem
-----------------
Cotton and Nitrile show a roll differently. Cotton has a coloured finished
edge and visible knitted material below it; black Nitrile has no equivalent
colour marker, so its decision must use cuff geometry and fold-edge evidence.
A single colour threshold would therefore not work for both materials.

Method
------
1. Build and return an independent glove mask from the raw photograph.
2. Use local texture to choose the Cotton or black Nitrile processing branch.
3. For Cotton, locate the yellow cuff edge, measure its upward movement from
   the stored Normal baseline, and check that a sufficiently deep and broad
   band of folded glove material remains below it.
4. For Nitrile, compare visible glove height/width with the stored Normal
   limit and detect a sufficiently strong, continuous fold edge in the cuff
   zone. Nearby visible skin is recorded as supporting evidence, not as a
   compulsory condition.
5. Use RGB thresholding, local standard deviation, morphology, connected
   components and bounding-box ratios to measure these cues.

Decision rule
-------------
The Cotton branch requires both upward cuff-edge movement and retained
fold-band evidence. The Nitrile branch requires both shortened-cuff geometry
and Normal-relative fold-edge evidence. A cuff-region box is returned only
when the applicable pair of conditions agrees.

Known limitations
-----------------
The Normal measurements are fixed calibration values in the configuration;
the program does not request a new Normal reference image at run time. The
detector is calibrated only for the assigned Cotton and black Nitrile images,
and it recognises this evidence pattern rather than classifying every possible
cuff defect or material.

Owner: TS. Key decision thresholds and score scales live in
``PipelineConfig.improper_roll`` and ``PipelineConfig.skin_colour``.
"""

from __future__ import annotations

import cv2
import numpy as np

from .ts_support.config import PipelineConfig, SkinColourConfig, get_config
from .ts_support.features import BBox, DefectResult
from .ts_support.preprocessing import preprocess, resize_to_limit
from .ts_support.segmentation import (
    SegmentationResult,
    basic_global_threshold,
    segment_glove,
)


def _odd_kernel_size(value: float, minimum: int = 3) -> int:
    """Round a scale-relative value to a valid odd morphology size."""
    size = max(minimum, int(round(value)))
    return size if size % 2 == 1 else size + 1


def _median_local_texture(
    image_bgr: np.ndarray,
    glove_selection: np.ndarray,
    window_size: int,
) -> float:
    """Median local standard deviation inside the segmented glove."""
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_mean = cv2.blur(grey, (window_size, window_size))
    local_square_mean = cv2.blur(grey * grey, (window_size, window_size))
    local_texture = np.sqrt(
        np.maximum(local_square_mean - local_mean * local_mean, 0.0)
    )
    return float(np.median(local_texture[glove_selection]))


def _skin_pixels_rgb(
    image_bgr: np.ndarray,
    cfg: SkinColourConfig,
) -> np.ndarray:
    """Simple RGB skin-colour threshold used only near the glove cuff."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    channel_range = (
        np.maximum.reduce((red, green, blue))
        - np.minimum.reduce((red, green, blue))
    )
    skin_pixels = (
        (red > cfg.red_min)
        & (green > cfg.green_min)
        & (blue > cfg.blue_min)
        & (red > green)
        & (green > blue)
        & ((red - green) > cfg.red_green_difference_min)
        & ((red - green) < cfg.red_green_difference_max)
        & ((red - blue) > cfg.red_blue_difference_min)
        & (channel_range > cfg.channel_range_min)
        & (((red + green + blue) / 3) < cfg.mean_brightness_max)
    )
    return np.where(skin_pixels, 255, 0).astype(np.uint8)


def _largest_component(
    mask: np.ndarray,
    minimum_area: int,
) -> tuple[np.ndarray, BBox | None, int]:
    """Return the largest connected component above ``minimum_area``."""
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask, 8)
    best_label = 0
    best_area = 0
    for label in range(1, count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if area >= minimum_area and area > best_area:
            best_label = label
            best_area = area

    if not best_label:
        return np.zeros_like(mask), None, 0

    component = np.where(labels == best_label, 255, 0).astype(np.uint8)
    box = (
        int(statistics[best_label, cv2.CC_STAT_LEFT]),
        int(statistics[best_label, cv2.CC_STAT_TOP]),
        int(statistics[best_label, cv2.CC_STAT_WIDTH]),
        int(statistics[best_label, cv2.CC_STAT_HEIGHT]),
    )
    return component, box, best_area


def _refine_cotton_segmentation(
    segmentation: SegmentationResult,
    image_bgr: np.ndarray,
    config: PipelineConfig,
) -> SegmentationResult:
    """Combine colour and texture silhouettes, then bridge yarn-row gaps."""
    cfg = config.improper_roll
    shortest_side = min(segmentation.mask.shape)
    vertical_size = _odd_kernel_size(
        shortest_side * cfg.cotton_mask_vertical_close_fraction,
        minimum=5,
    )
    compact_size = _odd_kernel_size(
        shortest_side * cfg.cotton_mask_compact_close_fraction,
        minimum=5,
    )
    median_size = _odd_kernel_size(
        shortest_side * cfg.cotton_mask_median_fraction,
        minimum=3,
    )

    # The generic selector chooses texture for these knitted gloves. White
    # yarn can locally resemble the desk, so a separate border-relative RGB
    # distance mask supplies the parts where texture alone opens the outline.
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    image_height, image_width = rgb.shape[:2]
    border_width = max(
        2,
        round(min(image_height, image_width) * config.segmentation.border_fraction),
    )
    border_pixels = np.concatenate(
        (
            rgb[:border_width, :, :].reshape(-1, 3),
            rgb[-border_width:, :, :].reshape(-1, 3),
            rgb[:, :border_width, :].reshape(-1, 3),
            rgb[:, -border_width:, :].reshape(-1, 3),
        ),
        axis=0,
    )
    background_colour = np.median(border_pixels, axis=0)
    colour_distance = np.linalg.norm(rgb - background_colour, axis=2)
    colour_mask = basic_global_threshold(colour_distance)

    open_size = _odd_kernel_size(config.segmentation.open_kernel)
    close_size = _odd_kernel_size(config.segmentation.close_kernel)
    colour_mask = cv2.morphologyEx(
        colour_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
    )
    colour_mask = cv2.morphologyEx(
        colour_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    colour_mask, _, _ = _largest_component(
        colour_mask,
        minimum_area=cfg.cotton_colour_min_component_pixels,
    )
    colour_contours, _ = cv2.findContours(
        colour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if colour_contours:
        colour_solid = np.zeros_like(colour_mask)
        cv2.drawContours(
            colour_solid,
            [max(colour_contours, key=cv2.contourArea)],
            -1,
            255,
            cv2.FILLED,
        )
    else:
        colour_solid = np.zeros_like(colour_mask)

    # Colour distance also sees exposed skin. It is only permitted to
    # repair a narrow neighbourhood around the texture-selected glove, so
    # the arm below the cuff cannot become part of the glove silhouette.
    support_size = _odd_kernel_size(
        shortest_side * cfg.cotton_mask_support_dilate_fraction,
        minimum=5,
    )
    texture_neighbourhood = cv2.dilate(
        segmentation.mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (support_size, support_size)
        ),
    )
    colour_solid = cv2.bitwise_and(colour_solid, texture_neighbourhood)
    refined = cv2.bitwise_or(segmentation.mask, colour_solid)
    vertical_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (median_size, vertical_size)
    )
    compact_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (compact_size, compact_size)
    )
    refined = cv2.morphologyEx(
        refined, cv2.MORPH_CLOSE, vertical_element
    )
    refined = cv2.morphologyEx(
        refined, cv2.MORPH_CLOSE, compact_element
    )
    refined = cv2.medianBlur(refined, median_size)
    contours, _ = cv2.findContours(
        refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return segmentation

    contour = max(contours, key=cv2.contourArea)
    solid = np.zeros_like(refined)
    cv2.drawContours(solid, [contour], -1, 255, cv2.FILLED)
    return SegmentationResult(
        mask=solid,
        contour=contour,
        bbox=cv2.boundingRect(contour),
        area=float(cv2.contourArea(contour)),
        cue=f"{segmentation.cue}+cotton_refinement",
        source_image=segmentation.source_image,
    )


def _cuff_box(
    glove_box: BBox,
    start_fraction: float,
    height_fraction: float,
    image_shape: tuple[int, ...],
) -> BBox:
    """Box the lower cuff region so the UI points at the actual defect."""
    glove_x, glove_y, glove_width, glove_height = glove_box
    image_height, image_width = image_shape[:2]
    x = max(0, glove_x)
    y = max(0, glove_y + round(glove_height * start_fraction))
    width = min(glove_width, image_width - x)
    height = min(
        max(1, round(glove_height * height_fraction)), image_height - y
    )
    return x, y, width, height


def _cotton_measurements(
    source_image: np.ndarray,
    segmentation: SegmentationResult,
    config: PipelineConfig,
) -> tuple[bool, float, float, float, np.ndarray]:
    """Measure Cotton cuff shift and material retained below its edge."""
    cfg = config.improper_roll
    glove_x, glove_y, glove_width, glove_height = segmentation.bbox
    rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB).astype(np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    yellow_pixels = (
        (segmentation.mask > 0)
        & (red > cfg.cotton_yellow_red_min)
        & (green > cfg.cotton_yellow_green_min)
        & (blue < cfg.cotton_yellow_blue_max)
        & (
            (red - blue)
            > cfg.cotton_yellow_red_blue_difference_min
        )
        & (
            (green - blue)
            > cfg.cotton_yellow_green_blue_difference_min
        )
    )
    yellow_mask = np.where(yellow_pixels, 255, 0).astype(np.uint8)
    yellow_search_start = glove_y + round(
        glove_height * cfg.cotton_yellow_search_start_fraction
    )
    yellow_mask[:yellow_search_start, :] = 0
    yellow_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    yellow_mask = cv2.morphologyEx(
        yellow_mask, cv2.MORPH_OPEN, yellow_element
    )
    band_mask, band_box, band_area = _largest_component(
        yellow_mask, cfg.cotton_band_min_area
    )
    if band_box is None or band_area < cfg.cotton_band_min_area:
        return False, 1.0, 0.0, 0.0, band_mask

    _, band_y, _, band_height = band_box
    band_centre_y = band_y + band_height / 2.0
    band_y_fraction = (band_centre_y - glove_y) / max(glove_height, 1)
    band_bottom = band_y + band_height - 1

    # A real roll leaves a visible double layer below the moved cuff edge.
    # A cut/unfinished cuff can also be short, but it does not leave this
    # broad retained-material band. This is the second independent cue.
    fold_material_mask = np.zeros_like(segmentation.mask)
    fold_material_mask[band_bottom + 1 :, :] = segmentation.mask[
        band_bottom + 1 :, :
    ]
    fold_depth_fraction = (
        glove_y + glove_height - 1 - band_bottom
    ) / max(glove_height, 1)
    fold_area_fraction = np.count_nonzero(fold_material_mask) / max(
        np.count_nonzero(segmentation.mask), 1
    )

    band_shift_fraction = (
        cfg.cotton_normal_band_y_fraction - band_y_fraction
    )
    glove_aspect = glove_height / max(glove_width, 1)
    found = (
        glove_aspect < cfg.cotton_max_aspect
        and band_shift_fraction >= cfg.cotton_min_band_shift_fraction
        and fold_depth_fraction >= cfg.cotton_min_fold_depth_fraction
        and fold_area_fraction >= cfg.cotton_min_fold_area_fraction
    )
    debug_mask = cv2.bitwise_or(band_mask, fold_material_mask)
    return (
        found,
        float(band_y_fraction),
        float(fold_depth_fraction),
        float(fold_area_fraction),
        debug_mask,
    )


def _horizontal_fold_edge(
    source_image: np.ndarray,
    segmentation: SegmentationResult,
    config: PipelineConfig,
) -> tuple[float, float, float, np.ndarray]:
    """Find the strongest continuous horizontal edge in the cuff zone."""
    cfg = config.improper_roll
    glove_x, glove_y, glove_width, glove_height = segmentation.bbox
    grey = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    vertical_gradient = np.abs(
        cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    )

    x_start = max(
        0,
        glove_x + round(glove_width * cfg.nitrile_edge_x_start_fraction),
    )
    x_end = min(
        source_image.shape[1],
        glove_x + round(glove_width * cfg.nitrile_edge_x_end_fraction),
    )
    y_start = max(
        0, glove_y + round(glove_height * cfg.nitrile_edge_start_fraction)
    )
    y_end = min(
        source_image.shape[0],
        glove_y + round(glove_height * cfg.nitrile_edge_end_fraction),
    )
    if x_end <= x_start or y_end <= y_start:
        return 0.0, 0.0, 1.0, np.zeros_like(segmentation.mask)

    best_score = 0.0
    best_continuity = 0.0
    best_row = y_start
    for row in range(y_start, y_end):
        row_values = vertical_gradient[row, x_start:x_end]
        row_score = float(np.mean(row_values))
        row_continuity = float(
            np.mean(row_values > cfg.nitrile_edge_pixel_threshold)
        )
        if row_score > best_score:
            best_score = row_score
            best_continuity = row_continuity
            best_row = row

    edge_y_fraction = (best_row - glove_y) / max(glove_height, 1)
    edge_mask = np.zeros_like(segmentation.mask)
    half_band = max(
        2,
        round(glove_height * cfg.nitrile_edge_half_band_fraction),
    )
    band_start = max(y_start, best_row - half_band)
    band_end = min(y_end, best_row + half_band + 1)
    edge_pixels = (
        vertical_gradient[band_start:band_end, x_start:x_end]
        > cfg.nitrile_edge_pixel_threshold
    )
    edge_mask[band_start:band_end, x_start:x_end] = np.where(
        edge_pixels, 255, 0
    ).astype(np.uint8)
    return best_score, best_continuity, float(edge_y_fraction), edge_mask


def _nitrile_measurements(
    source_image: np.ndarray,
    segmentation: SegmentationResult,
    config: PipelineConfig,
) -> tuple[bool, float, float, float, float, float, np.ndarray]:
    """Measure cuff shift from skin plus independent fold-edge evidence."""
    cfg = config.improper_roll
    glove_x, glove_y, glove_width, glove_height = segmentation.bbox

    skin_mask = _skin_pixels_rgb(source_image, config.skin_colour)
    skin_mask = cv2.medianBlur(skin_mask, 5)
    clean_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(
        skin_mask, cv2.MORPH_OPEN, clean_element
    )
    skin_mask = cv2.morphologyEx(
        skin_mask, cv2.MORPH_CLOSE, clean_element, iterations=2
    )

    cuff_neighbourhood = np.zeros_like(skin_mask)
    x_start = max(
        0,
        glove_x + round(glove_width * cfg.skin_region_x_start_fraction),
    )
    x_end = min(
        skin_mask.shape[1],
        glove_x + round(glove_width * cfg.skin_region_x_end_fraction),
    )
    y_start = max(
        0,
        glove_y + round(glove_height * cfg.skin_region_y_start_fraction),
    )
    y_end = min(
        skin_mask.shape[0],
        glove_y + round(glove_height * cfg.skin_region_y_end_fraction),
    )
    cuff_neighbourhood[y_start:y_end, x_start:x_end] = 255

    contact_size = _odd_kernel_size(
        min(source_image.shape[:2]) * cfg.contact_kernel_fraction,
        minimum=9,
    )
    contact_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (contact_size, contact_size)
    )
    near_glove = cv2.dilate(segmentation.mask, contact_element)
    skin_mask[(cuff_neighbourhood == 0) | (near_glove == 0)] = 0

    minimum_area = max(
        cfg.skin_min_component_pixels,
        round(segmentation.area * cfg.skin_min_component_fraction),
    )
    kept_skin_mask, skin_box, skin_area = _largest_component(
        skin_mask, minimum_area
    )
    if skin_box is None:
        skin_top_fraction = 1.0
    else:
        _, skin_y, _, _ = skin_box
        skin_top_fraction = (skin_y - glove_y) / max(glove_height, 1)

    skin_area_fraction = skin_area / max(segmentation.area, 1.0)
    glove_aspect = glove_height / max(glove_width, 1)
    (
        edge_score,
        edge_continuity,
        edge_y_fraction,
        edge_mask,
    ) = _horizontal_fold_edge(source_image, segmentation, config)
    minimum_edge_score = (
        cfg.nitrile_normal_edge_score + cfg.nitrile_min_edge_increase
    )
    # A roll may cover the wrist completely, so visible skin is supporting
    # evidence rather than a compulsory condition.  The decision requires two
    # independent structural cues: the visible glove is shorter than the
    # Normal reference and a continuous fold edge is present in the cuff zone.
    shortened_cuff = glove_aspect < cfg.nitrile_max_aspect
    fold_edge_present = (
        edge_score >= minimum_edge_score
        and edge_continuity >= cfg.nitrile_min_edge_continuity
    )
    found = shortened_cuff and fold_edge_present
    debug_mask = cv2.bitwise_or(kept_skin_mask, edge_mask)
    return (
        found,
        float(skin_top_fraction),
        float(skin_area_fraction),
        float(edge_score),
        float(edge_continuity),
        float(edge_y_fraction),
        debug_mask,
    )


def detect(
    image: np.ndarray,
    config: PipelineConfig | None = None,
) -> DefectResult:
    """Run the complete Improper Roll workflow on one raw image."""
    config = config or get_config()
    source_image = resize_to_limit(image, config.preprocess.max_dimension)
    image = preprocess(image, config.preprocess)
    cfg = config.improper_roll
    independent_segmentation = segment_glove(image, config.segmentation)
    if independent_segmentation is None:
        return DefectResult(
            False,
            "improper_roll",
            details="independent Improper Roll glove mask failed",
            debug_mask=np.zeros(image.shape[:2], dtype=np.uint8),
            analysis_mask=np.zeros(image.shape[:2], dtype=np.uint8),
        )
    independent_segmentation.source_image = source_image
    segmentation = independent_segmentation
    glove_selection = segmentation.mask > 0
    if np.count_nonzero(glove_selection) < cfg.minimum_analysis_pixels:
        return DefectResult(
            False,
            "improper_roll",
            details="segmented glove too small to analyse",
            debug_mask=np.zeros_like(segmentation.mask),
            analysis_mask=segmentation.mask,
        )

    texture_window = _odd_kernel_size(
        min(source_image.shape[:2]) * cfg.texture_window_fraction,
        minimum=9,
    )
    median_texture = _median_local_texture(
        source_image, glove_selection, texture_window
    )
    is_cotton = median_texture > cfg.cotton_texture_threshold
    if is_cotton:
        segmentation = _refine_cotton_segmentation(
            segmentation, image, config
        )

    if is_cotton:
        (
            found,
            band_y_fraction,
            fold_depth_fraction,
            fold_area_fraction,
            debug_mask,
        ) = _cotton_measurements(source_image, segmentation, config)
        material_name = "Cotton"
        band_shift_fraction = (
            cfg.cotton_normal_band_y_fraction - band_y_fraction
        )
        evidence = (
            f"Normal edge {cfg.cotton_normal_band_y_fraction:.1%}, "
            f"current edge {band_y_fraction:.1%} "
            f"(upward shift {band_shift_fraction:.1%}); retained fold "
            f"depth {fold_depth_fraction:.1%}, area "
            f"{fold_area_fraction:.1%}"
        )
        agreement_text = "cuff shift and retained fold-band evidence"
        severity = max(
            0.0,
            band_shift_fraction,
        )
        score = (
            min(1.0, severity / cfg.cotton_score_scale)
            if found
            else 0.0
        )
    else:
        (
            found,
            skin_top_fraction,
            skin_area_fraction,
            edge_score,
            edge_continuity,
            edge_y_fraction,
            debug_mask,
        ) = _nitrile_measurements(source_image, segmentation, config)
        material_name = "black Nitrile"
        evidence = (
            f"glove height/width {segmentation.bbox[3] / max(segmentation.bbox[2], 1):.2f} "
            f"(Normal-relative maximum {cfg.nitrile_max_aspect:.2f}); "
            f"fold edge at {edge_y_fraction:.1%}, score "
            f"{edge_score:.1f} versus Normal "
            f"{cfg.nitrile_normal_edge_score:.1f}, continuity "
            f"{edge_continuity:.1%}; supporting cuff-adjacent skin "
            f"{skin_area_fraction:.2%}"
        )
        agreement_text = (
            "shortened-cuff geometry and Normal-relative fold-edge evidence"
        )
        glove_aspect = (
            segmentation.bbox[3] / max(segmentation.bbox[2], 1)
        )
        score = min(
            1.0,
            cfg.nitrile_aspect_score_weight
            * max(0.0, cfg.nitrile_max_aspect - glove_aspect)
            / cfg.nitrile_aspect_score_scale
            + cfg.nitrile_edge_score_weight
            * edge_continuity
            / cfg.nitrile_continuity_score_scale,
        ) if found else 0.0

    locations = []
    if found:
        locations = [
            _cuff_box(
                segmentation.bbox,
                cfg.cuff_box_start_fraction,
                cfg.cuff_box_height_fraction,
                source_image.shape,
            )
        ]

    return DefectResult(
        defect_found=found,
        defect_type="improper_roll",
        locations=locations,
        score=score,
        details=(
            f"{material_name} selected from local texture; {evidence}; "
            f"{agreement_text} must both agree"
        ),
        debug_mask=debug_mask,
        analysis_mask=segmentation.mask,
    )

detect.owns_pipeline = True
