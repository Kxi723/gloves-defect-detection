"""Plastic Contamination detection on blue Latex and black Nitrile gloves.

Defect definition
-----------------
Plastic Contamination is represented by visible transparent plastic film on
the photographed glove. The assigned test set uses blue Latex and black
Nitrile materials.

Detection problem
-----------------
Transparent plastic partly retains the colour of the glove below it. Its most
useful evidence is therefore a material-dependent combination of colour,
brightness and local texture rather than one fixed colour. Reflections and
folds can also split one physical plastic sheet into several image regions.

Method
------
1. Build and return an independent glove mask from the raw photograph.
2. Estimate whether the glove is blue Latex, black Nitrile or light knitted
   material using blue-channel ratio and local texture. Light knitted gloves
   are outside this detector's assigned materials.
3. Erode the mask to obtain the glove interior, divide it into finger and palm
   zones, and exclude the cuff.
4. Calculate HSI saturation/intensity and local texture. Compare each zone
   with its own glove reference so that Latex and Nitrile use material-specific
   candidate rules.
5. Apply morphological opening/closing and connected-component shape, area
   and quality checks to retain plausible plastic regions.

Decision rule
-------------
Plastic Contamination is reported when at least one connected candidate
survives the material-specific component checks. The score is based on the
accepted candidate area relative to the segmented glove area.

Known limitations
-----------------
The defect mask shows pixels that satisfy the visual rules; it is not a ground
truth outline of the physical plastic. Very transparent, low-contrast or
strongly folded sections may be only partly detected, while glare can resemble
plastic. The calibrated material scope is blue Latex and black Nitrile.

Owner: TS. Key decision thresholds and score scales live in
``PipelineConfig.plastic_contamination``.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .ts_support.config import PipelineConfig, get_config
from .ts_support.features import BBox, DefectResult
from .ts_support.preprocessing import preprocess, resize_to_limit
from .ts_support.segmentation import (
    SegmentationResult,
    hsi_saturation_and_intensity,
    segment_glove,
)


def _odd_kernel_size(value: float, minimum: int = 3) -> int:
    """Round a scale-relative value to a valid odd morphology size."""
    size = max(minimum, int(round(value)))
    return size if size % 2 == 1 else size + 1


def _local_texture_ratio(
    image_bgr: np.ndarray,
    selection: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, float]:
    """Local standard deviation divided by the glove's median texture."""
    _, intensity = hsi_saturation_and_intensity(image_bgr)
    local_mean = cv2.blur(intensity, (window_size, window_size))
    local_square_mean = cv2.blur(
        intensity * intensity, (window_size, window_size)
    )
    local_texture = np.sqrt(
        np.maximum(local_square_mean - local_mean * local_mean, 0.0)
    )
    median_texture = max(float(np.median(local_texture[selection])), 1e-6)
    return local_texture / median_texture, median_texture


def _keep_regions(
    mask: np.ndarray,
    minimum_area: int,
    minimum_extent: float,
    maximum_elongation: float,
    relative_component_fraction: float,
    merge_gap: int,
    quality_image: np.ndarray | None = None,
    minimum_quality: float | None = None,
) -> tuple[np.ndarray, List[BBox], float]:
    """Keep plausible components, filling only their enclosed small gaps."""
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask, 8)
    kept = np.zeros_like(mask)
    plausible_components: list[tuple[np.ndarray, BBox, int]] = []
    for label in range(1, count):
        x = int(statistics[label, cv2.CC_STAT_LEFT])
        y = int(statistics[label, cv2.CC_STAT_TOP])
        width = int(statistics[label, cv2.CC_STAT_WIDTH])
        height = int(statistics[label, cv2.CC_STAT_HEIGHT])
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        raw_extent = area / max(float(width * height), 1.0)
        if raw_extent < minimum_extent:
            continue
        if quality_image is not None and minimum_quality is not None:
            component_quality = float(np.median(quality_image[labels == label]))
            if component_quality < minimum_quality:
                continue

        # Thresholds often leave pinholes where a transparent sheet catches
        # several different reflections. Fill the external outline of this
        # already-connected component, but never bridge separate components.
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
        filled_area = int(np.count_nonzero(filled))
        elongation = max(width, height) / max(min(width, height), 1)
        if elongation > maximum_elongation:
            continue
        plausible_components.append((filled, (x, y, width, height), filled_area))

    # Fragments of one transparent sheet can be disconnected because its
    # middle has the same colour as the glove. Merge components only when
    # their slightly expanded boxes intersect, then fill their joint hull.
    groups: list[tuple[np.ndarray, BBox, int]] = []
    for component, box, area in plausible_components:
        x, y, width, height = box
        merged_index = None
        for index, (_, group_box, _) in enumerate(groups):
            gx, gy, gw, gh = group_box
            separated = (
                x > gx + gw + merge_gap
                or gx > x + width + merge_gap
                or y > gy + gh + merge_gap
                or gy > y + height + merge_gap
            )
            if not separated:
                merged_index = index
                break
        if merged_index is None:
            groups.append((component.copy(), box, area))
            continue

        group_mask, group_box, _ = groups[merged_index]
        group_mask = cv2.bitwise_or(group_mask, component)
        points = cv2.findNonZero(group_mask)
        if points is None:
            continue
        hull = cv2.convexHull(points)
        joined = np.zeros_like(mask)
        cv2.drawContours(joined, [hull], -1, 255, cv2.FILLED)
        joined_box = cv2.boundingRect(hull)
        groups[merged_index] = (
            joined,
            joined_box,
            int(np.count_nonzero(joined)),
        )

    largest_area = max((item[2] for item in groups), default=0)
    relative_minimum = largest_area * relative_component_fraction
    locations: List[BBox] = []
    total_area = 0.0
    for component, box, area in groups:
        if area < relative_minimum:
            continue
        kept[component > 0] = 255
        locations.append(box)
        total_area += area

    return kept, locations, total_area


def _blue_latex_glove_mask(
    source_image: np.ndarray,
    segmentation: SegmentationResult,
    glove_interior: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """Recover the blue glove silhouette beneath transparent plastic."""
    cfg = config.plastic_contamination
    rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    blue_ratio = blue / np.maximum(rgb.sum(axis=2), 1.0)
    blue_pixels = (
        (blue_ratio > cfg.blue_support_ratio)
        & (blue > red * cfg.blue_support_red_multiplier)
        & (blue > green * cfg.blue_support_green_multiplier)
    )

    glove_x, glove_y, glove_width, glove_height = segmentation.bbox
    padding = cfg.blue_support_padding_pixels
    search = np.zeros_like(blue_pixels)
    search[
        max(0, glove_y - padding) : min(
            source_image.shape[0], glove_y + glove_height + padding
        ),
        max(0, glove_x - padding) : min(
            source_image.shape[1], glove_x + glove_width + padding
        ),
    ] = True
    blue_mask = np.where(blue_pixels & search, 255, 0).astype(np.uint8)
    close_size = _odd_kernel_size(
        min(source_image.shape[:2]) * cfg.blue_support_close_fraction,
        minimum=5,
    )
    close_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    blue_mask = cv2.morphologyEx(
        blue_mask, cv2.MORPH_CLOSE, close_element, iterations=2
    )

    # Background objects with a similar blue colour are not allowed in:
    # every retained component must overlap this detector's glove interior.
    count, labels, _, _ = cv2.connectedComponentsWithStats(blue_mask, 8)
    support = np.zeros_like(blue_mask)
    interior_selection = glove_interior > 0
    for label in range(1, count):
        component = labels == label
        if (
            np.count_nonzero(component & interior_selection)
            > cfg.blue_support_min_component_pixels
        ):
            support[component] = 255
    combined = cv2.bitwise_or(support, segmentation.mask)
    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return segmentation.mask.copy()
    detector_mask = np.zeros_like(combined)
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(detector_mask, [largest], -1, 255, cv2.FILLED)
    return detector_mask


def _zone_masks(
    support: np.ndarray,
    bbox: BBox,
    config: PipelineConfig,
    *,
    palm_end_fraction: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split detector support into finger, palm and excluded cuff zones."""
    cfg = config.plastic_contamination
    _, glove_y, _, glove_height = bbox
    finger_end = glove_y + round(glove_height * cfg.finger_end_fraction)
    effective_palm_end = (
        cfg.palm_end_fraction
        if palm_end_fraction is None
        else palm_end_fraction
    )
    palm_end = glove_y + round(glove_height * effective_palm_end)

    finger = support > 0
    finger[finger_end:, :] = False
    palm = support > 0
    palm[:finger_end, :] = False
    palm[palm_end:, :] = False
    cuff = support > 0
    cuff[:palm_end, :] = False
    return finger, palm, cuff


def _zone_reference_candidate(
    zone: np.ndarray,
    saturation: np.ndarray,
    intensity: np.ndarray,
    texture_ratio: np.ndarray,
    *,
    minimum_pixels: int,
    saturation_drop: float | None = None,
    intensity_gain: float,
    max_texture_ratio: float | None = None,
    smooth_intensity_gain: float | None = None,
    smooth_max_saturation: float | None = None,
) -> np.ndarray:
    """Threshold a zone relative to that zone's median surface values."""
    if np.count_nonzero(zone) < minimum_pixels:
        return np.zeros_like(zone)
    reference_saturation = float(np.median(saturation[zone]))
    reference_intensity = float(np.median(intensity[zone]))
    gain = intensity - reference_intensity
    candidate = zone & (gain > intensity_gain)
    if saturation_drop is not None:
        colour_candidate = (
            zone
            & ((reference_saturation - saturation) > saturation_drop)
            & (gain > intensity_gain)
        )
        candidate = colour_candidate
    if (
        max_texture_ratio is not None
        and smooth_intensity_gain is not None
        and smooth_max_saturation is not None
    ):
        smooth_candidate = (
            zone
            & (texture_ratio < max_texture_ratio)
            & (gain > smooth_intensity_gain)
            & (saturation < smooth_max_saturation)
        )
        candidate = candidate | smooth_candidate
    return candidate


def detect(
    image: np.ndarray,
    config: PipelineConfig | None = None,
) -> DefectResult:
    """Run the complete Plastic Contamination workflow on one raw image."""
    config = config or get_config()
    source_image = resize_to_limit(image, config.preprocess.max_dimension)
    image = preprocess(image, config.preprocess)
    cfg = config.plastic_contamination
    independent_segmentation = segment_glove(image, config.segmentation)
    if independent_segmentation is None:
        return DefectResult(
            False,
            "plastic_contamination",
            details="independent Plastic glove mask failed",
            debug_mask=np.zeros(image.shape[:2], dtype=np.uint8),
            analysis_mask=np.zeros(image.shape[:2], dtype=np.uint8),
        )
    independent_segmentation.source_image = source_image
    segmentation = independent_segmentation
    shortest_side = min(source_image.shape[:2])

    interior_size = _odd_kernel_size(
        shortest_side * cfg.interior_kernel_fraction, minimum=5
    )
    interior_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (interior_size, interior_size)
    )
    glove_interior = cv2.erode(segmentation.mask, interior_element)
    selection = glove_interior > 0
    if np.count_nonzero(selection) < cfg.minimum_analysis_pixels:
        return DefectResult(
            False,
            "plastic_contamination",
            details="glove interior too small to analyse",
            debug_mask=np.zeros_like(segmentation.mask),
            analysis_mask=segmentation.mask,
        )

    rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb_sum = np.maximum(rgb.sum(axis=2), 1.0)
    blue_ratio = rgb[:, :, 2] / rgb_sum
    median_blue_ratio = float(np.median(blue_ratio[selection]))

    texture_window = _odd_kernel_size(
        shortest_side * cfg.texture_window_fraction, minimum=9
    )
    texture_ratio, median_texture = _local_texture_ratio(
        source_image, selection, texture_window
    )

    is_blue_latex = median_blue_ratio >= cfg.blue_latex_ratio_threshold
    is_light_knitted = (
        not is_blue_latex
        and median_texture >= cfg.knitted_texture_threshold
    )
    if is_light_knitted:
        return DefectResult(
            False,
            "plastic_contamination",
            details=(
                "light knitted material; Plastic Contamination is assigned "
                "to blue Latex and black Nitrile"
            ),
            debug_mask=np.zeros_like(segmentation.mask),
            analysis_mask=segmentation.mask,
        )

    if is_blue_latex:
        detector_glove_mask = _blue_latex_glove_mask(
            source_image, segmentation, glove_interior, config
        )
        analysis_support = cv2.erode(detector_glove_mask, interior_element)
    else:
        detector_glove_mask = segmentation.mask
        analysis_support = glove_interior
    selection = analysis_support > 0
    finger_zone, palm_zone, _cuff_zone = _zone_masks(
        analysis_support, segmentation.bbox, config
    )
    saturation, intensity = hsi_saturation_and_intensity(source_image)
    texture_ratio, _ = _local_texture_ratio(
        source_image, selection, texture_window
    )

    if is_blue_latex:
        finger_candidate = _zone_reference_candidate(
            finger_zone,
            saturation,
            intensity,
            texture_ratio,
            minimum_pixels=cfg.minimum_analysis_pixels,
            saturation_drop=cfg.latex_finger_saturation_drop,
            intensity_gain=cfg.latex_finger_intensity_gain,
            max_texture_ratio=cfg.latex_finger_max_texture_ratio,
            smooth_intensity_gain=cfg.latex_finger_smooth_intensity_gain,
            smooth_max_saturation=cfg.latex_smooth_max_saturation,
        )
        palm_candidate = _zone_reference_candidate(
            palm_zone,
            saturation,
            intensity,
            texture_ratio,
            minimum_pixels=cfg.minimum_analysis_pixels,
            saturation_drop=cfg.latex_palm_saturation_drop,
            intensity_gain=cfg.latex_palm_intensity_gain,
            max_texture_ratio=cfg.latex_palm_max_texture_ratio,
            smooth_intensity_gain=cfg.latex_palm_smooth_intensity_gain,
            smooth_max_saturation=cfg.latex_smooth_max_saturation,
        )
        candidate_pixels = (
            (finger_candidate | palm_candidate)
            & (saturation < cfg.latex_max_candidate_saturation)
        )
        minimum_component_fraction = cfg.latex_min_component_fraction
        minimum_extent = cfg.latex_min_extent
        maximum_elongation = cfg.latex_max_elongation
        relative_component_fraction = cfg.latex_relative_component_fraction
        minimum_quality = None
        material_name = "blue Latex"
    else:
        # The photographed Nitrile sheets sit lower on the palm than the
        # Latex sheets. Rebuild only this detector's zones with a slightly
        # lower palm boundary; this does not change the Latex branch.
        finger_zone, palm_zone, _cuff_zone = _zone_masks(
            analysis_support,
            segmentation.bbox,
            config,
            palm_end_fraction=cfg.nitrile_palm_end_fraction,
        )
        finger_candidate = _zone_reference_candidate(
            finger_zone,
            saturation,
            intensity,
            texture_ratio,
            minimum_pixels=cfg.minimum_analysis_pixels,
            intensity_gain=cfg.nitrile_finger_intensity_gain,
        )
        palm_candidate = _zone_reference_candidate(
            palm_zone,
            saturation,
            intensity,
            texture_ratio,
            minimum_pixels=cfg.minimum_analysis_pixels,
            intensity_gain=cfg.nitrile_palm_intensity_gain,
        )
        # Transparent film produces local highlights with more variation than
        # the surrounding smooth Nitrile. Requiring both relative brightness
        # and local texture prevents ordinary broad lighting changes from
        # becoming candidates.
        candidate_pixels = (
            (finger_candidate | palm_candidate)
            & (texture_ratio > cfg.nitrile_min_pixel_texture_ratio)
        )
        minimum_component_fraction = cfg.nitrile_min_component_fraction
        minimum_extent = cfg.nitrile_min_extent
        maximum_elongation = cfg.nitrile_max_elongation
        relative_component_fraction = cfg.nitrile_relative_component_fraction
        minimum_quality = cfg.nitrile_min_texture_ratio
        material_name = "black Nitrile"

    candidate_mask = np.where(candidate_pixels, 255, 0).astype(np.uint8)
    open_size = _odd_kernel_size(
        shortest_side * cfg.open_kernel_fraction, minimum=3
    )
    close_fraction = (
        cfg.close_kernel_fraction
        if is_blue_latex
        else cfg.nitrile_close_kernel_fraction
    )
    close_size = _odd_kernel_size(shortest_side * close_fraction, minimum=3)
    open_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_size, open_size)
    )
    close_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    candidate_mask = cv2.morphologyEx(
        candidate_mask, cv2.MORPH_OPEN, open_element
    )
    candidate_mask = cv2.morphologyEx(
        candidate_mask, cv2.MORPH_CLOSE, close_element
    )

    minimum_area = max(
        cfg.minimum_component_pixels,
        round(segmentation.area * minimum_component_fraction),
    )
    merge_gap = max(
        cfg.minimum_merge_pixels,
        round(shortest_side * cfg.component_merge_fraction),
    )
    kept_mask, locations, candidate_area = _keep_regions(
        candidate_mask,
        minimum_area=minimum_area,
        minimum_extent=minimum_extent,
        maximum_elongation=maximum_elongation,
        relative_component_fraction=relative_component_fraction,
        merge_gap=merge_gap,
        quality_image=texture_ratio,
        minimum_quality=minimum_quality,
    )
    found = bool(locations)
    candidate_fraction = candidate_area / max(segmentation.area, 1.0)

    return DefectResult(
        defect_found=found,
        defect_type="plastic_contamination",
        locations=locations if found else [],
        score=(
            min(1.0, candidate_fraction / cfg.score_fraction_scale)
            if found
            else 0.0
        ),
        details=(
            f"{material_name}; finger/palm references applied and cuff "
            f"excluded; {len(locations)} connected plastic region(s), "
            f"total {candidate_fraction:.2%} of glove"
        ),
        debug_mask=kept_mask,
        analysis_mask=detector_glove_mask,
    )

detect.owns_pipeline = True
