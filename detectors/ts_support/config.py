

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PreprocessConfig:


    max_dimension: int = 1024
    median_kernel: int = 3


@dataclass
class SegmentationConfig:


    border_fraction: float = 0.04
    min_area_fraction: float = 0.12
    max_area_fraction: float = 0.90
    open_kernel: int = 5
    close_kernel: int = 9
    texture_window: int = 9


@dataclass
class SkinColourConfig:


    red_min: int = 85
    green_min: int = 45
    blue_min: int = 25
    red_green_difference_min: int = 12
    red_green_difference_max: int = 95
    red_blue_difference_min: int = 28
    channel_range_min: int = 18
    mean_brightness_max: float = 220.0


@dataclass
class FingerNotEnoughConfig:


    cuff_start_fraction: float = 0.84
    contact_kernel_fraction: float = 0.035
    min_component_pixels: int = 40
    min_component_fraction: float = 0.001
    min_exposed_area_fraction: float = 0.0022
    marginal_area_ratio: float = 0.80
    min_skeleton_branch_fraction: float = 0.010
    area_score_scale: float = 0.04
    skeleton_score_scale: float = 0.12


@dataclass
class PlasticContaminationConfig:


    interior_kernel_fraction: float = 0.016
    texture_window_fraction: float = 0.015
    open_kernel_fraction: float = 0.003
    close_kernel_fraction: float = 0.010
    component_merge_fraction: float = 0.040
    minimum_analysis_pixels: int = 100
    minimum_component_pixels: int = 40
    minimum_merge_pixels: int = 2
    score_fraction_scale: float = 0.10


    finger_end_fraction: float = 0.48
    palm_end_fraction: float = 0.82


    blue_latex_ratio_threshold: float = 0.42
    knitted_texture_threshold: float = 18.0


    blue_support_ratio: float = 0.39
    blue_support_red_multiplier: float = 1.12
    blue_support_green_multiplier: float = 1.02
    blue_support_close_fraction: float = 0.012
    blue_support_padding_pixels: int = 5
    blue_support_min_component_pixels: int = 100


    latex_finger_saturation_drop: float = 0.110
    latex_finger_intensity_gain: float = 10.0
    latex_finger_max_texture_ratio: float = 0.48
    latex_finger_smooth_intensity_gain: float = 7.0
    latex_palm_saturation_drop: float = 0.100
    latex_palm_intensity_gain: float = 10.0
    latex_palm_max_texture_ratio: float = 0.58
    latex_palm_smooth_intensity_gain: float = 6.0
    latex_smooth_max_saturation: float = 0.28
    latex_max_candidate_saturation: float = 0.30
    latex_min_component_fraction: float = 0.006
    latex_min_extent: float = 0.24
    latex_max_elongation: float = 2.25
    latex_relative_component_fraction: float = 0.10


    nitrile_palm_end_fraction: float = 0.88
    nitrile_finger_intensity_gain: float = 20.0
    nitrile_palm_intensity_gain: float = 20.0
    nitrile_min_pixel_texture_ratio: float = 1.10
    nitrile_close_kernel_fraction: float = 0.020
    nitrile_min_texture_ratio: float = 0.80
    nitrile_min_component_fraction: float = 0.004
    nitrile_min_extent: float = 0.25
    nitrile_max_elongation: float = 2.00
    nitrile_relative_component_fraction: float = 0.08


@dataclass
class ImproperRollConfig:


    blue_latex_ratio_threshold: float = 0.42
    texture_window_fraction: float = 0.015
    cotton_texture_threshold: float = 18.0
    minimum_analysis_pixels: int = 100


    cotton_mask_support_dilate_fraction: float = 0.035
    cotton_mask_vertical_close_fraction: float = 0.055
    cotton_mask_compact_close_fraction: float = 0.014
    cotton_mask_median_fraction: float = 0.009
    cotton_colour_min_component_pixels: int = 100


    cotton_yellow_red_min: int = 115
    cotton_yellow_green_min: int = 85
    cotton_yellow_blue_max: int = 115
    cotton_yellow_red_blue_difference_min: int = 45
    cotton_yellow_green_blue_difference_min: int = 25
    cotton_yellow_search_start_fraction: float = 0.48
    cotton_normal_band_y_fraction: float = 0.969
    cotton_min_band_shift_fraction: float = 0.049
    cotton_min_fold_depth_fraction: float = 0.090
    cotton_min_fold_area_fraction: float = 0.070
    cotton_max_aspect: float = 1.42
    cotton_band_min_area: int = 30
    cotton_score_scale: float = 0.30


    contact_kernel_fraction: float = 0.055
    skin_min_component_pixels: int = 80
    skin_min_component_fraction: float = 0.002
    skin_region_x_start_fraction: float = 0.10
    skin_region_x_end_fraction: float = 0.90
    skin_region_y_start_fraction: float = 0.52
    skin_region_y_end_fraction: float = 1.12


    nitrile_max_aspect: float = 1.35
    nitrile_normal_edge_score: float = 17.0
    nitrile_min_edge_increase: float = 8.0
    nitrile_min_edge_continuity: float = 0.10
    nitrile_edge_start_fraction: float = 0.65
    nitrile_edge_end_fraction: float = 0.93
    nitrile_edge_pixel_threshold: float = 60.0
    nitrile_edge_x_start_fraction: float = 0.08
    nitrile_edge_x_end_fraction: float = 0.92
    nitrile_edge_half_band_fraction: float = 0.025
    nitrile_aspect_score_weight: float = 0.50
    nitrile_edge_score_weight: float = 0.50
    nitrile_aspect_score_scale: float = 0.30
    nitrile_continuity_score_scale: float = 0.25

    cuff_box_start_fraction: float = 0.68
    cuff_box_height_fraction: float = 0.28


@dataclass
class PipelineConfig:


    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    skin_colour: SkinColourConfig = field(default_factory=SkinColourConfig)
    finger_not_enough: FingerNotEnoughConfig = field(
        default_factory=FingerNotEnoughConfig
    )
    plastic_contamination: PlasticContaminationConfig = field(
        default_factory=PlasticContaminationConfig
    )
    improper_roll: ImproperRollConfig = field(default_factory=ImproperRollConfig)


def get_config() -> PipelineConfig:


    return PipelineConfig()
