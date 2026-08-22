
from __future__ import annotations

from dataclasses import dataclass, field


# preprocessing
@dataclass
class PreprocessConfig:


    max_dimension: int = 1024


    white_balance: bool = True


    bilateral_diameter: int = 7
    bilateral_sigma_color: float = 50
    bilateral_sigma_space: float = 50


# segmentation
@dataclass
class SegmentationConfig:


    border_fraction: float = 0.04

    min_area_fraction: float = 0.08
    max_area_fraction: float = 0.90


    open_kernel: int = 5
    close_kernel: int = 9


    min_hole_area_fraction: float = 0.0004


    texture_window: int = 9


    illumination_border_fraction: float = 0.06


    background_mad_floor: float = 4.0
    background_z_clip: float = 8.0


# tearing
@dataclass
class TearingConfig:


    min_hole_area_fraction: float = 0.0008
    max_hole_area_fraction: float = 0.25


    max_hole_elongation: float = 4.5
    min_hole_extent: float = 0.35


    min_defect_depth_ratio: float = 0.35
    max_defect_angle_deg: float = 60.0


    valley_endpoint_tip_ratio: float = 0.7


    fingertip_radius_ratio: float = 0.55


    showthrough_margin_ratio: float = 0.05


    showthrough_z_threshold: float = 3.0


    showthrough_mad_floor: float = 1.5

    min_showthrough_area_fraction: float = 0.0015
    max_showthrough_area_fraction: float = 0.05

    max_showthrough_fingertip_fraction: float = 1.5

    min_showthrough_extent: float = 0.30
    max_showthrough_elongation: float = 4.5

    showthrough_min_backdrop_distance: float = 8.0

@dataclass
class DirtConfig:


    interior_margin_ratio: float = 0.10


    z_threshold: float = 2.0

    min_area_fraction: float = 0.010
    min_extent: float = 0.40


    max_texture_ratio: float = 0.25
    texture_window: int = 9

    min_glove_chroma: float = 12.0
    min_off_hue_distance: float = 15.0


# general tearing
@dataclass
class GeneralTearingConfig:
    min_candidate_area_fraction: float = 0.0020
    small_candidate_area_fraction: float = 0.0100
    small_candidate_min_edge_density: float = 0.045
    min_saturation: float = 30.0
    min_value: float = 45.0
    reject_bright_low_saturation: bool = True
    bright_value_threshold: float = 195.0
    low_saturation_threshold: float = 40.0
    frame_margin_fraction: float = 0.008
    confidence_full_area_fraction: float = 0.025


    light_glove_value_cutoff: float = 105.0
    dark_skin_hue_max: float = 25.0
    dark_skin_saturation_min: float = 35.0
    dark_skin_saturation_max: float = 100.0
    dark_skin_value_min: float = 130.0
    dark_min_bbox_area_fraction: float = 0.008


    light_skin_hue_max: float = 20.0
    light_skin_saturation_min: float = 30.0
    light_skin_value_min: float = 120.0
    light_saturation_residual_min: float = 20.0
    light_min_component_area_fraction: float = 0.0012
    light_max_component_area_fraction: float = 0.02
    light_min_short_side_pixels: int = 10
    light_cuff_exclusion_fraction: float = 0.30
    light_local_sigma: float = 12.0


# spotting
@dataclass
class SpottingConfig:
    interior_margin_ratio: float = 0.08
    blur_kernel: int = 3


    light_glove_value_cutoff: float = 100.0
    light_dark_delta: float = 60.0
    light_absolute_value_max: float = 90.0


    dark_bright_delta: float = 75.0
    dark_low_saturation_max: float = 35.0
    dark_high_saturation_min: float = 80.0
    dark_colour_value_delta: float = 0.0


    min_area_fraction: float = 0.00015
    max_area_fraction: float = 0.0045
    min_extent: float = 0.30
    max_elongation: float = 3.0
    min_circularity: float = 0.20


    min_spot_count: int = 3
    min_spread_ratio: float = 2.0
    min_total_area_fraction: float = 0.0023
    full_confidence_count: int = 8
    full_confidence_spread_ratio: float = 5.0

    open_kernel: int = 3


# incomplete beading
@dataclass
class BeadingConfig:


    skin_hue_max: int = 25
    skin_saturation_min: int = 38
    skin_saturation_max: int = 180
    skin_value_min: int = 70
    skin_open_kernel: int = 5
    skin_close_kernel: int = 9


    min_cuff_depth_fraction: float = 0.18
    max_cuff_depth_fraction: float = 0.40


    min_profile_points: int = 20
    profile_sample_target: int = 80
    min_roughness_ratio: float = 0.07
    full_confidence_roughness_ratio: float = 0.20
    boundary_thickness: int = 3


@dataclass
class FoldConfig:

    interior_margin_ratio: float = 0.12


    clahe_clip_limit: float = 2.5
    clahe_tile_grid: int = 8


    fine_sigma_ratio: float = 0.020
    coarse_sigma_ratio: float = 0.120


    z_threshold: float = 2.2


    min_length_ratio: float = 0.55
    min_elongation: float = 3.0


    min_crease_count: int = 1

    palm_radius_ratio: float = 1.4

    ridge_bridge_ratio: float = 0.18
    ridge_bridge_angles: int = 12


    bridge_min_elongation: float = 2.5


    bridged_min_length_ratio: float = 0.80


    group_angle_degrees: float = 22.0
    group_collinear_degrees: float = 25.0
    group_max_gap: float = 0.55

    max_lightness_delta: float = -8.0

    stripe_tensor_sigma_ratio: float = 0.035
    stripe_deviation_degrees: float = 25.0


    use_chroma_residual: bool = True


@dataclass
class FingertipConfig:

    expected_fingers: int = 5


    min_tip_distance_ratio: float = 1.35
    tip_merge_separation_ratio: float = 0.45

    tip_frame_cut_reach_ratio: float = 0.5


@dataclass
class PipelineConfig:

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    tearing: TearingConfig = field(default_factory=TearingConfig)
    general_tearing: GeneralTearingConfig = field(default_factory=GeneralTearingConfig)
    dirt: DirtConfig = field(default_factory=DirtConfig)
    spotting: SpottingConfig = field(default_factory=SpottingConfig)
    beading: BeadingConfig = field(default_factory=BeadingConfig)
    fold: FoldConfig = field(default_factory=FoldConfig)
    fingertip: FingertipConfig = field(default_factory=FingertipConfig)


def get_config() -> PipelineConfig:
    return PipelineConfig()
