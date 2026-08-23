from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessConfig:

    working_long_edge: int = 720

    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8

    median_ksize: int = 5


@dataclass(frozen=True)
class SegmentationConfig:

    border_band_frac: float = 0.03

    min_separability: float = 0.20

    open_ksize_frac: float = 0.010
    close_ksize_frac: float = 0.008

    min_glove_area_frac: float = 0.03

    max_contour_raggedness: float = 2.35


@dataclass(frozen=True)
class FeatureConfig:

    interior_erode_frac: float = 0.025


@dataclass(frozen=True)
class WrinkleConfig:

    tile_frac: float = 0.05

    tile_min_cover: float = 0.6

    tile_weber_threshold: float = 0.80

    wrinkled_tile_frac: float = 0.119


@dataclass(frozen=True)
class StainConfig:

    local_window_frac: float = 0.03

    residual_sigma: float = 6.0

    max_elongation: float = 2.0
    min_fill: float = 0.6

    min_component_area_frac: float = 0.00002

    min_stain_area_frac: float = 0.00106

    min_bright_fill: float = 0.40

    skip_when_worn: bool = True

    worn_skin_area_frac: float = 0.01


@dataclass(frozen=True)
class OversizeConfig:

    cuff_band_frac: float = 0.030

    min_cuff_to_wrist_ratio: float = 1.80


@dataclass(frozen=True)
class PipelineConfig:
    preprocess: PreprocessConfig = PreprocessConfig()
    segmentation: SegmentationConfig = SegmentationConfig()
    features: FeatureConfig = FeatureConfig()
    wrinkle: WrinkleConfig = WrinkleConfig()
    stain: StainConfig = StainConfig()
    oversize: OversizeConfig = OversizeConfig()


def get_config() -> PipelineConfig:
    return PipelineConfig()
