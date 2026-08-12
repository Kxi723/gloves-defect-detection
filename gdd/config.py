"""
Central configuration for the glove inspection pipeline.

Every tunable threshold lives here, grouped by the stage or detector that
uses it, so calibration never means editing algorithm code.

Two rules keep these numbers meaningful on photos nobody has seen yet.
Sizes are expressed as fractions of the palm radius or the glove area,
never in pixels, so they survive a change of camera or framing. And where
a threshold needs a reference level, that level is measured from the
glove in the photo itself rather than hard-coded, so it adapts to colour,
material and lighting on its own.

Usage:
    from gdd.config import get_config
    cfg = get_config()
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Per-stage configuration blocks
# --------------------------------------------------------------------------- #

@dataclass
class PreprocessConfig:
    """Parameters for noise reduction + illumination normalisation."""

    # Images larger than this (longest side, pixels) are downscaled first.
    # Phone photos are typically 3000-4000 px; ~1000 px keeps shape detail
    # while making filtering fast and thresholds resolution-independent.
    max_dimension: int = 1024

    # Gray-world white balance neutralises colour casts from indoor lighting.
    white_balance: bool = True

    # Bilateral filter: smooths sensor noise while keeping glove edges sharp
    # (a plain Gaussian would blur the very edges segmentation relies on).
    bilateral_diameter: int = 7        # pixel neighbourhood diameter
    bilateral_sigma_color: float = 50  # how different colours may be to mix
    bilateral_sigma_space: float = 50  # how far pixels may be to mix

    # CLAHE is NOT part of the shared chain; the only detector that wants
    # it owns its settings in FoldConfig below.


@dataclass
class SegmentationConfig:
    """Parameters for separating the glove from an arbitrary background."""

    # Width of the image-border strip (fraction of image size) sampled to
    # estimate the background colour. Assumes the glove does not cover the
    # entire frame edge-to-edge.
    border_fraction: float = 0.04

    # Candidate-mask scoring: a plausible glove occupies this fraction of
    # the frame. Masks outside the range are heavily penalised.
    #
    # A glove photographed to fill the frame is never 3% of it, and a loose
    # lower bound lets fragments win: on the cotton photos a mask covering
    # 5-8% of the frame (one stripe of the knit) beat the real silhouette.
    # Measured over the 15 photos, 0.10 / 0.13 / 0.16 all give the same
    # result, so this sits on a plateau rather than a knife edge. Together
    # with the texture cue and post-cleanup scoring it took the number of
    # plausible silhouettes from 11/15 to 15/15; the three work as a set.
    min_area_fraction: float = 0.12
    max_area_fraction: float = 0.90

    # Morphological cleanup (elliptical kernel, size in px on the resized
    # image). Opening removes speckle noise; closing bridges small gaps.
    open_kernel: int = 5
    close_kernel: int = 9

    # Internal cavities smaller than this fraction of the glove area are
    # treated as segmentation noise and filled unconditionally.
    min_hole_area_fraction: float = 0.0004

    # Window for the texture cue's local standard deviation, in px on the
    # resized image.
    texture_window: int = 9


@dataclass
class TearingConfig:
    """Parameters shared by the tearing / fingertip-tearing detectors."""

    # A hole counts as a tear if its area is within this fraction range of
    # the glove area (too small = noise, too large = segmentation failure).
    min_hole_area_fraction: float = 0.0008
    max_hole_area_fraction: float = 0.25

    # Shape gate separating real holes from residual crease artefacts. A
    # fold that survives thresholding is a long thin sliver; a tear is a
    # compact blob. Elongation is the ratio of the fitted ellipse axes and
    # extent is area / bounding-box area, both scale invariant.
    max_hole_elongation: float = 4.5
    min_hole_extent: float = 0.35

    # Boundary tears show up as deep, narrow convexity defects. Depth is
    # measured relative to the palm radius (scale invariant).
    min_defect_depth_ratio: float = 0.35   # depth / palm_radius
    max_defect_angle_deg: float = 60.0     # tears are narrow notches

    # Natural finger valleys are ALSO deep and narrow; they are excluded
    # by geometry instead: a valley's hull chord spans two different
    # fingertips. A defect endpoint counts as "at a fingertip" when it is
    # within this multiple of the palm radius of one.
    valley_endpoint_tip_ratio: float = 0.7

    # Fingertip classification: a defect belongs to a fingertip if it lies
    # within this multiple of the palm radius from a fingertip point.
    fingertip_radius_ratio: float = 0.55


@dataclass
class DirtConfig:
    """Parameters for the dirt / stain detector (appearance based)."""

    # Analysis is restricted to the glove interior, shrunk by this fraction
    # of the palm radius. The rim of a glove carries its own shading and
    # the segmentation boundary is never pixel exact, so both would
    # otherwise read as colour anomalies.
    interior_margin_ratio: float = 0.10

    # A pixel is anomalous when its lightness sits this many robust
    # standard deviations from the glove's own median. Robust statistics
    # (median + MAD) are used so the stain itself, and any pattern woven
    # into the glove, cannot drag the reference off.
    z_threshold: float = 2.0

    # A stain must cover at least this fraction of the glove and be
    # reasonably compact; speckle from knit texture is neither.
    min_area_fraction: float = 0.002
    min_extent: float = 0.50

    # THE decisive test. A stain lies ON TOP of the material and hides its
    # weave, so local texture energy inside it collapses; a shadow dims the
    # material without hiding anything, so its texture is unchanged.
    # Measured over the 15-photo set, texture inside the region divided by
    # the glove's median texture came out at 0.06-0.12 for the five real
    # stains and 0.92-1.54 for shadows on undamaged gloves - nearly an
    # order of magnitude apart, which is why this gate carries the
    # detector rather than the lightness threshold above.
    max_texture_ratio: float = 0.25
    texture_window: int = 9


@dataclass
class FoldConfig:
    """Parameters for the fold / crease detector (appearance based)."""

    interior_margin_ratio: float = 0.12

    # CLAHE, applied by this detector alone. A fold is a *relative*
    # brightness ridge, so equalised local contrast makes a faint crease in
    # a dim corner as detectable as a bright one, worth 17 points of
    # precision here. Every other stage measures absolute deviations and is
    # harmed by equalisation, which is why it is not in the shared chain.
    clahe_clip_limit: float = 2.5      # contrast amplification cap
    clahe_tile_grid: int = 8           # grid of local histogram tiles

    # Band-pass scales, expressed as fractions of the palm radius so they
    # track glove size and image resolution. The fine sigma blurs away the
    # manufactured grip texture; the coarse sigma removes the slow
    # illumination gradient. What survives in between is fold-sized.
    fine_sigma_ratio: float = 0.020
    coarse_sigma_ratio: float = 0.120

    # Ridge strength threshold, in robust standard deviations of the
    # band-pass response inside the glove.
    z_threshold: float = 2.2

    # A crease is a long thin ridge. Candidates must be at least this long
    # relative to the palm radius and at least this elongated, which is
    # what separates a fold from a blob of shading.
    min_length_ratio: float = 0.55
    min_elongation: float = 3.0

    # Number of qualifying creases needed before the glove is called
    # defective (a single faint ridge is usually just how the glove sits).
    min_crease_count: int = 1

    # Only creases within this many palm radii of the palm centre count.
    # The defect being inspected for is a fold across the PALM; creases out
    # near the cuff, the knuckles or the finger joints are how any glove
    # behaves when handled. Measured over the 15-photo set, restricting the
    # search this way cut false positives on undamaged palms from 6 to 2
    # while keeping the same 4 true detections.
    #
    # CALIBRATION NOTE: 1.4 was chosen by sweeping 0.8/1.1/1.4/1.8 over
    # only 15 photos, all of them defective. Re-check it once defect-free
    # samples exist; it is the least evidenced number in this file.
    palm_radius_ratio: float = 1.4


@dataclass
class FingertipConfig:
    """Parameters for locating the fingertips of a glove.

    Used by the tearing_at_finger detector, both to place the fingertip
    regions it searches and to recognise the natural valleys between
    fingers so they are not mistaken for tears.
    """

    expected_fingers: int = 5

    # Hull points further than this multiple of the palm radius from the
    # palm centre are fingertip candidates; candidates closer together
    # than the separation ratio collapse into a single tip.
    min_tip_distance_ratio: float = 1.35
    tip_merge_separation_ratio: float = 0.45


@dataclass
class PipelineConfig:
    """Top-level bundle handed to every pipeline stage and detector."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    tearing: TearingConfig = field(default_factory=TearingConfig)
    dirt: DirtConfig = field(default_factory=DirtConfig)
    fold: FoldConfig = field(default_factory=FoldConfig)
    fingertip: FingertipConfig = field(default_factory=FingertipConfig)


# --------------------------------------------------------------------------- #
# Material presets
# --------------------------------------------------------------------------- #
# Each preset lists only the values that differ from the defaults above.
# Calibrate these on the team's own photo set; the structure is
# {config block name: {field name: value}}.

def get_config() -> PipelineConfig:
    """Build the pipeline configuration.

    There is deliberately no per-material variant. An earlier version
    carried presets keyed on glove material, which was dropped for two
    reasons. Every preset would have been fitted to five photos of one
    specific glove, so it would perform *worse* than a neutral default on
    an unseen glove of the same material. And keying presets on material
    hard-codes the assumption that a given defect only ever appears on a
    given material, which is false — a fold crease on a cotton glove is
    still a fold crease.

    Thresholds are instead written to be material independent: sizes are
    fractions of the palm radius or the glove area, and reference values
    are measured from each photo's own glove (see `gdd/features.py`).
    """
    return PipelineConfig()
