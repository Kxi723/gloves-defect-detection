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

    # Background-model cue. Border strip width used both to fit the
    # illumination surface and to sample the background colour; wider than
    # `border_fraction` above because a surface fit needs more support than
    # a median does.
    illumination_border_fraction: float = 0.06

    # Robust spread floor and per-channel clip for the background model,
    # in LAB units. Without the floor the near-zero chroma spread of a grey
    # desk turns any coloured pixel into a 27-sigma outlier, and that tail
    # drags Otsu past the glove itself.
    background_mad_floor: float = 4.0
    background_z_clip: float = 8.0


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

    # ---- show-through channel ------------------------------------------ #
    # A through-tear exposes whatever is BEHIND the glove. Laid flat on a
    # backdrop that is the background, and the hole channel above sees it.
    # Worn, it is the hand, which fills the opening so completely that the
    # silhouette barely changes and no hole ever forms. Measured on the
    # four worn nitrile photos, the tear moved the outline so little that
    # neither the hole nor the notch channel produced a single candidate.
    # Same physical event, different backing, so it needs its own channel.
    #
    # What it looks for is a patch inside the glove outline that is not
    # glove MATERIAL, found by colour rather than by shape.

    # The rim is excluded, as everywhere else in the project, but by much
    # less than the dirt detector uses: a fingertip tear sits at the very
    # end of a finger and a 0.10R margin erodes it away.
    showthrough_margin_ratio: float = 0.05

    # CHROMA, not lightness, is what separates the exposed hand from the
    # things that also brighten or darken a glove. A specular highlight, a
    # shadow and a crease all move a patch along the glove's own lightness
    # axis and leave its hue where it was; what is behind the glove has no
    # reason to share that hue. Deviation is measured in robust standard
    # deviations of the glove's own a/b, so it adapts to glove colour
    # instead of hard-coding what skin looks like — which also survives the
    # gray-world white balance that defeated an absolute skin-colour test
    # (see the note in gdd/pipeline.py).
    showthrough_z_threshold: float = 3.0

    # Floor on the chroma spread, in LAB units, for the same reason the
    # background model needs one: a near-neutral glove has almost no chroma
    # variation, so without a floor its own sensor noise scores 20 sigma.
    showthrough_mad_floor: float = 1.5

    # Area window as a fraction of the glove. Measured over the set, the
    # three cleanly-segmented tears came out at 0.29%, 0.40% and 0.40%,
    # while the chroma speckle that survives on undamaged gloves near a
    # fingertip sat at 0.08%, 0.08% and 0.09% — so the floor sits in a
    # roughly two-fold gap on both sides. CALIBRATION NOTE: three positives
    # is thin evidence. The ceiling is deliberately loose and only rejects
    # wholesale segmentation failure; the forearm is kept out by the
    # fingertip gate, not by its size.
    min_showthrough_area_fraction: float = 0.0015
    max_showthrough_area_fraction: float = 0.05

    # THE size gate, and the one that needs no fitting: an opening in a
    # finger cannot be bigger than the finger. Expressed against the
    # cross-section of the fingertip the patch sits on, measured from the
    # photo's own distance transform, so it is independent of glove size,
    # resolution and how much cuff the glove happens to have.
    #
    # This is what tells a tear from a patch of differently-coloured
    # MATERIAL at the fingertips. On dirty_latex_4 — a grey knit glove
    # with blue latex dipped tips — the blue tips are a large chroma
    # outlier against the glove's mostly-grey median, and they sit exactly
    # where this detector looks. They measured 3.25 fingertips of area;
    # the four nitrile tears measured 0.25, 0.46, 0.50 and 0.70.
    #
    # The ceiling is 1.5 rather than the 1.0 that the bare physical
    # statement implies, because the cross-section is measured from the
    # SILHOUETTE and a finger bent towards the camera is foreshortened,
    # so the measurement understates the real finger. tear_latex_2 is
    # exactly that photo — the finger is curled so the tear faces the
    # lens — and it measured 1.00, passing by nothing at all. 1.5 gives
    # that pose room while still clearing the 3.25 case by more than
    # twofold.
    #
    # Comparing a patch with its LOCAL surroundings was tried first and
    # fails: the ring around a dipped fingertip straddles the dip
    # boundary, so its local contrast (48.3) came out HIGHER than any real
    # tear (7.8 to 12.2).
    max_showthrough_fingertip_fraction: float = 1.5

    # A tear is a compact opening. The one undamaged glove that produced a
    # large chroma region near a fingertip (good_cotton_5, 1.68% of the
    # glove) was a sprawling stripe of knit with an extent of 0.11.
    min_showthrough_extent: float = 0.30
    max_showthrough_elongation: float = 4.5

    # ---- is it really BEHIND the glove? -------------------------------- #
    # A patch inside the outline is only tear evidence if it is something
    # seen THROUGH the glove. It can also be a piece of backdrop that
    # segmentation welded onto the silhouette, which happened on
    # tear_nitrile_1 where a band of glare on the wall crossed the
    # fingertips. Both gates below compare the patch with the backdrop,
    # which segmentation already measures from the image border strip.
    #
    # Matching the backdrop's colour. Then it either IS backdrop that got
    # swallowed, or it is backdrop genuinely visible through a hole — and
    # the hole channel already owns that case, so nothing is lost by
    # declining it here. Measured in LAB units: the swallowed wall patch
    # came out at 2.3 from the backdrop while the three real tears were at
    # 13.5, 17.8 and 18.5.
    showthrough_min_backdrop_distance: float = 8.0

    # REMOVED, and worth recording so nobody re-adds it. A second gate
    # rejected any patch brighter than the backdrop, on the argument that
    # nothing seen through an opening can out-shine the open backdrop
    # beside it. That argument only holds when the thing behind the tear
    # IS the backdrop, which means a glove laid flat and empty. On a WORN
    # glove what shows through is the hand, lit from the front like the
    # rest of the scene, and skin out-reflects a grey desk easily — and
    # worn is the case this channel exists for.
    #
    # It cost a real detection. On tear_latex_1 the exposed skin measured
    # L=208 against a backdrop of L=134, so the gate rejected the one
    # thing it was supposed to find. The same photo also showed the gate
    # was numerically unsound: it divided by (backdrop L - glove L), which
    # was 134-132 = 2 there, so the ratio exploded to 38. It only looked
    # safe on nitrile because that denominator happened to be 75.
    #
    # Removing it changed nothing on the 28-photo set. What it was built
    # for — a band of wall glare welded into the mask on the original
    # tear_nitrile_1 — is a capture failure that produced an unusable
    # photo anyway, and that photo has since been re-shot.


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
    #
    # Both were re-fitted on 2026-08-17 once cotton samples existed, by
    # sweeping the pair over the whole 27-photo set.
    #
    # AREA 0.002 -> 0.010. The old value admitted a 0.44% region on
    # good_latex_2 — the blue latex fingertip cap, which really is darker
    # and smoother than the grey knit body it is measured against, so no
    # appearance rule can reject it (see the two-tone note in dirty.py).
    # Excluding it by size costs nothing here: the five real stains cover
    # 1.5% to 7.2%, the nearest false region 0.44%.
    #
    # EXTENT 0.50 -> 0.40. A triangle inscribed in its bounding box has an
    # extent of exactly 0.50, and dirt laid on a glove is often roughly
    # wedge shaped, so the old gate sat precisely on the geometry it was
    # meant to admit: dirty_cotton_1 cleared it by 0.0021. That is not a
    # margin. 0.40 is the LOOSEST value that still excludes the 5.18%
    # patch of backdrop the ragged mask swallows beside the thumb on
    # dirty_cotton_2 (extent 0.36) — at 0.35 that patch is accepted, and
    # because the photo is labelled dirty it would score as a true
    # positive while boxing the wrong thing.
    min_area_fraction: float = 0.010
    min_extent: float = 0.40

    # FIRST way in. A stain lies ON TOP of the material and hides its
    # weave, so local texture energy inside it collapses; a shadow dims the
    # material without hiding anything, so its texture is unchanged.
    # Measured over the 15-photo set, texture inside the region divided by
    # the glove's median texture came out at 0.06-0.12 for the five real
    # stains and 0.92-1.54 for shadows on undamaged gloves - nearly an
    # order of magnitude apart.
    max_texture_ratio: float = 0.25
    texture_window: int = 9

    # SECOND way in, for dirt that does not flatten anything. The test
    # above assumes dirt is fine or greasy enough to fill the weave. A
    # COARSE powder does the opposite: it heaps up, the material still
    # shows between the grains, and the region ends up as textured as the
    # glove. The cream coffee powder on dirty_latex_1 measured 0.87, so
    # the texture gate rejected it outright even though its size, shape
    # and colour all passed.
    #
    # What still separates it is hue. Lighting can only slide a surface's
    # colour along the ray from neutral grey through the surface's own
    # hue - a highlight washes it toward neutral, a shadow deepens it -
    # so any sideways departure from that ray is colour the illumination
    # cannot explain. See `features.off_hue_distance`. Measured over the
    # candidate regions on the blue latex glove, the coffee powder scored
    # 34.7 while the two specular highlights on undamaged gloves scored
    # 5.1 and 6.4.
    #
    # The ray has no direction when the glove itself is near neutral, so
    # the test only applies above a minimum glove chroma. Nitrile and the
    # grey knit side measure 4.0-5.1 and are correctly excluded; the blue
    # latex measures 22-25.
    #
    # CALIBRATION NOTE: one positive photo only. Both numbers sit on a
    # plateau - every combination of min_glove_chroma in 8-20 and
    # min_off_hue_distance in 10-25 gives the same 1 detection and 0 added
    # false positives over the other 21 photos - but a plateau measured
    # against a single stain is still a single stain. Re-check once more
    # dirty photos exist.
    min_glove_chroma: float = 12.0
    min_off_hue_distance: float = 15.0


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

    # ---- joining a broken crease --------------------------------------- #
    # A crease's contrast varies along its length, so parts of it dip below
    # the z threshold and it arrives as several collinear fragments. The
    # shared 9x9 closing only bridges about nine pixels, far less than those
    # gaps, so the reported length was the longest FRAGMENT rather than the
    # crease. On fold_latex_2 the accepted piece was 1.00R while three more
    # fragments (0.89R, 0.51R, 0.40R) lay along the same line.
    #
    # Closing with a LINE-shaped element, repeated over a set of angles and
    # unioned, bridges gaps along a direction without fattening across it,
    # which a disc of the same size would do. Length is a fraction of the
    # palm radius like everything else here.
    ridge_bridge_ratio: float = 0.18
    ridge_bridge_angles: int = 12

    # Only fragments at least this elongated may be bridged. Without it,
    # bridging invented creases on three undamaged cotton gloves, because a
    # striped knit's speckle is collinear by construction and a directional
    # closing threads it into a convincing line.
    bridge_min_elongation: float = 2.5

    # Length a BRIDGED candidate must reach, as opposed to the plain
    # min_length_ratio applied to unjoined ones. Joining can only lengthen a
    # candidate, so one that barely passes has borrowed its length from a
    # neighbour rather than earned it.
    bridged_min_length_ratio: float = 0.80

    # ---- extending an accepted crease ---------------------------------- #
    # Applied only after a crease has passed every gate, so these cannot
    # create a detection and cannot affect precision. A fragment joins when
    # its direction agrees with the crease's within `angle`, the line
    # between their centres runs along that direction within `collinear`
    # (end to end, not side by side), and the gap is under `max_gap`.
    group_angle_degrees: float = 22.0
    group_collinear_degrees: float = 25.0
    group_max_gap: float = 0.55

    # ---- reflection rejection ------------------------------------------ #
    # A pressed crease is a SHADOW. The band-pass keeps both signs, because
    # a fold can show a bright crest beside its dark trough, but the region
    # as a whole is darker than the glove around it. A specular highlight is
    # the opposite, and that is what the one remaining false positive was:
    # the lit edge of the latex coating on good_latex_5.
    #
    # Measured over the fold set, median lightness inside the crease minus
    # the glove median came out at -36, -37, -47 and -49 for the four real
    # creases and +49 for the highlight. No overlap, and a wide gap. This
    # matters more than it looks: the highlight is 1.08R long while two of
    # the real creases are 0.55R and 0.93R, so LENGTH can never separate
    # them however it is tuned.
    max_lightness_delta: float = -8.0

    # ---- stripe-distortion channel ------------------------------------- #
    # For woven gloves the band-pass channel is blind: the knit's stripes
    # are 10-20 px bands, exactly fold-scale, so they inflate the robust
    # spread that sets the threshold. Measured headroom (99th percentile of
    # the response over the gate) was 0.82-0.94 on undamaged cotton and
    # 0.81-0.92 on FOLDED cotton - identical, and never above 1.0, so the
    # detector could not fire on that material at all.
    #
    # Folding a woven fabric bends its yarn lines, and an undamaged glove
    # has none of that. Local direction comes from the structure tensor,
    # and this is the angle by which it must depart from the glove's own
    # dominant direction. An ABSOLUTE angle, deliberately: a percentile
    # gate keeps its top few per cent of pixels whether or not anything is
    # wrong, so it cannot express "this glove is undisturbed".
    stripe_tensor_sigma_ratio: float = 0.035
    stripe_deviation_degrees: float = 25.0

    # ---- chroma-residual channel --------------------------------------- #
    # The stripes are two yarn COLOURS, so part of them lives in chroma,
    # while a fold shades whichever yarn lies under it and lives only in
    # lightness. Regressing the lightness band-pass on the chroma band-pass
    # and keeping the residual suppresses the weave. On its own the residual
    # does not separate (the yarns differ mostly in lightness, not hue, so
    # the correlation is only 0.5-0.6); it works because the residual leaves
    # the fold as one long line and the weave as speckle, which the existing
    # length and elongation gates then tell apart.
    use_chroma_residual: bool = True


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

    # A candidate is discarded when the glove runs off the image edge
    # within this many palm radii of it, because the end of that
    # protrusion was never photographed. On a worn glove the forearm
    # stump is the farthest hull point of all and was being reported as a
    # fingertip; see :func:`gdd.features.locate_fingertips`. Half a palm
    # radius is short enough to leave a thumb photographed close to the
    # edge alone (measured at 0.05R from the frame on tear_nitrile_4, and
    # correctly kept).
    tip_frame_cut_reach_ratio: float = 0.5


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
