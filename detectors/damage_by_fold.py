"""
Damage by fold — a permanent crease pressed across the palm.

The problem
-----------
A work glove is covered in detail at two very different scales. The
manufactured grip coating on latex is a dense, fine, high-frequency
crinkle. A fold is a broad ridge an order of magnitude wider. On top of
both sits a slow illumination gradient across the photo. Thresholding the
image directly finds all three at once and cannot tell them apart.

The method
----------
Three independent channels look for the same defect, because no single one
of them survives every material. A glove is called folded if ANY channel
finds a qualifying crease.

1. **Shading ridge** (the original channel; works on coated latex).
   Equalise local contrast (CLAHE), band-pass the lightness with a
   difference of Gaussians — the fine sigma blurs the grip texture away,
   the coarse sigma removes the illumination gradient, so what survives is
   fold-sized by construction — then keep the strongest ridges by robust
   z-score.

2. **Stripe distortion** (works on woven and knitted gloves).
   Folding a woven fabric bends its yarn lines. Local direction comes from
   the structure tensor, and a crease shows up as an elongated region where
   that direction departs from the glove's own dominant one.

3. **Chroma residual** (a weave-suppressed version of channel 1).
   The weave is partly a difference of yarn COLOUR, a fold is purely
   shading, so regressing the lightness band-pass on the chroma band-pass
   and keeping the residual removes much of the weave.

Every candidate then passes the same three gates: it must lie within
``palm_radius_ratio`` of the palm centre, it must be long and elongated
(a crease is a line, a patch of shading is a blob), and it must be a
SHADOW rather than a highlight.

Why each gate exists, measured
------------------------------
* The palm restriction is what makes this defect separable at all. Every
  handled glove carries creases at the cuff and over the finger joints, so
  searching the whole glove flagged all five undamaged nitrile photos.
* The shadow gate removes the last false positive, the lit edge of the
  latex coating on good_latex_5. It is not a refinement of the length
  gate — it replaces it for this case, because that highlight is 1.08R
  long while two genuine creases are 0.55R and 0.93R, so no length
  threshold can separate them.
* Channels 2 and 3 exist because channel 1 is structurally blind on the
  striped cotton knit, where the weave inflates the very reference used to
  threshold it. See ``FoldConfig`` for the measurements.

Owner: Jason. Tunables live in ``PipelineConfig.fold``.

CALIBRATION NOTE: the three channels were developed against a 16-photo set
containing 5 folded gloves, and those same 5 photos were used to choose the
thresholds. That is not a held-out result. The mechanisms are physical
rather than fitted, but the numbers still need confirming on photos that
took no part in tuning.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from gdd.config import FoldConfig, PipelineConfig
from gdd.features import (
    BBox, DefectResult, glove_interior, palm_center_and_radius, robust_stats,
)
from gdd.preprocessing import normalize_illumination
from gdd.segmentation import SegmentationResult


# --------------------------------------------------------------------------- #
# Channel 1: shading ridge
# --------------------------------------------------------------------------- #

def _band_pass(channel: np.ndarray, palm_radius: float,
               cfg: FoldConfig) -> np.ndarray:
    """Difference of two Gaussian-smoothed copies, at fold scale.

    Both sigmas are fractions of the palm radius, so the filter tracks
    glove size and image resolution rather than pixel counts.
    """
    fine = max(1.0, cfg.fine_sigma_ratio * palm_radius)
    coarse = max(fine + 1.0, cfg.coarse_sigma_ratio * palm_radius)
    return (cv2.GaussianBlur(channel, (0, 0), fine)
            - cv2.GaussianBlur(channel, (0, 0), coarse))


def fold_ridge_response(image: np.ndarray, interior: np.ndarray,
                        palm_radius: float, cfg: FoldConfig) -> np.ndarray:
    """Band-pass filtered lightness, isolating fold-scale structure.

    CLAHE is applied here and nowhere else in the pipeline; see
    ``FoldConfig.clahe_clip_limit`` for why this detector is the exception.
    The result keeps its SIGN, which the shadow gate later depends on.
    """
    equalized = normalize_illumination(image, cfg.clahe_clip_limit,
                                       cfg.clahe_tile_grid)
    lightness = cv2.cvtColor(equalized, cv2.COLOR_BGR2LAB)[:, :, 0]
    response = _band_pass(lightness.astype(np.float32), palm_radius, cfg)
    response[interior == 0] = 0.0
    return response


# --------------------------------------------------------------------------- #
# Channel 2: stripe distortion
# --------------------------------------------------------------------------- #

def stripe_deviation(image: np.ndarray, interior: np.ndarray,
                     palm_radius: float, cfg: FoldConfig) -> np.ndarray:
    """Degrees by which the local weave direction departs from the glove's.

    Orientation is handled in doubled-angle form. A yarn line at 179 deg
    and one at 1 deg are the same direction, and averaging raw angles would
    call them opposite; doubling makes them coincide.
    """
    equalized = normalize_illumination(image, cfg.clahe_clip_limit,
                                       cfg.clahe_tile_grid)
    gray = cv2.cvtColor(equalized, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sigma = max(1.0, cfg.stripe_tensor_sigma_ratio * palm_radius)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)

    cos2 = jxx - jyy
    sin2 = 2.0 * jxy
    magnitude = np.maximum(np.hypot(cos2, sin2), 1e-6)

    inside = interior > 0
    if not inside.any():
        return np.zeros(gray.shape, np.float32)

    # The glove's own dominant direction, summed over the glove so that
    # strongly directional areas carry the vote.
    total_c = float(cos2[inside].sum())
    total_s = float(sin2[inside].sum())
    total = max(float(np.hypot(total_c, total_s)), 1e-6)
    total_c, total_s = total_c / total, total_s / total

    aligned = np.clip((cos2 / magnitude) * total_c
                      + (sin2 / magnitude) * total_s, -1.0, 1.0)
    deviation = np.degrees(np.arccos(aligned)) / 2.0     # 0..90 degrees
    deviation[~inside] = 0.0
    return deviation


# --------------------------------------------------------------------------- #
# Channel 3: chroma residual
# --------------------------------------------------------------------------- #

def chroma_residual(image: np.ndarray, interior: np.ndarray,
                    palm_radius: float, cfg: FoldConfig) -> np.ndarray:
    """Lightness structure that the weave's colour pattern cannot explain."""
    equalized = normalize_illumination(image, cfg.clahe_clip_limit,
                                       cfg.clahe_tile_grid)
    lab = cv2.cvtColor(equalized, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = np.abs(_band_pass(lab[:, :, 0], palm_radius, cfg))
    chroma = np.hypot(_band_pass(lab[:, :, 1], palm_radius, cfg),
                      _band_pass(lab[:, :, 2], palm_radius, cfg))

    inside = interior > 0
    residual = np.zeros(lightness.shape, np.float32)
    if np.count_nonzero(inside) < 100:
        return residual

    design = np.stack([chroma[inside], np.ones(int(inside.sum()), np.float32)],
                      axis=1)
    coefficients, *_ = np.linalg.lstsq(design, lightness[inside], rcond=None)
    residual[inside] = lightness[inside] - design @ coefficients
    return residual


# --------------------------------------------------------------------------- #
# Shared candidate extraction
# --------------------------------------------------------------------------- #

def _clean(binary: np.ndarray) -> np.ndarray:
    """Close along the ridge so speckle does not break it, then despeckle."""
    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return cv2.morphologyEx(
        closed, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def _line_kernel(length: int, angle_degrees: float) -> np.ndarray:
    """A one-pixel-wide line through the centre of a square kernel."""
    kernel = np.zeros((length, length), np.uint8)
    centre = length // 2
    radians = np.deg2rad(angle_degrees)
    for step in np.linspace(-centre, centre, 2 * length):
        x = int(round(centre + step * np.cos(radians)))
        y = int(round(centre + step * np.sin(radians)))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    return kernel


def _line_like(binary: np.ndarray, min_elongation: float) -> np.ndarray:
    """Keep only components that are already line-shaped.

    This gate has to come BEFORE bridging, and it is not optional. On a
    striped knit the leftover speckle is collinear by construction, because
    it lies along the stripes, so a directional closing joins it into
    convincing long lines: bridging without this filter invented creases of
    1.53R, 1.21R and 0.73R on three undamaged cotton gloves. Bridging is
    meant to rejoin a broken LINE, so only line-shaped pieces may take part.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros_like(binary)
    for index in range(1, count):
        width = stats[index, cv2.CC_STAT_WIDTH]
        height = stats[index, cv2.CC_STAT_HEIGHT]
        area = stats[index, cv2.CC_STAT_AREA]
        if area < 12:
            continue
        # Ratio of the diagonal of the bounding box to the mean thickness
        # of the component, a cheap elongation that needs no ellipse fit.
        span = float(np.hypot(width, height))
        thickness = area / max(span, 1.0)
        if span / max(thickness, 1e-6) >= min_elongation:
            keep[labels == index] = 255
    return keep


def _bridged_variants(binary: np.ndarray, palm_radius: float,
                      cfg: FoldConfig) -> List[np.ndarray]:
    """One closed copy per direction, deliberately NOT unioned.

    A line-shaped element bridges gaps along its own direction without
    fattening across it — but only one direction at a time. Unioning the
    twelve results puts back exactly the thickening a disc would have
    caused: on fold_latex_2 that did join the fragments, 1.00R becoming 1.86R,
    yet elongation collapsed from 7.6 to 2.3 and the crease was then thrown
    out for being a blob.

    So each direction is kept as its own candidate map. The one aligned
    with the crease yields a long thin contour; the others yield nothing
    that passes, and no direction can dilute another.
    """
    seeds = _line_like(binary, cfg.bridge_min_elongation)
    length = max(5, int(cfg.ridge_bridge_ratio * palm_radius)) | 1
    variants = [binary]
    for index in range(cfg.ridge_bridge_angles):
        angle = 180.0 * index / cfg.ridge_bridge_angles
        variants.append(cv2.bitwise_or(
            binary,
            cv2.morphologyEx(seeds, cv2.MORPH_CLOSE,
                             _line_kernel(length, angle))))
    return variants


def _shaped_creases(binary: np.ndarray, palm_region: np.ndarray,
                    palm_radius: float, cfg: FoldConfig,
                    bridge: bool = False) -> List[Tuple[np.ndarray, float]]:
    """Contours that are long enough and thin enough to be a crease.

    The centre of each contour must lie in ``palm_region``. Searching the
    whole glove and applying the palm test only to the centre WAS tried, so
    that a crease running past the disc could be measured in full — about
    2000 of fold_latex_2's thresholded pixels sit outside it. It lengthened the
    real creases but cost two false positives (undamaged cotton and
    nitrile), taking precision from 100% to 71%, so the search stays inside
    the palm and the reported length is the part of the crease that lies
    there. The centre test is kept because it costs nothing and documents
    the intent.
    """
    cleaned = _clean(binary)
    variants = (_bridged_variants(cleaned, palm_radius, cfg) if bridge
                else [cleaned])

    out: List[Tuple[np.ndarray, float]] = []
    for position, variant in enumerate(variants):
        # Bridging can only ever ADD length, so a bridged candidate that
        # merely scrapes past the ordinary gate has not been shown to be a
        # crease — it is a short fragment that borrowed length from its
        # neighbours. Variant 0 is the unbridged map and keeps the normal
        # gate; the joined ones must clear a higher bar. Without this,
        # good_latex_4 produced a 0.57R false crease against a 0.55R gate,
        # while the genuine bridged creases measure 2.05R, 2.83R and 2.90R.
        minimum = palm_radius * (cfg.min_length_ratio if position == 0
                                 else cfg.bridged_min_length_ratio)
        contours, _ = cv2.findContours(variant, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 5:
                continue
            # A fitted ellipse gives length (major axis) and elongation.
            (cx, cy), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
            major = max(axis_a, axis_b)
            minor = max(min(axis_a, axis_b), 1e-6)
            if major < minimum or major / minor < cfg.min_elongation:
                continue
            row, col = int(round(cy)), int(round(cx))
            if not (0 <= row < palm_region.shape[0]
                    and 0 <= col < palm_region.shape[1]
                    and palm_region[row, col]):
                continue
            out.append((contour, float(major)))
    return out


def _fragment_pool(binary: np.ndarray, cfg: FoldConfig) -> List[np.ndarray]:
    """Line-shaped pieces of this channel, accepted or not.

    These are only ever used to lengthen a crease that already passed every
    gate, never to justify one, so admitting sub-threshold pieces here is
    safe.
    """
    contours, _ = cv2.findContours(_clean(binary), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    pool: List[np.ndarray] = []
    for contour in contours:
        if len(contour) < 5 or cv2.contourArea(contour) < 30:
            continue
        (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        major, minor = max(axis_a, axis_b), max(min(axis_a, axis_b), 1e-6)
        if major / minor >= cfg.bridge_min_elongation:
            pool.append(contour)
    return pool


def _axis_of(contour: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Centre, unit direction along the long axis, and length."""
    (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    radians = np.deg2rad(angle)
    direction = np.array([-np.sin(radians), np.cos(radians)], np.float32)
    return (np.array([cx, cy], np.float32), direction,
            float(max(axis_a, axis_b)))


def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
    """Unsigned angle in degrees between two undirected axes."""
    return float(np.degrees(np.arccos(
        np.clip(abs(float(np.dot(first, second))), 0.0, 1.0))))


def _extend_along_line(contour: np.ndarray, pool: List[np.ndarray],
                       palm_radius: float,
                       cfg: FoldConfig) -> Tuple[BBox, float]:
    """Lengthen an ACCEPTED crease using fragments that continue its line.

    A crease's contrast varies along its length, so parts of it fall below
    the threshold and it arrives in pieces; the reported length was that of
    the longest piece. fold_cotton_5's crease measured 0.55R that way against a
    true extent nearer 1.4R.

    Morphological closing was tried for this and had to be abandoned on the
    weave-sensitive channels: it joins whatever lies along the kernel's
    direction and cannot ask whether two pieces are END TO END or SIDE BY
    SIDE. On a striped knit the leftover speckle sits on adjacent parallel
    stripes -- side by side -- and closing threaded it into false creases on
    three undamaged gloves.

    This asks. A fragment is absorbed only when its own direction agrees
    with the crease's AND the line joining their centres runs along that
    direction rather than across it. Being applied after acceptance, it can
    only lengthen a crease that was already found; it can never create one,
    so precision cannot move.
    """
    centre, direction, length = _axis_of(contour)
    members = [contour]
    for other in pool:
        if other is contour or len(other) < 5:
            continue
        other_centre, other_direction, other_length = _axis_of(other)
        if _angle_between(direction, other_direction) > cfg.group_angle_degrees:
            continue
        link = other_centre - centre
        distance = float(np.linalg.norm(link))
        if distance < 1e-6:
            continue
        if _angle_between(link / distance,
                          direction) > cfg.group_collinear_degrees:
            continue
        if distance - (length + other_length) / 2.0 > cfg.group_max_gap * palm_radius:
            continue
        members.append(other)

    points = np.vstack([m.reshape(-1, 2) for m in members]).astype(np.float32)
    along = (points - points.mean(axis=0)) @ direction
    extent = float(along.max() - along.min())
    return cv2.boundingRect(points.astype(np.int32)), max(extent, length)


def _is_shadow(contour: np.ndarray, lightness: np.ndarray,
               glove_median: float, cfg: FoldConfig) -> Tuple[bool, float]:
    """Is this region darker than the glove, i.e. a crease and not a glare?"""
    stencil = np.zeros(lightness.shape, np.uint8)
    cv2.drawContours(stencil, [contour], -1, 255, thickness=cv2.FILLED)
    inside = stencil > 0
    if not inside.any():
        return False, 0.0
    delta = float(np.median(lightness[inside])) - glove_median
    return delta <= cfg.max_lightness_delta, delta


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def detect(image: np.ndarray, segmentation: SegmentationResult,
           config: PipelineConfig) -> DefectResult:
    """Detect a fold crease across the palm. See the module docstring."""
    cfg = config.fold
    interior = glove_interior(segmentation, cfg.interior_margin_ratio)
    palm_center, palm_radius = palm_center_and_radius(segmentation.mask)
    if np.count_nonzero(interior) < 100 or palm_radius < 10:
        return DefectResult(False, "damage_by_fold",
                            details="glove interior too small to analyse")

    # Search only the palm; measure every reference over the whole glove.
    palm_disc = np.zeros_like(interior)
    cv2.circle(palm_disc, palm_center,
               int(cfg.palm_radius_ratio * palm_radius), 255, cv2.FILLED)
    palm_region = cv2.bitwise_and(interior, palm_disc)
    if np.count_nonzero(palm_region) < 100:
        return DefectResult(False, "damage_by_fold",
                            details="palm region too small to analyse")

    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    glove_median = float(np.median(lightness[interior > 0]))

    candidates: List[Tuple[np.ndarray, float, str]] = []
    pools: dict = {}

    # --- channel 1: shading ridge ------------------------------------- #
    # Ridges AND valleys: a fold shows as a bright crest beside a dark
    # trough, and which one dominates depends on where the light is. The
    # shadow gate below judges the region as a whole, not pixel by pixel,
    # so the crest stays attached to its own trough.
    response = fold_ridge_response(image, interior, palm_radius, cfg)
    _, spread = robust_stats(response[interior > 0])
    ridge = ((np.abs(response) > cfg.z_threshold * spread)
             & (palm_region > 0)).astype(np.uint8) * 255
    # Fragment bridging is enabled HERE ONLY. The residual and weave
    # channels work on woven gloves, whose leftover speckle runs along the
    # stripes and is therefore both collinear and elongated already; a
    # directional closing threads it into convincing creases, and it
    # invented three on undamaged cotton (1.53R, 1.21R, 0.73R). Filtering
    # the fragments by elongation first did not help, because the speckle
    # passes that test too. The shading channel does not have the problem:
    # its residue is isotropic, and the shadow gate screens it besides.
    pools["shading"] = _fragment_pool(ridge, cfg)
    for contour, major in _shaped_creases(ridge, palm_region, palm_radius,
                                          cfg, bridge=True):
        candidates.append((contour, major, "shading"))

    # --- channel 2: stripe distortion --------------------------------- #
    deviation = stripe_deviation(image, interior, palm_radius, cfg)
    bent = ((deviation > cfg.stripe_deviation_degrees)
            & (palm_region > 0)).astype(np.uint8) * 255
    # The weave channel needs heavier morphology than the other two. Its
    # raw map is a field of small disturbed patches wherever the knit is
    # merely uneven, and the shared 5x5 opening leaves enough of them
    # touching to fake an elongated blob -- that produced a false crease of
    # 0.56R on good_cotton_2. Closing at 15 then opening at 9 keeps a real
    # crease band whole while clearing that speckle.
    bent = cv2.morphologyEx(
        bent, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bent = cv2.morphologyEx(
        bent, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    pools["weave"] = _fragment_pool(bent, cfg)
    for contour, major in _shaped_creases(bent, palm_region, palm_radius, cfg):
        candidates.append((contour, major, "weave"))

    # --- channel 3: chroma residual ----------------------------------- #
    if cfg.use_chroma_residual:
        residual = chroma_residual(image, interior, palm_radius, cfg)
        _, residual_spread = robust_stats(residual[interior > 0])
        weak = ((residual > cfg.z_threshold * residual_spread)
                & (palm_region > 0)).astype(np.uint8) * 255
        pools["residual"] = _fragment_pool(weak, cfg)
        for contour, major in _shaped_creases(weak, palm_region, palm_radius,
                                              cfg):
            candidates.append((contour, major, "residual"))

    # --- shadow gate, then de-duplicate across channels ---------------- #
    creases: List[BBox] = []
    lengths: List[float] = []
    channels: List[str] = []
    rejected_bright = 0
    claimed = np.zeros(interior.shape, np.uint8)
    for contour, major, channel in sorted(candidates, key=lambda c: -c[1]):
        shadow, _ = _is_shadow(contour, lightness, glove_median, cfg)
        if not shadow:
            rejected_bright += 1
            continue
        stencil = np.zeros(interior.shape, np.uint8)
        cv2.drawContours(stencil, [contour], -1, 255, thickness=cv2.FILLED)
        # Two channels finding the same crease is one crease, not two.
        overlap = np.count_nonzero(cv2.bitwise_and(stencil, claimed))
        if overlap > 0.4 * max(np.count_nonzero(stencil), 1):
            continue
        claimed = cv2.bitwise_or(claimed, stencil)
        # Only now, once this crease is definitely a detection, gather the
        # sub-threshold fragments that continue its line so the reported
        # box and length cover the whole crease rather than its brightest
        # section.
        box, extent = _extend_along_line(contour, pools.get(channel, []),
                                         palm_radius, cfg)
        creases.append(box)
        lengths.append(max(major, extent))
        channels.append(channel)

    found = len(creases) >= cfg.min_crease_count
    longest = max(lengths) / palm_radius if lengths else 0.0
    if creases:
        seen = ", ".join(sorted(set(channels)))
        detail = (f"{len(creases)} palm crease(s), longest {longest:.2f}R, "
                  f"via {seen}")
    elif rejected_bright:
        detail = (f"{rejected_bright} bright ridge(s) rejected as glare "
                  f"rather than a crease")
    else:
        detail = (f"no palm crease longer than {cfg.min_length_ratio:g}R "
                  f"(searched {cfg.palm_radius_ratio:g}R around the palm "
                  f"centre)")

    return DefectResult(
        defect_found=found,
        defect_type="damage_by_fold",
        locations=creases if found else [],
        # A crease spanning 1.5 palm radii is treated as full confidence.
        score=min(1.0, longest / 1.5) if found else 0.0,
        details=detail,
    )
