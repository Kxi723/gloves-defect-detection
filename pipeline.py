"""Stage by stage tracing of the three glove defect detectors.

The detector modules stay untouched. Everything here calls their own functions
and captures the intermediate images, so the UI can replay a run as a sequence
of frames (preprocess, segmentation, analysis, verdict).
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
PHOTO_DIR = PROJECT_ROOT / "gloves"
OUTPUT_DIR = PROJECT_ROOT / "output"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STAGE_MAX_DIM = 860


@dataclass(frozen=True)
class DefectSpec:
    slug: str
    module: str
    title: str
    accent: str
    glow: str
    prefix: str

    def load(self) -> ModuleType:
        return importlib.import_module("detectors." + self.module)


DEFECTS: List[DefectSpec] = [
    DefectSpec(
        slug="fold",
        module="jason_damage_by_fold",
        title="DAMAGE BY FOLD",
        accent="#3ddbd9",
        glow="#0b3c46",
        prefix="fold",
    ),
    DefectSpec(
        slug="dirty",
        module="jason_dirty",
        title="DIRTY",
        accent="#ffb020",
        glow="#4a3208",
        prefix="dirty",
    ),
    DefectSpec(
        slug="tear",
        module="jason_tearing_at_finger",
        title="TEARING AT FINGERTIP",
        accent="#ff5d8f",
        glow="#4a1030",
        prefix="tearfinger",
    ),
]

DEFECTS_BY_SLUG: Dict[str, DefectSpec] = {spec.slug: spec for spec in DEFECTS}

PHASES = ("preprocess", "segmentation", "analysis", "verdict")
PHASE_LABELS = {
    "preprocess": "PREPROCESS",
    "segmentation": "SEGMENTATION",
    "analysis": "ANALYSIS",
    "verdict": "VERDICT",
}


@dataclass
class Stage:
    key: str
    phase: str
    title: str
    caption: str
    image: Optional[np.ndarray]
    kind: str = "photo"
    packed: Optional[bytes] = None

    def pack(self, quality: int = 92) -> None:
        """Hold the frame as JPEG bytes, so every finished photo can stay in memory."""
        if self.image is None:
            return
        ok, buffer = cv2.imencode(".jpg", self.image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            self.packed = buffer.tobytes()
            self.image = None

    def frame(self) -> np.ndarray:
        if self.image is not None:
            return self.image
        return cv2.imdecode(np.frombuffer(self.packed, np.uint8), cv2.IMREAD_COLOR)


@dataclass
class Trace:
    path: Path
    name: str
    spec: DefectSpec
    stages: List[Stage] = field(default_factory=list)
    found: bool = False
    score: float = 0.0
    details: str = ""
    verdict: str = "PASS"
    elapsed: float = 0.0
    annotated: Optional[np.ndarray] = None

    @property
    def expected(self) -> bool:
        """True when the filename says this photo should carry this defect."""
        return self.name.lower().startswith(self.spec.prefix)

    @property
    def correct(self) -> bool:
        return self.found == self.expected


# ------------------------------------------------------------------ file work

def imread_unicode(path: Path) -> Optional[np.ndarray]:
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def imread_preview(path: Path) -> Optional[np.ndarray]:
    """Half resolution decode, for showing a photo before its trace exists."""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_REDUCED_COLOR_2)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    ok, buffer = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        buffer.tofile(str(path))
    except OSError:
        return False
    return True


def photos_in(folder: Path = PHOTO_DIR) -> List[Path]:
    if not folder.is_dir():
        return []
    found = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(found, key=lambda p: p.name.lower())


# ------------------------------------------------------------ frame rendering

def fit(image: np.ndarray, limit: int = STAGE_MAX_DIM) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= limit:
        return image
    scale = limit / longest
    size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def hex_bgr(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (blue, green, red)


def _heat(values: np.ndarray, inside: Optional[np.ndarray] = None,
          cmap: int = cv2.COLORMAP_TURBO, percentile: float = 99.0) -> np.ndarray:
    """False colour view of a float response map, normalised inside the glove."""
    data = values.astype(np.float32)
    pool = data[inside > 0] if inside is not None and np.any(inside) else data.ravel()
    low = float(np.percentile(pool, 100.0 - percentile))
    high = float(np.percentile(pool, percentile))
    if high - low < 1e-6:
        high = low + 1e-6
    scaled = np.clip((data - low) / (high - low), 0.0, 1.0)
    coloured = cv2.applyColorMap((scaled * 255).astype(np.uint8), cmap)
    if inside is not None:
        coloured[inside == 0] = (14, 12, 10)
    return coloured


def _tint(mask: np.ndarray, colour: Tuple[int, int, int],
          backdrop: Tuple[int, int, int] = (18, 16, 14)) -> np.ndarray:
    canvas = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    canvas[:] = backdrop
    canvas[mask > 0] = colour
    return canvas


def _overlay(base: np.ndarray, mask: np.ndarray, colour: Tuple[int, int, int],
             alpha: float = 0.42, outline: bool = True) -> np.ndarray:
    canvas = base.copy()
    active = mask > 0
    if np.any(active):
        tint = np.zeros_like(canvas)
        tint[:] = colour
        blended = cv2.addWeighted(canvas, 1.0 - alpha, tint, alpha, 0)
        canvas[active] = blended[active]
        if outline:
            binary = np.where(active, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, colour, 2)
    return canvas


def _dim(image: np.ndarray, factor: float = 0.42) -> np.ndarray:
    return (image.astype(np.float32) * factor).astype(np.uint8)


def _montage(panels: List[Tuple[str, np.ndarray]], winner: str, shape: Tuple[int, int],
             accent: Tuple[int, int, int], columns: int = 3) -> np.ndarray:
    """Grid of the candidate masks with the chosen cue ringed in the accent colour."""
    height, width = shape
    rows = int(math.ceil(len(panels) / columns))
    cell_w = max(width // columns, 60)
    cell_h = max(height // rows, 60)
    canvas = np.zeros((cell_h * rows, cell_w * columns, 3), np.uint8)
    canvas[:] = (16, 14, 12)
    for index, (label, mask) in enumerate(panels):
        row, column = divmod(index, columns)
        chosen = label == winner
        colour = accent if chosen else (108, 104, 100)
        tile = cv2.resize(_tint(mask, colour), (cell_w - 8, cell_h - 8),
                          interpolation=cv2.INTER_NEAREST)
        y, x = row * cell_h + 4, column * cell_w + 4
        canvas[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
        if chosen:
            cv2.rectangle(canvas, (x, y), (x + tile.shape[1] - 1, y + tile.shape[0] - 1), accent, 2)
        text = label.upper() + ("  [chosen]" if chosen else "")
        cv2.putText(canvas, text, (x + 8, y + tile.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, colour, 1, cv2.LINE_AA)
    return canvas


def _markers(base: np.ndarray, points, colour: Tuple[int, int, int], radius: int = 14) -> np.ndarray:
    canvas = base.copy()
    for index, point in enumerate(points, start=1):
        centre = (int(point[0]), int(point[1]))
        cv2.circle(canvas, centre, radius, colour, 2, cv2.LINE_AA)
        cv2.circle(canvas, centre, 2, colour, cv2.FILLED, cv2.LINE_AA)
        cv2.putText(canvas, str(index), (centre[0] + radius + 4, centre[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return canvas


def _boxes(base: np.ndarray, boxes, colour: Tuple[int, int, int], labels=None) -> np.ndarray:
    canvas = base.copy()
    for index, box in enumerate(boxes):
        x, y, w, h = (int(v) for v in box)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 2)
        tag = str(index + 1) if labels is None else str(labels[index])
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y - 6, th + 6)
        cv2.rectangle(canvas, (x, top - th - 5), (x + tw + 10, top + 4), colour, cv2.FILLED)
        cv2.putText(canvas, tag, (x + 5, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (12, 12, 12), 1, cv2.LINE_AA)
    return canvas


def _solid(mask: np.ndarray) -> np.ndarray:
    """Outer outline filled in, so comparing two masks shows boundary changes only
    and not the pinholes of a knit glove."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(mask)
    cv2.drawContours(solid, contours, -1, 255, cv2.FILLED)
    return solid


def _pct(part: float, whole: float) -> str:
    return "{:.1%}".format(part / whole) if whole > 0 else "n/a"


# ---------------------------------------------------------------- the tracers

def _preprocess_stages(module: ModuleType, raw: np.ndarray, prepared: np.ndarray,
                       cfg) -> Iterator[Stage]:
    pcfg = cfg.preprocess
    height, width = raw.shape[:2]
    yield Stage("capture", "preprocess", "Capture",
                "{}x{} px, straight off the camera".format(width, height), raw, "photo")

    sized = module.resize_to_limit(raw, pcfg.max_dimension)
    yield Stage("resize", "preprocess", "Resize",
                "longest side capped at {} px, so every later threshold means the same thing "
                "on every photo".format(pcfg.max_dimension),
                sized, "photo")

    white_balance = getattr(pcfg, "white_balance", True)
    balanced = module.gray_world_white_balance(sized) if white_balance else sized
    if white_balance:
        shift = float(cv2.absdiff(balanced, sized).mean())
        yield Stage("white_balance", "preprocess", "Gray world white balance",
                    "channels pulled to a common gray, cancelling the lamp cast "
                    "({:.1f} levels)".format(shift),
                    balanced, "photo")

    # the module's own preprocess output, so this frame is exactly what the detector sees
    yield Stage("bilateral", "preprocess", "Bilateral filter",
                "grain smoothed away, real edges keep their contrast "
                "(d={}, sigma {:.0f})".format(pcfg.bilateral_diameter, pcfg.bilateral_sigma_color),
                prepared, "photo")


def _segmentation_stages(module: ModuleType, prepared: np.ndarray, segmentation, cfg,
                         accent: Tuple[int, int, int]) -> Iterator[Stage]:
    scfg = cfg.segmentation
    lab = cv2.cvtColor(prepared, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(prepared, cv2.COLOR_BGR2HSV)
    backdrop = module.estimate_background_lab(lab, scfg.border_fraction)

    distance = np.linalg.norm(lab.astype(np.float32) - backdrop.astype(np.float32), axis=2)
    yield Stage("bg_distance", "segmentation", "Background distance",
                "Lab distance from the backdrop sampled along the frame border, "
                "L{:.0f} a{:.0f} b{:.0f}".format(*backdrop),
                _heat(distance), "heat")

    otsu = module._channel_otsu_masks(hsv)
    candidates = [
        ("bg_model", module._background_model_mask(prepared, scfg)),
        ("bg_distance", module._background_distance_mask(lab, scfg.border_fraction)),
        ("saturation", otsu[0]),
        ("value", otsu[1]),
        ("value_inverted", otsu[2]),
        ("texture", module._texture_energy_mask(prepared, scfg.texture_window)),
    ]
    winner = segmentation.cue if segmentation is not None else ""
    yield Stage("cues", "segmentation", "Cue vote",
                "six ways of calling background, scored on area, compactness and how well the "
                "outline lands on real edges, winner {}".format(
                    winner or "none, the glove was not separable"),
                _montage(candidates, winner, prepared.shape[:2], accent), "mask")

    if segmentation is None:
        return

    chosen = dict(candidates).get(winner, candidates[0][1])
    cleaned = module._morphological_cleanup(chosen, scfg)
    yield Stage("cleanup", "segmentation", "Morphological cleanup",
                "opening {} px drops the speckle, closing {} px seals the hairline "
                "gaps".format(scfg.open_kernel, scfg.close_kernel),
                _tint(cleaned, accent), "mask")

    seed = module._largest_blob(chosen, scfg)
    if seed is None:
        seed = segmentation.mask
    yield Stage("component", "segmentation", "Largest component",
                "only the biggest blob is kept, as the seed for the next two passes",
                _tint(seed, accent), "mask")

    # mirrors segment_glove, which keeps the surface pass only when it scores as a glove
    surfaced = seed
    residual = module._backdrop_residual_mask(prepared, seed, scfg)
    if residual is not None and module._score_candidate(residual, scfg) > -1.0:
        blob = module._largest_blob(residual, scfg)
        if blob is not None:
            surfaced = blob
    dropped = cv2.bitwise_and(_solid(seed), cv2.bitwise_not(_solid(surfaced)))
    view = _overlay(_dim(prepared, 0.5), surfaced, accent, 0.35)
    view = _overlay(view, dropped, (80, 80, 255), 0.55)
    yield Stage("surface", "segmentation", "Backdrop surface",
                "a smooth surface is fitted to the cloth so the lamp falloff is predicted, "
                "{} of the seed turned out to be lit backdrop (red)".format(
                    _pct(float(np.count_nonzero(dropped)), float(max(np.count_nonzero(seed), 1)))),
                view, "overlay")

    final = segmentation.mask
    trimmed = cv2.bitwise_and(_solid(surfaced), cv2.bitwise_not(final))
    added = cv2.bitwise_and(final, cv2.bitwise_not(_solid(surfaced)))
    view = _overlay(_dim(prepared, 0.5), segmentation.mask_raw, accent, 0.35)
    view = _overlay(view, trimmed, (80, 80, 255), 0.6)
    view = _overlay(view, added, (235, 235, 235), 0.6)
    area = float(max(np.count_nonzero(final), 1))
    yield Stage("grabcut", "segmentation", "GrabCut refine",
                "the boundary is re cut from the colour of both sides, {} trimmed (red) and {} "
                "added (white)".format(_pct(float(np.count_nonzero(trimmed)), area),
                                       _pct(float(np.count_nonzero(added)), area)),
                view, "overlay")

    hole_area = float(np.count_nonzero(segmentation.holes_mask))
    outlined = _overlay(_dim(prepared, 0.55), segmentation.mask, accent, 0.30)
    cv2.drawContours(outlined, [segmentation.contour], -1, accent, 3, cv2.LINE_AA)
    if hole_area > 0:
        outlined = _overlay(outlined, segmentation.holes_mask, (60, 90, 255), 0.55)
    yield Stage("outline", "segmentation", "Glove found",
                "small holes filled, and the glove covers {} of the frame".format(
                    _pct(segmentation.area, prepared.shape[0] * prepared.shape[1])),
                outlined, "overlay")


def _fold_stages(module: ModuleType, prepared: np.ndarray, segmentation, cfg,
                 accent: Tuple[int, int, int]) -> Iterator[Stage]:
    fcfg = cfg.fold
    interior = module.glove_interior(segmentation, fcfg.interior_margin_ratio)
    centre, palm_radius = module.palm_center_and_radius(segmentation.mask)
    if np.count_nonzero(interior) < 100 or palm_radius < 10:
        return

    palm_disc = np.zeros_like(interior)
    cv2.circle(palm_disc, centre, int(fcfg.palm_radius_ratio * palm_radius), 255, cv2.FILLED)
    palm_region = cv2.bitwise_and(interior, palm_disc)
    search = _overlay(_dim(prepared, 0.5), palm_region, accent, 0.30)
    cv2.circle(search, centre, int(fcfg.palm_radius_ratio * palm_radius), accent, 2, cv2.LINE_AA)
    cv2.circle(search, centre, 4, accent, cv2.FILLED, cv2.LINE_AA)
    yield Stage("palm", "analysis", "Palm search region",
                "only the palm is searched, the rim is shaved by {:g}R to skip edge shading and "
                "the circle at {:g}R keeps out fingers and cuff (R={:.0f} px)".format(
                    fcfg.interior_margin_ratio, fcfg.palm_radius_ratio, palm_radius),
                search, "overlay")

    lightness = cv2.cvtColor(prepared, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    glove_median = float(np.median(lightness[interior > 0]))

    response = module.fold_ridge_response(prepared, interior, palm_radius, fcfg)
    _, spread = module.robust_stats(response[interior > 0])
    yield Stage("shading", "analysis", "Channel 1, shading ridge",
                "difference of Gaussians on lightness, a crease is a dark valley",
                _heat(np.abs(response), interior), "heat")

    ridge = ((np.abs(response) > fcfg.z_threshold * spread) & (palm_region > 0)).astype(np.uint8) * 255
    boundary = module.material_boundary(prepared, interior, palm_radius, fcfg)
    ridge = cv2.bitwise_and(ridge, cv2.bitwise_not(boundary))
    yield Stage("shading_binary", "analysis", "Shading ridge, thresholded",
                "above {:g} robust sigma, material boundary subtracted".format(fcfg.z_threshold),
                _tint(ridge, accent), "mask")

    deviation = module.stripe_deviation(prepared, interior, palm_radius, fcfg)
    bent = ((deviation > fcfg.stripe_deviation_degrees) & (palm_region > 0)).astype(np.uint8) * 255
    bent = cv2.morphologyEx(bent, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bent = cv2.morphologyEx(bent, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    yield Stage("weave", "analysis", "Channel 2, weave bend",
                "structure tensor weave angle, bent over {:g} degrees off its own "
                "run".format(fcfg.stripe_deviation_degrees),
                _heat(deviation, interior, cv2.COLORMAP_PARULA), "heat")
    yield Stage("weave_binary", "analysis", "Weave bend, thresholded",
                "closed then opened into one strip",
                _tint(bent, accent), "mask")

    weak = np.zeros_like(ridge)
    if fcfg.use_chroma_residual:
        residual = module.chroma_residual(prepared, interior, palm_radius, fcfg)
        _, residual_spread = module.robust_stats(residual[interior > 0])
        weak = ((residual > fcfg.z_threshold * residual_spread) & (palm_region > 0)).astype(np.uint8) * 255
        yield Stage("residual", "analysis", "Channel 3, chroma residual",
                    "on latex a crease shifts colour more than lightness, the a/b residual",
                    _heat(residual, interior, cv2.COLORMAP_MAGMA), "heat")

    pools = {"shading": ridge, "weave": bent, "residual": weak}
    stack = _dim(prepared, 0.45)
    for name, colour in (("shading", (255, 210, 60)), ("weave", (90, 255, 140)),
                         ("residual", (200, 120, 255))):
        stack = _overlay(stack, pools[name], colour, 0.5, outline=False)
    yield Stage("channels", "analysis", "Three channels stacked",
                "shading yellow, weave green, residual violet, a real crease lights up "
                "in more than one",
                stack, "overlay")

    candidates = []
    for channel, binary in (("shading", ridge), ("weave", bent), ("residual", weak)):
        for contour, major in module._shaped_creases(binary, palm_region, palm_radius, fcfg,
                                                     bridge=channel == "shading"):
            candidates.append((contour, major, channel))

    shaped = np.zeros_like(interior)
    for contour, _major, _channel in candidates:
        cv2.drawContours(shaped, [contour], -1, 255, cv2.FILLED)
    yield Stage("shaped", "analysis", "Line shaped fragments",
                "{} fragment(s) past elongation {:g} and length {:g}R".format(
                    len(candidates), fcfg.min_elongation, fcfg.min_length_ratio),
                _overlay(_dim(prepared, 0.5), shaped, accent, 0.5), "overlay")

    kept = np.zeros_like(interior)
    rejected = np.zeros_like(interior)
    for contour, _major, _channel in candidates:
        shadow, _delta = module._is_shadow(contour, lightness, glove_median, fcfg)
        cv2.drawContours(kept if shadow else rejected, [contour], -1, 255, cv2.FILLED)
    gated = _overlay(_dim(prepared, 0.5), rejected, (80, 80, 255), 0.5)
    gated = _overlay(gated, kept, accent, 0.5)
    yield Stage("shadow_gate", "analysis", "Shadow versus glare gate",
                "a crease is {:g} levels darker than the glove, brighter is glare and "
                "dropped (red)".format(abs(fcfg.max_lightness_delta)),
                gated, "overlay")

    # mirrors the detector: claim creases longest first, extend each along its line,
    # and only a line spanning enough of the palm counts as a fold
    fragments = {name: module._fragment_pool(binary, fcfg) for name, binary in pools.items()}
    claimed = np.zeros(interior.shape, np.uint8)
    spanning, short = [], []
    for contour, major, channel in sorted(candidates, key=lambda c: -c[1]):
        shadow, _delta = module._is_shadow(contour, lightness, glove_median, fcfg)
        if not shadow:
            continue
        stencil = np.zeros(interior.shape, np.uint8)
        cv2.drawContours(stencil, [contour], -1, 255, cv2.FILLED)
        if np.count_nonzero(cv2.bitwise_and(stencil, claimed)) > 0.4 * max(np.count_nonzero(stencil), 1):
            continue
        claimed = cv2.bitwise_or(claimed, stencil)
        box, extent = module._extend_along_line(contour, fragments.get(channel, []), palm_radius, fcfg)
        reach = max(major, extent) / palm_radius
        (spanning if reach >= fcfg.min_span_ratio else short).append((box, "{:.2f}R".format(reach)))
    spans = _boxes(_dim(prepared, 0.5), [b for b, _ in short], (90, 90, 255),
                   labels=[s for _, s in short])
    spans = _boxes(spans, [b for b, _ in spanning], accent, labels=[s for _, s in spanning])
    yield Stage("span", "analysis", "Span gate",
                "a fold runs across the palm, so a line has to span {:g}R to count, {} do and "
                "{} fall short (red)".format(fcfg.min_span_ratio, len(spanning), len(short)),
                spans, "overlay")


def _dirty_stages(module: ModuleType, prepared: np.ndarray, segmentation, cfg,
                  accent: Tuple[int, int, int]) -> Iterator[Stage]:
    dcfg = cfg.dirt
    interior = module.glove_interior(segmentation, dcfg.interior_margin_ratio)
    if np.count_nonzero(interior) < 100:
        return
    selection = interior > 0

    yield Stage("interior", "analysis", "Interior band",
                "the mask is eroded by {:g}R so the rim shading is never read as dirt".format(
                    dcfg.interior_margin_ratio),
                _overlay(_dim(prepared, 0.5), interior, accent, 0.28), "overlay")

    lightness = cv2.cvtColor(prepared, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    median, spread = module.robust_stats(lightness[selection])
    z_score = np.abs(lightness - median) / spread
    yield Stage("zmap", "analysis", "Lightness deviation",
                "robust sigma from the median lightness of the glove itself "
                "({:.0f} plus or minus {:.1f})".format(median, spread),
                _heat(z_score, interior), "heat")

    candidate = ((z_score > dcfg.z_threshold) & selection).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    yield Stage("candidate", "analysis", "Off colour candidates",
                "everything past {:g} sigma, opened and closed into solid blobs".format(
                    dcfg.z_threshold),
                _tint(candidate, accent), "mask")

    texture = module.local_texture_energy(prepared, dcfg.texture_window)
    glove_texture = max(float(np.median(texture[selection])), 1e-6)
    yield Stage("texture", "analysis", "Local texture energy",
                "powder and paint sit on the weave and flatten it "
                "(glove median {:.1f})".format(glove_texture),
                _heat(texture, interior, cv2.COLORMAP_PARULA), "heat")

    chroma = module.lab_chroma(prepared)
    glove_chroma = module.median_chroma(chroma, selection)
    hue_route = math.hypot(*glove_chroma) > dcfg.min_glove_chroma
    hue_distance = np.linalg.norm(chroma - np.array(glove_chroma, np.float32), axis=2)
    yield Stage("chroma", "analysis", "Colour distance",
                "a/b distance from the colour of the glove itself, hue route {}".format(
                    "open" if hue_route else "closed, this glove is too neutral"),
                _heat(hue_distance, interior, cv2.COLORMAP_MAGMA), "heat")

    depth, palm_radius = module.interior_depth_map(segmentation)
    kept, dropped, notes = [], [], []
    for bbox, _area, members in module.components_as_boxes(
            candidate, min_area=dcfg.min_area_fraction * segmentation.area, min_extent=dcfg.min_extent):
        depth_ratio = module.region_depth_ratio(depth, palm_radius, members)
        if depth_ratio < dcfg.min_depth_ratio:
            dropped.append((bbox, "edge {:.2f}R".format(depth_ratio)))
            continue
        texture_ratio = float(np.median(texture[members])) / glove_texture
        off_hue = module.off_hue_distance(module.median_chroma(chroma, members), glove_chroma)
        covered = texture_ratio <= dcfg.max_texture_ratio
        foreign = hue_route and off_hue > dcfg.min_off_hue_distance
        if not (covered or foreign):
            dropped.append((bbox, "tex {:.2f}x".format(texture_ratio)))
            continue
        kept.append(bbox)
        notes.append("tex {:.2f}x".format(texture_ratio) if covered else "hue {:.0f}".format(off_hue))

    judged = _boxes(_dim(prepared, 0.5), [b for b, _ in dropped], (90, 90, 255),
                    labels=[note for _, note in dropped])
    judged = _boxes(judged, kept, accent, labels=notes)
    yield Stage("judged", "analysis", "Two tests per blob",
                "kept if texture free below {:g}x or over {:g} off hue, {} rejected (red)".format(
                    dcfg.max_texture_ratio, dcfg.min_off_hue_distance, len(dropped)),
                judged, "overlay")


def _tear_stages(module: ModuleType, prepared: np.ndarray, segmentation, cfg,
                 accent: Tuple[int, int, int]) -> Iterator[Stage]:
    tcfg = cfg.tearing
    fcfg = cfg.fingertip
    centre, palm_radius = module.palm_center_and_radius(segmentation.mask)
    if palm_radius <= 1:
        return

    distance = cv2.distanceTransform(segmentation.mask, cv2.DIST_L2, 5)
    yield Stage("distance", "analysis", "Distance transform",
                "distance to the nearest edge, the ridge of it is the hand skeleton "
                "(palm R={:.0f} px)".format(palm_radius),
                _heat(distance, segmentation.mask), "heat")

    tips = module.locate_fingertips(segmentation, fcfg.min_tip_distance_ratio,
                                    fcfg.tip_merge_separation_ratio, fcfg.expected_fingers,
                                    fcfg.tip_frame_cut_reach_ratio)
    max_tip_distance = tcfg.fingertip_radius_ratio * palm_radius
    tip_view = _dim(prepared, 0.5)
    for tip in tips:
        cv2.circle(tip_view, tip, int(max_tip_distance), (70, 70, 70), 1, cv2.LINE_AA)
    tip_view = _markers(tip_view, tips, accent)
    yield Stage("tips", "analysis", "Fingertip localisation",
                "{} tip(s) at least {:g}R out from the palm centre, only findings inside the "
                "rings will count".format(len(tips), fcfg.min_tip_distance_ratio),
                tip_view, "overlay")

    if not tips:
        return

    holes = module.find_holes(segmentation,
                              min_area=tcfg.min_hole_area_fraction * segmentation.area,
                              max_area=tcfg.max_hole_area_fraction * segmentation.area,
                              max_elongation=tcfg.max_hole_elongation,
                              min_extent=tcfg.min_hole_extent)
    yield Stage("holes", "analysis", "Holes in the silhouette",
                "{} gap(s) between {:.2%} and {:.0%} of the glove, where the backdrop shows "
                "through".format(len(holes), tcfg.min_hole_area_fraction,
                                 tcfg.max_hole_area_fraction),
                _boxes(_dim(prepared, 0.5), [bbox for _c, bbox, _a in holes], accent), "overlay")

    lab = cv2.cvtColor(prepared, cv2.COLOR_BGR2LAB).astype(np.float32)
    margin = max(3, int(tcfg.showthrough_margin_ratio * palm_radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    interior = cv2.erode(segmentation.mask, kernel)
    if np.count_nonzero(interior) >= 100:
        a_median, a_spread = module.robust_stats(lab[:, :, 1][interior > 0])
        b_median, b_spread = module.robust_stats(lab[:, :, 2][interior > 0])
        a_spread = max(a_spread, tcfg.showthrough_mad_floor)
        b_spread = max(b_spread, tcfg.showthrough_mad_floor)
        deviation = np.sqrt(((lab[:, :, 1] - a_median) / a_spread) ** 2
                            + ((lab[:, :, 2] - b_median) / b_spread) ** 2)
        yield Stage("showthrough", "analysis", "Show through chroma",
                    "a worn glove keeps its outline when it tears, so the tell is colour, "
                    "the a/b deviation",
                    _heat(deviation, interior, cv2.COLORMAP_MAGMA), "heat")

    patches = module.find_showthrough_patches(prepared, segmentation, cfg, tips)
    yield Stage("patches", "analysis", "Show through patches",
                "{} patch(es) past {:g} sigma and clear of the backdrop colour".format(
                    len(patches), tcfg.showthrough_z_threshold),
                _boxes(_dim(prepared, 0.5), [bbox for bbox, _c, _n in patches], accent,
                       labels=["{:.2f}".format(c) for _b, c, _n in patches]), "overlay")

    notches = []
    valley_reach = tcfg.valley_endpoint_tip_ratio * palm_radius
    for start, end, far, depth in module.convexity_defect_list(segmentation.contour):
        if module.is_finger_valley(start, end, tips, valley_reach):
            continue
        depth_ratio = depth / palm_radius
        angle = module.angle_at(far, start, end)
        if depth_ratio >= tcfg.min_defect_depth_ratio and angle <= tcfg.max_defect_angle_deg:
            half = max(8, int(0.15 * palm_radius))
            notches.append(module.bbox_around(far, half, segmentation.mask.shape))
    yield Stage("notches", "analysis", "Contour notches",
                "{} notch(es) deeper than {:g}R and sharper than {:.0f} degrees, finger valleys "
                "excluded".format(len(notches), tcfg.min_defect_depth_ratio,
                                  tcfg.max_defect_angle_deg),
                _boxes(_dim(prepared, 0.5), notches, accent), "overlay")

    evidence = module.find_tear_evidence(prepared, segmentation, cfg)
    near, far_off = [], []
    for (x, y, w, h), confidence, _note in evidence:
        spot = (x + w // 2, y + h // 2)
        target = near if any(math.dist(spot, tip) <= max_tip_distance for tip in tips) else far_off
        target.append(((x, y, w, h), "{:.2f}".format(confidence)))
    gate_view = _dim(prepared, 0.5)
    for tip in tips:
        cv2.circle(gate_view, tip, int(max_tip_distance), (70, 70, 70), 1, cv2.LINE_AA)
    gate_view = _boxes(gate_view, [b for b, _ in far_off], (90, 90, 255),
                       labels=[c for _, c in far_off])
    gate_view = _boxes(gate_view, [b for b, _ in near], accent, labels=[c for _, c in near])
    yield Stage("gate", "analysis", "Fingertip gate",
                "{} of {} finding(s) within {:g}R of a tip, the rest dropped (red)".format(
                    len(near), len(evidence), tcfg.fingertip_radius_ratio),
                gate_view, "overlay")


_ANALYSIS = {
    "fold": _fold_stages,
    "dirty": _dirty_stages,
    "tear": _tear_stages,
}


def annotate(prepared: np.ndarray, segmentation, result, accent: Tuple[int, int, int]) -> np.ndarray:
    canvas = prepared.copy()
    if segmentation is not None:
        cv2.drawContours(canvas, [segmentation.contour], -1, (90, 90, 90), 2, cv2.LINE_AA)
    colour = accent if result.defect_found else (110, 200, 110)
    canvas = _boxes(canvas, result.locations, colour)
    return canvas


def trace(spec: DefectSpec, path: Path, raw: Optional[np.ndarray] = None) -> Trace:
    """Run one photo through one detector and collect every intermediate frame."""
    import time as _time

    module = spec.load()
    cfg = module.Config()
    accent = hex_bgr(spec.accent)
    record = Trace(path=path, name=path.name, spec=spec)
    started = _time.perf_counter()

    if raw is None:
        raw = imread_unicode(path)
    if raw is None:
        record.verdict = "READ-FAIL"
        record.details = "the file could not be decoded as an image"
        return record

    prepared = module.preprocess(raw, cfg.preprocess)
    segmentation = module.segment_glove(prepared, cfg.segmentation)

    for stage in _preprocess_stages(module, raw, prepared, cfg):
        record.stages.append(stage)
    for stage in _segmentation_stages(module, prepared, segmentation, cfg, accent):
        record.stages.append(stage)

    if segmentation is None:
        record.verdict = "SEG-FAIL"
        record.details = "the glove could not be separated from the background"
        record.elapsed = _time.perf_counter() - started
        record.annotated = prepared
        record.stages.append(Stage("verdict", "verdict", "Segmentation failed",
                                   record.details, prepared, "photo"))
        for stage in record.stages:
            stage.image = fit(stage.image)
        return record

    try:
        for stage in _ANALYSIS[spec.slug](module, prepared, segmentation, cfg, accent):
            # analysis looks at part of the glove only, so the full outline stays drawn
            # to show the segmentation itself has not changed
            cv2.drawContours(stage.image, [segmentation.contour], -1, (175, 175, 175), 2,
                             cv2.LINE_AA)
            record.stages.append(stage)
    except Exception as error:  # a trace panel must never break the run
        record.stages.append(Stage("trace_error", "analysis", "Trace interrupted",
                                   "this panel could not be rendered: {}".format(error),
                                   prepared, "photo"))

    result = module.detect(prepared, segmentation, cfg)
    record.found = bool(result.defect_found)
    record.score = float(result.score)
    record.details = result.details
    record.verdict = "DEFECT" if record.found else "PASS"
    record.annotated = annotate(prepared, segmentation, result, accent)
    record.elapsed = _time.perf_counter() - started
    record.stages.append(Stage("verdict", "verdict",
                               "Verdict, {}".format(record.verdict.lower()),
                               result.details, record.annotated, "overlay"))

    for stage in record.stages:
        stage.image = fit(stage.image)
    return record
