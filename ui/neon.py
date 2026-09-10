"""Dark console look for the inspection studio.

Palette plus the PIL drawing helpers the studio needs (glow cards, HUD frames,
gradient bars, letterboxed photo panels). Everything returns a cached
ImageTk.PhotoImage so the canvases can just swap images around.
"""
from __future__ import annotations

import ctypes
import math
import sys
from ctypes import wintypes
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk

SCALE = 1.0


def enable_hidpi() -> float:
    """Ask Windows for real pixels and remember the monitor scale for px()."""
    global SCALE
    if sys.platform != "win32":
        return SCALE
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    SCALE = max(1.0, _primary_monitor_dpi() / 96.0)
    return SCALE


def _primary_monitor_dpi() -> int:
    user32 = ctypes.windll.user32
    try:
        monitor = user32.MonitorFromPoint(wintypes.POINT(0, 0), 1)  # the primary monitor
        dpi_x, dpi_y = ctypes.c_uint(), ctypes.c_uint()
        ctypes.windll.shcore.GetDpiForMonitor(monitor, 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y))
        if dpi_x.value:
            return dpi_x.value
    except Exception:
        pass
    try:
        return user32.GetDpiForSystem()
    except Exception:
        return 96


def apply_scaling(root) -> None:
    root.tk.call("tk", "scaling", SCALE * 96.0 / 72.0)


def px(value: float) -> int:
    return int(round(value * SCALE))

# --------------------------------------------------------------------- colour

VOID = "#05070d"
BASE = "#0a0e17"
PANEL = "#111726"
PANEL_HI = "#161d30"
PANEL_SOFT = "#0d1320"
LINE = "#1e2739"
LINE_HI = "#2c3852"
INK = "#e8edf7"
INK_SOFT = "#98a4bd"
INK_FAINT = "#5b6780"
CYAN = "#3ddbd9"
AMBER = "#ffb020"
ROSE = "#ff5d8f"
GREEN = "#3ddc84"
RED = "#ff4d5e"
VIOLET = "#a78bfa"

FAMILY = "Segoe UI"
FAMILY_STRONG = "Segoe UI Semibold"
FAMILY_WIDE = "Bahnschrift SemiBold"
FAMILY_MONO = "Consolas"

_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
# fixed radii for the dirty smudge, so the blob is organic but never redraws differently
_SMUDGE = (0.180, 0.155, 0.170, 0.135, 0.165, 0.150, 0.185, 0.160,
           0.175, 0.140, 0.155, 0.130, 0.160, 0.150, 0.170, 0.165)
_CACHE: Dict[tuple, ImageTk.PhotoImage] = {}
_CACHE_LIMIT = 320


def font(size: int = 10, weight: str = "regular") -> tuple:
    family = {"strong": FAMILY_STRONG, "wide": FAMILY_WIDE, "mono": FAMILY_MONO}.get(weight, FAMILY)
    return (family, size)


def rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def hexa(colour: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, int(c))) for c in colour))


def mix(first: str, second: str, amount: float) -> str:
    """Blend two hex colours, amount 0 keeps the first, 1 keeps the second."""
    a, b = rgb(first), rgb(second)
    return hexa(tuple(a[i] + (b[i] - a[i]) * amount for i in range(3)))


def fade(colour: str, amount: float, ground: str = BASE) -> str:
    return mix(ground, colour, max(0.0, min(1.0, amount)))


def _store(key: tuple, photo: ImageTk.PhotoImage) -> ImageTk.PhotoImage:
    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.clear()
    _CACHE[key] = photo
    return photo


# ------------------------------------------------------------------- surfaces

def panel(width: int, height: int, radius: int, fill: str,
          border: Optional[str] = None, border_width: int = 1,
          ground: str = BASE) -> ImageTk.PhotoImage:
    """Plain rounded panel on an opaque ground, so Tk needs no alpha."""
    key = ("panel", width, height, radius, fill, border, border_width, ground)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    scale = 3
    image = Image.new("RGB", (width * scale, height * scale), ground)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, width * scale - 1, height * scale - 1),
                           radius=radius * scale, fill=fill,
                           outline=border, width=border_width * scale if border else 0)
    return _store(key, ImageTk.PhotoImage(image.resize((width, height), _RESAMPLE)))


def glow_panel(width: int, height: int, radius: int, fill: str, border: str,
               glow: str, strength: float = 1.0, ground: str = BASE,
               inset: int = 18) -> ImageTk.PhotoImage:
    """Rounded panel wrapped in a soft outer bloom of the accent colour."""
    key = ("glow", width, height, radius, fill, border, glow, round(strength, 2), ground, inset)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    scale = 2
    w, h, r, pad = width * scale, height * scale, radius * scale, inset * scale
    body = (pad, pad, w - pad - 1, h - pad - 1)

    bloom = Image.new("RGB", (w, h), ground)
    draw = ImageDraw.Draw(bloom)
    draw.rounded_rectangle(body, radius=r, fill=mix(ground, glow, min(1.0, 0.85 * strength)))
    bloom = bloom.filter(ImageFilter.GaussianBlur(pad * 0.45))

    draw = ImageDraw.Draw(bloom)
    draw.rounded_rectangle(body, radius=r, fill=fill, outline=border, width=2 * scale)
    return _store(key, ImageTk.PhotoImage(bloom.resize((width, height), _RESAMPLE)))


def glow_panel_on(background: Image.Image, radius: int, fill: str, border: str, glow: str,
                  strength: float = 1.0, inset: int = 16) -> ImageTk.PhotoImage:
    """Same as glow_panel but bloomed onto a crop of the real backdrop, so the
    card has no visible square seam over a gradient."""
    width, height = background.size
    body = (inset, inset, width - inset - 1, height - inset - 1)
    canvas = background.copy()
    bloom = canvas.copy()
    ImageDraw.Draw(bloom).rounded_rectangle(body, radius=radius, fill=glow)
    bloom = bloom.filter(ImageFilter.GaussianBlur(inset * 0.62))
    canvas = Image.blend(canvas, bloom, min(0.85, 0.18 + 0.55 * strength))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(body, radius=radius, fill=fill, outline=border, width=2)
    return ImageTk.PhotoImage(canvas)


def gradient_bar(width: int, height: int, start: str, end: str, radius: int,
                 ground: str = BASE) -> ImageTk.PhotoImage:
    key = ("bar", width, height, start, end, radius, ground)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    if width < 1 or height < 1:
        width, height = max(width, 1), max(height, 1)
    ramp = Image.new("RGB", (max(width, 2), 1))
    a, b = rgb(start), rgb(end)
    pixels = ramp.load()
    for x in range(ramp.width):
        t = x / max(ramp.width - 1, 1)
        pixels[x, 0] = tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
    ramp = ramp.resize((width, height), Image.BILINEAR)
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, width - 1, height - 1),
                                           radius=min(radius, height // 2), fill=255)
    canvas = Image.new("RGB", (width, height), ground)
    canvas.paste(ramp, (0, 0), mask)
    return _store(key, ImageTk.PhotoImage(canvas))


def ring(size: int, thickness: int, fraction: float, track: str, accent: str,
         ground: str = BASE) -> ImageTk.PhotoImage:
    """Progress ring, fraction between 0 and 1."""
    key = ("ring", size, thickness, round(fraction, 3), track, accent, ground)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    scale = 3
    side = size * scale
    image = Image.new("RGB", (side, side), ground)
    draw = ImageDraw.Draw(image)
    box = (thickness * scale, thickness * scale, side - thickness * scale, side - thickness * scale)
    draw.ellipse(box, outline=track, width=thickness * scale)
    if fraction > 0.001:
        draw.arc(box, start=-90, end=-90 + 360 * min(fraction, 1.0),
                 fill=accent, width=thickness * scale)
    return _store(key, ImageTk.PhotoImage(image.resize((size, size), _RESAMPLE)))


def glyph(kind: str, size: int, accent: str, ground: str) -> ImageTk.PhotoImage:
    """One defect mark on one swatch of material, so the three read as a family.

    The swatch is the glove, drawn in a dim accent. What went wrong with it is
    drawn on top in the full accent.
    """
    key = ("glyph2", kind, size, accent, ground)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    scale = 3
    side = size * scale
    image = Image.new("RGB", (side, side), ground)
    draw = ImageDraw.Draw(image)
    stroke = max(int(side * 0.030), 3)
    dim = mix(ground, accent, 0.42)
    pad = side * 0.15
    swatch = (pad, pad, side - pad, side - pad)
    radius = side * 0.15

    def at(x: float, y: float):
        return (side * x, side * y)

    if kind == "fold":
        draw.rounded_rectangle(swatch, radius=radius, outline=dim, width=stroke)
        crease, shade = [], []
        for step in range(33):
            t = step / 32.0
            x = pad + t * (side - 2 * pad)
            y = side * 0.62 - t * side * 0.24 + math.sin(t * math.pi) * side * 0.028
            crease.append((x, y))
            shade.append((x, y + side * 0.075))
        draw.line(shade[3:-3], fill=mix(ground, accent, 0.60), width=stroke, joint="curve")
        draw.line(crease, fill=accent, width=stroke + 2 * scale, joint="curve")

    elif kind == "dirty":
        draw.rounded_rectangle(swatch, radius=radius, outline=dim, width=stroke)
        centre = at(0.44, 0.45)
        blob = []
        for index, reach in enumerate(_SMUDGE):
            angle = index / len(_SMUDGE) * 2 * math.pi
            blob.append((centre[0] + math.cos(angle) * side * reach,
                         centre[1] + math.sin(angle) * side * reach))
        draw.polygon(blob, fill=accent)
        for x, y, r, bright in ((0.685, 0.635, 0.046, True), (0.725, 0.44, 0.030, False),
                                (0.595, 0.735, 0.025, True), (0.315, 0.72, 0.021, False)):
            draw.ellipse((side * (x - r), side * (y - r), side * (x + r), side * (y + r)),
                         fill=accent if bright else dim)

    else:  # tear, an opening that breaks the top edge and runs to a point
        draw.rounded_rectangle(swatch, radius=radius, outline=dim, width=stroke)
        rip = [at(0.415, 0.11), at(0.385, 0.27), at(0.450, 0.39), at(0.395, 0.52),
               at(0.470, 0.63), at(0.500, 0.73),
               at(0.552, 0.60), at(0.500, 0.46), at(0.570, 0.34), at(0.525, 0.22),
               at(0.600, 0.11)]
        draw.polygon(rip, fill=ground)
        draw.line(rip + [rip[0]], fill=accent, width=stroke, joint="curve")

    return _store(key, ImageTk.PhotoImage(image.resize((size, size), _RESAMPLE)))


def hud_frame(width: int, height: int, accent: str, ground: str = PANEL_SOFT,
              arm: int = 26) -> ImageTk.PhotoImage:
    """Corner brackets around the viewport."""
    key = ("hud", width, height, accent, ground, arm)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    image = Image.new("RGB", (width, height), ground)
    draw = ImageDraw.Draw(image)
    edge = mix(ground, accent, 0.75)
    for x, y, dx, dy in ((0, 0, 1, 1), (width - 1, 0, -1, 1),
                         (0, height - 1, 1, -1), (width - 1, height - 1, -1, -1)):
        draw.line([(x, y), (x + dx * arm, y)], fill=edge, width=2)
        draw.line([(x, y), (x, y + dy * arm)], fill=edge, width=2)
    return _store(key, ImageTk.PhotoImage(image))


# ---------------------------------------------------------------- photo tiles

def _to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def letterbox(image_bgr: np.ndarray, width: int, height: int,
              ground: str = PANEL_SOFT) -> Image.Image:
    """Fit inside the box without cropping, centred on the panel colour."""
    h, w = image_bgr.shape[:2]
    scale = min(width / w, height / h)
    size = (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, size, interpolation=interp)
    canvas = Image.new("RGB", (width, height), ground)
    canvas.paste(_to_pil(resized), ((width - size[0]) // 2, (height - size[1]) // 2))
    return canvas


def photo(image_bgr: np.ndarray, width: int, height: int,
          ground: str = PANEL_SOFT) -> ImageTk.PhotoImage:
    return ImageTk.PhotoImage(letterbox(image_bgr, width, height, ground))


def thumb(image_bgr: np.ndarray, side: int, radius: int, ring_colour: Optional[str] = None,
          ground: str = PANEL, dim: float = 1.0) -> ImageTk.PhotoImage:
    """Square, centre cropped, rounded thumbnail for the filmstrip."""
    scale = 2
    box = side * scale
    h, w = image_bgr.shape[:2]
    factor = max(box / w, box / h)
    resized = cv2.resize(image_bgr, (max(int(round(w * factor)), 1), max(int(round(h * factor)), 1)),
                         interpolation=cv2.INTER_AREA)
    top, left = (resized.shape[0] - box) // 2, (resized.shape[1] - box) // 2
    cropped = resized[top:top + box, left:left + box]
    if dim < 1.0:
        cropped = (cropped.astype(np.float32) * dim).astype(np.uint8)
    tile = _to_pil(cropped)
    canvas = Image.new("RGB", (box, box), ground)
    mask = Image.new("L", (box, box), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, box - 1, box - 1), radius=radius * scale, fill=255)
    canvas.paste(tile, (0, 0), mask)
    if ring_colour:
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle((1, 1, box - 2, box - 2), radius=radius * scale,
                               outline=ring_colour, width=2 * scale)
    return ImageTk.PhotoImage(canvas.resize((side, side), _RESAMPLE))


_BACKDROPS: Dict[tuple, Image.Image] = {}


def backdrop_image(width: int, height: int, accent: str) -> Image.Image:
    """Dot grid with a soft accent bloom in the upper left, drawn once per size."""
    key = (width, height, accent)
    cached = _BACKDROPS.get(key)
    if cached is not None:
        return cached
    small = Image.new("RGB", (max(width // 6, 2), max(height // 6, 2)), VOID)
    draw = ImageDraw.Draw(small)
    draw.ellipse((-small.width * 0.25, -small.height * 0.55,
                  small.width * 0.62, small.height * 0.62),
                 fill=mix(VOID, accent, 0.16))
    draw.ellipse((small.width * 0.55, small.height * 0.5,
                  small.width * 1.35, small.height * 1.5),
                 fill=mix(VOID, VIOLET, 0.08))
    canvas = small.filter(ImageFilter.GaussianBlur(6)).resize((width, height), Image.BILINEAR)

    draw = ImageDraw.Draw(canvas)
    dot = mix(VOID, INK, 0.10)
    for y in range(0, height, 34):
        for x in range(0, width, 34):
            draw.point((x, y), fill=dot)
    if len(_BACKDROPS) > 4:
        _BACKDROPS.clear()
    _BACKDROPS[key] = canvas
    return canvas


def backdrop(width: int, height: int, accent: str) -> ImageTk.PhotoImage:
    return _store(("backdrop", width, height, accent),
                  ImageTk.PhotoImage(backdrop_image(width, height, accent)))


def ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 3 * t * t - 2 * t * t * t
