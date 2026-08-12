"""The visual language of the app: palette, type scale, and the drawing
helpers that make Tkinter look like it was designed on purpose.

Two things do most of the work here.

*Crispness.* Windows scales a DPI-unaware process by bitmap-stretching it,
which is why plain Tk apps look soft on a 125% display. `enable_hidpi` opts
the process in before the first window exists, then every pixel dimension
goes through `px()` and every font size through Tk's own scaling, so the
layout keeps its physical size while text and edges stay sharp.

*Soft geometry.* Tk has no rounded corners and no antialiasing. Pillow has
both, so surfaces (buttons, pills, thumbnails) are drawn as RGBA images at
4x and downsampled, then used as widget backgrounds. Tk 8.6 composites the
alpha against the widget's own background colour, so the corners really are
transparent rather than faked with a matching fill.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
# One accent, three status hues, and a neutral ramp. Every status colour has
# a "soft" partner used as the fill behind it, mixed to roughly the same
# lightness so no badge shouts louder than the others.

HEADER_BG = "#171b23"
HEADER_FG = "#ffffff"
HEADER_DIM = "#8b93a3"

APP_BG = "#edeff3"
CARD = "#ffffff"
LINE = "#e0e4ec"
LINE_SOFT = "#eef0f5"

INK = "#171b23"
INK_SOFT = "#5a6373"
INK_FAINT = "#98a1b1"

ACCENT = "#2f5fd9"
ACCENT_DARK = "#2549ac"
ACCENT_SOFT = "#e9eefb"

OK = "#1a7a4a"
OK_SOFT = "#e3f3ea"
BAD = "#c0392f"
BAD_SOFT = "#fbeae8"
WARN = "#9a6a12"
WARN_SOFT = "#fbf1dc"
MUTED = "#6b7484"
MUTED_SOFT = "#eceef3"

HOVER = "#f4f6fa"
SELECTED = "#f3f6fd"
SCROLL_THUMB = "#c6ccd8"
STAGE = "#f5f6fa"

FAMILY = "Segoe UI"
FAMILY_STRONG = "Segoe UI Semibold"
FAMILY_MONO = "Consolas"

_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS

SCALE = 1.0


# --------------------------------------------------------------------------- #
# High DPI
# --------------------------------------------------------------------------- #

def enable_hidpi() -> float:
    """Opt into per-monitor DPI before any window exists. Returns the scale."""
    global SCALE
    if sys.platform != "win32":
        return SCALE
    try:
        # 2 = per-monitor aware. Falls back to the older system-wide call on
        # Windows versions without shcore.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            return SCALE
    try:
        dpi = ctypes.windll.user32.GetDpiForSystem()
    except Exception:
        dpi = 96
    SCALE = max(1.0, dpi / 96.0)
    return SCALE


def apply_scaling(root) -> None:
    """Make point-sized fonts match the display, once a root window exists."""
    root.tk.call("tk", "scaling", SCALE * 96.0 / 72.0)


def px(value: float) -> int:
    """A pixel dimension, corrected for display scaling."""
    return int(round(value * SCALE))


def font(size: int = 10, strong: bool = False, mono: bool = False,
         bold: bool = False) -> tuple:
    family = FAMILY_MONO if mono else (FAMILY_STRONG if strong else FAMILY)
    return (family, size, "bold") if bold else (family, size)


# --------------------------------------------------------------------------- #
# Drawn surfaces
# --------------------------------------------------------------------------- #

_SURFACE_CACHE: Dict[tuple, ImageTk.PhotoImage] = {}
_SUPERSAMPLE = 4


def surface(width: int, height: int, radius: int, fill: Optional[str],
            outline: Optional[str] = None, outline_width: int = 1
            ) -> ImageTk.PhotoImage:
    """A rounded rectangle to sit behind a widget, cached by its parameters.

    Transparent where the corners are cut, so the widget's own background
    shows through and the shape reads as a rounded surface rather than a
    rectangle with painted-on corners.
    """
    key = (width, height, radius, fill, outline, outline_width)
    cached = _SURFACE_CACHE.get(key)
    if cached is not None:
        return cached

    s = _SUPERSAMPLE
    image = Image.new("RGBA", (width * s, height * s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (0, 0, width * s - 1, height * s - 1),
        radius=max(radius, 0) * s,
        fill=fill,
        outline=outline,
        width=outline_width * s if outline else 0,
    )
    photo = ImageTk.PhotoImage(image.resize((width, height), _RESAMPLE))
    _SURFACE_CACHE[key] = photo
    return photo


def _fit(image_bgr: np.ndarray, box_w: int, box_h: int) -> Image.Image:
    h, w = image_bgr.shape[:2]
    scale = min(box_w / w, box_h / h)
    size = (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, size, interpolation=interp)
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


def photo_thumb(image_bgr: np.ndarray, box: int, radius: int,
                ring: Optional[str] = None, check: bool = False
                ) -> ImageTk.PhotoImage:
    """A square thumbnail: cover-cropped, rounded, optionally ringed+ticked.

    Cover-cropping rather than letterboxing keeps a grid of photos on a
    common rhythm, which is what makes a contact sheet scannable.
    """
    s = 2                                   # supersample the mask and ring
    side = box * s
    h, w = image_bgr.shape[:2]
    scale = max(side / w, side / h)
    resized = cv2.resize(image_bgr,
                         (max(int(round(w * scale)), 1),
                          max(int(round(h * scale)), 1)),
                         interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    top, left = (rh - side) // 2, (rw - side) // 2
    cropped = resized[top:top + side, left:left + side]
    tile = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)).convert("RGBA")

    mask = Image.new("L", (side, side), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, side - 1, side - 1),
                                           radius=radius * s, fill=255)
    tile.putalpha(mask)

    if ring:
        draw = ImageDraw.Draw(tile)
        inset = max(s, 1)
        draw.rounded_rectangle((inset // 2, inset // 2,
                                side - 1 - inset // 2, side - 1 - inset // 2),
                               radius=radius * s, outline=ring, width=2 * s)
    if check:
        draw = ImageDraw.Draw(tile)
        r = int(side * 0.13)
        cx, cy = side - r - 4 * s, r + 4 * s
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=ring or ACCENT)
        draw.line([(cx - r * 0.45, cy), (cx - r * 0.1, cy + r * 0.38),
                   (cx + r * 0.5, cy - r * 0.4)],
                  fill="#ffffff", width=max(int(r * 0.32), 2), joint="curve")

    return ImageTk.PhotoImage(tile.resize((box, box), _RESAMPLE))


def photo_rounded(image_bgr: np.ndarray, box_w: int, box_h: int, radius: int
                  ) -> ImageTk.PhotoImage:
    """Fit an image into the box and round its corners (no crop)."""
    fitted = _fit(image_bgr, max(box_w, 24), max(box_h, 24)).convert("RGBA")
    w, h = fitted.size
    mask = Image.new("L", (w * 2, h * 2), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w * 2 - 1, h * 2 - 1),
                                           radius=radius * 2, fill=255)
    fitted.putalpha(mask.resize((w, h), _RESAMPLE))
    return ImageTk.PhotoImage(fitted)


def placeholder_tile(box: int, radius: int) -> ImageTk.PhotoImage:
    """The grey square shown while a thumbnail is still being decoded."""
    return surface(box, box, radius, MUTED_SOFT)
