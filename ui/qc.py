"""QC floor look for the Glove Defect Detection System.

Ink dark ground, one signal yellow, square edges and a hazard stripe along the
top. Holds the palette, the type, the small image helpers both screens share, and
the line screen, an accordion of three photo panels that opens a detector.
"""
from __future__ import annotations

import math
import threading
import tkinter as tk
import tkinter.font as tkfont
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

import pipeline
from ui import neon

TITLE = "Glove Defect Detection System"

INK = "#070A12"
INK_2 = "#0D1220"
RULE = "#20293C"
PAPER = "#F2F5FA"
DIM = "#93A0B8"
DIM_2 = "#6B7893"
SOFT = "#C6CEDC"
MARK = "#FFD100"
REJECT = "#FF4438"
PASS = "#00E08A"
PANE = "#05070C"

# name on screen, the kind of evidence it reads, and one line on how it decides
DETECTORS: Dict[str, Tuple[str, str, str]] = {
    "fold": ("Fold damage", "SHAPE",
             "Shading, weave and colour residual channels trace creases across the palm, "
             "and only a crease long enough to span the palm counts as a fold."),
    "dirty": ("Contamination", "TEXTURE",
              "Lightness outliers inside the glove, kept only where they break the weave "
              "or drift away from the glove's own hue."),
    "tear": ("Fingertip tear", "SHAPE",
             "Holes, contour notches and skin showing through, counted only when they "
             "sit at a located fingertip."),
}
# the photo behind each panel, and how far down it is framed
HERO = {"fold": ("Fold_Latex_1", 0.52), "dirty": ("Dirty_Latex_3", 0.46),
        "tear": ("TearFinger_Latex_1", 0.30)}

_TYPE = {
    "disp": ("Segoe UI Black", "Arial Black"),
    "sans": ("Segoe UI", "Arial"),
    "semi": ("Segoe UI Semibold", "Segoe UI"),
    "mono": ("Cascadia Mono", "Consolas"),
}
_FAMILIES: Optional[set] = None
_STRIPES: Dict[tuple, ImageTk.PhotoImage] = {}
_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def px(value: float) -> int:
    return neon.px(value)


def font(kind: str, size: float, bold: bool = False) -> tuple:
    """Sizes are design pixels, handed to Tk as real pixels."""
    global _FAMILIES
    if _FAMILIES is None:
        _FAMILIES = set(tkfont.families())
    preferred, fallback = _TYPE[kind]
    family = preferred if preferred in _FAMILIES else fallback
    return (family, -px(size), "bold") if bold else (family, -px(size))


def rgb(colour: str) -> Tuple[int, int, int]:
    colour = colour.lstrip("#")
    return int(colour[0:2], 16), int(colour[2:4], 16), int(colour[4:6], 16)


def mix(first: str, second: str, amount: float) -> str:
    return neon.mix(first, second, amount)


def clip(text: str, fnt: tuple, room: int) -> str:
    """One line, cut with an ellipsis rather than wrapped."""
    measure = tkfont.Font(font=fnt)
    if measure.measure(text) <= room:
        return text
    while text and measure.measure(text + "…") > room:
        text = text[:-1]
    return text + "…"


def stripes(width: int, height: int, first: str, second: str, band: int) -> ImageTk.PhotoImage:
    """Diagonal tape, the hazard stripe along the top and the foot of a stamp."""
    key = (width, height, first, second, band)
    cached = _STRIPES.get(key)
    if cached is not None:
        return cached
    ss = 3
    ys, xs = np.mgrid[0:max(height, 1) * ss, 0:max(width, 1) * ss].astype(np.float32)
    along = ((xs + ys) / math.sqrt(2.0)) % (2 * band * ss)
    pick = (along < band * ss)[..., None]
    image = np.where(pick, np.array(rgb(first), np.uint8), np.array(rgb(second), np.uint8))
    photo = ImageTk.PhotoImage(Image.fromarray(image.astype(np.uint8)).resize(
        (max(width, 1), max(height, 1)), _LANCZOS))
    if len(_STRIPES) > 24:
        _STRIPES.clear()
    _STRIPES[key] = photo
    return photo


def cover(bgr: np.ndarray, width: int, height: int, ypos: float = 0.5) -> Image.Image:
    """Fill the box and crop the overflow, like CSS object-fit cover."""
    h, w = bgr.shape[:2]
    scale = max(width / w, height / h)
    size = (max(int(round(w * scale)), width), max(int(round(h * scale)), height))
    grown = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    left = (size[0] - width) // 2
    top = int((size[1] - height) * ypos)
    crop = grown[top:top + height, left:left + width]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def letterbox(bgr: np.ndarray, width: int, height: int, ground: str = PANE) -> np.ndarray:
    """Fit the whole frame inside the box on a flat ground, as RGB."""
    canvas = np.empty((max(height, 1), max(width, 1), 3), np.uint8)
    canvas[:] = rgb(ground)
    h, w = bgr.shape[:2]
    scale = min(width / w, height / h)
    size = (max(int(w * scale), 1), max(int(h * scale), 1))
    fitted = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x, y = (width - size[0]) // 2, (height - size[1]) // 2
    canvas[y:y + size[1], x:x + size[0]] = cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB)
    return canvas


# ------------------------------------------------------------------ line screen

class LineScreen(tk.Canvas):
    """Three photo panels side by side. The one under the pointer opens wide and
    shows how its detector works, the other two stay narrow but keep their counts,
    so all three can be compared without opening anything."""

    OPEN, SHUT = 0.48, 0.26

    def __init__(self, app, parent: tk.Widget) -> None:
        super().__init__(parent, bg=INK, highlightthickness=0, bd=0)
        self.app = app
        self.specs = list(pipeline.DEFECTS)
        self.open = 1
        self._frac = [self.OPEN if i == self.open else self.SHUT for i in range(len(self.specs))]
        self._sources: Dict[str, np.ndarray] = {}
        self._loaded = False
        self._shown_loaded = False
        self._full: Dict[str, tuple] = {}
        self._images: List[ImageTk.PhotoImage] = []
        self._hazard = None
        self._empty = False
        self._awake = False
        self.bind("<Configure>", lambda _e: self.paint())
        self.bind("<Motion>", self._on_motion)
        self.bind("<Button-1>", self._on_click)
        threading.Thread(target=self._load, daemon=True).start()

    # -- lifecycle

    def _load(self) -> None:
        """The panel photos are decoded off the UI thread."""
        photos = self.app.photos
        for spec in self.specs:
            stem = HERO.get(spec.slug, ("", 0.5))[0].lower()
            path = next((p for p in photos if p.stem.lower() == stem), None)
            if path is None:
                path = next((p for p in photos if p.stem.lower().startswith(spec.prefix)), None)
            if path is not None:
                image = pipeline.imread_preview(path)
                if image is not None:
                    self._sources[spec.slug] = image
        self._loaded = True

    def wake(self) -> None:
        if not self._awake:
            self._awake = True
            self.app.animator.add(self._tick)
            self.bind_all("<Left>", lambda _e: self._move(-1))
            self.bind_all("<Right>", lambda _e: self._move(1))
            self.bind_all("<Return>", lambda _e: self._enter(self.open))
        self.paint()

    def sleep(self) -> None:
        self._awake = False
        self.app.animator.remove(self._tick)
        for sequence in ("<Left>", "<Right>", "<Return>"):
            try:
                self.unbind_all(sequence)
            except Exception:
                pass

    def flash_empty(self) -> None:
        self._empty = True
        self.paint()

    def _set_hover(self, index: int) -> None:
        self.open = max(0, min(index, len(self.specs) - 1))

    def _tick(self, delta: float) -> None:
        if self._loaded and not self._shown_loaded:
            self._shown_loaded = True
            self.paint()
            return
        changed = False
        for i in range(len(self._frac)):
            goal = self.OPEN if i == self.open else self.SHUT
            diff = goal - self._frac[i]
            if abs(diff) > 0.0005:
                self._frac[i] += diff * min(1.0, delta * 9.0)
                changed = True
            else:
                self._frac[i] = goal
        if changed:
            self._paint_panels()

    # -- data

    def _counts(self, slug: str) -> Tuple[int, int]:
        saved = self.app.store.get(slug)
        if not saved:
            return 0, 0
        summary = saved["summary"]
        return len(summary), sum(1 for entry in summary.values() if entry["found"])

    # -- geometry

    def _geo(self):
        width, height = max(self.winfo_width(), 10), max(self.winfo_height(), 10)
        top = px(6)
        head = top + px(88)
        foot = height - px(44)
        return width, height, top, head, foot

    def _boxes(self) -> List[Tuple[int, int, int, int]]:
        width, _height, _top, head, foot = self._geo()
        total = sum(self._frac)
        boxes, x = [], 0.0
        for fraction in self._frac:
            w = width * fraction / total
            boxes.append((int(round(x)), head + 1, int(round(x + w)), foot))
            x += w
        return boxes

    def _panel_at(self, x: int, y: int) -> int:
        for i, (x0, y0, x1, y1) in enumerate(self._boxes()):
            if x0 <= x < x1 and y0 <= y < y1:
                return i
        return -1

    # -- painting

    def paint(self) -> None:
        self.delete("all")
        width, height, top, head, foot = self._geo()
        self._hazard = stripes(width, top, MARK, INK, px(13))
        self.create_image(0, 0, image=self._hazard, anchor="nw")
        self.create_line(0, head, width, head, fill=RULE)
        self.create_line(0, foot, width, foot, fill=RULE)

        mid = (top + head) // 2
        self.create_text(px(34), mid, text=TITLE, anchor="w", font=font("disp", 27), fill=PAPER)

        n = len(self.app.photos)
        done = rejected = 0
        for spec in self.specs:
            d, r = self._counts(spec.slug)
            done += d
            rejected += r
        stats = [("PHOTOS", "{:02d}".format(n)), ("REJECTED", "{:02d}".format(rejected)),
                 ("REMAINING", "{:02d}".format(max(n * len(self.specs) - done, 0)))]
        x = width - px(34)
        for label, value in reversed(stats):
            a = self.create_text(x, mid + px(15), text=value, anchor="se", font=font("mono", 26, True),
                                 fill=PAPER)
            b = self.create_text(x, mid - px(13), text=label, anchor="se", font=font("sans", 10.5),
                                 fill=DIM_2)
            span = max(self.bbox(a)[2] - self.bbox(a)[0], self.bbox(b)[2] - self.bbox(b)[0])
            x -= span + px(26)

        # foot
        fy = (foot + height) // 2
        materials = sorted({p.stem.split("_")[1].lower() for p in self.app.photos if p.stem.count("_") >= 2})
        self.create_text(px(34), fy, text="gloves/", anchor="w", font=font("mono", 11.5), fill=DIM_2)
        self.create_text(px(34) + px(90), fy, text="  ·  ".join(materials), anchor="w",
                         font=font("mono", 11.5), fill=DIM_2)
        if self._empty or not self.app.photos:
            self.create_text(width - px(34), fy, text="● NO PHOTOS IN gloves/", anchor="e",
                             font=font("mono", 11.5), fill=REJECT)
        else:
            self.create_text(width - px(34), fy, text="● READY", anchor="e", font=font("mono", 11.5),
                             fill=PASS)
        self._paint_panels()

    def _panel_image(self, slug: str, width: int, height: int) -> Optional[Image.Image]:
        """Each photo is framed and veiled once at its open width, and a narrower
        panel is a centre crop of that, so the accordion never rescales a photo."""
        full_w = int(self.winfo_width() * self.OPEN) + 4
        key = (full_w, height)
        cached = self._full.get(slug)
        if cached is None or cached[0] != key:
            source = self._sources.get(slug)
            if source is None:
                return None
            ypos = HERO.get(slug, ("", 0.5))[1]
            image = np.asarray(cover(source, full_w, height, ypos)).astype(np.float32)
            image = image * 0.92 + np.array(rgb(INK_2), np.float32) * 0.08
            veil = np.interp(np.linspace(0.0, 1.0, height), [0.0, 0.24, 0.48, 1.0],
                             [0.6, 0.0, 0.25, 0.95]).astype(np.float32)[:, None, None]
            image = image * (1.0 - veil) + np.array(rgb(INK), np.float32) * veil
            cached = (key, Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)))
            self._full[slug] = cached
        full = cached[1]
        left = max((full.width - width) // 2, 0)
        return full.crop((left, 0, left + width, height))

    def _paint_panels(self) -> None:
        self.delete("panels")
        self._images = []
        n = len(self.app.photos)
        boxes = self._boxes()
        for i, (spec, (x0, y0, x1, y1)) in enumerate(zip(self.specs, boxes)):
            w, h = x1 - x0, y1 - y0
            if w < 4 or h < 4:
                continue
            name, kind, blurb = DETECTORS.get(spec.slug, (spec.title, "", ""))
            image = self._panel_image(spec.slug, w, h)
            if image is not None:
                photo = ImageTk.PhotoImage(image)
                self._images.append(photo)
                self.create_image(x0, y0, image=photo, anchor="nw", tags="panels")
            else:
                self.create_rectangle(x0, y0, x1, y1, fill=INK_2, outline="", tags="panels")
            if i < len(boxes) - 1:
                self.create_line(x1, y0, x1, y1, fill=RULE, tags="panels")

            self.create_text(x0 + px(24), y0 + px(20), text="{:02d}".format(i + 1), anchor="nw",
                             font=font("mono", 13, True), fill=MARK, tags="panels")
            tag = self.create_text(x1 - px(33), y0 + px(23), text=kind, anchor="ne",
                                   font=font("mono", 11), fill=PAPER, tags="panels")
            bx0, by0, bx1, by1 = self.bbox(tag)
            self.create_rectangle(bx0 - px(9), by0 - px(3), bx1 + px(9), by1 + px(3),
                                  outline=mix(INK, PAPER, 0.4), tags="panels")

            done, rejected = self._counts(spec.slug)
            if self._frac[i] > 0.40:
                self._paint_info(x0, y1, x1, name, blurb, n, done, rejected)
            else:
                self.create_text(x0 + px(22), y1 - px(30), text=name, angle=90, anchor="nw",
                                 font=font("disp", 28), fill=PAPER, tags="panels")
                if done == 0:
                    status, colour = "NOT RUN", DIM_2
                elif rejected:
                    status, colour = "{} REJECTED".format(rejected), REJECT
                elif done >= n:
                    status, colour = "ALL PASS", PASS
                else:
                    status, colour = "{} / {} DONE".format(done, n), DIM
                self.create_text(x1 - px(22), y1 - px(26), text=status, anchor="se",
                                 font=font("mono", 11.5, True), fill=colour, tags="panels")
                self.create_text(x1 - px(22), y1 - px(45), text="{} photos".format(n), anchor="se",
                                 font=font("mono", 11.5), fill=DIM, tags="panels")
            if i == self.open:
                self.create_rectangle(x0, y1 - px(5), x1, y1, fill=MARK, outline="", tags="panels")

    def _paint_info(self, x0: int, y1: int, x1: int, name: str, blurb: str,
                    n: int, done: int, rejected: int) -> None:
        left, right = x0 + px(28), x1 - px(28)
        base = y1 - px(26)
        bh = px(38)
        label = "Start review" if done == 0 else ("Open results" if done >= n else "Continue review")
        go_font = font("disp", 13)
        bw = tkfont.Font(font=go_font).measure(label) + 2 * px(17)
        self.create_rectangle(right - bw, base - bh, right, base, fill=MARK, outline="", tags="panels")
        self.create_text(right - bw // 2, base - bh // 2, text=label, font=go_font, fill=INK, tags="panels")
        count = self.create_text(left, base - bh // 2, text="{} photos  ·  {} reviewed".format(n, done),
                                 anchor="w", font=font("mono", 12.5), fill=DIM, tags="panels")
        if done:
            self.create_text(self.bbox(count)[2] + px(16), base - bh // 2,
                             text="{} REJECTED".format(rejected), anchor="w",
                             font=font("mono", 12.5, True), fill=REJECT if rejected else DIM,
                             tags="panels")
        line_y = base - bh - px(14)
        self.create_line(left, line_y, right, line_y, fill=mix(INK, PAPER, 0.22), tags="panels")
        body = self.create_text(left, line_y - px(18), text=blurb, anchor="sw",
                                width=min(px(360), right - left), font=font("sans", 13.5), fill=SOFT,
                                tags="panels")
        self.create_text(left, self.bbox(body)[1] - px(10), text=name, anchor="sw",
                         font=font("disp", 40), fill=PAPER, tags="panels")

    # -- input

    def _on_motion(self, event) -> None:
        index = self._panel_at(event.x, event.y)
        self.configure(cursor="hand2" if index >= 0 else "")
        if index >= 0 and index != self.open:
            self.open = index
            self._paint_panels()

    def _on_click(self, event) -> None:
        index = self._panel_at(event.x, event.y)
        if index >= 0:
            self._enter(index)

    def _move(self, delta: int) -> None:
        self._set_hover(self.open + delta)
        self._paint_panels()

    def _enter(self, index: int) -> None:
        if 0 <= index < len(self.specs):
            self.app.start_run(self.specs[index])
