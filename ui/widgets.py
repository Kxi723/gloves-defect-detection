"""The widget kit the app is assembled from.

Everything here is a plain Tk widget wearing a Pillow-drawn surface, rather
than a ttk widget fighting its theme engine. That trade buys rounded
corners, real hover states and per-widget colour control, and costs the few
lines of bookkeeping in each class below.
"""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from typing import Callable, List, Optional

import numpy as np

from ui import theme
from ui.theme import px


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #

class Button(tk.Label):
    """A flat rounded button with hover, press and disabled states."""

    VARIANTS = {
        # variant: (fill, text, outline, hover fill)
        "primary": (theme.ACCENT, "#ffffff", None, theme.ACCENT_DARK),
        "secondary": (theme.CARD, theme.INK, theme.LINE, theme.HOVER),
        "ghost": (None, theme.INK_SOFT, None, theme.MUTED_SOFT),
    }

    def __init__(self, parent, text: str, command: Optional[Callable] = None,
                 variant: str = "secondary", parent_bg: str = theme.CARD,
                 pad: int = 13, height: int = 30, stretch: bool = False,
                 size: int = 9):
        fill, fg, outline, hover = self.VARIANTS[variant]
        self._command = command
        self._fill, self._fg, self._outline, self._hover = fill, fg, outline, hover
        self._parent_bg = parent_bg
        self._enabled = True
        self._stretch = stretch
        self._height = px(height)
        self._radius = px(6)
        self._font = tkfont.Font(family=theme.FAMILY, size=size, weight="bold")
        self._width = self._font.measure(text) + px(pad) * 2

        super().__init__(parent, text=text, font=self._font, fg=fg,
                         bg=parent_bg, bd=0, highlightthickness=0,
                         compound="center", cursor="hand2")
        self._paint(hover=False)
        self.bind("<Enter>", lambda _e: self._enabled and self._paint(hover=True))
        self.bind("<Leave>", lambda _e: self._enabled and self._paint(hover=False))
        self.bind("<Button-1>", self._on_press)
        if stretch:
            self.bind("<Configure>", self._on_resize)

    def _paint(self, hover: bool) -> None:
        fill = self._hover if hover else self._fill
        if fill is None:
            fill = self._parent_bg if not hover else self._hover
        self._surface = theme.surface(self._width, self._height, self._radius,
                                      fill, self._outline)
        self.configure(image=self._surface)

    def _on_resize(self, event) -> None:
        if event.width > 1 and event.width != self._width:
            self._width = event.width
            self._paint(hover=False)

    def _on_press(self, _event=None) -> None:
        if self._enabled and self._command is not None:
            self._command()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.configure(cursor="hand2" if enabled else "arrow",
                       fg=self._fg if enabled else theme.INK_FAINT)
        if enabled:
            self._paint(hover=False)
        else:
            self._surface = theme.surface(self._width, self._height,
                                          self._radius, theme.MUTED_SOFT, None)
            self.configure(image=self._surface)

    def set_text(self, text: str) -> None:
        self.configure(text=text)
        if not self._stretch:
            self._width = self._font.measure(text) + px(26)
            self._paint(hover=False)


class Pill(tk.Label):
    """A small status badge: rounded fill, short uppercase text."""

    def __init__(self, parent, text: str, fg: str, bg: str,
                 parent_bg: str = theme.CARD, size: int = 8):
        font = tkfont.Font(family=theme.FAMILY, size=size, weight="bold")
        width = font.measure(text) + px(16)
        height = px(19)
        self._surface = theme.surface(width, height, height // 2, bg)
        super().__init__(parent, text=text, image=self._surface, font=font,
                         fg=fg, bg=parent_bg, bd=0, highlightthickness=0,
                         compound="center")


class Divider(tk.Frame):
    def __init__(self, parent, color: str = theme.LINE):
        super().__init__(parent, bg=color, height=1)


class ClipLabel(tk.Frame):
    """One line of text that ellipsizes instead of stretching its parent.

    Tk has no ellipsis mode, and a Label wide enough for its text pushes the
    layout around, so the text is measured and cut to whatever width the
    geometry manager actually granted.
    """

    def __init__(self, parent, text: str = "", font: Optional[tuple] = None,
                 fg: str = theme.INK, bg: str = theme.CARD):
        self._font = tkfont.Font(font=font or theme.font(10))
        super().__init__(parent, bg=bg, width=px(40),
                         height=self._font.metrics("linespace") + px(2))
        self.pack_propagate(False)
        self._label = tk.Label(self, text=text, font=self._font, fg=fg, bg=bg,
                               anchor="w", bd=0, highlightthickness=0)
        self._label.pack(fill="both", expand=True)
        self._full = text
        self.bind("<Configure>", lambda _e: self._refit())

    def set_text(self, text: str) -> None:
        self._full = text
        self._refit()

    def set_style(self, fg: Optional[str] = None, bg: Optional[str] = None) -> None:
        if fg:
            self._label.configure(fg=fg)
        if bg:
            self.configure(bg=bg)
            self._label.configure(bg=bg)

    def _refit(self) -> None:
        available = self.winfo_width()
        if available <= 1:
            return
        if self._font.measure(self._full) <= available:
            self._label.configure(text=self._full)
            return
        ellipsis = "…"
        low, high = 0, len(self._full)
        while low < high:
            mid = (low + high + 1) // 2
            if self._font.measure(self._full[:mid] + ellipsis) <= available:
                low = mid
            else:
                high = mid - 1
        self._label.configure(text=self._full[:low].rstrip() + ellipsis)


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #

class Card(tk.Frame):
    """A white panel with a hairline border, an optional header and a body."""

    def __init__(self, parent):
        super().__init__(parent, bg=theme.LINE)
        self.inner = tk.Frame(self, bg=theme.CARD)
        self.inner.pack(fill="both", expand=True, padx=1, pady=1)
        self.body = self.inner

    def add_header(self, step: str, title: str) -> tk.Frame:
        """Numbered step badge + section title, with a rule underneath."""
        head = tk.Frame(self.inner, bg=theme.CARD)
        head.pack(fill="x", padx=px(14), pady=(px(11), px(10)))

        badge_font = tkfont.Font(family=theme.FAMILY, size=8, weight="bold")
        side = px(18)
        badge_surface = theme.surface(side, side, side // 2, theme.ACCENT_SOFT)
        badge = tk.Label(head, text=step, image=badge_surface, font=badge_font,
                         fg=theme.ACCENT, bg=theme.CARD, compound="center",
                         bd=0)
        badge.image = badge_surface
        badge.pack(side="left")

        tk.Label(head, text=title.upper(), font=theme.font(9, bold=True),
                 fg=theme.INK, bg=theme.CARD).pack(side="left", padx=(px(8), 0))

        Divider(self.inner).pack(fill="x")
        self.body = tk.Frame(self.inner, bg=theme.CARD)
        self.body.pack(fill="both", expand=True)
        return head


class ScrollFrame(tk.Frame):
    """A vertically scrolling area with a slim, self-drawn scrollbar."""

    def __init__(self, parent, bg: str = theme.CARD, padding: int = 0):
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, bd=0, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.bar = tk.Canvas(self, bg=bg, width=px(8), bd=0,
                             highlightthickness=0)
        self.bar.pack(side="right", fill="y")
        self._thumb = self.bar.create_rectangle(0, 0, 0, 0,
                                                fill=theme.SCROLL_THUMB,
                                                outline="")

        self.inner = tk.Frame(self.canvas, bg=bg)
        self._window = self.canvas.create_window((0, 0), window=self.inner,
                                                 anchor="nw")
        self._padding = px(padding)

        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.bind("<Enter>", lambda _e: self._bind_wheel(True))
        self.bind("<Leave>", lambda _e: self._bind_wheel(False))

    # -- geometry ------------------------------------------------------- #

    def _on_inner(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._draw_bar()

    def _on_canvas(self, event) -> None:
        self.canvas.itemconfigure(self._window,
                                  width=event.width - self._padding)
        self._draw_bar()

    def _draw_bar(self) -> None:
        view_h = self.canvas.winfo_height()
        content_h = max(self.inner.winfo_reqheight(), 1)
        if content_h <= view_h or view_h <= 1:
            self.bar.coords(self._thumb, 0, 0, 0, 0)
            return
        first, last = self.canvas.yview()
        top = int(first * view_h)
        bottom = max(int(last * view_h), top + px(24))
        self.bar.coords(self._thumb, px(2), top, px(6), bottom)

    # -- wheel ---------------------------------------------------------- #

    def _bind_wheel(self, active: bool) -> None:
        if active:
            self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        else:
            self.canvas.unbind_all("<MouseWheel>")

    def _on_wheel(self, event) -> None:
        if self.inner.winfo_reqheight() <= self.canvas.winfo_height():
            return
        self.canvas.yview_scroll(int(-event.delta / 120) * 3, "units")
        self._draw_bar()

    def scroll_to_top(self) -> None:
        self.canvas.yview_moveto(0.0)
        self._draw_bar()


# --------------------------------------------------------------------------- #
# Rows and tiles
# --------------------------------------------------------------------------- #

class DefectRow(tk.Frame):
    """One entry of the defect menu."""

    def __init__(self, parent, index: int, label: str, pending: bool,
                 on_click: Callable[[int], None]):
        super().__init__(parent, bg=theme.CARD, height=px(34))
        self.pack_propagate(False)
        self.index, self.pending = index, pending
        self._selected = False

        self._bar = tk.Frame(self, bg=theme.CARD, width=px(3))
        self._bar.pack(side="left", fill="y")

        fg = theme.INK_FAINT if pending else theme.INK
        self._text = tk.Label(self, text=label, font=theme.font(10),
                              fg=fg, bg=theme.CARD, anchor="w", bd=0)
        self._text.pack(side="left", padx=(px(11), 0))

        self._pill = None
        if pending:
            self._pill = Pill(self, "PENDING", theme.MUTED, theme.MUTED_SOFT)
            self._pill.pack(side="right", padx=(0, px(12)))

        for widget in (self, self._text, self._bar):
            widget.bind("<Button-1>", lambda _e: on_click(index))
            widget.bind("<Enter>", lambda _e: self._hover(True))
            widget.bind("<Leave>", lambda _e: self._hover(False))
            widget.configure(cursor="hand2")
        if self._pill is not None:
            self._pill.bind("<Button-1>", lambda _e: on_click(index))

    def _paint(self, bg: str) -> None:
        for widget in (self, self._text):
            widget.configure(bg=bg)
        if self._pill is not None:
            self._pill.configure(bg=bg)

    def _hover(self, entering: bool) -> None:
        if self._selected:
            return
        self._paint(theme.HOVER if entering else theme.CARD)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._paint(theme.SELECTED if selected else theme.CARD)
        self._bar.configure(bg=theme.ACCENT if selected else theme.CARD)
        if not self.pending:
            self._text.configure(
                fg=theme.ACCENT if selected else theme.INK,
                font=theme.font(10, strong=selected))


class PhotoTile(tk.Frame):
    """One photo in the contact sheet: thumbnail, caption, selected state."""

    def __init__(self, parent, index: int, name: str, box: int,
                 on_click: Callable[[int, object], None]):
        super().__init__(parent, bg=theme.CARD)
        self.index, self.box = index, box
        self._selected = False
        self._image_bgr: Optional[np.ndarray] = None
        self._normal = self._chosen = theme.placeholder_tile(box, px(8))

        self._thumb = tk.Label(self, image=self._normal, bg=theme.CARD, bd=0,
                               highlightthickness=0, cursor="hand2")
        self._thumb.pack()
        self._caption = ClipLabel(self, name, font=theme.font(8),
                                  fg=theme.INK_SOFT, bg=theme.CARD)
        self._caption.configure(width=box)
        self._caption.pack(fill="x", pady=(px(4), 0))

        for widget in (self, self._thumb, self._caption):
            widget.bind("<Button-1>", lambda e: on_click(index, e))

    def set_image(self, image_bgr: np.ndarray) -> None:
        self._image_bgr = image_bgr
        radius = px(8)
        self._normal = theme.photo_thumb(image_bgr, self.box, radius)
        self._chosen = theme.photo_thumb(image_bgr, self.box, radius,
                                         ring=theme.ACCENT, check=True)
        self._thumb.configure(image=self._chosen if self._selected
                              else self._normal)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._thumb.configure(image=self._chosen if selected else self._normal)
        self._caption.set_style(fg=theme.ACCENT if selected else theme.INK_SOFT)


class ResultRow(tk.Frame):
    """One inspected photo: thumbnail, name, evidence, status badge."""

    STATUS_COLORS = {
        "MATCH": (theme.BAD, theme.BAD_SOFT),
        "NO MATCH": (theme.OK, theme.OK_SOFT),
        "SEG-FAIL": (theme.MUTED, theme.MUTED_SOFT),
        "PENDING": (theme.MUTED, theme.MUTED_SOFT),
        "ERROR": (theme.BAD, theme.BAD_SOFT),
    }

    def __init__(self, parent, index: int, name: str, status: str, score: float,
                 evidence: str, thumb_bgr: Optional[np.ndarray],
                 on_click: Callable[[int], None]):
        super().__init__(parent, bg=theme.CARD, height=px(52))
        self.pack_propagate(False)
        self.index = index
        self._selected = False
        self._parts: List[tk.Widget] = [self]

        self._bar = tk.Frame(self, bg=theme.CARD, width=px(3))
        self._bar.pack(side="left", fill="y")

        side = px(36)
        if thumb_bgr is not None:
            self._thumb_image = theme.photo_thumb(thumb_bgr, side, px(6))
        else:
            self._thumb_image = theme.placeholder_tile(side, px(6))
        thumb = tk.Label(self, image=self._thumb_image, bg=theme.CARD, bd=0)
        thumb.pack(side="left", padx=(px(10), px(10)))
        self._parts.append(thumb)

        fg, bg = self.STATUS_COLORS.get(status, (theme.MUTED, theme.MUTED_SOFT))
        badge_holder = tk.Frame(self, bg=theme.CARD)
        badge_holder.pack(side="right", padx=(px(8), px(12)))
        self._parts.append(badge_holder)
        self._pill = Pill(badge_holder, status, fg, bg)
        self._pill.pack(anchor="e")
        self._score = tk.Label(badge_holder,
                               text=f"score {score:.2f}" if score else "",
                               font=theme.font(8, mono=True), fg=theme.INK_FAINT,
                               bg=theme.CARD)
        self._score.pack(anchor="e", pady=(px(2), 0))
        self._parts.append(self._score)

        text = tk.Frame(self, bg=theme.CARD)
        text.pack(side="left", fill="both", expand=True)
        self._parts.append(text)
        self._name = ClipLabel(text, name, font=theme.font(10, strong=True),
                               fg=theme.INK, bg=theme.CARD)
        self._name.pack(fill="x")
        self._evidence = ClipLabel(text, evidence.replace("\n", " "),
                                   font=theme.font(8), fg=theme.INK_SOFT,
                                   bg=theme.CARD)
        self._evidence.pack(fill="x")

        for widget in self._parts + [self._pill, self._name, self._evidence]:
            widget.bind("<Button-1>", lambda _e: on_click(index))
            widget.bind("<Enter>", lambda _e: self._hover(True))
            widget.bind("<Leave>", lambda _e: self._hover(False))
            widget.configure(cursor="hand2")

    def _paint(self, bg: str) -> None:
        for widget in self._parts:
            widget.configure(bg=bg)
        self._pill.configure(bg=bg)
        self._score.configure(bg=bg)
        self._name.set_style(bg=bg)
        self._evidence.set_style(bg=bg)

    def _hover(self, entering: bool) -> None:
        if self._selected:
            return
        self._paint(theme.HOVER if entering else theme.CARD)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._paint(theme.SELECTED if selected else theme.CARD)
        self._bar.configure(bg=theme.ACCENT if selected else theme.CARD)
