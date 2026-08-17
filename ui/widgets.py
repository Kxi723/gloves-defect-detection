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

def _ellipsize(text: str, font: tkfont.Font, available: int) -> str:
    """The longest prefix of `text` that fits, with an ellipsis if it was cut."""
    if available <= 0 or font.measure(text) <= available:
        return text
    ellipsis = "…"
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if font.measure(text[:mid] + ellipsis) <= available:
            low = mid
        else:
            high = mid - 1
    return text[:low].rstrip() + ellipsis


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
        self._pending_width = 0
        self._repaint_job = None

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
        """Repaint once the drag settles, not on every pixel of it.

        Each repaint draws a rounded rectangle at 4x and downsamples it, so
        doing that per <Configure> event both burns time and fills the
        surface cache with widths nobody will see again.
        """
        if event.width <= 1 or event.width == self._width:
            return
        self._pending_width = event.width
        if self._repaint_job is not None:
            self.after_cancel(self._repaint_job)
        self._repaint_job = self.after(90, self._apply_width)

    def _apply_width(self) -> None:
        self._repaint_job = None
        if self._pending_width and self._pending_width != self._width:
            self._width = self._pending_width
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
        self._fitted_width = -1
        self.bind("<Configure>", lambda _e: self._refit())

    def set_text(self, text: str) -> None:
        self._full = text
        self._fitted_width = -1        # force a re-measure of the new string
        self._refit()

    def set_style(self, fg: Optional[str] = None, bg: Optional[str] = None) -> None:
        if fg:
            self._label.configure(fg=fg)
        if bg:
            self.configure(bg=bg)
            self._label.configure(bg=bg)

    def _refit(self) -> None:
        available = self.winfo_width()
        if available <= 1 or available == self._fitted_width:
            return      # a <Configure> that did not change our width
        self._fitted_width = available
        self._label.configure(text=_ellipsize(self._full, self._font, available))


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
        # width/height 1 on purpose: a Tk Canvas asks for 378x266 by
        # default, and that request was silently setting the minimum width
        # of whichever column the scroller sat in.
        self.canvas = tk.Canvas(self, bg=bg, bd=0, highlightthickness=0,
                                width=1, height=1)
        self.canvas.pack(side="left", fill="both", expand=True)
        self._last_width = -1

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
        if event.width != self._last_width:
            self._last_width = event.width
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


class ResultList(tk.Frame):
    """The inspected photos, drawn as canvas items rather than as widgets.

    Each row was originally a Frame holding about ten nested widgets. That
    reads well but Tk re-runs its geometry manager over every one of them on
    every <Configure>, and the cost is linear in the number of rows:
    measured on this layout, 5 rows added 45 ms to a window resize, 20 rows
    118 ms and 60 rows 218 ms — and sixty photos is the normal batch here.

    A canvas does no geometry management for its items, so the whole list is
    one widget no matter how many rows it holds, and a resize only has to
    re-place the right-hand items and re-ellipsize two strings per row.
    The contact sheet was always cheap for exactly this reason.
    """

    STATUS_COLORS = {
        "MATCH": (theme.BAD, theme.BAD_SOFT),
        "NO MATCH": (theme.OK, theme.OK_SOFT),
        "SEG-FAIL": (theme.MUTED, theme.MUTED_SOFT),
        "PENDING": (theme.MUTED, theme.MUTED_SOFT),
        "ERROR": (theme.BAD, theme.BAD_SOFT),
    }

    def __init__(self, parent, on_select: Callable[[int], None],
                 row_height: int = 52):
        super().__init__(parent, bg=theme.CARD)
        self._on_select = on_select
        self._row_h = px(row_height)

        self.canvas = tk.Canvas(self, bg=theme.CARD, bd=0, highlightthickness=0,
                                width=1, height=1)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.bar = tk.Canvas(self, bg=theme.CARD, width=px(8), bd=0,
                             highlightthickness=0)
        self.bar.pack(side="right", fill="y")
        self._bar_thumb = self.bar.create_rectangle(0, 0, 0, 0,
                                                    fill=theme.SCROLL_THUMB,
                                                    outline="")

        self._rows: List[dict] = []
        self._selected = -1
        self._hover = -1
        self._width = 0
        self._placeholder: Optional[int] = None

        self._name_font = tkfont.Font(font=theme.font(10, strong=True))
        self._evidence_font = tkfont.Font(font=theme.font(8))
        self._score_font = tkfont.Font(font=theme.font(8, mono=True))
        self._pill_font = tkfont.Font(family=theme.FAMILY, size=8, weight="bold")

        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda _e: self._set_hover(-1))
        self.canvas.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", lambda _e: self._bind_wheel(True))
        self.bind("<Leave>", lambda _e: self._bind_wheel(False))

    # -- content -------------------------------------------------------- #

    def clear(self, placeholder: str = "") -> None:
        self.canvas.delete("all")
        self._rows = []
        self._selected = self._hover = -1
        self.canvas.configure(scrollregion=(0, 0, 0, 0))
        self._placeholder = None
        if placeholder:
            self._placeholder = self.canvas.create_text(
                0, px(28), text=placeholder, font=theme.font(9),
                fill=theme.INK_FAINT, anchor="n")
            self._place_placeholder()
        self._draw_bar()

    def append(self, name: str, status: str, score: float, evidence: str,
               thumb_bgr: Optional[np.ndarray]) -> int:
        if self._placeholder is not None:
            self.canvas.delete(self._placeholder)
            self._placeholder = None

        index = len(self._rows)
        top = index * self._row_h
        side = px(36)
        canvas = self.canvas

        fg, bg = self.STATUS_COLORS.get(status, (theme.MUTED, theme.MUTED_SOFT))
        pill_w = self._pill_font.measure(status) + px(16)
        pill_h = px(19)

        row = {
            "name": name,
            "evidence": evidence.replace("\n", " "),
            "thumb": (theme.photo_thumb(thumb_bgr, side, px(6))
                      if thumb_bgr is not None
                      else theme.placeholder_tile(side, px(6))),
            "pill_image": theme.surface(pill_w, pill_h, pill_h // 2, bg),
            "pill_w": pill_w,
        }
        row["bg"] = canvas.create_rectangle(0, top, 1, top + self._row_h,
                                            fill=theme.CARD, outline="")
        row["bar"] = canvas.create_rectangle(0, top, px(3), top + self._row_h,
                                             fill=theme.CARD, outline="")
        row["image"] = canvas.create_image(px(10), top + self._row_h // 2,
                                           image=row["thumb"], anchor="w")
        text_x = px(10) + side + px(10)
        row["title"] = canvas.create_text(text_x, top + px(16), text=name,
                                          font=self._name_font, fill=theme.INK,
                                          anchor="w")
        row["detail"] = canvas.create_text(text_x, top + px(34),
                                           text=row["evidence"],
                                           font=self._evidence_font,
                                           fill=theme.INK_SOFT, anchor="w")
        row["pill_bg"] = canvas.create_image(0, top + px(18),
                                             image=row["pill_image"],
                                             anchor="e")
        row["pill_text"] = canvas.create_text(0, top + px(18), text=status,
                                              font=self._pill_font, fill=fg,
                                              anchor="e")
        row["score"] = canvas.create_text(
            0, top + px(38), text=f"score {score:.2f}" if score else "",
            font=self._score_font, fill=theme.INK_FAINT, anchor="e")
        row["line"] = canvas.create_line(0, top + self._row_h, 1,
                                         top + self._row_h, fill=theme.LINE_SOFT)
        self._rows.append(row)

        canvas.configure(scrollregion=(0, 0, 0, len(self._rows) * self._row_h))
        self._layout_row(index)
        self._draw_bar()
        return index

    def select(self, index: int) -> None:
        previous, self._selected = self._selected, index
        for position in {previous, index}:
            if 0 <= position < len(self._rows):
                self._paint_row(position)

    # -- geometry ------------------------------------------------------- #

    def _on_configure(self, event) -> None:
        # Height-only changes still have to redraw the scrollbar, which is
        # why that call sits outside the width check.
        if event.width != self._width:
            self._width = event.width
            for index in range(len(self._rows)):
                self._layout_row(index)
            self._place_placeholder()
        self._draw_bar()

    def _place_placeholder(self) -> None:
        if self._placeholder is not None:
            self.canvas.coords(self._placeholder,
                               max(self._width, 2) // 2, px(28))

    def _layout_row(self, index: int) -> None:
        """Place the width-dependent parts of one row."""
        width = max(self._width, px(240))
        row = self._rows[index]
        top = index * self._row_h
        canvas = self.canvas

        canvas.coords(row["bg"], 0, top, width, top + self._row_h)
        canvas.coords(row["line"], 0, top + self._row_h, width,
                      top + self._row_h)

        right = width - px(12)
        canvas.coords(row["pill_bg"], right, top + px(18))
        canvas.coords(row["pill_text"], right - px(8), top + px(18))
        canvas.coords(row["score"], right, top + px(38))

        text_x = px(10) + px(36) + px(10)
        available = max(right - row["pill_w"] - px(12) - text_x, px(40))
        canvas.itemconfigure(row["title"],
                             text=_ellipsize(row["name"], self._name_font,
                                             available))
        canvas.itemconfigure(row["detail"],
                             text=_ellipsize(row["evidence"],
                                             self._evidence_font, available))

    def _paint_row(self, index: int) -> None:
        selected = index == self._selected
        fill = theme.SELECTED if selected else (
            theme.HOVER if index == self._hover else theme.CARD)
        self.canvas.itemconfigure(self._rows[index]["bg"], fill=fill)
        self.canvas.itemconfigure(self._rows[index]["bar"],
                                  fill=theme.ACCENT if selected else fill)

    # -- interaction ---------------------------------------------------- #

    def _index_at(self, event) -> int:
        index = int(self.canvas.canvasy(event.y) // self._row_h)
        return index if 0 <= index < len(self._rows) else -1

    def _set_hover(self, index: int) -> None:
        if index == self._hover:
            return
        previous, self._hover = self._hover, index
        for position in (previous, index):
            if 0 <= position < len(self._rows):
                self._paint_row(position)
        self.canvas.configure(cursor="hand2" if index >= 0 else "arrow")

    def _on_motion(self, event) -> None:
        self._set_hover(self._index_at(event))

    def _on_click(self, event) -> None:
        index = self._index_at(event)
        if index >= 0:
            self._on_select(index)

    # -- scrolling ------------------------------------------------------ #

    def _bind_wheel(self, active: bool) -> None:
        if active:
            self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        else:
            self.canvas.unbind_all("<MouseWheel>")

    def _on_wheel(self, event) -> None:
        if len(self._rows) * self._row_h <= self.canvas.winfo_height():
            return
        self.canvas.yview_scroll(int(-event.delta / 120), "units")
        self._draw_bar()

    def _draw_bar(self) -> None:
        view_h = self.canvas.winfo_height()
        content_h = len(self._rows) * self._row_h
        if content_h <= view_h or view_h <= 1:
            self.bar.coords(self._bar_thumb, 0, 0, 0, 0)
            return
        first, last = self.canvas.yview()
        top = int(first * view_h)
        bottom = max(int(last * view_h), top + px(24))
        self.bar.coords(self._bar_thumb, px(2), top, px(6), bottom)
