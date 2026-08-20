from __future__ import annotations
import tkinter as tk
import tkinter.font as tkfont
from typing import Callable, List, Optional
import numpy as np
from ui import theme
from ui.theme import px
from ui.widgets import Button, ClipLabel, Divider, Pill, ellipsize

STATUS_COLORS = {"MATCH": (theme.BAD, theme.BAD_SOFT), "NO MATCH": (theme.OK, theme.OK_SOFT), "SEG-FAIL": (theme.MUTED, theme.MUTED_SOFT), "PENDING": (theme.MUTED, theme.MUTED_SOFT), "ERROR": (theme.BAD, theme.BAD_SOFT)}
THUMB = 54
THUMB_GAP = 8

class ComparePage(tk.Frame):
    def __init__(self, parent, on_back: Callable[[], None], load_original: Callable[[int], Optional[np.ndarray]], on_save: Optional[Callable[[int], None]] = None):
        super().__init__(parent, bg=theme.APP_BG)
        self._on_back = on_back
        self._load_original = load_original
        self._on_save = on_save
        self._rows: List = []
        self._index = -1
        self._photos = {}
        self._render_job: Optional[str] = None
        self._last_render = None
        self._strip_items: List[dict] = []
        self._strip_font = tkfont.Font(font=theme.font(8))
        self._build_toolbar()
        self._build_stages()
        self._build_filmstrip()

    def _build_toolbar(self) -> None:
        bar = tk.Frame(self, bg=theme.CARD, height=px(52))
        bar.pack(fill="x")
        bar.pack_propagate(False)
        Button(bar, "Back to results", self._on_back, variant="secondary", pad=12).pack(side="left", padx=(px(14), px(6)), pady=px(10))
        self._title = tk.Label(bar, text="", font=theme.font(12, strong=True), fg=theme.INK, bg=theme.CARD)
        self._title.pack(side="left", padx=(px(8), px(10)))
        self._pill_holder = tk.Frame(bar, bg=theme.CARD)
        self._pill_holder.pack(side="left")
        self._pill: Optional[Pill] = None
        self._score = tk.Label(bar, text="", font=theme.font(9, mono=True), fg=theme.INK_FAINT, bg=theme.CARD)
        self._score.pack(side="left", padx=px(10))
        if self._on_save is not None:
            Button(bar, "Save this comparison", lambda: self._on_save(self._index), variant="secondary", pad=12).pack(side="right", padx=(px(6), px(14)))
        Button(bar, "Next", lambda: self.step(1), variant="ghost", pad=10).pack(side="right", padx=(0, px(6)))
        self._position = tk.Label(bar, text="", font=theme.font(9), fg=theme.INK_FAINT, bg=theme.CARD)
        self._position.pack(side="right", padx=px(6))
        Button(bar, "Previous", lambda: self.step(-1), variant="ghost", pad=10).pack(side="right")
        Divider(self, theme.LINE).pack(fill="x")

    def _build_stages(self) -> None:
        wrap = tk.Frame(self, bg=theme.APP_BG)
        wrap.pack(fill="both", expand=True, padx=px(14), pady=px(12))
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1, uniform="stage")
        wrap.columnconfigure(1, weight=1, uniform="stage")
        self._left_stage, self._left_image = self._make_stage(wrap, 0, "ORIGINAL", "as photographed")
        self._right_stage, self._right_image = self._make_stage(wrap, 1, "DETECTED", "outline in green, findings boxed")
        self._caption = ClipLabel(self, "", font=theme.font(9), fg=theme.INK_SOFT, bg=theme.APP_BG)
        self._caption.pack(fill="x", padx=px(20), pady=(0, px(8)))
        self._left_stage.bind("<Configure>", self._on_resize)

    def _make_stage(self, parent, column: int, title: str, subtitle: str):
        card = tk.Frame(parent, bg=theme.LINE)
        card.grid(row=0, column=column, sticky="nsew", padx=(0, px(12)) if column == 0 else 0)
        inner = tk.Frame(card, bg=theme.CARD)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        head = tk.Frame(inner, bg=theme.CARD)
        head.pack(fill="x", padx=px(14), pady=(px(10), px(9)))
        tk.Label(head, text=title, font=theme.font(9, bold=True), fg=theme.INK, bg=theme.CARD).pack(side="left")
        tk.Label(head, text=subtitle, font=theme.font(8), fg=theme.INK_FAINT, bg=theme.CARD).pack(side="left", padx=(px(8), 0))
        Divider(inner, theme.LINE_SOFT).pack(fill="x")
        stage = tk.Frame(inner, bg=theme.STAGE)
        stage.pack(fill="both", expand=True, padx=px(10), pady=px(10))
        label = tk.Label(stage, bg=theme.STAGE, bd=0, text="", font=theme.font(9), fg=theme.INK_FAINT)
        label.place(relx=0.5, rely=0.5, anchor="center")
        return stage, label

    def _build_filmstrip(self) -> None:
        Divider(self, theme.LINE).pack(fill="x")
        holder = tk.Frame(self, bg=theme.CARD, height=px(THUMB + 34))
        holder.pack(fill="x")
        holder.pack_propagate(False)
        self._strip = tk.Canvas(holder, bg=theme.CARD, bd=0, highlightthickness=0, width=1, height=1)
        self._strip.pack(fill="both", expand=True, padx=px(14), pady=px(8))
        self._strip.bind("<Button-1>", self._on_strip_click)
        self._strip.bind("<MouseWheel>", self._on_strip_wheel)

    def show(self, rows: List, index: int) -> None:
        if rows is not self._rows:
            self._rows = rows
            self._build_strip_items()
        if not rows:
            return
        self._index = max(0, min(index, len(rows) - 1))
        self._last_render = None
        self._update_header()
        self._paint_strip()
        self._scroll_strip_into_view()
        self._render()

    def step(self, delta: int) -> None:
        if not self._rows:
            return
        self.show(self._rows, (self._index + delta) % len(self._rows))

    @property
    def index(self) -> int:
        return self._index

    def _update_header(self) -> None:
        row = self._rows[self._index]
        self._title.configure(text=row.name)
        self._position.configure(text=f"{self._index + 1} / {len(self._rows)}")
        self._score.configure(text=f"score {row.score:.2f}" if row.score else "")
        if self._pill is not None:
            self._pill.destroy()
        fg, bg = STATUS_COLORS.get(row.status, (theme.MUTED, theme.MUTED_SOFT))
        self._pill = Pill(self._pill_holder, row.status, fg, bg)
        self._pill.pack()
        if row.warnings:
            self._caption.set_style(fg=theme.WARN)
            self._caption.set_text("!  " + "; ".join(row.warnings) + "    " + row.evidence.replace("\n", " "))
        else:
            self._caption.set_style(fg=theme.INK_SOFT)
            self._caption.set_text(row.evidence.replace("\n", " "))

    def _on_resize(self, _event=None) -> None:
        if self._render_job is not None:
            self.after_cancel(self._render_job)
        self._render_job = self.after(140, self._render)

    def _render(self) -> None:
        self._render_job = None
        if not (0 <= self._index < len(self._rows)):
            return
        width = max(self._left_stage.winfo_width() - px(12), px(160))
        height = max(self._left_stage.winfo_height() - px(12), px(120))
        if (self._index, width, height) == self._last_render:
            return
        self._last_render = (self._index, width, height)
        row = self._rows[self._index]
        self._draw(self._left_image, "left", self._load_original(self._index), width, height, "original could not be read")
        self._draw(self._right_image, "right", row.annotated, width, height, "nothing was drawn for this photo")

    def _draw(self, label: tk.Label, key: str, image, width: int, height: int, missing: str) -> None:
        if image is None:
            label.configure(image="", text=missing)
            self._photos.pop(key, None)
            return
        photo = theme.photo_rounded(image, width, height, px(6))
        self._photos[key] = photo
        label.configure(image=photo, text="")

    def _build_strip_items(self) -> None:
        self._strip.delete("all")
        self._strip_items = []
        box = px(THUMB)
        step = box + px(THUMB_GAP)
        for position, row in enumerate(self._rows):
            x = position * step
            plain = (theme.photo_thumb(row.annotated, box, px(6)) if row.annotated is not None else theme.placeholder_tile(box, px(6)))
            chosen = (theme.photo_thumb(row.annotated, box, px(6), ring=theme.ACCENT) if row.annotated is not None else theme.placeholder_tile(box, px(6)))
            item = {"plain": plain, "chosen": chosen}
            item["image"] = self._strip.create_image(x, 0, anchor="nw", image=plain)
            fg, _ = STATUS_COLORS.get(row.status, (theme.MUTED, theme.MUTED_SOFT))
            item["dot"] = self._strip.create_oval(x + box - px(12), px(4), x + box - px(4), px(12), fill=fg, outline=theme.CARD)
            item["label"] = self._strip.create_text(x + box // 2, box + px(3), anchor="n", text=ellipsize(row.name, self._strip_font, box + px(8)), font=self._strip_font, fill=theme.INK_FAINT)
            self._strip_items.append(item)
        self._strip.configure(
            scrollregion=(0, 0, max(len(self._rows) * step - px(THUMB_GAP), 1), box + px(18)))

    def _paint_strip(self) -> None:
        for position, item in enumerate(self._strip_items):
            current = position == self._index
            self._strip.itemconfigure(item["image"], image=item["chosen"] if current else item["plain"])
            self._strip.itemconfigure(item["label"], fill=theme.ACCENT if current else theme.INK_FAINT)

    def _scroll_strip_into_view(self) -> None:
        if not self._strip_items:
            return
        step = px(THUMB) + px(THUMB_GAP)
        total = max(len(self._strip_items) * step, 1)
        view = self._strip.winfo_width()
        if view <= 1 or total <= view:
            self._strip.xview_moveto(0.0)
            return
        centre = self._index * step + step / 2
        self._strip.xview_moveto(
            min(max((centre - view / 2) / total, 0.0), 1.0))

    def _on_strip_click(self, event) -> None:
        step = px(THUMB) + px(THUMB_GAP)
        position = int(self._strip.canvasx(event.x) // step)
        if 0 <= position < len(self._rows):
            self.show(self._rows, position)

    def _on_strip_wheel(self, event) -> None:
        self._strip.xview_scroll(int(-event.delta / 120) * 2, "units")
