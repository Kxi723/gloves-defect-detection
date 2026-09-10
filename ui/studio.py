"""Glove Defect Detection System.

Two screens. The launcher offers the three defects, and picking one starts a run
over every photo in the gloves folder, replaying each detector one step at a time so
the preprocessing and the segmentation are visible rather than implied.
"""
from __future__ import annotations

import math
import queue
import random
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

import pipeline
from pipeline import DEFECTS, DefectSpec, Stage, Trace
from ui import neon

TICK_MS = 33
STAGE_DWELL_MS = 520
VERDICT_DWELL_MS = 1700
APP_TITLE = "GLOVE DEFECT DETECTION SYSTEM"
PLAYBACK_SPEED = 2.0  # frames play at twice the base dwell, the detector always runs flat out


def _clip(text: str, measure: "tkfont.Font", room: int) -> str:
    """Single line, cut with an ellipsis rather than wrapped."""
    if measure.measure(text) <= room:
        return text
    trimmed = text
    while trimmed and measure.measure(trimmed + "…") > room:
        trimmed = trimmed[:-1]
    return trimmed + "…"


# --------------------------------------------------------------------- engine

class TraceWorker:
    """Runs pipeline.trace off the UI thread, newest request first."""

    def __init__(self, spec: DefectSpec, paths: List[Path]) -> None:
        self.spec = spec
        self.paths = paths
        self.results: "queue.Queue[tuple]" = queue.Queue()
        self._requests: "queue.LifoQueue[Optional[int]]" = queue.LifoQueue()
        self._pending: set = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def ensure(self, index: int, known: Dict[int, Trace]) -> None:
        if index < 0 or index >= len(self.paths):
            return
        if index in known or index in self._pending:
            return
        self._pending.add(index)
        self._requests.put(index)

    def forget(self, index: int) -> None:
        self._pending.discard(index)

    def stop(self) -> None:
        self._stop.set()
        self._requests.put(None)

    def _run(self) -> None:
        while not self._stop.is_set():
            index = self._requests.get()
            if index is None or self._stop.is_set():
                break
            try:
                trace = pipeline.trace(self.spec, self.paths[index])
            except Exception as error:  # keep the run alive on a bad photo
                trace = Trace(path=self.paths[index], name=self.paths[index].name,
                              spec=self.spec, verdict="ERROR",
                              details="the detector raised {}".format(error))
            # finished photos are kept for review, so shrink them off the UI thread
            if trace.annotated is not None:
                trace.annotated = pipeline.fit(trace.annotated, 220)
            for stage in trace.stages:
                stage.pack()
            self.results.put((index, trace))


class Animator:
    """One timer for every moving thing on screen."""

    def __init__(self, widget: tk.Misc) -> None:
        self.widget = widget
        self._callbacks: List = []
        self._job = None
        self._last = time.perf_counter()

    def add(self, callback) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove(self, callback) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def start(self) -> None:
        if self._job is None:
            self._last = time.perf_counter()
            self._job = self.widget.after(TICK_MS, self._tick)

    def stop(self) -> None:
        if self._job is not None:
            self.widget.after_cancel(self._job)
            self._job = None

    def _tick(self) -> None:
        now = time.perf_counter()
        delta = min(now - self._last, 0.12)
        self._last = now
        for callback in list(self._callbacks):
            try:
                callback(delta)
            except tk.TclError:
                return
        self._job = self.widget.after(TICK_MS, self._tick)


# ---------------------------------------------------------------------- shell

class Studio:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        neon.enable_hidpi()
        neon.apply_scaling(root)
        root.title(APP_TITLE.title())
        root.configure(bg=neon.VOID)
        root.minsize(neon.px(1180), neon.px(740))
        self._centre(neon.px(1420), neon.px(900))

        self.animator = Animator(root)
        self.animator.start()

        self.photos: List[Path] = pipeline.photos_in()
        self.container = tk.Frame(root, bg=neon.VOID)
        self.container.pack(fill="both", expand=True)

        self.home = HomeScreen(self, self.container)
        self.run: Optional[RunScreen] = None
        self.show_home()

        root.bind("<Escape>", lambda _e: self.show_home())

    def _centre(self, width: int, height: int) -> None:
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = max((screen_w - width) // 2, 0)
        y = max((screen_h - height) // 3, 0)
        self.root.geometry("{}x{}+{}+{}".format(width, height, x, y))

    def show_home(self) -> None:
        if self.run is not None:
            self.run.close()
            self.run = None
        self.home.place(in_=self.container, relx=0, rely=0, relwidth=1, relheight=1)
        self.home.wake()

    def start_run(self, spec: DefectSpec) -> None:
        if not self.photos:
            self.home.flash_empty()
            return
        self.home.sleep()
        self.home.place_forget()
        self.run = RunScreen(self, self.container, spec, self.photos)
        self.run.place(in_=self.container, relx=0, rely=0, relwidth=1, relheight=1)


# ----------------------------------------------------------------------- home

class HomeScreen(tk.Canvas):
    CARD_W, CARD_H, GAP = 300, 254, 30

    def __init__(self, app: Studio, parent: tk.Widget) -> None:
        super().__init__(parent, bg=neon.VOID, highlightthickness=0, bd=0)
        self.app = app
        self._backdrop = None
        self._particles = []
        self._cards: List[dict] = []
        self._hover = -1
        self._phase = 0.0
        self._empty_flash = 0.0
        self._size = (0, 0)
        self.bind("<Configure>", self._on_resize)
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", lambda _e: self._set_hover(-1))
        self.bind("<Button-1>", self._on_click)

    # -- lifecycle

    def wake(self) -> None:
        self.app.animator.add(self._tick)

    def sleep(self) -> None:
        self.app.animator.remove(self._tick)

    # -- layout

    def _on_resize(self, event) -> None:
        if (event.width, event.height) == self._size:
            return
        self._size = (event.width, event.height)
        self._rebuild()

    def _rebuild(self) -> None:
        self.delete("all")
        width, height = self._size
        if width < 40 or height < 40:
            return

        self._ground = neon.backdrop_image(width, height, neon.CYAN)
        self._backdrop = neon.backdrop(width, height, neon.CYAN)
        self.create_image(0, 0, image=self._backdrop, anchor="nw", tags="bg")

        self._seed_particles(width, height)
        for particle in self._particles:
            particle["id"] = self.create_oval(0, 0, 0, 0, outline="", fill=particle["colour"])

        cards_w = neon.px(3 * self.CARD_W + 2 * self.GAP)
        scale = min((width - neon.px(70)) / cards_w,
                    (height - neon.px(190)) / neon.px(self.CARD_H))
        scale = max(min(scale, 1.0), 0.55)
        card_w = int(neon.px(self.CARD_W) * scale)
        card_h = int(neon.px(self.CARD_H) * scale)
        gap = int(neon.px(self.GAP) * scale)
        total = 3 * card_w + 2 * gap
        left = (width - total) // 2

        head_h = int(neon.px(92) * scale)
        block = head_h + card_h
        head_y = max((height - block) // 2, neon.px(24))
        top = head_y + head_h

        size = int(30 * scale)
        room = width - neon.px(60)
        while size > 11 and tkfont.Font(font=neon.font(size, "wide")).measure(APP_TITLE) > room:
            size -= 1
        self.create_text(width // 2, head_y + int(neon.px(24) * scale), text=APP_TITLE,
                         font=neon.font(size, "wide"), fill=neon.INK, tags="head")

        self._cards = []
        for index, spec in enumerate(DEFECTS):
            x = left + index * (card_w + gap)
            self._cards.append(self._build_card(spec, index, x, top, card_w, card_h, scale))

        if not self.app.photos:  # the only case where the folder is worth mentioning
            self.create_text(width // 2, min(top + card_h + int(neon.px(38) * scale),
                                             height - neon.px(22)),
                             text="no photos in  gloves/", font=neon.font(int(9 * scale), "mono"),
                             fill=neon.RED, tags=("foot",))

    def _build_card(self, spec: DefectSpec, index: int, x: int, y: int,
                    width: int, height: int, scale: float) -> dict:
        tag = "card{}".format(index)
        body = "cardtext{}".format(index)
        card = {"spec": spec, "tag": tag, "body": body, "box": (x, y, width, height),
                "hover": 0.0, "images": {}, "scale": scale, "lift": 0,
                "ground": self._ground.crop((x, max(y - neon.px(4), 0),
                                             x + width, max(y - neon.px(4), 0) + height))}
        image = self._card_image(card, 0.0)
        card["images"][0.0] = image
        card["image_id"] = self.create_image(x, y, image=image, anchor="nw", tags=(tag, "card"))

        centre = x + width // 2
        unit = lambda value: int(neon.px(value) * scale)  # noqa: E731
        card["icon_y"] = unit(101)
        card["icon_size"] = unit(118)
        icon = neon.glyph(spec.slug, card["icon_size"], spec.accent, self._card_fill(spec, 0.0))
        card["icon"] = icon
        card["icon_id"] = self.create_image(centre, y + card["icon_y"], image=icon,
                                            anchor="center", tags=(tag, "card"))
        card["chip_id"] = self.create_text(centre, y + unit(201), text=spec.title,
                                           font=neon.font(int(15 * scale), "wide"),
                                           fill=neon.INK, tags=(tag, body, "card"))
        return card

    @staticmethod
    def _card_fill(spec: DefectSpec, strength: float) -> str:
        return neon.mix(neon.PANEL, spec.glow, 0.34 + 0.42 * strength)

    def _card_image(self, card: dict, strength: float):
        spec = card["spec"]
        return neon.glow_panel_on(card["ground"], radius=neon.px(18),
                                  fill=self._card_fill(spec, strength),
                                  border=neon.mix(neon.LINE_HI, spec.accent, 0.30 + 0.70 * strength),
                                  glow=spec.accent, strength=strength, inset=neon.px(15))

    # -- particles

    def _seed_particles(self, width: int, height: int) -> None:
        self._particles = []
        for _ in range(46):
            self._particles.append({
                "x": random.uniform(0, width),
                "y": random.uniform(0, height),
                "vx": random.uniform(-14, 14),
                "vy": random.uniform(-26, -6),
                "r": random.uniform(0.8, 2.4),
                "colour": neon.fade(random.choice((neon.CYAN, neon.VIOLET, neon.INK)),
                                    random.uniform(0.10, 0.34), neon.VOID),
                "id": None,
            })

    # -- animation

    def _tick(self, delta: float) -> None:
        width, height = self._size
        if width < 40:
            return
        self._phase += delta
        for particle in self._particles:
            particle["x"] += particle["vx"] * delta
            particle["y"] += particle["vy"] * delta
            if particle["y"] < -6:
                particle["y"] = height + 6
                particle["x"] = random.uniform(0, width)
            if particle["x"] < -6:
                particle["x"] = width + 6
            elif particle["x"] > width + 6:
                particle["x"] = -6
            r = particle["r"]
            self.coords(particle["id"], particle["x"] - r, particle["y"] - r,
                        particle["x"] + r, particle["y"] + r)

        for index, card in enumerate(self._cards):
            target = 1.0 if index == self._hover else 0.0
            current = card["hover"]
            if abs(current - target) < 0.01:
                card["hover"] = target
            else:
                card["hover"] = current + (target - current) * min(1.0, delta * 9.0)
            self._paint_card(card)

    def _paint_card(self, card: dict) -> None:
        x, y, width, height = card["box"]
        level = round(card["hover"] * 4) / 4.0
        image = card["images"].get(level)
        if image is None:
            image = self._card_image(card, level)
            card["images"][level] = image
        self.itemconfigure(card["image_id"], image=image)

        lift = int(card["hover"] * neon.px(7))
        self.coords(card["image_id"], x, y - lift)
        breathe = math.sin(self._phase * 3.1) * neon.px(2) * card["hover"]
        icon = neon.glyph(card["spec"].slug, card["icon_size"], card["spec"].accent,
                          self._card_fill(card["spec"], level))
        card["icon"] = icon
        self.itemconfigure(card["icon_id"], image=icon)
        self.coords(card["icon_id"], x + width // 2, y + card["icon_y"] - lift + breathe)
        if lift != card["lift"]:
            self.move(card["body"], 0, card["lift"] - lift)
            card["lift"] = lift
        self.itemconfigure(card["chip_id"],
                           fill=neon.mix(neon.INK, card["spec"].accent, card["hover"]))

    # -- input

    def _card_at(self, x: int, y: int) -> int:
        for index, card in enumerate(self._cards):
            cx, cy, width, height = card["box"]
            if cx <= x <= cx + width and cy - 10 <= y <= cy + height:
                return index
        return -1

    def _on_motion(self, event) -> None:
        self._set_hover(self._card_at(event.x, event.y))

    def _set_hover(self, index: int) -> None:
        if index == self._hover:
            return
        self._hover = index
        self.configure(cursor="hand2" if index >= 0 else "")

    def _on_click(self, event) -> None:
        index = self._card_at(event.x, event.y)
        if index >= 0:
            self.app.start_run(self._cards[index]["spec"])

    def flash_empty(self) -> None:
        self.itemconfigure("foot", fill=neon.RED)


# ------------------------------------------------------------------ run screen

class RunScreen(tk.Frame):
    """Left plays the photo being processed live, the folder list keeps every
    finished photo, and picking one of those opens it for reading without running
    the detector again."""

    SIDE_W = 348
    TOP_H = 76
    TIMELINE_H = 96
    BOTTOM_H = 104

    def __init__(self, app: Studio, parent: tk.Widget, spec: DefectSpec, paths: List[Path]) -> None:
        super().__init__(parent, bg=neon.BASE)
        self.app = app
        self.spec = spec
        self.paths = paths
        self.accent = spec.accent

        self.traces: Dict[int, Trace] = {}
        self.summary: Dict[int, dict] = {}
        self.live = 0              # the photo the run is working through
        self.index = 0             # the photo on screen
        self.stage_i = -1          # the frame on screen
        self.reviewing = False     # True while reading a finished photo
        self.complete = False
        self.playing = True
        self.speed = PLAYBACK_SPEED
        self._live_stage = -1      # where the live photo was left when review began
        self._dwell_left = 0.0
        self._elapsed = 0.0
        self._pulse = 0.0
        self._log: List[dict] = []
        self._reveal = 1.0
        self._reveal_from: Optional[Image.Image] = None
        self._reveal_to: Optional[Image.Image] = None
        self._labels = ("", "", "", "")
        self._labels_prev = None
        self._frame_photo = None
        self._current_pil: Optional[Image.Image] = None
        self._thumbs: Dict[int, tuple] = {}
        self._thumb_cache: Dict[int, np.ndarray] = {}
        self._source_cache: Dict[int, np.ndarray] = {}
        self._film_offset = 0.0
        self._film_follow = True
        self._film_dirty = True
        self._top_stamp = -1
        self._hover_button = ""
        self._bar_value = 0.0

        self.worker = TraceWorker(spec, paths)
        self._build()
        self.worker.ensure(0, self.traces)
        self.worker.ensure(1, self.traces)
        self._log_line("studio", "loaded {} photo(s) from gloves/".format(len(paths)))
        self._log_line("studio", "detector {}".format(spec.module))
        self.app.animator.add(self._tick)
        self._poll = self.after(60, self._drain)

    # -- construction

    def _build(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.top = tk.Canvas(self, bg=neon.BASE, highlightthickness=0, bd=0,
                             height=neon.px(self.TOP_H))
        self.top.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.top.bind("<Configure>", lambda _e: self._paint_top())
        self.top.bind("<Button-1>", self._on_top_click)
        self.top.bind("<Motion>", self._on_top_motion)

        self.viewport = tk.Canvas(self, bg=neon.PANEL_SOFT, highlightthickness=0, bd=0)
        self.viewport.grid(row=1, column=0, sticky="nsew",
                           padx=(neon.px(18), neon.px(10)), pady=(0, neon.px(8)))
        self.viewport.bind("<Configure>", lambda _e: self._on_viewport_resize())

        self.timeline = tk.Canvas(self, bg=neon.BASE, highlightthickness=0, bd=0,
                                  height=neon.px(self.TIMELINE_H))
        self.timeline.grid(row=2, column=0, sticky="ew", padx=(neon.px(18), neon.px(10)))
        self.timeline.bind("<Configure>", lambda _e: self._paint_timeline())
        self.timeline.bind("<Button-1>", self._on_timeline_click)

        side = tk.Frame(self, bg=neon.BASE, width=neon.px(self.SIDE_W))
        side.grid(row=1, column=1, rowspan=2, sticky="nsew", padx=(0, neon.px(18)))
        side.grid_propagate(False)
        side.grid_rowconfigure(0, weight=3)
        side.grid_rowconfigure(1, weight=2)
        side.grid_columnconfigure(0, weight=1)

        self.film = tk.Canvas(side, bg=neon.BASE, highlightthickness=0, bd=0)
        self.film.grid(row=0, column=0, sticky="nsew", pady=(0, neon.px(10)))
        self.film.bind("<Configure>", lambda _e: self._paint_film())
        self.film.bind("<Button-1>", self._on_film_click)
        self.film.bind("<Motion>", self._on_film_motion)
        self.film.bind("<MouseWheel>", self._on_film_wheel)

        self.telemetry = tk.Canvas(side, bg=neon.BASE, highlightthickness=0, bd=0)
        self.telemetry.grid(row=1, column=0, sticky="nsew", pady=(0, neon.px(14)))
        self.telemetry.bind("<Configure>", lambda _e: self._paint_log())

        self.bottom = tk.Canvas(self, bg=neon.BASE, highlightthickness=0, bd=0,
                                height=neon.px(self.BOTTOM_H))
        self.bottom.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.bottom.bind("<Configure>", lambda _e: self._paint_bottom())
        self.bottom.bind("<Button-1>", self._on_bottom_click)
        self.bottom.bind("<Motion>", self._on_bottom_motion)

        self.bind_all("<space>", self._toggle_play_event)
        self.bind_all("<Right>", lambda _e: self._step(1))
        self.bind_all("<Left>", lambda _e: self._step(-1))

    def close(self) -> None:
        self.app.animator.remove(self._tick)
        try:
            self.after_cancel(self._poll)
        except Exception:
            pass
        for sequence in ("<space>", "<Right>", "<Left>"):
            try:
                self.unbind_all(sequence)
            except Exception:
                pass
        self.worker.stop()
        self.destroy()

    # -- data flow

    def _drain(self) -> None:
        changed = False
        while True:
            try:
                index, trace = self.worker.results.get_nowait()
            except queue.Empty:
                break
            self.worker.forget(index)
            self.traces[index] = trace
            if trace.annotated is not None:
                preview = trace.annotated
            elif trace.stages:
                preview = pipeline.fit(trace.stages[0].frame(), 220)
            else:
                preview = np.zeros((8, 8, 3), np.uint8)
            self._thumb_cache[index] = preview
            self._thumbs.pop(index, None)
            changed = True
            if index == self.live and not self.reviewing and self.stage_i < 0:
                self._begin_trace()
        if changed:
            self._film_dirty = True
        self._poll = self.after(60, self._drain)

    def _begin_trace(self) -> None:
        """Start playing the live photo from its first frame."""
        trace = self.traces.get(self.live)
        if trace is None or not trace.stages:
            return
        self.index = self.live
        self.stage_i = 0
        self._dwell_left = STAGE_DWELL_MS / 1000.0
        self._log_line("photo", trace.name, accent=True)
        self.worker.ensure(self.live + 1, self.traces)
        self._show_stage(trace.stages[0])
        self._paint_timeline()
        self._paint_bottom()

    def _advance(self) -> None:
        trace = self.traces.get(self.live)
        if trace is None or not trace.stages:
            return
        if self.stage_i + 1 < len(trace.stages):
            self.stage_i += 1
            self._show_stage(trace.stages[self.stage_i])
            last = self.stage_i == len(trace.stages) - 1
            self._dwell_left = (VERDICT_DWELL_MS if last else STAGE_DWELL_MS) / 1000.0
            if last:
                self._finish_trace(trace)
            self._paint_timeline()
            self._paint_bottom()
        else:
            self._next_photo()

    def _finish_trace(self, trace: Trace) -> None:
        self.summary[self.live] = {"verdict": trace.verdict, "found": trace.found,
                                   "expected": trace.expected, "correct": trace.correct,
                                   "score": trace.score, "elapsed": trace.elapsed}
        self._log_line("verdict", "{}   {}".format(trace.verdict, trace.details),
                       colour=neon.RED if trace.found else neon.GREEN)
        self._film_dirty = True

    def _next_photo(self) -> None:
        if self.live + 1 >= len(self.paths):
            self.complete = True
            self.playing = False
            self._log_line("studio", "run complete, {} photo(s) inspected".format(len(self.paths)))
            self._paint_top()
            self._paint_bottom()
            self._film_dirty = True
            return
        self.live += 1
        self.index = self.live
        self.stage_i = -1
        self._reveal = 1.0
        self._film_follow = True
        self.worker.ensure(self.live, self.traces)
        self.worker.ensure(self.live + 1, self.traces)
        trace = self.traces.get(self.live)
        if trace is not None and trace.stages:
            self._begin_trace()
        else:
            self._show_waiting()
        self._paint_all()

    def _review(self, index: int) -> None:
        """Open a finished photo on its verdict. Nothing runs again."""
        trace = self.traces.get(index)
        if index not in self.summary or trace is None or not trace.stages:
            return
        if not self.reviewing and not self.complete:
            self._live_stage = self.stage_i
        self.reviewing = True
        self.index = index
        self.stage_i = len(trace.stages) - 1
        self._film_follow = True
        self._show_stage(trace.stages[self.stage_i])
        self._paint_all()

    def _go_live(self) -> None:
        """Back to the photo the run is working through, where it was left."""
        if self.complete:
            return
        self.reviewing = False
        self.playing = True
        self.index = self.live
        self._film_follow = True
        trace = self.traces.get(self.live)
        self.stage_i = self._live_stage
        if trace is not None and trace.stages and 0 <= self.stage_i < len(trace.stages):
            self._show_stage(trace.stages[self.stage_i])
        else:
            self.stage_i = -1
            self._show_waiting()
        self._paint_all()

    def _step(self, delta: int) -> None:
        """Walk through the finished photos, ending on the live one."""
        stops = sorted(self.summary)
        if not self.complete and self.live not in stops:
            stops.append(self.live)
        if not stops:
            return
        here = self.index if self.index in stops else self.live
        position = stops.index(here) if here in stops else len(stops) - 1
        target = stops[max(0, min(position + delta, len(stops) - 1))]
        if target == self.index and self.reviewing == (target != self.live or self.complete):
            return
        if target == self.live and not self.complete:
            self._go_live()
        else:
            self._review(target)

    def _paint_all(self) -> None:
        self._paint_top()
        self._paint_timeline()
        self._paint_bottom()
        self._film_dirty = True

    # -- viewport

    def _viewport_box(self):
        width = max(self.viewport.winfo_width(), 10)
        height = max(self.viewport.winfo_height(), 10)
        return width, height

    def _on_viewport_resize(self) -> None:
        trace = self.traces.get(self.index)
        if trace is not None and trace.stages and 0 <= self.stage_i < len(trace.stages):
            self._show_stage(trace.stages[self.stage_i], animate=False)
        else:
            self._show_waiting()

    def _source_image(self, index: int) -> Optional[np.ndarray]:
        cached = self._source_cache.get(index)
        if cached is None:
            cached = pipeline.imread_preview(self.paths[index])
            if cached is None:
                return None
            self._source_cache[index] = cached
            if len(self._source_cache) > 6:
                for stale in [k for k in self._source_cache if abs(k - index) > 3]:
                    self._source_cache.pop(stale, None)
        return cached

    def _show_waiting(self) -> None:
        width, height = self._viewport_box()
        pad, top, gap, pane_w, pane_h = self._pane_geometry()
        canvas = Image.new("RGB", (width, height), neon.PANEL_SOFT)
        source = self._source_image(self.index)
        if source is not None:
            canvas.paste(neon.letterbox(source, pane_w, pane_h, neon.PANEL_SOFT), (pad, top))
        self._current_pil = canvas
        self._reveal = 1.0
        self._set_labels("QUEUED", "Analysing " + self.paths[self.index].name,
                         "the detector is working on this photo, the frames appear here as "
                         "they are produced", "")
        self._paint_viewport()

    def _pane_geometry(self):
        """Source on the left, current stage on the right, both letterboxed."""
        width, height = self._viewport_box()
        pad = neon.px(16)
        top = neon.px(50)
        caption = neon.px(80)
        gap = neon.px(14)
        pane_w = max((width - 2 * pad - gap) // 2, 40)
        pane_h = max(height - top - caption, 40)
        return pad, top, gap, pane_w, pane_h

    def _show_stage(self, stage: Stage, animate: bool = True) -> None:
        width, height = self._viewport_box()
        pad, top, gap, pane_w, pane_h = self._pane_geometry()
        trace = self.traces.get(self.index)
        frame = stage.frame()
        source = trace.stages[0].frame() if trace is not None and trace.stages else frame

        target = Image.new("RGB", (width, height), neon.PANEL_SOFT)
        target.paste(neon.letterbox(source, pane_w, pane_h, neon.PANEL_SOFT), (pad, top))
        target.paste(neon.letterbox(frame, pane_w, pane_h, neon.PANEL_SOFT),
                     (pad + pane_w + gap, top))
        if self._reveal < 1.0 and self._reveal_to is not None:
            # a faster speed can outrun the sweep, so land the previous one first
            self._current_pil = self._reveal_to
            self._reveal_to = None
            self._reveal = 1.0
        if animate and self._current_pil is not None and self._current_pil.size == target.size:
            self._reveal_from = self._current_pil
            self._reveal_to = target
            self._reveal = 0.22
        else:
            self._reveal = 1.0
            self._current_pil = target
        total = len(trace.stages) if trace is not None and trace.stages else 0
        counter = "STAGE {:02d} / {:02d}".format(self.stage_i + 1, total) if total else ""
        if counter and self.reviewing:
            counter = "REVIEW   " + counter
        self._set_labels(pipeline.PHASE_LABELS.get(stage.phase, stage.phase), stage.title,
                         stage.caption, counter)
        self._paint_viewport()

    def _set_labels(self, phase: str, title: str, caption: str, counter: str) -> None:
        """The caption follows the sweep, so it never describes a frame that is
        still mostly the previous one."""
        self._labels_prev = self._labels
        self._labels = (phase, title, caption, counter)

    def _paint_viewport(self) -> None:
        canvas = self.viewport
        canvas.delete("all")
        width, height = self._viewport_box()
        phase, title, caption, counter = (
            self._labels_prev if self._reveal < 0.45 and self._labels_prev else self._labels)
        source = self._current_pil
        if self._reveal < 1.0 and self._reveal_to is not None and self._reveal_from is not None:
            source = self._reveal_from.copy()
            cut = int(self._reveal_to.height * neon.ease_out(self._reveal))
            if cut > 0:
                source.paste(self._reveal_to.crop((0, 0, self._reveal_to.width, cut)), (0, 0))
                draw = ImageDraw.Draw(source)
                draw.line([(0, cut - 1), (source.width, cut - 1)], fill=self.accent, width=2)
        if source is None:
            return
        self._frame_photo = ImageTk.PhotoImage(source)
        canvas.create_image(0, 0, image=self._frame_photo, anchor="nw")

        # corner brackets in the accent while live, plain grey while reading
        bracket = self.accent if not self.reviewing else neon.INK_FAINT
        arm = neon.px(26)
        for x, y, dx, dy in ((2, 2, 1, 1), (width - 3, 2, -1, 1),
                             (2, height - 3, 1, -1), (width - 3, height - 3, -1, -1)):
            canvas.create_line(x, y, x + dx * arm, y, fill=bracket, width=2)
            canvas.create_line(x, y, x, y + dy * arm, fill=bracket, width=2)

        pad, top, gap, pane_w, pane_h = self._pane_geometry()
        right_x = pad + pane_w + gap
        label_y = top - neon.px(16)
        canvas.create_text(pad, label_y, text="SOURCE", anchor="w",
                           font=neon.font(9, "mono"), fill=neon.INK_FAINT)
        canvas.create_text(pad + pane_w, label_y, text=self.paths[self.index].name, anchor="e",
                           font=neon.font(9, "mono"), fill=neon.INK_FAINT)

        badge_w = tkfont.Font(font=neon.font(9, "mono")).measure(phase) + neon.px(18)
        canvas.create_rectangle(right_x, label_y - neon.px(9), right_x + badge_w,
                                label_y + neon.px(9),
                                fill=neon.mix(neon.PANEL_SOFT, self.accent, 0.20),
                                outline=self.accent)
        canvas.create_text(right_x + neon.px(9), label_y, text=phase, anchor="w",
                           font=neon.font(9, "mono"), fill=self.accent)

        if counter:
            canvas.create_text(right_x + pane_w, label_y, text=counter,
                               anchor="e", font=neon.font(9, "mono"), fill=neon.INK_SOFT)
        else:
            cx, cy = right_x + pane_w // 2, top + pane_h // 2
            for slot in range(3):
                beat = 0.5 + 0.5 * math.sin(self._pulse * 4.0 - slot * 0.8)
                radius = neon.px(4) + neon.px(3) * beat
                offset = (slot - 1) * neon.px(22)
                canvas.create_oval(cx + offset - radius, cy - radius,
                                   cx + offset + radius, cy + radius,
                                   fill=neon.mix(neon.PANEL_SOFT, self.accent, 0.25 + 0.6 * beat),
                                   outline="")

        base = top + pane_h + neon.px(26)
        canvas.create_text(pad, base, text=title, anchor="w",
                           font=neon.font(15, "wide"), fill=neon.INK)
        canvas.create_text(pad, base + neon.px(22), text=caption, anchor="nw",
                           width=max(width - 2 * pad, 60), font=neon.font(9),
                           fill=neon.INK_SOFT)

    # -- top bar

    def _paint_top(self) -> None:
        canvas = self.top
        canvas.delete("all")
        width = max(canvas.winfo_width(), 10)
        height = neon.px(self.TOP_H)
        left = neon.px(18)

        canvas.create_text(left, height // 2 - neon.px(4), text="◂  BACK", anchor="w",
                           font=neon.font(10, "wide"), fill=neon.INK_SOFT, tags="back")
        canvas.create_text(left, height // 2 + neon.px(14), text="Esc", anchor="w",
                           font=neon.font(8, "mono"), fill=neon.INK_FAINT)

        x = left + neon.px(110)
        canvas.create_oval(x, height // 2 - neon.px(15), x + neon.px(9), height // 2 - neon.px(6),
                           fill=self.accent, outline="")
        canvas.create_text(x + neon.px(20), height // 2 - neon.px(11), text=self.spec.title,
                           anchor="w", font=neon.font(15, "wide"), fill=neon.INK)
        canvas.create_text(x + neon.px(20), height // 2 + neon.px(12),
                           text="detectors/{}.py".format(self.spec.module), anchor="w",
                           font=neon.font(8, "mono"), fill=neon.INK_FAINT)

        done = len(self.summary)
        matched = sum(1 for entry in self.summary.values() if entry["correct"])
        right = width - neon.px(18)
        canvas.create_text(right, height // 2 - neon.px(11),
                           text="PHOTO {:02d} / {:02d}".format(self.index + 1, len(self.paths)),
                           anchor="e", font=neon.font(11, "wide"), fill=neon.INK)
        canvas.create_text(right, height // 2 + neon.px(11),
                           text="{}  ·  filename agrees on {} of {}".format(
                               self._clock(), matched, done) if done else self._clock(),
                           anchor="e", font=neon.font(8, "mono"), fill=neon.INK_FAINT)

        bar_x0 = x + neon.px(20)
        bar_x1 = right - neon.px(180)
        if bar_x1 > bar_x0 + neon.px(40):
            y = height - neon.px(9)
            canvas.create_line(bar_x0, y, bar_x1, y, fill=neon.LINE, width=3)
            span = (bar_x1 - bar_x0) * self._bar_value
            if span > 1:
                canvas.create_line(bar_x0, y, bar_x0 + span, y, fill=self.accent, width=3)
                canvas.create_oval(bar_x0 + span - 3, y - 3, bar_x0 + span + 3, y + 3,
                                   fill=self.accent, outline="")
        canvas.create_line(0, height - 1, width, height - 1, fill=neon.LINE)

    def _clock(self) -> str:
        return "{:02d}:{:02d}".format(int(self._elapsed) // 60, int(self._elapsed) % 60)

    def _on_top_click(self, event) -> None:
        if event.x < neon.px(100) and abs(event.y - neon.px(self.TOP_H) // 2) < neon.px(24):
            self.app.show_home()

    def _on_top_motion(self, event) -> None:
        over = event.x < neon.px(100) and abs(event.y - neon.px(self.TOP_H) // 2) < neon.px(24)
        self.top.configure(cursor="hand2" if over else "")
        self.top.itemconfigure("back", fill=neon.INK if over else neon.INK_SOFT)

    # -- timeline

    def _paint_timeline(self) -> None:
        canvas = self.timeline
        canvas.delete("all")
        width = max(canvas.winfo_width(), 10)
        trace = self.traces.get(self.index)
        canvas.create_text(0, neon.px(10), text="PIPELINE", anchor="nw",
                           font=neon.font(9, "mono"), fill=neon.INK_FAINT)
        if trace is None or not trace.stages or self.stage_i < 0:
            canvas.create_text(0, neon.px(40), text="waiting for the first frames of this photo",
                               anchor="nw", font=neon.font(9), fill=neon.INK_FAINT)
            return

        stages = trace.stages
        finished = self.index in self.summary
        top = neon.px(34)
        gap = neon.px(4)
        chip_w = max((width - gap * (len(stages) - 1)) / len(stages), neon.px(8))
        phase_runs = []
        for index, stage in enumerate(stages):
            x = index * (chip_w + gap)
            active = index == self.stage_i
            if active:
                colour = self.accent
            elif finished or index < self.stage_i:
                colour = neon.mix(neon.LINE_HI, self.accent, 0.55)
            else:
                colour = neon.LINE
            canvas.create_rectangle(x, top, x + chip_w, top + neon.px(9),
                                    fill=colour, outline="", tags=("chip", "chip{}".format(index)))
            if active:
                canvas.create_rectangle(x, top - neon.px(4), x + chip_w, top + neon.px(13),
                                        outline=self.accent, width=1)
            if not phase_runs or phase_runs[-1][0] != stage.phase:
                phase_runs.append([stage.phase, x, x + chip_w, index])
            else:
                phase_runs[-1][2] = x + chip_w

        label_y = top + neon.px(24)
        for phase, x0, _x1, first in phase_runs:
            reached = finished or self.stage_i >= first
            canvas.create_line(x0, label_y - neon.px(6), x0, label_y + neon.px(18),
                               fill=neon.LINE_HI)
            canvas.create_text(x0 + neon.px(7), label_y + neon.px(2),
                               text=pipeline.PHASE_LABELS.get(phase, phase), anchor="w",
                               font=neon.font(8, "mono"),
                               fill=self.accent if reached else neon.INK_FAINT)
        if 0 <= self.stage_i < len(stages):
            canvas.create_text(width, neon.px(10),
                               text=stages[self.stage_i].key, anchor="ne",
                               font=neon.font(8, "mono"), fill=neon.INK_FAINT)

    def _on_timeline_click(self, event) -> None:
        trace = self.traces.get(self.index)
        if trace is None or not trace.stages or self.stage_i < 0:
            return
        width = max(self.timeline.winfo_width(), 10)
        index = int(event.x / max(width, 1) * len(trace.stages))
        index = max(0, min(index, len(trace.stages) - 1))
        if not self.reviewing and self.index not in self.summary:
            # scrubbing the live photo holds it there until play is pressed
            index = min(index, self.stage_i)
            self.playing = False
        self.stage_i = index
        self._show_stage(trace.stages[index])
        self._paint_timeline()
        self._paint_bottom()

    # -- folder list

    def _film_metrics(self):
        return neon.px(56), neon.px(68)

    def _row_at(self, y: int) -> int:
        _side, row_h = self._film_metrics()
        index = int((y - neon.px(26) + self._film_offset) // row_h)
        return index if 0 <= index < len(self.paths) else -1

    def _row_state(self, index: int) -> str:
        if index in self.summary:
            return "done"
        if index == self.live and not self.complete:
            return "live"
        return "queued"

    def _tile(self, index: int, path: Path, side: int, current: bool, lit: bool):
        """Folder thumbnails are rebuilt only when their state changes."""
        state = (side, current, lit, index in self.traces)
        cached = self._thumbs.get(index)
        if cached is not None and cached[0] == state:
            return cached[1]
        image = self._thumb_cache.get(index)
        if image is None:
            raw = pipeline.imread_preview(path)
            image = pipeline.fit(raw, 220) if raw is not None else np.zeros((8, 8, 3), np.uint8)
            self._thumb_cache[index] = image
        tile = neon.thumb(image, side, 8,
                          ring_colour=self.accent if current else None,
                          dim=1.0 if lit else 0.45)
        self._thumbs[index] = (state, tile)
        return tile

    def _paint_film(self) -> None:
        canvas = self.film
        canvas.delete("all")
        width = max(canvas.winfo_width(), 10)
        height = max(canvas.winfo_height(), 10)
        side, row_h = self._film_metrics()
        top = neon.px(26)
        visible = height - top
        needed = row_h * len(self.paths)

        for index, path in enumerate(self.paths):
            y = top + index * row_h - self._film_offset
            if y < top - row_h or y > height:
                continue
            current = index == self.index
            state = self._row_state(index)
            tile = self._tile(index, path, side, current, state != "queued")
            canvas.create_image(0, y, image=tile, anchor="nw")

            text_x = side + neon.px(12)
            canvas.create_text(text_x, y + neon.px(14), text=path.name, anchor="w",
                               font=neon.font(9, "strong" if current else "regular"),
                               fill=neon.INK if state != "queued" else neon.INK_FAINT)
            if state == "done":
                entry = self.summary[index]
                colour = neon.RED if entry["found"] else neon.GREEN
                canvas.create_oval(text_x, y + neon.px(30), text_x + neon.px(8), y + neon.px(38),
                                   fill=colour, outline="")
                label = "{}   {}".format(entry["verdict"],
                                         "as named" if entry["correct"] else "differs from the name")
                canvas.create_text(text_x + neon.px(15), y + neon.px(34), text=label, anchor="w",
                                   font=neon.font(8, "mono"), fill=colour)
            elif state == "live":
                paused = self.reviewing or not self.playing
                canvas.create_text(text_x, y + neon.px(34),
                                   text="paused, click to resume" if paused else "processing",
                                   anchor="w", font=neon.font(8, "mono"), fill=self.accent)
            else:
                canvas.create_text(text_x, y + neon.px(34), text="queued", anchor="w",
                                   font=neon.font(8, "mono"), fill=neon.INK_FAINT)

        if needed > visible:
            track_h = visible * visible / needed
            track_y = top + (self._film_offset / max(needed - visible, 1)) * (visible - track_h)
            canvas.create_rectangle(width - 3, track_y, width - 1, track_y + track_h,
                                    fill=neon.LINE_HI, outline="")

        # drawn last so a scrolled row never runs underneath the heading
        canvas.create_rectangle(0, 0, width, top - neon.px(4), fill=neon.BASE, outline="")
        canvas.create_text(0, neon.px(4), text="FOLDER", anchor="nw",
                           font=neon.font(9, "mono"), fill=neon.INK_FAINT)
        canvas.create_text(width, neon.px(4),
                           text="{} of {} done".format(len(self.summary), len(self.paths)),
                           anchor="ne", font=neon.font(9, "mono"), fill=neon.INK_FAINT)

    def _on_film_click(self, event) -> None:
        index = self._row_at(event.y)
        state = self._row_state(index) if index >= 0 else ""
        if state == "done":
            self._review(index)
        elif state == "live":
            self._go_live()

    def _on_film_motion(self, event) -> None:
        index = self._row_at(event.y)
        clickable = index >= 0 and self._row_state(index) != "queued"
        self.film.configure(cursor="hand2" if clickable else "")

    def _on_film_wheel(self, event) -> None:
        _side, row_h = self._film_metrics()
        height = max(self.film.winfo_height(), 10) - neon.px(26)
        needed = row_h * len(self.paths)
        self._film_follow = False
        self._film_offset = max(0.0, min(self._film_offset - event.delta / 3.0,
                                         max(needed - height, 0.0)))
        self._paint_film()

    # -- telemetry

    def _log_line(self, tag: str, text: str, colour: Optional[str] = None,
                  accent: bool = False) -> None:
        self._log.append({"tag": tag, "text": text, "age": 0.0,
                          "colour": colour or (self.accent if accent else neon.INK_SOFT)})
        del self._log[:-160]
        self._paint_log()

    def _paint_log(self) -> None:
        canvas = self.telemetry
        canvas.delete("all")
        width = max(canvas.winfo_width(), 10)
        height = max(canvas.winfo_height(), 10)
        canvas.create_text(0, neon.px(4), text="TELEMETRY", anchor="nw",
                           font=neon.font(9, "mono"), fill=neon.INK_FAINT)
        line_h = neon.px(15)
        top = neon.px(26)
        text_x = neon.px(52)
        rows = max(int((height - top) / line_h), 1)
        measure = tkfont.Font(font=neon.font(8, "mono"))
        room = max(width - text_x, 40)
        y = height - line_h
        for entry in reversed(self._log[-rows:]):
            fresh = min(entry["age"] / 0.35, 1.0)
            colour = neon.mix(neon.BASE, entry["colour"], 0.35 + 0.65 * fresh)
            canvas.create_text(0, y, text="{:>7}".format(entry["tag"]), anchor="w",
                               font=neon.font(8, "mono"), fill=neon.INK_FAINT)
            canvas.create_text(text_x, y, text=_clip(entry["text"], measure, room), anchor="w",
                               font=neon.font(8, "mono"), fill=colour)
            y -= line_h
            if y < top:
                break

    # -- bottom bar

    def _play_label(self) -> str:
        if self.complete:
            return "■"
        if self.reviewing or not self.playing:
            return "▸"
        return "❚❚"

    def _buttons(self):
        width = max(self.bottom.winfo_width(), 10)
        height = neon.px(self.BOTTOM_H)
        y = height // 2
        w, h = neon.px(46), neon.px(34)
        gap = neon.px(8)
        specs = [("prev", "◂◂"), ("play", self._play_label()), ("next", "▸▸"),
                 ("save", "SAVE")]
        widths = [w, w, w, neon.px(70)]
        total = sum(widths) + gap * (len(specs) - 1)
        x = width - neon.px(18) - total
        boxes = []
        for (key, label), bw in zip(specs, widths):
            boxes.append((key, label, x, y - h // 2, bw, h))
            x += bw + gap
        return boxes

    def _paint_bottom(self, hover: str = "") -> None:
        canvas = self.bottom
        canvas.delete("all")
        width = max(canvas.winfo_width(), 10)
        height = neon.px(self.BOTTOM_H)
        canvas.create_line(0, 0, width, 0, fill=neon.LINE)

        # the verdict of a live photo only shows once its last frame is up
        entry = self.summary.get(self.index)
        trace = self.traces.get(self.index)
        left = neon.px(18)
        if entry:
            found = entry["found"]
            colour = neon.RED if found else neon.GREEN
            label = "DEFECT" if found else "PASS"
            glow = 0.25 + 0.20 * (0.5 + 0.5 * math.sin(self._pulse * 3.4)) if found else 0.22
            pill_w = neon.px(112)
            canvas.create_rectangle(left, height // 2 - neon.px(19), left + pill_w,
                                    height // 2 + neon.px(19),
                                    fill=neon.mix(neon.BASE, colour, glow), outline=colour)
            canvas.create_text(left + pill_w // 2, height // 2, text=label,
                               font=neon.font(14, "wide"), fill=colour)
            detail_x = left + pill_w + neon.px(16)
            canvas.create_text(detail_x, height // 2 - neon.px(11),
                               text=(trace.details if trace is not None else ""), anchor="w",
                               width=max(width - detail_x - neon.px(320), 80),
                               font=neon.font(9), fill=neon.INK)
            canvas.create_text(detail_x, height // 2 + neon.px(15),
                               text="score {:.2f}   ·   {:.2f}s   ·   filename says {}".format(
                                   entry["score"], entry["elapsed"],
                                   "defect" if entry["expected"] else "clean"),
                               anchor="w", font=neon.font(8, "mono"), fill=neon.INK_FAINT)
        else:
            canvas.create_text(left, height // 2, text="running the pipeline…", anchor="w",
                               font=neon.font(11), fill=neon.INK_SOFT)

        for key, label, x, y, w, h in self._buttons():
            active = key == hover
            disabled = key == "play" and self.complete
            fill = neon.mix(neon.PANEL, self.accent, 0.30 if active and not disabled else 0.10)
            canvas.create_rectangle(x, y, x + w, y + h, fill=fill,
                                    outline=self.accent if active and not disabled else neon.LINE_HI)
            canvas.create_text(x + w // 2, y + h // 2, text=label,
                               font=neon.font(10, "wide" if key == "save" else "regular"),
                               fill=neon.INK_FAINT if disabled else (
                                   neon.INK if active else neon.INK_SOFT))

    def _button_at(self, x: int, y: int) -> str:
        for key, _label, bx, by, bw, bh in self._buttons():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return key
        return ""

    def _on_bottom_motion(self, event) -> None:
        key = self._button_at(event.x, event.y)
        self.bottom.configure(cursor="hand2" if key else "")
        self._hover_button = key
        self._paint_bottom(key)

    def _on_bottom_click(self, event) -> None:
        key = self._button_at(event.x, event.y)
        if key == "prev":
            self._step(-1)
        elif key == "next":
            self._step(1)
        elif key == "play":
            self._toggle_play()
        elif key == "save":
            self._save_frame()

    def _toggle_play_event(self, _event) -> None:
        self._toggle_play()

    def _toggle_play(self) -> None:
        if self.complete:
            return
        if self.reviewing:
            self._go_live()
            return
        self.playing = not self.playing
        if self.playing and self.stage_i < 0:
            self._begin_trace()
        self._film_dirty = True
        self._paint_bottom()

    def _save_frame(self) -> None:
        trace = self.traces.get(self.index)
        if trace is None or not trace.stages or self.stage_i < 0:
            return
        stage = trace.stages[self.stage_i]
        folder = pipeline.OUTPUT_DIR / self.spec.slug
        target = folder / "{}_{:02d}_{}.png".format(Path(trace.name).stem, self.stage_i + 1, stage.key)
        if pipeline.imwrite_unicode(target, stage.frame()):
            self._log_line("saved", str(target.relative_to(pipeline.PROJECT_ROOT)), accent=True)
        else:
            self._log_line("saved", "could not write {}".format(target), colour=neon.RED)

    # -- the clock

    def _tick(self, delta: float) -> None:
        self._elapsed += delta
        self._pulse += delta

        if self._reveal < 1.0:
            self._reveal = min(1.0, self._reveal + delta * 3.2 * self.speed)
            if self._reveal >= 1.0 and self._reveal_to is not None:
                self._current_pil = self._reveal_to
                self._reveal_to = None
            self._paint_viewport()
        elif self.stage_i < 0 and int(self._elapsed * 12) % 2 == 0:
            self._paint_viewport()  # keeps the waiting dots moving

        # progress follows the run, not whatever photo is being read
        live_trace = self.traces.get(self.live)
        live_stage = self._live_stage if self.reviewing else self.stage_i
        total = len(live_trace.stages) if live_trace is not None and live_trace.stages else 1
        within = (live_stage + 1) / total if live_stage >= 0 else 0.0
        target = 1.0 if self.complete else (self.live + within) / max(len(self.paths), 1)
        self._bar_value += (target - self._bar_value) * min(1.0, delta * 4.0)

        if self.playing and not self.reviewing and not self.complete:
            if self.stage_i < 0:
                if live_trace is not None and live_trace.stages:
                    self._begin_trace()
            else:
                self._dwell_left -= delta * self.speed
                if self._dwell_left <= 0:
                    self._advance()

        now = int(self._elapsed * 2)
        if now != self._top_stamp:
            self._top_stamp = now
            self._paint_top()

        for entry in self._log:
            if entry["age"] < 0.4:
                entry["age"] += delta
                self._paint_log()
                break

        if self.summary.get(self.index, {}).get("found"):
            self._paint_bottom(self._hover_button)

        if self._film_follow:
            _side, row_h = self._film_metrics()
            visible = max(self.film.winfo_height() - neon.px(26), row_h)
            needed = row_h * len(self.paths)
            goal = max(0.0, min((self.index + 0.5) * row_h - visible / 2,
                                max(needed - visible, 0.0)))
            if abs(goal - self._film_offset) > 0.4:
                self._film_offset += (goal - self._film_offset) * min(1.0, delta * 8.0)
                self._film_dirty = True
            else:
                self._film_follow = False
        if self._film_dirty:
            self._film_dirty = False
            self._paint_film()


def main() -> int:
    root = tk.Tk()
    Studio(root)
    root.mainloop()
    return 0
