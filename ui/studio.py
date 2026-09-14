"""Glove Defect Detection System.

Two screens in one industrial QC floor look (see ui/qc.py). The line screen offers
the three detectors, and picking one opens the inspect screen, which runs the
detector over every photo in the gloves folder and replays each one a step at a
time, so the preprocessing and the segmentation are visible rather than implied.
Finished photos stay in the batch strip and reopen for reading without the
detector running again, and they are kept when you go back to the line screen.
"""
from __future__ import annotations

import queue
import re
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

import pipeline
from pipeline import DefectSpec, Stage, Trace
from ui import neon
from ui.qc import (DETECTORS, DIM, DIM_2, INK, INK_2, MARK, PANE, PAPER, PASS, REJECT, RULE,
                   TITLE, LineScreen, clip, cover, font, letterbox, mix, px, stripes)

TICK_MS = 33
STAGE_DWELL_MS = 520
VERDICT_DWELL_MS = 1700
PLAYBACK_SPEED = 2.0  # frames play at twice the base dwell, the detector always runs flat out

PHASE_NAMES = {"preprocess": "Preprocess", "segmentation": "Segment glove",
               "analysis": "Analyse", "verdict": "Verdict"}
PHASE_NOTES = {"preprocess": "Resize · balance · denoise", "segmentation": "Cue vote · GrabCut",
               "verdict": "Decide · mark"}
ANALYSIS_NOTES = {"fold": "Shading · weave · residual", "dirty": "Outliers · texture · hue",
                  "tear": "Tips · holes · show-through"}


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
        root.title(TITLE)
        root.configure(bg=INK)
        root.minsize(px(1180), px(740))
        self._centre(px(1440), px(900))

        self.animator = Animator(root)
        self.animator.start()

        self.photos: List[Path] = pipeline.photos_in()
        self.store: Dict[str, dict] = {}   # what each detector has finished, kept across visits
        self.container = tk.Frame(root, bg=INK)
        self.container.pack(fill="both", expand=True)

        self.home = LineScreen(self, self.container)
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
        self.run = RunScreen(self, self.container, spec, self.photos, self.store.get(spec.slug))
        self.run.place(in_=self.container, relx=0, rely=0, relwidth=1, relheight=1)


# -------------------------------------------------------------- inspect screen

class RunScreen(tk.Frame):
    """The main pane plays the photo being processed live, the batch strip keeps
    every finished photo, and picking one of those opens it for reading without
    running the detector again. Everything is drawn on one canvas, each region
    under its own tag so it can be redrawn alone."""

    RAIL_W = 384
    FILM_H = 108

    def __init__(self, app: Studio, parent: tk.Widget, spec: DefectSpec, paths: List[Path],
                 saved: Optional[dict] = None) -> None:
        super().__init__(parent, bg=INK)
        self.app = app
        self.spec = spec
        self.paths = paths
        self.name = DETECTORS.get(spec.slug, (spec.title, "", ""))[0]

        self.traces: Dict[int, Trace] = dict(saved["traces"]) if saved else {}
        self.summary: Dict[int, dict] = dict(saved["summary"]) if saved else {}
        pending = [i for i in range(len(paths)) if i not in self.summary]
        self.complete = not pending
        self.live = pending[0] if pending else len(paths) - 1   # the photo the run is working through
        self.index = self.live     # the photo on screen
        self.stage_i = -1          # the frame on screen
        self.reviewing = False     # True while reading a finished photo
        self.playing = not self.complete
        self.speed = PLAYBACK_SPEED
        self._live_stage = -1      # where the live photo was left when review began
        self._dwell_left = 0.0
        self._elapsed = 0.0
        self._pulse = 0.0
        self._closed = False
        self._log: List[dict] = []
        self._reveal = 1.0
        self._reveal_from: Optional[np.ndarray] = None
        self._reveal_to: Optional[np.ndarray] = None
        self._current: Optional[np.ndarray] = None
        self._labels = ("", "")
        self._labels_prev: Optional[tuple] = None
        self._geo_cache: Optional[tuple] = None
        self._images: Dict[str, object] = {}
        self._source_cache: Dict[int, np.ndarray] = {}
        self._previews: Dict[int, np.ndarray] = {}
        self._cache: Dict[tuple, ImageTk.PhotoImage] = {}
        self._hits: Dict[str, list] = {}
        self._hover = ""
        self._ref_index = -1
        self._prog = None
        self._film_dirty = True
        self._bar_value = len(self.summary) / max(len(paths), 1)

        self.canvas = tk.Canvas(self, bg=INK, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._on_resize())
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Leave>", lambda _e: self._set_hover(""))
        self.bind_all("<space>", self._toggle_play_event)
        self.bind_all("<Right>", lambda _e: self._step(1))
        self.bind_all("<Left>", lambda _e: self._step(-1))

        self.worker = TraceWorker(spec, paths)
        threading.Thread(target=self._load_previews, daemon=True).start()
        self._log_line("studio", "loaded {} photo(s) from gloves/".format(len(paths)))
        self._log_line("studio", "detector {}".format(spec.module))
        if self.summary:
            self._log_line("studio", "{} photo(s) kept from the last visit".format(len(self.summary)))
        if self.complete:
            self._review(self.live)
        else:
            self.worker.ensure(self.live, self.traces)
            self.worker.ensure(self.live + 1, self.traces)
        self.app.animator.add(self._tick)
        self._poll = self.after(60, self._drain)

    def close(self) -> None:
        self._closed = True
        self.app.store[self.spec.slug] = {"traces": self.traces, "summary": self.summary}
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

    def _load_previews(self) -> None:
        """Batch thumbnails are decoded off the UI thread."""
        for index, path in enumerate(self.paths):
            if self._closed:
                return
            image = pipeline.imread_preview(path)
            if image is not None:
                self._previews[index] = pipeline.fit(image, 220)
                self._film_dirty = True

    # -- geometry

    def _geo(self) -> dict:
        c = self.canvas
        W, H = max(c.winfo_width(), 10), max(c.winfo_height(), 10)
        if self._geo_cache is not None and self._geo_cache[0] == (W, H):
            return self._geo_cache[1]
        g: Dict[str, object] = {"W": W, "H": H}
        bar_top, bar_bottom = px(6), px(6) + px(64)
        film_top = H - px(self.FILM_H)
        rail_left = W - px(self.RAIL_W)
        g["bar"] = (0, bar_top, W, bar_bottom)
        g["film"] = (0, film_top, W, H)
        g["rail"] = (rail_left, bar_bottom, W, film_top)
        mx0, my0, mx1, my1 = px(18), bar_bottom + px(14), rail_left - px(18), film_top - px(16)
        g["tools"] = (mx0, my0, mx1, my0 + px(24))
        strip_top = my1 - px(118)
        cy0, cy1 = my0 + px(24) + px(12), strip_top - px(12)
        ref_w = int((mx1 - mx0 - px(12)) * 0.34)
        g["ref"] = (mx0, cy0, mx0 + ref_w, cy1)
        g["main"] = (mx0 + ref_w + px(12), cy0, mx1, cy1)
        thumbs_w = 5 * px(62) + 4 * px(8)
        g["stages"] = (mx1 - thumbs_w, strip_top, mx1, my1)
        g["pipe"] = (mx0, strip_top, mx1 - thumbs_w - px(12), my1)
        y = bar_bottom
        for name, height in (("stamp", 162), ("figure", 92), ("mtx", 112), ("truth", 42)):
            g[name] = (rail_left, y, W, y + px(height))
            y += px(height)
        g["rfoot"] = (rail_left, film_top - px(56), W, film_top)
        g["log"] = (rail_left, y, W, film_top - px(56))
        self._geo_cache = ((W, H), g)
        return g

    @staticmethod
    def _inner(box) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = box
        return x0 + px(16), y0 + px(32), x1 - px(16), y1 - px(24)

    def _remember(self, key: tuple, make) -> ImageTk.PhotoImage:
        photo = self._cache.get(key)
        if photo is None:
            if len(self._cache) > 480:
                self._cache.clear()
            photo = make()
            self._cache[key] = photo
        return photo

    # -- data flow

    def _drain(self) -> None:
        while True:
            try:
                index, trace = self.worker.results.get_nowait()
            except queue.Empty:
                break
            self.worker.forget(index)
            self.traces[index] = trace
            self._film_dirty = True
            if index == self.live and not self.reviewing and self.stage_i < 0:
                self._begin_trace()
        self._poll = self.after(60, self._drain)

    def _begin_trace(self) -> None:
        """Start playing the live photo from its first frame."""
        trace = self.traces.get(self.live)
        if trace is None or not trace.stages:
            return
        self.index = self.live
        self.stage_i = 0
        self._dwell_left = STAGE_DWELL_MS / 1000.0
        self._log_line("photo", trace.name, colour=MARK)
        self.worker.ensure(self.live + 1, self.traces)
        self._paint_ref()
        self._show_stage(trace.stages[0])
        self._paint_bar()
        self._paint_rail()

    def _advance(self) -> None:
        trace = self.traces.get(self.live)
        if trace is None or not trace.stages:
            return
        if self.stage_i + 1 < len(trace.stages):
            self.stage_i += 1
            last = self.stage_i == len(trace.stages) - 1
            self._dwell_left = (VERDICT_DWELL_MS if last else STAGE_DWELL_MS) / 1000.0
            if last:
                self._finish_trace(trace)
            self._show_stage(trace.stages[self.stage_i])
            self._paint_rail()
        else:
            self._next_photo()

    def _finish_trace(self, trace: Trace) -> None:
        self.summary[self.live] = {"verdict": trace.verdict, "found": trace.found,
                                   "expected": trace.expected, "correct": trace.correct,
                                   "score": trace.score, "elapsed": trace.elapsed}
        self._log_line("verdict", "{}   {}".format(trace.verdict, trace.details),
                       colour=REJECT if trace.found else PASS)
        self._film_dirty = True
        self._paint_bar()

    def _next_photo(self) -> None:
        if self.live + 1 >= len(self.paths):
            self.complete = True
            self.playing = False
            self._log_line("studio", "run complete, {} photo(s) inspected".format(len(self.paths)))
            self._paint_all()
            return
        self.live += 1
        self.index = self.live
        self.stage_i = -1
        self._reveal = 1.0
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
        self._show_stage(trace.stages[self.stage_i])
        self._paint_all()

    def _go_live(self) -> None:
        """Back to the photo the run is working through, where it was left."""
        if self.complete:
            return
        self.reviewing = False
        self.playing = True
        self.index = self.live
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

    def _jump(self, target: int) -> None:
        """Open one step of the photo on screen. Scrubbing the live photo holds it
        there until play is pressed, and it can never run ahead of the detector."""
        trace = self.traces.get(self.index)
        if trace is None or not trace.stages or self.stage_i < 0:
            return
        target = max(0, min(target, len(trace.stages) - 1))
        if not self.reviewing and self.index not in self.summary:
            target = min(target, self.stage_i)
            self.playing = False
        self.stage_i = target
        self._show_stage(trace.stages[target])
        self._paint_bar()
        self._paint_rail()

    def _row_state(self, index: int) -> str:
        if index in self.summary:
            return "done"
        if index == self.live and not self.complete:
            return "live"
        return "queued"

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

    # -- painting

    def _on_resize(self) -> None:
        self._geo_cache = None
        self._paint_chrome()
        trace = self.traces.get(self.index)
        if trace is not None and trace.stages and 0 <= self.stage_i < len(trace.stages):
            self._show_stage(trace.stages[self.stage_i], animate=False)
        else:
            self._show_waiting()
        self._paint_ref()
        self._paint_all()
        self._paint_log()

    def _paint_all(self) -> None:
        self._paint_bar()
        self._paint_ref()
        self._paint_steps()
        self._paint_rail()
        self._film_dirty = True

    def _paint_chrome(self) -> None:
        c = self.canvas
        c.delete("chrome")
        g = self._geo()
        W = g["W"]
        self._images["hazard"] = stripes(W, px(6), MARK, INK, px(13))
        c.create_image(0, 0, image=self._images["hazard"], anchor="nw", tags="chrome")
        x0, y0, x1, y1 = g["rail"]
        c.create_rectangle(x0, y0, x1, y1, fill=INK_2, outline="", tags="chrome")
        c.create_line(x0, y0, x0, y1, fill=RULE, tags="chrome")
        c.create_line(0, g["bar"][3] - 1, W, g["bar"][3] - 1, fill=RULE, tags="chrome")
        c.create_line(0, g["film"][1], W, g["film"][1], fill=RULE, tags="chrome")
        for name in ("figure", "mtx", "truth"):
            y = g[name][3] - 1
            c.create_line(x0, y, x1, y, fill=RULE, tags="chrome")
        c.create_line(x0, g["rfoot"][1], x1, g["rfoot"][1], fill=RULE, tags="chrome")
        c.tag_lower("chrome")

    # -- bar

    def _paint_bar(self) -> None:
        c = self.canvas
        c.delete("bar")
        g = self._geo()
        _x0, y0, x1, y1 = g["bar"]
        mid = (y0 + y1) // 2
        n = len(self.paths)

        colour = PAPER if self._hover == "back" else DIM
        bx = px(24)
        c.create_line(bx + px(9), mid - px(5), bx + px(4), mid, bx + px(9), mid + px(5), fill=colour,
                      width=max(px(1.8), 1), capstyle="round", joinstyle="round", tags="bar")
        back = c.create_text(bx + px(20), mid, text="Detectors", anchor="w", font=font("sans", 13),
                             fill=colour, tags="bar")
        back_right = c.bbox(back)[2]
        self._hits["back"] = [(px(14), y0, back_right + px(10), y1, "back")]
        tick = back_right + px(24)
        c.create_rectangle(tick, mid - px(13), tick + px(3), mid + px(13), fill=MARK, outline="", tags="bar")
        c.create_text(tick + px(17), mid, text=self.name, anchor="w", font=font("disp", 22), fill=PAPER,
                      tags="bar")

        x = x1 - px(18)
        total = c.create_text(x, mid, text=" / {:02d}".format(n), anchor="e", font=font("mono", 12.5),
                              fill=DIM, tags="bar")
        count = c.create_text(c.bbox(total)[0], mid, text="{:02d}".format(self.index + 1), anchor="e",
                              font=font("mono", 12.5, True), fill=PAPER, tags="bar")
        x = c.bbox(count)[0] - px(26)
        if self.complete:
            status, colour = "COMPLETE {} / {}".format(n, n), PASS
        elif self.playing and not self.reviewing:
            status, colour = "RUNNING {} / {}".format(self.live + 1, n), MARK
        else:
            status, colour = "PAUSED {} / {}".format(self.live + 1, n), DIM
        label = c.create_text(x, mid, text=status, anchor="e", font=font("mono", 11.5), fill=colour,
                              tags="bar")
        right = c.bbox(label)[0] - px(11)
        left = right - px(130)
        c.create_rectangle(left, mid - px(2), right, mid + px(2), fill=RULE, outline="", tags="bar")
        c.create_rectangle(left, mid - px(2), left + (right - left) * self._bar_value, mid + px(2),
                           fill=PASS if self.complete else MARK, outline="", tags=("bar", "progfill"))
        self._prog = (left, right, mid)

    # -- tools row

    def _paint_tools(self) -> None:
        c = self.canvas
        c.delete("tools")
        g = self._geo()
        x0, y0, x1, y1 = g["tools"]
        mid = (y0 + y1) // 2
        trace = self.traces.get(self.index)
        if trace is not None and trace.stages and 0 <= self.stage_i < len(trace.stages):
            stage = trace.stages[self.stage_i]
            items = [("STEP", DIM_2, 10.5, False),
                     ("{:02d} / {:02d}".format(self.stage_i + 1, len(trace.stages)), PAPER, 11, True),
                     (PHASE_NAMES.get(stage.phase, stage.phase.title()).upper(), MARK, 10.5, True)]
            if self.reviewing:
                items.append(("REVIEWING A FINISHED PHOTO", DIM, 10.5, False))
        else:
            items = [("WAITING FOR THE DETECTOR", DIM_2, 10.5, False)]
        x = x0
        for text, colour, size, bold in items:
            item = c.create_text(x, mid, text=text, anchor="w", font=font("mono", size, bold), fill=colour,
                                 tags="tools")
            x = c.bbox(item)[2] + px(12)

        keys = [("←", True), ("→", True), ("photo", False), ("Space", True), ("play", False)]
        x = x1
        for text, boxed in reversed(keys):
            if boxed:
                item = c.create_text(x - px(5), mid, text=text, anchor="e", font=font("mono", 10.5),
                                     fill=DIM, tags="tools")
                bx0, by0, bx1, by1 = c.bbox(item)
                c.create_rectangle(bx0 - px(5), by0, bx1 + px(5), by1, outline=RULE, tags="tools")
                x = bx0 - px(10)
            else:
                item = c.create_text(x - px(4), mid, text=text, anchor="e", font=font("mono", 10.5),
                                     fill=DIM_2, tags="tools")
                x = c.bbox(item)[0] - px(14)

    # -- the two panes

    def _tab(self, tag: str, x: int, y: int, text: str, fill: str, ink: str, bold: bool, room: int) -> None:
        c = self.canvas
        fnt = font("mono", 11, bold)
        text = clip(text, fnt, max(room - px(22), 20))
        item = c.create_text(x + px(11), y + px(11), text=text, anchor="w", font=fnt, fill=ink, tags=tag)
        right = c.bbox(item)[2] + px(11)
        back = c.create_rectangle(x, y, right, y + px(22), fill=fill, outline="", tags=tag)
        c.tag_lower(back, item)

    def _source_frame(self, index: int) -> Optional[np.ndarray]:
        trace = self.traces.get(index)
        if trace is not None and trace.stages:
            return trace.stages[0].frame()
        return self._source_image(index)

    def _paint_ref(self) -> None:
        c = self.canvas
        c.delete("ref")
        g = self._geo()
        x0, y0, x1, y1 = g["ref"]
        c.create_rectangle(x0, y0, x1, y1, fill=PANE, outline=RULE, tags="ref")
        ix0, iy0, ix1, iy1 = self._inner(g["ref"])
        source = self._source_frame(self.index)
        if source is not None and ix1 - ix0 > 8 and iy1 - iy0 > 8:
            has_trace = self.index in self.traces
            photo = self._remember(("ref", self.index, has_trace, ix1 - ix0, iy1 - iy0),
                                   lambda: ImageTk.PhotoImage(Image.fromarray(
                                       letterbox(source, ix1 - ix0, iy1 - iy0))))
            self._images["ref"] = photo
            c.create_image(ix0, iy0, image=photo, anchor="nw", tags="ref")
        self._tab("ref", x0, y0, "SOURCE", RULE, PAPER, False, x1 - x0)
        trace = self.traces.get(self.index)
        meta = self.paths[self.index].name
        if trace is not None and trace.size[0]:
            meta += "  ·  {}×{}".format(*trace.size)
        c.create_text(x0 + px(13), y1 - px(9), text=clip(meta, font("mono", 10.5), x1 - x0 - px(26)),
                      anchor="sw", font=font("mono", 10.5), fill=DIM_2, tags="ref")
        self._ref_index = self.index

    def _show_waiting(self) -> None:
        self._current = None
        self._reveal = 1.0
        self._reveal_to = None
        self._set_labels("ANALYZING", "the detector is working on this photo, its steps appear here "
                                      "as they are produced")
        if self._ref_index != self.index:
            self._paint_ref()
        self._paint_main()
        self._paint_steps()

    def _show_stage(self, stage: Stage, animate: bool = True) -> None:
        g = self._geo()
        ix0, iy0, ix1, iy1 = self._inner(g["main"])
        target = letterbox(stage.frame(), max(ix1 - ix0, 8), max(iy1 - iy0, 8))
        if self._reveal < 1.0 and self._reveal_to is not None:
            self._current = self._reveal_to  # land a crossfade that is still running
            self._reveal_to = None
            self._reveal = 1.0
        if animate and self._current is not None and self._current.shape == target.shape:
            self._reveal_from = self._current
            self._reveal_to = target
            self._reveal = 0.0
        else:
            self._reveal = 1.0
            self._current = target
        self._set_labels("{:02d} · {}".format(self.stage_i + 1, stage.title.upper()), stage.caption)
        if self._ref_index != self.index:
            self._paint_ref()
        self._paint_main()
        self._paint_steps()

    def _set_labels(self, tab: str, meta: str) -> None:
        """The labels change halfway through the crossfade, so they never describe
        a frame that is still mostly the previous one."""
        self._labels_prev = self._labels
        self._labels = (tab, meta)

    def _paint_main(self) -> None:
        c = self.canvas
        c.delete("main")
        g = self._geo()
        x0, y0, x1, y1 = g["main"]
        c.create_rectangle(x0, y0, x1, y1, fill=PANE, outline=RULE, tags="main")
        ix0, iy0, ix1, iy1 = self._inner(g["main"])
        tab, meta = self._labels_prev if self._reveal < 0.5 and self._labels_prev else self._labels
        frame = self._current
        if self._reveal < 1.0 and self._reveal_to is not None and self._reveal_from is not None:
            mix_t = neon.ease_in_out(self._reveal)
            frame = cv2.addWeighted(self._reveal_from, 1.0 - mix_t, self._reveal_to, mix_t, 0)
        if frame is not None:
            self._images["main"] = ImageTk.PhotoImage(Image.fromarray(frame))
            c.create_image(ix0, iy0, image=self._images["main"], anchor="nw", tags="main")
        else:
            cx, cy = (ix0 + ix1) // 2, (iy0 + iy1) // 2
            for slot in range(3):
                beat = 0.5 + 0.5 * np.sin(self._pulse * 4.0 - slot * 0.8)
                half = px(4)
                sx = cx + (slot - 1) * px(18)
                c.create_rectangle(sx - half, cy - half, sx + half, cy + half, outline="",
                                   fill=mix(PANE, MARK, 0.25 + 0.75 * beat), tags="main")
        self._tab("main", x0, y0, tab, MARK, INK, True, x1 - x0)
        c.create_text(x0 + px(13), y1 - px(9), text=clip(meta, font("mono", 10.5), x1 - x0 - px(26)),
                      anchor="sw", font=font("mono", 10.5), fill=DIM_2, tags="main")

    # -- pipeline and stage thumbnails

    def _paint_steps(self) -> None:
        self._paint_tools()
        self._paint_pipe()
        self._paint_stages()

    def _paint_pipe(self) -> None:
        c = self.canvas
        c.delete("pipe")
        g = self._geo()
        x0, y0, x1, y1 = g["pipe"]
        c.create_rectangle(x0, y0, x1, y1, outline=RULE, tags="pipe")
        c.create_text(x0 + px(16), y0 + px(12), text="PIPELINE", anchor="nw", font=font("disp", 12.5),
                      fill=PAPER, tags="pipe")
        trace = self.traces.get(self.index)
        order: List[str] = []
        first: Dict[str, int] = {}
        if trace is not None and trace.stages:
            for i, stage in enumerate(trace.stages):
                if stage.phase not in first:
                    first[stage.phase] = i
                    order.append(stage.phase)
            c.create_text(x1 - px(16), y0 + px(14), anchor="ne", font=font("mono", 10.5), fill=DIM,
                          text="{} STEPS  ·  {:.2f} S".format(len(trace.stages), trace.elapsed), tags="pipe")
        if not order:
            order = list(PHASE_NAMES)
        current = -1
        if trace is not None and trace.stages and 0 <= self.stage_i < len(trace.stages):
            phase = trace.stages[self.stage_i].phase
            current = order.index(phase) if phase in order else -1

        col_w = (x1 - x0 - px(32)) / max(len(order), 1)
        top = y1 - px(12) - px(48)
        square = px(9)
        hits = []
        for k, phase in enumerate(order):
            cx = int(x0 + px(16) + k * col_w)
            done, now = k < current, k == current
            if now:
                c.create_rectangle(cx - px(4), top - px(4), cx + square + px(4), top + square + px(4),
                                   fill=mix(INK, MARK, 0.18), outline="", tags="pipe")
            c.create_rectangle(cx, top, cx + square, top + square, outline="", tags="pipe",
                               fill=PASS if done else (MARK if now else RULE))
            if k < len(order) - 1:
                c.create_line(cx + px(12), top + px(4), int(cx + col_w - px(14)), top + px(4),
                              fill=PASS if done else RULE, tags="pipe")
            note = ANALYSIS_NOTES.get(self.spec.slug, "") if phase == "analysis" else PHASE_NOTES.get(phase, "")
            c.create_text(cx, top + px(17), text=PHASE_NAMES.get(phase, phase.title()), anchor="nw",
                          font=font("semi", 12.5), fill=PAPER, tags="pipe")
            c.create_text(cx, top + px(35), text=clip(note, font("sans", 10.5), int(col_w - px(14))),
                          anchor="nw", font=font("sans", 10.5), fill=DIM_2, tags="pipe")
            if phase in first:
                hits.append((cx - px(6), top - px(10), int(cx + col_w - px(8)), y1, first[phase]))
        self._hits["pipe"] = hits

    def _paint_stages(self) -> None:
        """Thumbnails of the steps in the current phase, the one on screen marked."""
        c = self.canvas
        c.delete("stages")
        self._hits["stages"] = []
        g = self._geo()
        x0, y0, _x1, _y1 = g["stages"]
        trace = self.traces.get(self.index)
        if trace is None or not trace.stages or not 0 <= self.stage_i < len(trace.stages):
            return
        stages = trace.stages
        phase = stages[self.stage_i].phase
        members = [i for i, stage in enumerate(stages) if stage.phase == phase]
        at = members.index(self.stage_i)
        start = max(0, min(at - 2, len(members) - 5))
        tw, th, gap = px(62), px(80), px(8)
        hits = []
        for k, si in enumerate(members[start:start + 5]):
            tx = x0 + k * (tw + gap)
            on = si == self.stage_i
            photo = self._remember(("thumb", self.index, si, tw, th),
                                   lambda si=si: ImageTk.PhotoImage(cover(stages[si].frame(), tw - 2, th - 2)))
            c.create_rectangle(tx, y0, tx + tw, y0 + th, fill=PANE, outline=MARK if on else RULE,
                               tags="stages")
            c.create_image(tx + 1, y0 + 1, image=photo, anchor="nw", tags="stages")
            caption = clip(stages[si].key.replace("_", " ").upper(), font("mono", 9.5), tw)
            c.create_text(tx, y0 + th + px(5), text=caption, anchor="nw", font=font("mono", 9.5),
                          fill=MARK if on else DIM, tags="stages")
            hits.append((tx, y0, tx + tw, y0 + th + px(20), si))
        self._hits["stages"] = hits

    # -- rail

    def _paint_rail(self) -> None:
        self._paint_stamp()
        self._paint_figure()
        self._paint_mtx()
        self._paint_truth()
        self._paint_rfoot()

    def _paint_stamp(self) -> None:
        c = self.canvas
        c.delete("stamp")
        g = self._geo()
        x0, y0, x1, y1 = g["stamp"]
        entry = self.summary.get(self.index)
        trace = self.traces.get(self.index)
        if entry and trace is not None:
            ground, ink = (REJECT if entry["found"] else PASS), INK
            label, head, body = "DETECTOR VERDICT", ("REJECT" if entry["found"] else "PASS"), trace.details
        else:
            ground, ink = RULE, PAPER
            label = "IN PROGRESS"
            head = "RUNNING" if self.playing and not self.reviewing else "PAUSED"
            if trace is not None and trace.stages and 0 <= self.stage_i < len(trace.stages):
                stage = trace.stages[self.stage_i]
                body = "Step {} of {}, {}.".format(self.stage_i + 1, len(trace.stages),
                                                   PHASE_NAMES.get(stage.phase, stage.phase).lower())
            else:
                body = "Waiting for the detector to finish this photo."
        if len(body) > 96:
            body = body[:95].rstrip() + "…"
        c.create_rectangle(x0, y0, x1, y1, fill=ground, outline="", tags="stamp")
        c.create_text(x0 + px(22), y0 + px(16), text=label, anchor="nw", font=font("mono", 10.5, True),
                      fill=mix(ground, ink, 0.75), tags="stamp")
        c.create_text(x0 + px(19), y0 + px(28), text=head, anchor="nw", font=font("disp", 52), fill=ink,
                      tags="stamp")
        c.create_text(x0 + px(22), y0 + px(104), text=body, anchor="nw", width=x1 - x0 - px(44),
                      font=font("semi", 12.5), fill=ink, tags="stamp")
        self._images["tape"] = stripes(x1 - x0, px(5), mix(ground, INK, 0.85), ground, px(11))
        c.create_image(x0, y1 - px(5), image=self._images["tape"], anchor="nw", tags="stamp")

    def _figure(self) -> Tuple[str, str, str]:
        """The one number that matters for this detector."""
        entry = self.summary.get(self.index)
        trace = self.traces.get(self.index)
        if not entry or trace is None:
            return "–", "", "waiting for\nthe verdict"
        if self.spec.slug == "dirty":
            found = re.search(r"([\d.]+)% of the glove", trace.details)
            return "{:.2f}".format(float(found.group(1)) if found else 0.0), "%", "of the glove\nsurface affected"
        if self.spec.slug == "fold":
            return "{:.2f}".format(entry["score"]), "", "crease score,\n1.00 at a 1.5R span"
        if self.spec.slug == "tear":
            return "{:.2f}".format(entry["score"]), "", "strongest tear\nconfidence"
        return "{:.2f}".format(entry["score"]), "", "detector\nscore"

    def _paint_figure(self) -> None:
        c = self.canvas
        c.delete("figure")
        g = self._geo()
        x0, _y0, _x1, y1 = g["figure"]
        value, unit, caption = self._figure()
        pending = value == "–"
        big = c.create_text(x0 + px(20), y1 - px(4), text=value, anchor="sw", font=font("disp", 62),
                            fill=DIM_2 if pending else PAPER, tags="figure")
        right = c.bbox(big)[2]
        if unit:
            item = c.create_text(right + px(6), y1 - px(22), text=unit, anchor="sw", font=font("disp", 22),
                                 fill=MARK, tags="figure")
            right = c.bbox(item)[2]
        c.create_text(right + px(12), y1 - px(24), text=caption, anchor="sw", font=font("sans", 11.5),
                      fill=DIM, tags="figure")

    def _paint_mtx(self) -> None:
        c = self.canvas
        c.delete("mtx")
        g = self._geo()
        x0, y0, x1, _y1 = g["mtx"]
        entry = self.summary.get(self.index)
        trace = self.traces.get(self.index)
        if entry and trace is not None:
            rows = [("Regions", str(trace.regions)), ("Score", "{:.2f}".format(entry["score"])),
                    ("Time", "{:.2f} s".format(entry["elapsed"]))]
        else:
            rows = [("Regions", "—"), ("Score", "—"), ("Time", "—")]
        row_h = px(35)
        for k, (name, value) in enumerate(rows):
            mid = y0 + px(4) + k * row_h + row_h // 2
            c.create_text(x0 + px(22), mid, text=name, anchor="w", font=font("sans", 12.5), fill=DIM,
                          tags="mtx")
            c.create_text(x1 - px(22), mid, text=value, anchor="e", font=font("mono", 13), fill=PAPER,
                          tags="mtx")
            if k < len(rows) - 1:
                y = y0 + px(4) + (k + 1) * row_h
                c.create_line(x0 + px(22), y, x1 - px(22), y, fill=mix(INK_2, RULE, 0.6), tags="mtx")

    def _paint_truth(self) -> None:
        c = self.canvas
        c.delete("truth")
        g = self._geo()
        x0, y0, x1, y1 = g["truth"]
        mid = (y0 + y1) // 2
        expected = self.paths[self.index].stem.lower().startswith(self.spec.prefix)
        says = c.create_text(x0 + px(22), mid, text="Filename says", anchor="w", font=font("sans", 12),
                             fill=DIM, tags="truth")
        c.create_text(c.bbox(says)[2] + px(6), mid, text=self.spec.prefix if expected else "clean",
                      anchor="w", font=font("mono", 12), fill=PAPER, tags="truth")
        entry = self.summary.get(self.index)
        if entry:
            text, colour = ("AGREES", PASS) if entry["correct"] else ("DIFFERS", REJECT)
        else:
            text, colour = "—", DIM_2
        c.create_text(x1 - px(22), mid, text=text, anchor="e", font=font("mono", 11, True), fill=colour,
                      tags="truth")

    def _paint_rfoot(self) -> None:
        c = self.canvas
        c.delete("rfoot")
        g = self._geo()
        x0, y0, x1, y1 = g["rfoot"]
        mid = (y0 + y1) // 2
        bw, bh = px(36), px(34)
        hits = []
        for k, key in enumerate(("prev", "play", "next")):
            bx = x0 + px(16) + k * (bw + px(8))
            by = mid - bh // 2
            hot = self._hover == "rfoot:" + key
            if key == "play":
                disabled = self.complete
                c.create_rectangle(bx, by, bx + bw, by + bh, fill=RULE if disabled else MARK,
                                   outline=RULE if disabled else MARK, tags="rfoot")
                ink = DIM_2 if disabled else INK
                cx, cy = bx + bw // 2, by + bh // 2
                if self.playing and not self.reviewing and not self.complete:
                    for dx in (-px(3), px(3)):
                        c.create_rectangle(cx + dx - px(1.5), cy - px(5), cx + dx + px(1.5), cy + px(5),
                                           fill=ink, outline="", tags="rfoot")
                else:
                    c.create_polygon(cx - px(3), cy - px(5.5), cx - px(3), cy + px(5.5), cx + px(5), cy,
                                     fill=ink, outline="", tags="rfoot")
            else:
                c.create_rectangle(bx, by, bx + bw, by + bh, outline=DIM_2 if hot else RULE, tags="rfoot")
                cx, cy = bx + bw // 2, by + bh // 2
                sign = -1 if key == "prev" else 1
                c.create_line(cx - sign * px(2), cy - px(5), cx + sign * px(3), cy, cx - sign * px(2),
                              cy + px(5), fill=PAPER if hot else DIM, width=max(px(1.8), 1),
                              capstyle="round", joinstyle="round", tags="rfoot")
            hits.append((bx, by, bx + bw, by + bh, key))
        hot = self._hover == "rfoot:save"
        label = c.create_text(x1 - px(16) - px(15), mid, text="Save result", anchor="e",
                              font=font("semi", 12.5), fill=PAPER, tags="rfoot")
        lx0, _ly0, lx1, _ly1 = c.bbox(label)
        box = (lx0 - px(15), mid - bh // 2, lx1 + px(15), mid + bh // 2)
        c.create_rectangle(*box, outline=DIM_2 if hot else RULE, tags="rfoot")
        hits.append(box + ("save",))
        self._hits["rfoot"] = hits

    def _paint_log(self) -> None:
        c = self.canvas
        c.delete("log")
        g = self._geo()
        x0, y0, x1, y1 = g["log"]
        c.create_text(x0 + px(22), y0 + px(15), text="LOG", anchor="nw", font=font("disp", 12), fill=PAPER,
                      tags="log")
        line_h = px(17)
        text_x = x0 + px(22) + px(60)
        fnt = font("mono", 10)
        room = max(x1 - text_x - px(18), 40)
        y = y1 - px(14)
        for entry in reversed(self._log):
            if y < y0 + px(44):
                break
            fresh = min(entry["age"] / 0.35, 1.0)
            c.create_text(x0 + px(22), y, text=entry["tag"], anchor="w", font=fnt, fill=DIM_2, tags="log")
            c.create_text(text_x, y, text=clip(entry["text"], fnt, room), anchor="w", font=fnt,
                          fill=mix(INK_2, entry["colour"], 0.35 + 0.65 * fresh), tags="log")
            y -= line_h

    def _log_line(self, tag: str, text: str, colour: Optional[str] = None) -> None:
        self._log.append({"tag": tag, "text": text, "age": 0.0, "colour": colour or DIM})
        del self._log[:-160]
        if self._geo_cache is not None:
            self._paint_log()

    # -- batch strip

    def _paint_film(self) -> None:
        c = self.canvas
        c.delete("film")
        g = self._geo()
        W = g["W"]
        _x0, y0, _x1, y1 = g["film"]
        n = len(self.paths)
        rejected = sum(1 for entry in self.summary.values() if entry["found"])
        passed = len(self.summary) - rejected
        hy = y0 + px(19)
        title = c.create_text(px(18), hy, text="BATCH", anchor="w", font=font("disp", 12), fill=PAPER,
                              tags="film")
        x = c.bbox(title)[2] + px(14)
        chips = [("ALL {}".format(n), None, True), ("REJECT {}".format(rejected), REJECT, False),
                 ("PASS {}".format(passed), PASS, False), ("QUEUED {}".format(n - len(self.summary)), DIM_2, False)]
        for text, dot, on in chips:
            fnt = font("mono", 10.5, on)
            item = c.create_text(x + px(9) + (px(12) if dot else 0), hy, text=text, anchor="w", font=fnt,
                                 fill=INK if on else DIM, tags="film")
            right = c.bbox(item)[2] + px(9)
            box = c.create_rectangle(x, hy - px(10), right, hy + px(10), fill=PAPER if on else "",
                                     outline=PAPER if on else RULE, tags="film")
            c.tag_lower(box, item)
            if dot:
                c.create_rectangle(x + px(9), hy - px(3), x + px(15), hy + px(3), fill=dot, outline="",
                                   tags="film")
            x = right + px(6)
        c.create_text(W - px(18), hy, text="SORTED BY FILENAME", anchor="e", font=font("mono", 10.5),
                      fill=DIM, tags="film")

        fy0, fy1 = hy + px(18), y1 - px(10)
        fw, gap, strip = px(54), px(7), px(15)
        hits = []
        for i in range(n):
            fx = px(18) + i * (fw + gap)
            if fx + fw > W - px(18):
                break
            preview = self._previews.get(i)
            if preview is not None:
                photo = self._remember(("frame", i, fw, fy1 - fy0),
                                       lambda p=preview: ImageTk.PhotoImage(cover(p, fw - 2, fy1 - fy0 - 2)))
                c.create_image(fx + 1, fy0 + 1, image=photo, anchor="nw", tags="film")
            else:
                c.create_rectangle(fx, fy0, fx + fw, fy1, fill=PANE, outline="", tags="film")
            state = self._row_state(i)
            if state == "done":
                found = self.summary[i]["found"]
                ground, ink, letter = (REJECT, INK, "R") if found else (PASS, INK, "P")
            elif state == "live":
                ground, ink, letter = RULE, MARK, "···"
            else:
                ground, ink, letter = RULE, DIM, "—"
            c.create_rectangle(fx + 1, fy1 - strip, fx + fw, fy1, fill=ground, outline="", tags="film")
            c.create_text(fx + fw // 2, fy1 - strip // 2, text=letter, font=font("mono", 9, True), fill=ink,
                          tags="film")
            on = i == self.index
            if on:
                c.create_rectangle(fx - px(2), fy0 - px(2), fx + fw + px(2), fy1 + px(2),
                                   outline=mix(INK, MARK, 0.3), width=px(2), tags="film")
            c.create_rectangle(fx, fy0, fx + fw, fy1, outline=MARK if on else RULE, tags="film")
            hits.append((fx, fy0, fx + fw, fy1, i))
        self._hits["film"] = hits

    # -- input

    def _hit(self, x: int, y: int) -> Tuple[str, object]:
        for group in ("back", "rfoot", "pipe", "stages", "film"):
            for x0, y0, x1, y1, key in self._hits.get(group, []):
                if x0 <= x < x1 and y0 <= y < y1:
                    return group, key
        return "", None

    def _set_hover(self, hover: str) -> None:
        if hover != self._hover:
            self._hover = hover
            self._paint_bar()
            self._paint_rfoot()

    def _on_motion(self, event) -> None:
        group, key = self._hit(event.x, event.y)
        clickable = group in ("back", "rfoot", "pipe", "stages") or (
            group == "film" and self._row_state(key) != "queued")
        self.canvas.configure(cursor="hand2" if clickable else "")
        self._set_hover("back" if group == "back" else ("rfoot:" + key if group == "rfoot" else ""))

    def _on_click(self, event) -> None:
        group, key = self._hit(event.x, event.y)
        if group == "back":
            self.app.show_home()
        elif group == "rfoot":
            if key == "prev":
                self._step(-1)
            elif key == "next":
                self._step(1)
            elif key == "play":
                self._toggle_play()
            elif key == "save":
                self._save_frame()
        elif group in ("pipe", "stages"):
            self._jump(key)
        elif group == "film":
            state = self._row_state(key)
            if state == "done":
                self._review(key)
            elif state == "live":
                self._go_live()

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
        self._paint_bar()
        self._paint_rail()

    def _save_frame(self) -> None:
        trace = self.traces.get(self.index)
        if trace is None or not trace.stages or self.stage_i < 0:
            return
        stage = trace.stages[self.stage_i]
        folder = pipeline.OUTPUT_DIR / self.spec.slug
        target = folder / "{}_{:02d}_{}.png".format(Path(trace.name).stem, self.stage_i + 1, stage.key)
        if pipeline.imwrite_unicode(target, stage.frame()):
            self._log_line("saved", str(target.relative_to(pipeline.PROJECT_ROOT)), colour=MARK)
        else:
            self._log_line("saved", "could not write {}".format(target), colour=REJECT)

    # -- the clock

    def _tick(self, delta: float) -> None:
        self._elapsed += delta
        self._pulse += delta

        if self._reveal < 1.0:
            self._reveal = min(1.0, self._reveal + delta * 3.2 * self.speed)
            if self._reveal >= 1.0 and self._reveal_to is not None:
                self._current = self._reveal_to
                self._reveal_to = None
            self._paint_main()
        elif self.stage_i < 0 and int(self._elapsed * 12) % 2 == 0:
            self._paint_main()  # keeps the waiting squares moving

        # progress follows the run, not whatever photo is being read
        live_trace = self.traces.get(self.live)
        live_stage = self._live_stage if self.reviewing else self.stage_i
        total = len(live_trace.stages) if live_trace is not None and live_trace.stages else 1
        within = (live_stage + 1) / total if live_stage >= 0 else 0.0
        target = 1.0 if self.complete else (self.live + within) / max(len(self.paths), 1)
        self._bar_value += (target - self._bar_value) * min(1.0, delta * 4.0)
        if self._prog is not None:
            left, right, mid = self._prog
            self.canvas.coords("progfill", left, mid - px(2), left + (right - left) * self._bar_value,
                               mid + px(2))

        if self.playing and not self.reviewing and not self.complete:
            if self.stage_i < 0:
                if live_trace is not None and live_trace.stages:
                    self._begin_trace()
            else:
                self._dwell_left -= delta * self.speed
                if self._dwell_left <= 0:
                    self._advance()

        for entry in self._log:
            if entry["age"] < 0.4:
                entry["age"] += delta
                self._paint_log()
                break

        if self._film_dirty and self._geo_cache is not None:
            self._film_dirty = False
            self._paint_film()


def main() -> int:
    root = tk.Tk()
    Studio(root)
    root.mainloop()
    return 0
