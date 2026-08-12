"""Glove Defect Detection - desktop UI.

Three cards, left to right, in the order the operator works:

    1. DEFECT   pick the defect to test for
    2. PHOTOS   open a folder, then pick the photos from the contact sheet
    3. RESULT   verdict per photo, with the annotated photo below

Picking a defect imports `detectors/<key>.py` and runs only that detector,
so each defect really is backed by its own file (see `detectors/__init__.py`
for the menu registry).

Run it with the interpreter that has OpenCV installed:

    C:\\Tool\\python\\python.exe app.py
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np

from detectors import DEFECTS, DefectSpec
from gdd.pipeline import GloveInspector
from ui import theme
from ui.theme import px
from ui.widgets import (Button, Card, ClipLabel, DefectRow, Divider, PhotoTile,
                        ResultRow, ScrollFrame)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# The assignment asks the operator to run a handful of photos at a time.
# Fewer is allowed, but the counter says so in red.
RECOMMENDED_MIN_IMAGES = 1

TILE = 104          # contact-sheet thumbnail edge, before DPI scaling
TILE_GAP = 10

PREVIEW_PLACEHOLDER = "Select a row to view the annotated photo"

PROJECT_ROOT = Path(__file__).resolve().parent
PHOTO_DIR = PROJECT_ROOT / "gloves"          # where the photo set lives
OUTPUT_DIR = PROJECT_ROOT / "output"         # where annotated copies go


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #

def imread_unicode(path: Path, reduced: bool = False) -> Optional[np.ndarray]:
    """Read an image whose path may contain non-ASCII characters.

    `cv2.imread` goes through the ANSI Windows API and returns None for a
    path with Chinese characters in it, which is exactly what a folder of
    student photos tends to have. `reduced` decodes at 1/8 scale, which is
    all a 104 px thumbnail needs and roughly an order of magnitude faster on
    a 4000 px phone photo.
    """
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    flag = cv2.IMREAD_REDUCED_COLOR_8 if reduced else cv2.IMREAD_COLOR
    image = cv2.imdecode(buffer, flag)
    if image is None and reduced:      # tiny images cannot be reduced
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    ok, buffer = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        return False
    try:
        buffer.tofile(str(path))
    except OSError:
        return False
    return True


# --------------------------------------------------------------------------- #
# One inspected photo
# --------------------------------------------------------------------------- #

@dataclass
class Row:
    name: str
    status: str
    score: float
    evidence: str
    annotated: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #

class DefectApp:

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Glove Defect Detection")
        # Fit the window to the screen rather than assuming one: the DPI-aware
        # geometry below is in physical pixels, so a fixed number that is
        # comfortable at 100% scaling overflows a 125% display.
        width = min(px(1360), self.root.winfo_screenwidth() - px(80))
        height = min(px(830), self.root.winfo_screenheight() - px(120))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(px(1120), px(700))
        self.root.configure(bg=theme.APP_BG)

        # Where the file dialogs open. Starts at the project's own photo
        # folder so the common case takes no navigation at all, then follows
        # the operator once they pick somewhere else.
        self._browse_dir = PHOTO_DIR if PHOTO_DIR.is_dir() else PROJECT_ROOT
        self._save_dir = OUTPUT_DIR if OUTPUT_DIR.is_dir() else PROJECT_ROOT

        self.pool: List[Path] = []
        self.tiles: List[PhotoTile] = []
        self.selected: Set[int] = set()
        self._anchor = 0

        self.rows: List[Row] = []
        self.row_widgets: List[ResultRow] = []
        self.active_row = -1
        self.current_spec: Optional[DefectSpec] = None

        self.preview_image: Optional[np.ndarray] = None
        self._preview_photo = None
        self._resize_job: Optional[str] = None
        self._columns = 0

        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._run_token = 0
        self._thumb_token = 0
        self._running = False

        self._build_header()
        self._build_body()
        self._build_statusbar()
        self.root.after(50, self._drain_queue)
        # After the first idle pass, so the sheet lays its tiles out against
        # a real window width instead of the placeholder one.
        self.root.after(80, self._autoload_photos)

    # ------------------------------------------------------------------ #
    # Chrome
    # ------------------------------------------------------------------ #

    def _build_header(self) -> None:
        head = tk.Frame(self.root, bg=theme.HEADER_BG, height=px(54))
        head.pack(fill="x")
        head.pack_propagate(False)

        left = tk.Frame(head, bg=theme.HEADER_BG)
        left.pack(side="left", padx=(px(20), 0))
        tk.Label(left, text="Glove Defect Detection",
                 font=theme.font(13, strong=True), fg=theme.HEADER_FG,
                 bg=theme.HEADER_BG).pack(anchor="w")
        tk.Label(left, text="classical image processing  ·  CT036-3-IPPR",
                 font=theme.font(8), fg=theme.HEADER_DIM,
                 bg=theme.HEADER_BG).pack(anchor="w")

        ready = sum(1 for spec in DEFECTS if spec.implemented)
        tk.Label(head, text=f"{ready} of {len(DEFECTS)} detectors live",
                 font=theme.font(9), fg=theme.HEADER_DIM,
                 bg=theme.HEADER_BG).pack(side="right", padx=(0, px(20)))

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self.root, bg=theme.APP_BG, height=px(26))
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status = tk.Label(bar, text="Pick a defect, then load photos.",
                               font=theme.font(9), fg=theme.INK_FAINT,
                               bg=theme.APP_BG, anchor="w")
        self.status.pack(side="left", padx=px(20))

    def _build_body(self) -> None:
        body = tk.Frame(self.root, bg=theme.APP_BG)
        body.pack(fill="both", expand=True, padx=px(16), pady=(px(14), px(6)))
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0, minsize=px(248))
        body.columnconfigure(1, weight=3, minsize=px(372))
        body.columnconfigure(2, weight=5, minsize=px(470))

        self._build_defect_card(body)
        self._build_photo_card(body)
        self._build_result_card(body)

    # ---- card 1: defects ---------------------------------------------- #

    def _build_defect_card(self, parent: tk.Frame) -> None:
        card = Card(parent)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, px(12)))
        card.add_header("1", "Defect")

        scroll = ScrollFrame(card.body)
        scroll.pack(fill="both", expand=True, pady=(px(6), 0))

        self.defect_rows: List[DefectRow] = []
        for index, spec in enumerate(DEFECTS):
            row = DefectRow(scroll.inner, index, spec.label,
                            not spec.implemented, self._on_defect_click)
            row.pack(fill="x")
            self.defect_rows.append(row)

        footer = tk.Frame(card.body, bg=theme.CARD)
        footer.pack(fill="x", padx=px(14), pady=(px(8), px(12)))
        tk.Label(footer,
                 text="Clicking a defect runs its own\nfile in detectors/.",
                 font=theme.font(8), fg=theme.INK_FAINT, bg=theme.CARD,
                 justify="left").pack(anchor="w")

    # ---- card 2: photos ----------------------------------------------- #

    def _build_photo_card(self, parent: tk.Frame) -> None:
        card = Card(parent)
        card.grid(row=0, column=1, sticky="nsew", padx=(0, px(12)))
        head = card.add_header("2", "Photos")
        self.count_label = tk.Label(head, text="none loaded",
                                    font=theme.font(9), fg=theme.INK_FAINT,
                                    bg=theme.CARD)
        self.count_label.pack(side="right")

        tools = tk.Frame(card.body, bg=theme.CARD)
        tools.pack(fill="x", padx=px(14), pady=(px(12), px(10)))
        Button(tools, "Open folder", self.open_folder,
               variant="secondary").pack(side="left")
        Button(tools, "Add files", self.add_files,
               variant="ghost").pack(side="left", padx=px(4))
        Button(tools, "Clear", self.clear_pool,
               variant="ghost").pack(side="right")

        picks = tk.Frame(card.body, bg=theme.CARD)
        picks.pack(fill="x", padx=px(14), pady=(0, px(8)))
        Button(picks, "Select all", self.select_all, variant="ghost",
               pad=8, height=24, size=8).pack(side="left")
        Button(picks, "Select none", self.select_none, variant="ghost",
               pad=8, height=24, size=8).pack(side="left", padx=px(4))
        tk.Label(picks, text="click to pick  ·  shift-click for a range",
                 font=theme.font(8), fg=theme.INK_FAINT,
                 bg=theme.CARD).pack(side="right")

        Divider(card.body, theme.LINE_SOFT).pack(fill="x")

        self.sheet = ScrollFrame(card.body, padding=4)
        self.sheet.pack(fill="both", expand=True, padx=(px(12), px(4)),
                        pady=px(10))
        self.sheet.canvas.bind("<Configure>", self._on_sheet_resize, add="+")

        self.sheet_empty = tk.Label(
            self.sheet.inner, text="No photos yet.\nOpen a folder to begin.",
            font=theme.font(9), fg=theme.INK_FAINT, bg=theme.CARD,
            justify="center")
        self.sheet_empty.pack(pady=px(40))

        footer = tk.Frame(card.body, bg=theme.CARD)
        footer.pack(fill="x", padx=px(14), pady=(0, px(14)))
        self.run_button = Button(footer, "Run detection", self.run_detection,
                                 variant="primary", stretch=True, height=34,
                                 size=10)
        self.run_button.pack(fill="x")

    # ---- card 3: results ---------------------------------------------- #

    def _build_result_card(self, parent: tk.Frame) -> None:
        card = Card(parent)
        card.grid(row=0, column=2, sticky="nsew")
        head = card.add_header("3", "Result")
        self.save_button = Button(head, "Save annotated", self.save_results,
                                  variant="ghost", pad=10, height=24, size=8)
        self.save_button.pack(side="right")
        self.save_button.set_enabled(False)

        body = card.body
        body.columnconfigure(0, weight=1)
        # The result list takes only the height its rows need (capped), so a
        # short run gives the annotated photo the rest of the card instead of
        # leaving a band of empty white.
        body.rowconfigure(2, weight=0)
        body.rowconfigure(4, weight=1)

        summary = tk.Frame(body, bg=theme.CARD)
        summary.grid(row=0, column=0, sticky="ew", padx=px(16),
                     pady=(px(14), px(12)))
        self.summary_title = tk.Label(summary, text="Nothing inspected yet",
                                      font=theme.font(15, strong=True),
                                      fg=theme.INK, bg=theme.CARD, anchor="w")
        self.summary_title.pack(fill="x")
        self.summary_detail = tk.Label(
            summary, text="Choose a defect on the left and photos in the "
                          "middle, then run.",
            font=theme.font(9), fg=theme.INK_SOFT, bg=theme.CARD, anchor="w")
        self.summary_detail.pack(fill="x", pady=(px(3), 0))

        Divider(body, theme.LINE_SOFT).grid(row=1, column=0, sticky="ew")

        self.result_scroll = ScrollFrame(body)
        self.result_scroll.pack_propagate(False)
        self.result_scroll.configure(height=px(96))
        self.result_scroll.grid(row=2, column=0, sticky="nsew",
                                padx=(px(4), px(4)), pady=px(4))
        self.result_empty = tk.Label(self.result_scroll.inner,
                                     text="Results appear here, one row per photo.",
                                     font=theme.font(9), fg=theme.INK_FAINT,
                                     bg=theme.CARD)
        self.result_empty.pack(pady=px(28))

        Divider(body, theme.LINE_SOFT).grid(row=3, column=0, sticky="ew")

        stage = tk.Frame(body, bg=theme.CARD)
        stage.grid(row=4, column=0, sticky="nsew", padx=px(14),
                   pady=(px(12), px(6)))
        stage.rowconfigure(0, weight=1)
        stage.columnconfigure(0, weight=1)
        self.stage = tk.Frame(stage, bg=theme.STAGE)
        self.stage.grid(row=0, column=0, sticky="nsew")
        self.preview = tk.Label(self.stage, bg=theme.STAGE,
                                text=PREVIEW_PLACEHOLDER,
                                font=theme.font(9), fg=theme.INK_FAINT, bd=0)
        self.preview.place(relx=0.5, rely=0.5, anchor="center")
        self.stage.bind("<Configure>", self._on_stage_resize)

        self.caption = ClipLabel(body, "", font=theme.font(8),
                                 fg=theme.INK_FAINT, bg=theme.CARD)
        self.caption.grid(row=5, column=0, sticky="ew", padx=px(16),
                          pady=(px(4), px(12)))

    # ------------------------------------------------------------------ #
    # Card 1 behaviour
    # ------------------------------------------------------------------ #

    def _on_defect_click(self, index: int) -> None:
        self.current_spec = DEFECTS[index]
        for position, row in enumerate(self.defect_rows):
            row.set_selected(position == index)
        if not self.selected:
            self._set_status(f"{self.current_spec.label}: now pick photos in "
                             f"the middle.")
            return
        self.run_detection()

    # ------------------------------------------------------------------ #
    # Card 2 behaviour
    # ------------------------------------------------------------------ #

    @staticmethod
    def _images_in(folder: Path) -> List[Path]:
        try:
            return sorted(p for p in folder.iterdir()
                          if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        except OSError:
            return []

    def _autoload_photos(self) -> None:
        """Fill the contact sheet from the project's own photo folder.

        Nothing is selected, unlike the explicit *Open folder*. Starting a
        session with every photo ticked means one stray click on a defect
        launches a run over the whole folder, and a run cannot be cancelled
        once it starts. The operator has expressed no intent yet at startup,
        so the sheet waits for them to pick.
        """
        if self.pool:
            return
        found = self._images_in(PHOTO_DIR)
        if not found:
            return
        self.pool = found
        self.selected = set()
        self._rebuild_sheet()
        self._set_status(f"{len(found)} photo(s) from {PHOTO_DIR.name}/  ·  "
                         f"pick the ones to inspect, then a defect.")

    def open_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose a folder of glove photos",
                                         initialdir=str(self._browse_dir))
        if not folder:
            return
        self._browse_dir = Path(folder)
        found = self._images_in(Path(folder))
        if not found:
            messagebox.showinfo("No photos", "That folder has no images in it.")
            return
        self.pool = found
        self.selected = set(range(len(found)))
        self._rebuild_sheet()
        self._set_status(f"Loaded {len(found)} photo(s) from {folder}")

    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Choose glove photos",
            initialdir=str(self._browse_dir),
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                       ("All files", "*.*")])
        if not paths:
            return
        self._browse_dir = Path(paths[0]).parent
        known = set(self.pool)
        for raw in paths:
            path = Path(raw)
            if path not in known:
                self.selected.add(len(self.pool))
                self.pool.append(path)
                known.add(path)
        self._rebuild_sheet()

    def clear_pool(self) -> None:
        self.pool, self.selected = [], set()
        self._rebuild_sheet()

    def select_all(self) -> None:
        self.selected = set(range(len(self.pool)))
        self._sync_tiles()

    def select_none(self) -> None:
        self.selected = set()
        self._sync_tiles()

    def selected_paths(self) -> List[Path]:
        return [self.pool[i] for i in sorted(self.selected)]

    def _rebuild_sheet(self) -> None:
        for tile in self.tiles:
            tile.destroy()
        self.tiles = []
        self._columns = 0
        self.sheet_empty.pack_forget()

        if not self.pool:
            self.sheet_empty.pack(pady=px(40))
            self._update_count()
            return

        box = px(TILE)
        for index, path in enumerate(self.pool):
            tile = PhotoTile(self.sheet.inner, index, path.name, box,
                             self._on_tile_click)
            self.tiles.append(tile)
        self._layout_sheet(force=True)
        self._sync_tiles()
        self.sheet.scroll_to_top()

        # Thumbnails are decoded off the UI thread; the tiles show their grey
        # placeholder until each one arrives.
        self._thumb_token += 1
        threading.Thread(target=self._thumb_worker,
                         args=(list(self.pool), self._thumb_token),
                         daemon=True).start()

    def _thumb_worker(self, paths: List[Path], token: int) -> None:
        for index, path in enumerate(paths):
            if token != self._thumb_token:
                return
            image = imread_unicode(path, reduced=True)
            if image is not None:
                self._queue.put(("thumb", token, (index, image)))

    def _layout_sheet(self, force: bool = False) -> None:
        if not self.tiles:
            return
        width = self.sheet.canvas.winfo_width()
        step = px(TILE) + px(TILE_GAP)
        columns = max(1, (width - px(4)) // step)
        if columns == self._columns and not force:
            return
        self._columns = columns
        for position, tile in enumerate(self.tiles):
            tile.grid(row=position // columns, column=position % columns,
                      padx=(0, px(TILE_GAP)), pady=(0, px(TILE_GAP)))

    def _on_sheet_resize(self, _event=None) -> None:
        self._layout_sheet()

    def _on_tile_click(self, index: int, event) -> None:
        shift = bool(getattr(event, "state", 0) & 0x0001)
        if shift and self.tiles:
            low, high = sorted((self._anchor, index))
            self.selected |= set(range(low, high + 1))
        else:
            self.selected ^= {index}     # plain click toggles one photo
        self._anchor = index
        self._sync_tiles()

    def _sync_tiles(self) -> None:
        for index, tile in enumerate(self.tiles):
            tile.set_selected(index in self.selected)
        self._update_count()

    def _update_count(self) -> None:
        if not self.pool:
            self.count_label.configure(text="none loaded", fg=theme.INK_FAINT)
            return
        chosen = len(self.selected)
        text = f"{chosen} of {len(self.pool)} selected"
        if chosen < RECOMMENDED_MIN_IMAGES:
            self.count_label.configure(
                text=f"{text}  ·  pick at least {RECOMMENDED_MIN_IMAGES}",
                fg=theme.BAD)
        else:
            self.count_label.configure(text=text, fg=theme.ACCENT)

    # ------------------------------------------------------------------ #
    # Running
    # ------------------------------------------------------------------ #

    def run_detection(self) -> None:
        if self._running:
            self._set_status("A detection run is already in progress.")
            return
        if self.current_spec is None:
            self._set_status("Pick a defect on the left first.")
            return
        paths = self.selected_paths()
        if not paths:
            self._set_status("Pick at least one photo in the middle.")
            return

        spec = self.current_spec
        self._clear_results()
        self._running = True
        self._run_token += 1
        self.run_button.set_text("Running…")
        self.run_button.set_enabled(False)
        self.save_button.set_enabled(False)
        self.summary_title.configure(text=f"{spec.label}", fg=theme.INK)
        self.summary_detail.configure(text=f"inspecting {len(paths)} photo(s)…")

        threading.Thread(target=self._worker,
                         args=(spec, paths, self._run_token),
                         daemon=True).start()

    def _worker(self, spec: DefectSpec, paths: List[Path], token: int) -> None:
        """Runs off the UI thread; talks back through `self._queue`."""
        try:
            module = spec.load()
            inspector = GloveInspector(include_builtin_detectors=False)
            inspector.register_detector(spec.key, module.detect)
        except Exception:
            self._queue.put(("fatal", token, traceback.format_exc()))
            return

        for index, path in enumerate(paths, start=1):
            self._queue.put(("progress", token, (index, len(paths), path.name)))
            image = imread_unicode(path)
            if image is None:
                self._queue.put(("row", token, Row(
                    name=path.name, status="ERROR", score=0.0,
                    evidence="could not read this file")))
                continue
            try:
                report = inspector.inspect(image, image_name=path.name)
                result = report.results.get(spec.key)
                annotated = inspector.annotate(report)
            except Exception as exc:
                self._queue.put(("row", token, Row(
                    name=path.name, status="ERROR", score=0.0,
                    evidence=f"{type(exc).__name__}: {exc}")))
                continue

            if not report.segmentation_ok or result is None:
                row = Row(name=path.name, status="SEG-FAIL", score=0.0,
                          evidence="the glove could not be separated from the "
                                   "background",
                          annotated=annotated, warnings=report.warnings)
            elif not spec.implemented:
                row = Row(name=path.name, status="PENDING", score=0.0,
                          evidence=result.details, annotated=annotated,
                          warnings=report.warnings)
            else:
                # The status is the detector's actual answer. A capture
                # warning no longer replaces it — it rides along in
                # row.warnings and is shown beside the result, because a
                # warning that fires on most photos would otherwise hide
                # every correct detection behind it.
                status = "MATCH" if result.defect_found else "NO MATCH"
                row = Row(name=path.name, status=status, score=result.score,
                          evidence=result.details, annotated=annotated,
                          warnings=report.warnings)
            self._queue.put(("row", token, row))

        self._queue.put(("done", token, spec))

    def _drain_queue(self) -> None:
        while True:
            try:
                kind, token, payload = self._queue.get_nowait()
            except queue.Empty:
                break
            if kind == "thumb":
                if token == self._thumb_token:
                    index, image = payload
                    if index < len(self.tiles):
                        self.tiles[index].set_image(image)
                        self.tiles[index].set_selected(index in self.selected)
                continue
            if token != self._run_token:
                continue        # a stale run the user has already replaced
            if kind == "progress":
                index, total, name = payload
                self._set_status(f"[{index}/{total}]  {name}")
            elif kind == "row":
                self._add_row(payload)
            elif kind == "done":
                self._finish_run(payload)
            elif kind == "fatal":
                self._running = False
                self.run_button.set_text("Run detection")
                self.run_button.set_enabled(True)
                messagebox.showerror("Detector failed to load", payload)
        self.root.after(50, self._drain_queue)

    def _add_row(self, row: Row) -> None:
        self.result_empty.pack_forget()
        index = len(self.rows)
        self.rows.append(row)
        widget = ResultRow(self.result_scroll.inner, index, row.name,
                           row.status, row.score, row.evidence, row.annotated,
                           self._select_row)
        widget.pack(fill="x")
        Divider(self.result_scroll.inner, theme.LINE_SOFT).pack(fill="x")
        self.row_widgets.append(widget)
        self.result_scroll.configure(
            height=min(len(self.rows) * px(53) + px(6), px(340)))
        if index == 0:
            self._select_row(0)

    def _finish_run(self, spec: DefectSpec) -> None:
        self._running = False
        self.run_button.set_text("Run detection")
        self.run_button.set_enabled(True)

        matched = sum(1 for r in self.rows if r.status == "MATCH")
        flagged = sum(1 for r in self.rows
                      if r.warnings or r.status in ("SEG-FAIL", "ERROR"))
        total = len(self.rows)

        if not spec.implemented:
            self.summary_title.configure(text=f"{spec.label} · not written yet",
                                         fg=theme.MUTED)
            self.summary_detail.configure(
                text=f"detectors/{spec.key}.py is still a placeholder, so "
                     f"these {total} photo(s) were not judged.")
        else:
            self.summary_title.configure(
                text=f"{matched} of {total} photo(s) matched {spec.label}",
                fg=theme.BAD if matched else theme.OK)
            detail = f"detectors/{spec.key}.py"
            if flagged:
                detail += f"  ·  {flagged} with a capture warning"
            self.summary_detail.configure(text=detail)

        self.save_button.set_enabled(
            any(r.annotated is not None for r in self.rows))
        self._set_status(f"Done. {total} photo(s) inspected.")

    def _clear_results(self) -> None:
        for child in self.result_scroll.inner.winfo_children():
            child.destroy()
        self.rows, self.row_widgets, self.active_row = [], [], -1
        self.result_empty = tk.Label(self.result_scroll.inner,
                                     text="Inspecting…", font=theme.font(9),
                                     fg=theme.INK_FAINT, bg=theme.CARD)
        self.result_empty.pack(pady=px(28))
        self.preview_image = None
        self.preview.configure(image="", text="")
        self._preview_photo = None
        self.caption.set_text("")

    # ------------------------------------------------------------------ #
    # Card 3 behaviour
    # ------------------------------------------------------------------ #

    def _select_row(self, index: int) -> None:
        if not 0 <= index < len(self.rows):
            return
        self.active_row = index
        for position, widget in enumerate(self.row_widgets):
            widget.set_selected(position == index)

        row = self.rows[index]
        self.preview_image = row.annotated
        self._render_preview()
        if row.warnings:
            self.caption.set_style(fg=theme.WARN)
            self.caption.set_text("!  " + "; ".join(row.warnings))
        else:
            self.caption.set_style(fg=theme.INK_FAINT)
            self.caption.set_text(row.evidence.replace("\n", " "))

    def _render_preview(self) -> None:
        if self.preview_image is None:
            # Reached on a cold start too, because the first <Configure> of
            # the stage fires before anything has been inspected.
            self.preview.configure(image="", text=PREVIEW_PLACEHOLDER)
            return
        self.stage.update_idletasks()
        width = max(self.stage.winfo_width() - px(20), px(200))
        height = max(self.stage.winfo_height() - px(20), px(160))
        self._preview_photo = theme.photo_rounded(self.preview_image, width,
                                                  height, px(6))
        self.preview.configure(image=self._preview_photo, text="")

    def _on_stage_resize(self, _event=None) -> None:
        # Redrawing on every pixel of a window drag is wasteful, so coalesce.
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(140, self._render_preview)

    def save_results(self) -> None:
        rows = [r for r in self.rows if r.annotated is not None]
        if not rows:
            return
        folder = filedialog.askdirectory(title="Save annotated photos into",
                                         initialdir=str(self._save_dir))
        if not folder:
            return
        target = Path(folder)
        self._save_dir = target
        written = 0
        for row in rows:
            name = f"{Path(row.name).stem}_{row.status.replace(' ', '-')}.png"
            if imwrite_unicode(target / name, row.annotated):
                written += 1
        self._set_status(f"Saved {written} annotated photo(s) to {target}")

    # ------------------------------------------------------------------ #

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)


def main() -> int:
    theme.enable_hidpi()
    root = tk.Tk()
    theme.apply_scaling(root)
    DefectApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
