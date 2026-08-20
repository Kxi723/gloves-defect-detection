"""Glove Defect Detection UI"""

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
from ui.compare import ComparePage
from ui.theme import px
from ui.widgets import (Button, Card, ClipLabel, DefectRow, Divider, PhotoTile, ResultList, ScrollFrame)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RECOMMENDED_MIN_IMAGES = 1
TILE = 104
TILE_GAP = 10
ROW_H = 52
PREVIEW_PLACEHOLDER = ("Select a row to preview it here\nclick to compare it with the original")
PROJECT_ROOT = Path(__file__).resolve().parent
PHOTO_DIR = PROJECT_ROOT / "gloves"
OUTPUT_DIR = PROJECT_ROOT / "output"
_REDUCED_FLAGS = {2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8,}

def imread_unicode(path: Path, reduce: int = 1) -> Optional[np.ndarray]:
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    flag = _REDUCED_FLAGS.get(reduce, cv2.IMREAD_COLOR)
    image = cv2.imdecode(buffer, flag)
    if image is None and flag != cv2.IMREAD_COLOR:
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image

def side_by_side(original: np.ndarray, annotated: np.ndarray, name: str = "") -> np.ndarray:
    height = max(annotated.shape[0], 480)
    def fit(image: np.ndarray) -> np.ndarray:
        scale = height / image.shape[0]
        width = max(int(round(image.shape[1] * scale)), 1)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    left, right = fit(original), fit(annotated)
    gap, strip = 16, 34
    canvas = np.full((height + strip, left.shape[1] + gap + right.shape[1], 3), 255, np.uint8)
    canvas[strip:, :left.shape[1]] = left
    canvas[strip:, left.shape[1] + gap:] = right
    label = f"ORIGINAL   {name}".strip()
    cv2.putText(canvas, label, (4, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "DETECTED", (left.shape[1] + gap + 4, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
    return canvas

def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    ok, buffer = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        return False
    try:
        buffer.tofile(str(path))
    except OSError:
        return False
    return True

@dataclass
class Row:
    name: str
    status: str
    score: float
    evidence: str
    annotated: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)
    path: Optional[Path] = None


class DefectApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Glove Defect Detection")
        width = min(px(1360), self.root.winfo_screenwidth() - px(80))
        height = min(px(830), self.root.winfo_screenheight() - px(120))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(px(1120), px(700))
        self.root.configure(bg=theme.APP_BG)
        self._browse_dir = PHOTO_DIR if PHOTO_DIR.is_dir() else PROJECT_ROOT
        self._save_dir = OUTPUT_DIR if OUTPUT_DIR.is_dir() else PROJECT_ROOT
        self.pool: List[Path] = []
        self.tiles: List[PhotoTile] = []
        self.selected: Set[int] = set()
        self._anchor = 0
        self.rows: List[Row] = []
        self.active_row = -1
        self.current_spec: Optional[DefectSpec] = None
        self._page = "inspect"
        self._original_cache: "dict[Path, np.ndarray]" = {}
        self.preview_image: Optional[np.ndarray] = None
        self._preview_photo = None
        self._preview_key: Optional[tuple] = None
        self._resize_job: Optional[str] = None
        self._sheet_job: Optional[str] = None
        self._columns = 0
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._run_token = 0
        self._thumb_token = 0
        self._running = False
        self._build_header()
        self._build_body()
        self._build_statusbar()
        self.root.after(50, self._drain_queue)
        self.root.after(80, self._autoload_photos)

    def _build_header(self) -> None:
        head = tk.Frame(self.root, bg=theme.HEADER_BG, height=px(54))
        head.pack(fill="x")
        head.pack_propagate(False)
        left = tk.Frame(head, bg=theme.HEADER_BG)
        left.pack(side="left", padx=(px(20), 0))
        tk.Label(left, text="Glove Defect Detection", font=theme.font(13, strong=True), fg=theme.HEADER_FG, bg=theme.HEADER_BG).pack(anchor="w")
        tk.Label(left, text="classical image processing  ·  CT036-3-IPPR", font=theme.font(8), fg=theme.HEADER_DIM, bg=theme.HEADER_BG).pack(anchor="w")
        ready = sum(1 for spec in DEFECTS if spec.implemented)
        tk.Label(head, text=f"{ready} of {len(DEFECTS)} detectors live", font=theme.font(9), fg=theme.HEADER_DIM, bg=theme.HEADER_BG).pack(side="right", padx=(0, px(20)))

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self.root, bg=theme.APP_BG, height=px(26))
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status = tk.Label(bar, text="Pick a defect, then load photos.", font=theme.font(9), fg=theme.INK_FAINT, bg=theme.APP_BG, anchor="w")
        self.status.pack(side="left", padx=px(20))

    def _build_body(self) -> None:
        pages = tk.Frame(self.root, bg=theme.APP_BG)
        pages.pack(fill="both", expand=True)
        pages.rowconfigure(0, weight=1)
        pages.columnconfigure(0, weight=1)
        self.inspect_page = tk.Frame(pages, bg=theme.APP_BG)
        self.inspect_page.grid(row=0, column=0, sticky="nsew")
        self.compare_page = ComparePage(pages, on_back=self.show_inspect, load_original=self._original_for, on_save=self.save_comparison)
        self.compare_page.grid(row=0, column=0, sticky="nsew")
        self.compare_page.grid_remove()  # not tkraise: a raised page is still laid out on every resize
        self.root.bind("<Escape>", lambda _e: self._page == "compare" and self.show_inspect())
        self.root.bind("<Left>", lambda _e: self._compare_step(-1))
        self.root.bind("<Right>", lambda _e: self._compare_step(1))
        self._build_inspect_page(self.inspect_page)

    def _build_inspect_page(self, page: tk.Frame) -> None:
        body = tk.Frame(page, bg=theme.APP_BG)
        body.pack(fill="both", expand=True, padx=px(16), pady=(px(14), px(6)))
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0, minsize=px(232))
        body.columnconfigure(1, weight=4, minsize=px(372))
        body.columnconfigure(2, weight=5, minsize=px(470))
        self._build_defect_card(body)
        self._build_photo_card(body)
        self._build_result_card(body)

    def _build_defect_card(self, parent: tk.Frame) -> None:
        card = Card(parent)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, px(12)))
        card.add_header("1", "Defect")
        scroll = ScrollFrame(card.body)
        scroll.pack(fill="both", expand=True, pady=(px(6), 0))
        self.defect_rows: List[DefectRow] = []
        for index, spec in enumerate(DEFECTS):
            row = DefectRow(scroll.inner, index, spec.label, not spec.implemented, self._on_defect_click)
            row.pack(fill="x")
            self.defect_rows.append(row)
        footer = tk.Frame(card.body, bg=theme.CARD)
        footer.pack(fill="x", padx=px(14), pady=(px(8), px(12)))
        tk.Label(footer, text="Clicking a defect runs its own\nfile in detectors/.", font=theme.font(8), fg=theme.INK_FAINT, bg=theme.CARD, justify="left").pack(anchor="w")

    def _build_photo_card(self, parent: tk.Frame) -> None:
        card = Card(parent)
        card.grid(row=0, column=1, sticky="nsew", padx=(0, px(12)))
        head = card.add_header("2", "Photos")
        self.count_label = tk.Label(head, text="none loaded", font=theme.font(9), fg=theme.INK_FAINT, bg=theme.CARD)
        self.count_label.pack(side="right") 
        tools = tk.Frame(card.body, bg=theme.CARD)
        tools.pack(fill="x", padx=px(14), pady=(px(12), px(10)))
        Button(tools, "Open folder", self.open_folder, variant="secondary").pack(side="left")
        Button(tools, "Add files", self.add_files, variant="ghost").pack(side="left", padx=px(4))
        Button(tools, "Clear", self.clear_pool, variant="ghost").pack(side="right") 
        picks = tk.Frame(card.body, bg=theme.CARD)
        picks.pack(fill="x", padx=px(14), pady=(0, px(8)))
        Button(picks, "Select all", self.select_all, variant="ghost", pad=8, height=24, size=8).pack(side="left")
        Button(picks, "Select none", self.select_none, variant="ghost", pad=8, height=24, size=8).pack(side="left", padx=px(4))
        tk.Label(picks, text="click to pick  ·  shift-click for a range", font=theme.font(8), fg=theme.INK_FAINT, bg=theme.CARD).pack(side="right")
        Divider(card.body, theme.LINE_SOFT).pack(fill="x")
        self.sheet = ScrollFrame(card.body, padding=4)
        self.sheet.pack(fill="both", expand=True, padx=(px(12), px(4)), pady=px(10))
        self.sheet.canvas.bind("<Configure>", self._on_sheet_resize, add="+")
        self.sheet_empty = tk.Label(self.sheet.inner, text="No photos yet.\nOpen a folder to begin.", font=theme.font(9), fg=theme.INK_FAINT, bg=theme.CARD, justify="center")
        self.sheet_empty.pack(pady=px(40))
        footer = tk.Frame(card.body, bg=theme.CARD)
        footer.pack(fill="x", padx=px(14), pady=(0, px(14)))
        holder = tk.Frame(footer, bg=theme.CARD, width=1, height=px(34))
        holder.pack_propagate(False)
        holder.pack(fill="x")
        self.run_button = Button(holder, "Run detection", self.run_detection, variant="primary", stretch=True, height=34, size=10)
        self.run_button.pack(fill="both", expand=True) 

    def _build_result_card(self, parent: tk.Frame) -> None:
        card = Card(parent)
        card.grid(row=0, column=2, sticky="nsew")
        head = card.add_header("3", "Result")
        self.save_button = Button(head, "Save annotated", self.save_results, variant="ghost", pad=10, height=24, size=8)
        self.save_button.pack(side="right")
        self.save_button.set_enabled(False)
        self.compare_button = Button(head, "Compare with original", self.open_compare, variant="secondary", pad=10, height=24, size=8)
        self.compare_button.pack(side="right", padx=(0, px(8)))
        self.compare_button.set_enabled(False)
        body = card.body
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=0)
        body.rowconfigure(4, weight=1)
        summary = tk.Frame(body, bg=theme.CARD)
        summary.grid(row=0, column=0, sticky="ew", padx=px(16), pady=(px(14), px(12)))
        self.summary_title = tk.Label(summary, text="Nothing inspected yet", font=theme.font(15, strong=True), fg=theme.INK, bg=theme.CARD, anchor="w")
        self.summary_title.pack(fill="x")
        self.summary_detail = tk.Label( summary, text="Choose a defect on the left and photos in the middle, then run.", font=theme.font(9), fg=theme.INK_SOFT, bg=theme.CARD, anchor="w")
        self.summary_detail.pack(fill="x", pady=(px(3), 0))
        Divider(body, theme.LINE_SOFT).grid(row=1, column=0, sticky="ew")
        self.result_list = ResultList(body, self._select_row, row_height=ROW_H, on_activate=self.open_compare)
        self.result_list.pack_propagate(False)
        self.result_list.configure(height=px(96))
        self.result_list.grid(row=2, column=0, sticky="nsew", padx=(px(4), px(4)), pady=px(4))
        self.result_list.clear("Results appear here, one row per photo.")
        Divider(body, theme.LINE_SOFT).grid(row=3, column=0, sticky="ew")
        stage = tk.Frame(body, bg=theme.CARD)
        stage.grid(row=4, column=0, sticky="nsew", padx=px(14), pady=(px(12), px(6)))
        stage.rowconfigure(0, weight=1)
        stage.columnconfigure(0, weight=1)
        self.stage = tk.Frame(stage, bg=theme.STAGE)
        self.stage.grid(row=0, column=0, sticky="nsew")
        self.preview = tk.Label(self.stage, bg=theme.STAGE, text=PREVIEW_PLACEHOLDER, font=theme.font(9), fg=theme.INK_FAINT, bd=0)
        self.preview.place(relx=0.5, rely=0.5, anchor="center")
        self.stage.bind("<Configure>", self._on_stage_resize)
        for widget in (self.stage, self.preview):
            widget.bind("<Button-1>", lambda _e: self.open_compare())
        self.caption = ClipLabel(body, "", font=theme.font(8), fg=theme.INK_FAINT, bg=theme.CARD)
        self.caption.grid(row=5, column=0, sticky="ew", padx=px(16), pady=(px(4), px(12)))

    def _on_defect_click(self, index: int) -> None:
        self.current_spec = DEFECTS[index]
        for position, row in enumerate(self.defect_rows):
            row.set_selected(position == index)
        if not self.selected:
            self._set_status(f"{self.current_spec.label}: now pick photos in the middle.")
            return
        self.run_detection()

    @staticmethod
    def _images_in(folder: Path) -> List[Path]:
        try:
            return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        except OSError:
            return []

    def _autoload_photos(self) -> None:
        if self.pool:
            return
        found = self._images_in(PHOTO_DIR)
        if not found:
            return
        self.pool = found
        self.selected = set()
        self._rebuild_sheet()
        self._set_status(f"{len(found)} photo(s) from {PHOTO_DIR.name}/  ·  pick the ones to inspect, then a defect.")

    def open_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose a folder of glove photos", initialdir=str(self._browse_dir))
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
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"), ("All files", "*.*")])
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
            tile = PhotoTile(self.sheet.inner, index, path.name, box, self._on_tile_click)
            self.tiles.append(tile)
        self._layout_sheet(force=True)
        self._sync_tiles()
        self.sheet.scroll_to_top()
        self._thumb_token += 1
        threading.Thread(target=self._thumb_worker, args=(list(self.pool), self._thumb_token), daemon=True).start()

    def _thumb_worker(self, paths: List[Path], token: int) -> None:
        for index, path in enumerate(paths):
            if token != self._thumb_token:
                return
            image = imread_unicode(path, reduce=8)
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
            tile.grid(row=position // columns, column=position % columns, padx=(0, px(TILE_GAP)), pady=(0, px(TILE_GAP)))

    def _on_sheet_resize(self, _event=None) -> None:
        if self._sheet_job is not None:
            self.root.after_cancel(self._sheet_job)
        self._sheet_job = self.root.after(120, self._relayout_sheet)

    def _relayout_sheet(self) -> None:
        self._sheet_job = None
        self._layout_sheet()

    def _on_tile_click(self, index: int, event) -> None:
        shift = bool(getattr(event, "state", 0) & 0x0001)
        if shift and self.tiles:
            low, high = sorted((self._anchor, index))
            self.selected |= set(range(low, high + 1))
        else:
            self.selected ^= {index}
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
            self.count_label.configure(text=f"{text}  ·  pick at least {RECOMMENDED_MIN_IMAGES}", fg=theme.BAD)
        else:
            self.count_label.configure(text=text, fg=theme.ACCENT)

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
        self.compare_button.set_enabled(False)
        self.summary_title.configure(text=f"{spec.label}", fg=theme.INK)
        self.summary_detail.configure(text=f"inspecting {len(paths)} photo(s)…")
        threading.Thread(target=self._worker, args=(spec, paths, self._run_token), daemon=True).start()

    def _worker(self, spec: DefectSpec, paths: List[Path], token: int) -> None:
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
                self._queue.put(("row", token, Row(name=path.name, status="ERROR", score=0.0, evidence="could not read this file", path=path)))
                continue
            try:
                report = inspector.inspect(image, image_name=path.name)
                result = report.results.get(spec.key)
                annotated = inspector.annotate(report)
            except Exception as exc:
                self._queue.put(("row", token, Row(name=path.name, status="ERROR", score=0.0, evidence=f"{type(exc).__name__}: {exc}", path=path)))
                continue
            if not report.segmentation_ok or result is None:
                row = Row(name=path.name, status="SEG-FAIL", score=0.0, evidence="the glove could not be separated from the background", annotated=annotated, warnings=report.warnings, path=path)
            elif not spec.implemented:
                row = Row(name=path.name, status="PENDING", score=0.0, evidence=result.details, annotated=annotated, warnings=report.warnings, path=path)
            else:
                status = "MATCH" if result.defect_found else "NO MATCH"
                row = Row(name=path.name, status=status, score=result.score, evidence=result.details, annotated=annotated, warnings=report.warnings, path=path)
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
                continue
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
        index = len(self.rows)
        self.rows.append(row)
        self.result_list.append(row.name, row.status, row.score, row.evidence, row.annotated)
        self.result_list.configure(height=min(len(self.rows) * px(ROW_H) + px(6), px(340)))
        if index == 0:
            self._select_row(0)

    def _finish_run(self, spec: DefectSpec) -> None:
        self._running = False
        self.run_button.set_text("Run detection")
        self.run_button.set_enabled(True)
        matched = sum(1 for r in self.rows if r.status == "MATCH")
        flagged = sum(1 for r in self.rows if r.warnings or r.status in ("SEG-FAIL", "ERROR"))
        total = len(self.rows)
        if not spec.implemented:
            self.summary_title.configure(text=f"{spec.label} · not written yet", fg=theme.MUTED)
            self.summary_detail.configure(text=f"detectors/{spec.key}.py is still a placeholder, so these {total} photo(s) were not judged.")
        else:
            self.summary_title.configure(text=f"{matched} of {total} photo(s) matched {spec.label}", fg=theme.BAD if matched else theme.OK)
            detail = f"detectors/{spec.key}.py"
            if flagged:
                detail += f"  ·  {flagged} with a capture warning"
            self.summary_detail.configure(text=detail)
        self.save_button.set_enabled(any(r.annotated is not None for r in self.rows))
        self.compare_button.set_enabled(bool(self.rows))
        self._set_status(f"Done. {total} photo(s) inspected.  Double-click a row to compare it with the original.")

    def _clear_results(self) -> None:
        self.result_list.clear("Inspecting…")
        self.result_list.configure(height=px(96))
        self.rows, self.active_row = [], -1
        self.preview_image = None
        self.preview.configure(image="", text="")
        self._preview_photo = None
        self.caption.set_text("")

    def _select_row(self, index: int) -> None:
        if not 0 <= index < len(self.rows):
            return
        self.active_row = index
        self.result_list.select(index)
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
            self.preview.configure(image="", text=PREVIEW_PLACEHOLDER)
            self._preview_key = None
            return
        width = max(self.stage.winfo_width() - px(20), px(200))
        height = max(self.stage.winfo_height() - px(20), px(160))
        key = (self.active_row, width, height)
        if key == self._preview_key:
            return
        self._preview_key = key
        self._preview_photo = theme.photo_rounded(self.preview_image, width, height, px(6))
        self.preview.configure(image=self._preview_photo, text="")

    def _on_stage_resize(self, _event=None) -> None:
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(140, self._render_preview)

    def save_results(self) -> None:
        rows = [r for r in self.rows if r.annotated is not None]
        if not rows:
            return
        folder = filedialog.askdirectory(title="Save annotated photos into", initialdir=str(self._save_dir))
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

    def show_inspect(self) -> None:
        self.compare_page.grid_remove()
        self.inspect_page.grid()
        self._page = "inspect"
        self._set_status("Back to the run. Double-click a row to compare again.")

    def open_compare(self, index: Optional[int] = None) -> None:
        if not self.rows:
            self._set_status("Run a detection first.")
            return
        target = self.active_row if index is None else index
        self.inspect_page.grid_remove()
        self.compare_page.grid()
        self.compare_page.show(self.rows, max(target, 0))
        self._page = "compare"
        self._set_status("Left and Right move between photos  ·  Esc goes back")

    def _compare_step(self, delta: int) -> None:
        if self._page == "compare":
            self.compare_page.step(delta)
            self._select_row(self.compare_page.index)

    def _original_for(self, index: int) -> Optional[np.ndarray]:
        if not 0 <= index < len(self.rows):
            return None
        path = self.rows[index].path
        if path is None:
            return None
        image = self._original_cache.get(path)
        if image is None:
            image = imread_unicode(path, reduce=2)
            if image is None:
                return None
            self._original_cache[path] = image
            while len(self._original_cache) > 6:
                self._original_cache.pop(next(iter(self._original_cache)))
        return image

    def save_comparison(self, index: int) -> None:
        if not 0 <= index < len(self.rows):
            return
        row = self.rows[index]
        original = self._original_for(index)
        if row.annotated is None or original is None:
            self._set_status("There is no pair to save for this photo.")
            return
        target = filedialog.asksaveasfilename(title="Save this comparison", defaultextension=".png", initialdir=str(self._save_dir), initialfile=f"{Path(row.name).stem}_compare.png", filetypes=[("PNG image", "*.png")])
        if not target:
            return
        self._save_dir = Path(target).parent
        combined = side_by_side(original, row.annotated, row.name)
        if imwrite_unicode(Path(target), combined):
            self._set_status(f"Saved the comparison to {target}")
        else:
            self._set_status("Could not write that file.")

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
