from __future__ import annotations
import time
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

@dataclass
class DefectResult:
    defect_found: bool
    defect_type: str
    locations: List[Any] = field(default_factory=list)
    score: float = 0.0
    details: str = ""

def assess_capture(seg) -> List[str]:
    warnings: List[str] = []
    h, w = seg.mask.shape
    border_px = np.concatenate([seg.mask[0, :], seg.mask[-1, :], seg.mask[:, 0], seg.mask[:, -1]])
    border_contact = float(np.count_nonzero(border_px)) / float(border_px.size)
    if border_contact > 0.02:
        warnings.append(f"glove runs off the frame edge ({border_contact:.0%} of the border); part of the outline was not photographed")
    return warnings

@dataclass
class InspectionReport:
    image_name: str
    segmentation_ok: bool
    segmentation: Optional[Any]
    results: Dict[str, DefectResult] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def any_defect(self) -> bool:
        return any(r.defect_found for r in self.results.values())

    @property
    def defect_types_found(self) -> List[str]:
        return [r.defect_type for r in self.results.values() if r.defect_found]

    def verdict(self) -> str:
        if not self.segmentation_ok:
            return "SEG-FAIL"
        return "DEFECT" if self.any_defect else "PASS"


class GloveInspector:
    def __init__(self, include_builtin_detectors: bool = True) -> None:
        self._modules: Dict[str, ModuleType] = {}
        self._last_normalized: Optional[np.ndarray] = None
        if include_builtin_detectors:
            self.register_implemented_defects()

    def register_module(self, name: str, module: ModuleType) -> None:
        if not hasattr(module, "detect"):
            raise TypeError(f"detector module {name!r} has no detect()")
        self._modules[name] = module

    register_detector = register_module

    def register_implemented_defects(self) -> None:
        from detectors import DEFECTS

        for spec in DEFECTS:
            if spec.implemented:
                self.register_module(spec.key, spec.load())

    @property
    def detector_names(self) -> List[str]:
        return list(self._modules)

    def inspect(self, image: np.ndarray, image_name: str = "image") -> InspectionReport:
        start = time.perf_counter()
        report = InspectionReport(image_name=image_name, segmentation_ok=False, segmentation=None)
        need_view = True
        for name, module in self._modules.items():
            try:
                config = getattr(module, "Config", None)
                drivable = (config is not None and hasattr(module, "preprocess") and hasattr(module, "segment_glove"))
                if drivable:
                    cfg = config()
                    prepared = module.preprocess(image, cfg.preprocess)
                    segmentation = module.segment_glove(prepared, cfg.segmentation)
                    if segmentation is None:
                        report.results[name] = DefectResult(False, name, details="the glove could not be separated from the background")
                        continue
                    result = module.detect(prepared, segmentation, cfg)
                else:
                    result = module.detect(image.copy())
                    prepared, segmentation = image.copy(), None
                    mask = getattr(result, "analysis_mask", None)

                    if (
                            isinstance(mask, np.ndarray)
                            and mask.size > 0
                            and mask.ndim in (2, 3)
                        ):
                        if mask.ndim == 3:
                            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
                        contours, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )

                        if contours:
                            contour = max(contours, key=cv2.contourArea)
                            segmentation = SimpleNamespace(
                                mask=mask,
                                mask_raw=mask.copy(),
                                contour=contour,
                                cue="detector_owned",
                            )
                            prepared = cv2.resize(
                                image,
                                (mask.shape[1], mask.shape[0]),
                                interpolation=cv2.INTER_AREA,
                            )
                report.results[name] = result
                if need_view and segmentation is not None:
                    report.segmentation_ok = True
                    report.segmentation = segmentation
                    report.warnings = assess_capture(segmentation)
                    self._last_normalized = prepared
                    need_view = False
            except Exception as exc:
                report.results[name] = DefectResult(
                    False, name, details=f"detector error: {exc}")
        report.elapsed_seconds = time.perf_counter() - start
        return report

    _COLORS = {
        "damage_by_fold": (0, 0, 255),
        "dirty": (0, 0, 255),
        "tearing_at_finger": (0, 0, 255),
    }

    _EXTRA_COLORS = [(255, 255, 0), (0, 255, 255), (128, 0, 255), (0, 128, 0)]

    _VERDICT_COLORS = {"PASS": (0, 170, 0), "DEFECT": (0, 0, 220), "SEG-FAIL": (60, 60, 60),}

    def _detector_color(self, defect_type: str, extra_index: int) -> tuple:
        color = self._COLORS.get(defect_type)
        if color is None:
            color = self._EXTRA_COLORS[extra_index % len(self._EXTRA_COLORS)]
        return color

    def annotate(self, report: InspectionReport, image: Optional[np.ndarray] = None) -> np.ndarray:
        canvas = (image if image is not None else self._last_normalized).copy()

        if not report.segmentation_ok or report.segmentation is None:
            cv2.putText(canvas, "SEGMENTATION FAILED", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return canvas

        cv2.drawContours(canvas, [report.segmentation.contour], -1, (0, 255, 0), 2)

        index = 0
        for extra_i, (name, result) in enumerate(report.results.items()):
            if not result.defect_found:
                continue
            color = self._detector_color(result.defect_type, extra_i)
            for (x, y, w, h) in result.locations:
                index += 1
                cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
                tag = str(index)
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                tx, ty = x, max(y - 4, th + 4)
                cv2.rectangle(canvas, (tx, ty - th - 4), (tx + tw + 8, ty + 4), color, cv2.FILLED)
                cv2.putText(canvas, tag, (tx + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return canvas

    def render_report(self, report: InspectionReport, original: Optional[np.ndarray] = None) -> np.ndarray:
        annotated = self.annotate(report)
        h, w = annotated.shape[:2]
        panels, titles = [], []
        if original is not None:
            panels.append(cv2.resize(original, (w, h), interpolation=cv2.INTER_AREA))
            titles.append("1. original")
        if report.segmentation is not None:
            panels.append(cv2.cvtColor(report.segmentation.mask_raw, cv2.COLOR_GRAY2BGR))
            titles.append(f"2. glove mask  [cue: {report.segmentation.cue}]")
        panels.append(annotated)
        titles.append("3. detection")
        labelled = []
        for panel, title in zip(panels, titles):
            strip = np.full((28, w, 3), 245, np.uint8)
            cv2.putText(strip, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
            labelled.append(np.vstack([strip, panel]))
        row = np.hstack(labelled)
        lines: List[tuple] = []
        verdict = report.verdict()
        lines.append((f"{report.image_name}   ->   {verdict}", self._VERDICT_COLORS.get(verdict, (0, 0, 0)), 0.75))
        for warning in report.warnings:
            lines.append((f"!  {warning}", (0, 140, 220), 0.5))
        index = 0
        for extra_i, (name, result) in enumerate(report.results.items()):
            color = self._detector_color(result.defect_type, extra_i)
            if result.defect_found:
                nums = ", ".join(str(index + i + 1) for i in range(len(result.locations)))
                index += len(result.locations)
                head = f"[X] {name}: FOUND"
                if nums:
                    head += f"  (box {nums})"
                head += f"  score={result.score:.2f}"
                lines.append((head, color, 0.55))
            else:
                lines.append((f"[ ] {name}: clear", (120, 120, 120), 0.55))
            lines.append((f"      {result.details}", (90, 90, 90), 0.45))
        legend_h = 18 + sum(int(26 * scale / 0.55) for _, _, scale in lines)
        legend = np.full((max(legend_h, 60), row.shape[1], 3), 250, np.uint8)
        y = 26
        for text, color, scale in lines:
            cv2.putText(legend, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if scale >= 0.7 else 1, cv2.LINE_AA)
            y += int(26 * scale / 0.55)
        return np.vstack([row, legend])
