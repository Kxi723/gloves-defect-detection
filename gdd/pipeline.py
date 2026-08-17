"""
Pipeline: chain preprocess -> segment -> defect detectors -> report.

`GloveInspector` is the single entry point for the GUI, the batch runner
and the evaluator alike. Detectors are plugins: any callable with the
shared signature

    detector(image, segmentation, config) -> DefectResult

can be registered under a name, so detectors written independently coexist
without any pipeline change.

Also provides `annotate()` and `render_report()`, which draw defect boxes,
labels and the report figure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from gdd.config import PipelineConfig, get_config
from gdd.preprocessing import preprocess
from gdd.segmentation import SegmentationResult, segment_glove
from gdd.features import DefectResult

# The plugin contract every detector must follow.
DetectorFn = Callable[[np.ndarray, SegmentationResult, PipelineConfig], DefectResult]


def assess_capture(
    image: np.ndarray, seg: SegmentationResult, config: PipelineConfig
) -> List[str]:
    """Note capture conditions that reduce confidence in the result.

    These are advisory only — see :meth:`InspectionReport.verdict` for why
    they no longer override the verdict.

    A silhouette running off the frame edge means part of the glove was
    never photographed, so anything shape based is working from an
    incomplete outline. It also tends to indicate a worn glove, since a
    worn glove has an arm leaving the frame, and a worn glove has
    legitimate deep concavities (finger gaps, wrist opening, thumb web)
    that are geometrically indistinguishable from a tear.

    Returns a list of human-readable warnings (empty means a clean capture).
    """
    warnings: List[str] = []
    h, w = seg.mask.shape

    # --- silhouette leaves the frame ------------------------------------- #
    border_px = np.concatenate([
        seg.mask[0, :], seg.mask[-1, :], seg.mask[:, 0], seg.mask[:, -1]
    ])
    border_contact = float(np.count_nonzero(border_px)) / float(border_px.size)
    if border_contact > 0.02:
        warnings.append(
            f"glove runs off the frame edge ({border_contact:.0%} of the "
            f"border); part of the outline was not photographed"
        )

    # A colour-based "is a bare hand inside?" test was tried here and
    # removed. Gray-world white balance desaturates skin when a large dark
    # glove dominates the frame, so the YCrCb/HSV skin rule scored 0% on
    # real worn-glove photos and 10% on synthetic photos containing no skin
    # at all — anti-correlated with the truth. The border test above is
    # purely geometric and catches the same photos, because a worn glove
    # has an arm leaving the frame.

    return warnings


@dataclass
class InspectionReport:
    """Aggregated result of inspecting one image."""

    image_name: str
    segmentation_ok: bool
    segmentation: Optional[SegmentationResult]
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
        """One-word quality verdict for summary tables.

        Capture warnings do NOT change this value. An earlier version
        returned "REVIEW" whenever :func:`assess_capture` flagged
        something, which hid the real answer: on the current photo set it
        fired on 13 of 15 images, including all five where the stain was
        detected correctly. A warning that fires on almost everything
        carries no information but still suppresses the result, so the
        warning is now reported alongside the verdict (see
        :attr:`warnings`) and the caller decides how much to trust it.
        """
        if not self.segmentation_ok:
            return "SEG-FAIL"
        return "DEFECT" if self.any_defect else "PASS"


class GloveInspector:
    """Runs the full inspection chain on one image at a time.

    Example (running one detector on one photo):

        inspector = GloveInspector(include_builtin_detectors=False)
        inspector.register_detector("dirty", dirty.detect)
        report = inspector.inspect(cv2.imread("glove.jpg"), "glove.jpg")
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        include_builtin_detectors: bool = True,
    ) -> None:
        self.config = config or get_config()
        self._detectors: Dict[str, DetectorFn] = {}
        if include_builtin_detectors:
            self.register_implemented_defects()

    def register_implemented_defects(self) -> None:
        """Register every defect marked implemented in ``detectors/``.

        The menu in ``detectors/__init__.py`` is the single source of truth
        for which detectors exist, so the GUI, the batch runner and the
        evaluator cannot drift apart. Imported lazily so that ``gdd`` does
        not depend on ``detectors`` at module load time — the GUI registers
        one detector at a time and never calls this.
        """
        from detectors import DEFECTS

        for spec in DEFECTS:
            if spec.implemented:
                self.register_detector(spec.key, spec.load().detect)

    # ------------------------------------------------------------------ #
    # Plugin management
    # ------------------------------------------------------------------ #

    def register_detector(self, name: str, detector: DetectorFn) -> None:
        """Add (or replace) a detector under a unique name."""
        if not callable(detector):
            raise TypeError(f"detector {name!r} is not callable")
        self._detectors[name] = detector

    @property
    def detector_names(self) -> List[str]:
        return list(self._detectors)

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #

    def inspect(self, image: np.ndarray, image_name: str = "image") -> InspectionReport:
        """Preprocess, segment, then run every registered detector.

        Segmentation and the detectors share one normalised image, so a
        box drawn by a detector lands on the pixels the segmenter saw. A
        detector that wants different normalisation applies it to the copy
        it is handed; see `detectors/damage_by_fold.py`.
        """
        start = time.perf_counter()

        normalized = preprocess(image, self.config.preprocess)
        segmentation = segment_glove(normalized, self.config.segmentation)

        report = InspectionReport(
            image_name=image_name,
            segmentation_ok=segmentation is not None,
            segmentation=segmentation,
        )

        if segmentation is not None:
            report.warnings = assess_capture(normalized, segmentation, self.config)
            for name, detector in self._detectors.items():
                try:
                    report.results[name] = detector(
                        normalized, segmentation, self.config
                    )
                except Exception as exc:  # a broken plugin must not kill the run
                    report.results[name] = DefectResult(
                        defect_found=False,
                        defect_type=name,
                        details=f"detector error: {exc}",
                    )

        report.elapsed_seconds = time.perf_counter() - start
        # Kept for annotation and the report figure.
        self._last_normalized = normalized

        return report

    # ------------------------------------------------------------------ #
    # Visualisation
    # ------------------------------------------------------------------ #

    # One fixed colour per defect type keeps report figures readable across
    # photos; a type with no entry here falls back to the extras below.
    # BGR. Fold is red because its box is usually read against the blue
    # latex coating, where the previous blue box nearly disappeared.
    # Tearing moved off pure red so that a figure showing both defects can
    # still tell them apart.
    _COLORS = {
        "damage_by_fold": (0, 0, 255),
        "dirty": (0, 0, 255),
        "tearing_at_finger": (0, 0, 255),
    }
    _EXTRA_COLORS = [(255, 255, 0), (0, 255, 255), (128, 0, 255), (0, 128, 0)]

    _VERDICT_COLORS = {
        "PASS": (0, 170, 0),
        "DEFECT": (0, 0, 220),
        "SEG-FAIL": (60, 60, 60),
    }

    def _detector_color(self, defect_type: str, extra_index: int) -> tuple:
        color = self._COLORS.get(defect_type)
        if color is None:
            color = self._EXTRA_COLORS[extra_index % len(self._EXTRA_COLORS)]
        return color

    def annotate(self, report: InspectionReport,
                 image: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw glove outline, numbered defect boxes and labels.

        Boxes are numbered rather than each carrying a long caption; the
        matching numbers appear in the legend built by
        :meth:`render_report`, which keeps the picture readable when many
        findings overlap.

        Args:
            report: The report returned by :meth:`inspect`.
            image:  Image to draw on; defaults to the preprocessed image of
                    the most recent :meth:`inspect` call.

        Returns:
            Annotated BGR image (a copy — the input is not modified).
        """
        canvas = (image if image is not None else self._last_normalized).copy()

        if not report.segmentation_ok or report.segmentation is None:
            cv2.putText(canvas, "SEGMENTATION FAILED", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return canvas

        # Glove outline (green) so figures show what was actually segmented.
        cv2.drawContours(canvas, [report.segmentation.contour], -1, (0, 255, 0), 2)

        index = 0
        for extra_i, (name, result) in enumerate(report.results.items()):
            if not result.defect_found:
                continue
            color = self._detector_color(result.defect_type, extra_i)
            for (x, y, w, h) in result.locations:
                index += 1
                cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
                # Filled number tag, so a box is identifiable even when the
                # boxes overlap heavily.
                tag = str(index)
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                tx, ty = x, max(y - 4, th + 4)
                cv2.rectangle(canvas, (tx, ty - th - 4), (tx + tw + 8, ty + 4),
                              color, cv2.FILLED)
                cv2.putText(canvas, tag, (tx + 4, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return canvas

    def render_report(self, report: InspectionReport,
                      original: Optional[np.ndarray] = None) -> np.ndarray:
        """Build the full figure: original | mask | annotated + a legend.

        This is what to put in the assignment report. Every detector gets a
        line stating whether it fired and the evidence it used, so a reader
        can see not just *that* something was flagged but *why*.
        """
        annotated = self.annotate(report)
        h, w = annotated.shape[:2]

        # --- panel row ---------------------------------------------------- #
        panels, titles = [], []
        if original is not None:
            panels.append(cv2.resize(original, (w, h), interpolation=cv2.INTER_AREA))
            titles.append("1. original")
        if report.segmentation is not None:
            panels.append(cv2.cvtColor(report.segmentation.mask_raw, cv2.COLOR_GRAY2BGR))
            # Naming the winning cue makes it visible which of the five
            # segmentation cues actually did the work on this photo.
            titles.append(f"2. glove mask  [cue: {report.segmentation.cue}]")
        panels.append(annotated)
        titles.append("3. detection")

        labelled = []
        for panel, title in zip(panels, titles):
            strip = np.full((28, w, 3), 245, np.uint8)
            cv2.putText(strip, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (30, 30, 30), 1, cv2.LINE_AA)
            labelled.append(np.vstack([strip, panel]))
        row = np.hstack(labelled)

        # --- legend ------------------------------------------------------- #
        lines: List[tuple] = []
        verdict = report.verdict()
        lines.append((f"{report.image_name}   ->   {verdict}",
                      self._VERDICT_COLORS.get(verdict, (0, 0, 0)), 0.75))

        for warning in report.warnings:
            lines.append((f"!  {warning}", (0, 140, 220), 0.5))

        index = 0
        for extra_i, (name, result) in enumerate(report.results.items()):
            color = self._detector_color(result.defect_type, extra_i)
            if result.defect_found:
                nums = ", ".join(
                    str(index + i + 1) for i in range(len(result.locations)))
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
            cv2.putText(legend, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        color, 2 if scale >= 0.7 else 1, cv2.LINE_AA)
            y += int(26 * scale / 0.55)

        return np.vstack([row, legend])
