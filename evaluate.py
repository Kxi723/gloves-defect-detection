"""
Measure detector accuracy against ground-truth labels.

Eyeballing annotated images cannot tell you whether a change helped or
just moved errors around, so every tuning decision should be checked here
instead.

Ground truth comes from the filename by default, using the convention

    <anything>_<material>_<n>.jpeg      e.g. kxi_latex_3.jpeg

together with the material -> defect mapping in DEFECT_BY_MATERIAL below.
Photos of undamaged gloves must be named with the material followed by
"good", e.g. ``kxi_latex_good_1.jpeg``; they are what makes precision
meaningful, because without them a detector that fires on everything
scores a perfect recall.

Usage:
    python evaluate.py                    # gloves/ , labels from filenames
    python evaluate.py -i photos --holdout 20
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2

from gdd.pipeline import GloveInspector, InspectionReport

# Which defect each material was photographed to demonstrate. Keys must
# match a DefectSpec.key in detectors/__init__.py. Update when the photo
# set changes.
DEFECT_BY_MATERIAL: Dict[str, str] = {
    "latex": "damage_by_fold",
    "cotton": "dirty",
    "nitrile": "tearing_at_finger",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_from_filename(path: Path) -> Tuple[Optional[str], Set[str]]:
    """(material, expected defect names) parsed from a filename.

    An empty defect set means the photo is of an undamaged glove, so every
    detector firing on it counts as a false positive.
    """
    stem = path.stem.lower()
    material = next((m for m in DEFECT_BY_MATERIAL if m in stem), None)
    if material is None:
        return None, set()
    if re.search(r"(^|_)(good|ok|clean|nodefect)(_|$)", stem):
        return material, set()
    return material, {DEFECT_BY_MATERIAL[material]}


def score_counts(reports: List[Tuple[InspectionReport, Set[str]]],
                 detector: str) -> Tuple[int, int, int, int]:
    """(true pos, false pos, false neg, true neg) for one detector."""
    tp = fp = fn = tn = 0
    for report, expected in reports:
        result = report.results.get(detector)
        fired = bool(result and result.defect_found)
        should = detector in expected
        if fired and should:
            tp += 1
        elif fired and not should:
            fp += 1
        elif not fired and should:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def ratio(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.0%}" if denominator else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", default="gloves")
    parser.add_argument(
        "--holdout", type=int, default=0, metavar="PERCENT",
        help="reserve this %% of photos as a test set and report it "
             "separately; tune only against the calibration half")
    args = parser.parse_args()

    folder = Path(args.input)
    images = sorted(p for p in folder.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"No images in {folder}", file=sys.stderr)
        return 1

    inspector = GloveInspector()
    evaluated: List[Tuple[InspectionReport, Set[str]]] = []
    unlabelled: List[str] = []

    for path in images:
        material, expected = label_from_filename(path)
        if material is None:
            unlabelled.append(path.name)
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        evaluated.append((inspector.inspect(image, path.name), expected))

    if unlabelled:
        print(f"Skipped {len(unlabelled)} file(s) with no recognisable "
              f"material in the name: {', '.join(unlabelled[:5])}")

    # --- split -------------------------------------------------------- #
    groups = {"all photos": evaluated}
    if args.holdout:
        cut = len(evaluated) - max(1, len(evaluated) * args.holdout // 100)
        groups = {"CALIBRATION (tune here)": evaluated[:cut],
                  "HELD OUT (report this)": evaluated[cut:]}

    for title, subset in groups.items():
        if not subset:
            continue
        print(f"\n=== {title} — {len(subset)} photo(s) ===")
        header = (f"{'detector':18} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}  "
                  f"{'precision':>9} {'recall':>7}")
        print(header)
        print("-" * len(header))
        for detector in inspector.detector_names:
            tp, fp, fn, tn = score_counts(subset, detector)
            print(f"{detector:18} {tp:3d} {fp:3d} {fn:3d} {tn:3d}  "
                  f"{ratio(tp, tp + fp):>9} {ratio(tp, tp + fn):>7}")

        negatives = sum(1 for _, exp in subset if not exp)
        if negatives == 0:
            print("\n  ! No undamaged-glove photos in this set. Precision "
                  "above is measured only against OTHER defect types, so it "
                  "overstates how well the system avoids false alarms.")

        flagged = sum(1 for r, _ in subset if r.warnings)
        if flagged:
            print(f"  ! {flagged} photo(s) carry a capture warning (part of "
                  f"the glove outside the frame). The warning is advisory "
                  f"only; those rows are scored normally above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
