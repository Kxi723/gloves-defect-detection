"""
Measure detector accuracy against ground-truth labels.

Eyeballing annotated images cannot tell you whether a change helped or
just moved errors around, so every tuning decision should be checked here
instead.

Ground truth comes from the filename, using the convention

    <defect>_<material>_<n>.jpg          e.g. fold_cotton_01.jpg

The defect keyword and the material are read independently, so the same
defect can be photographed on several materials — which the brief requires
— and any combination scores correctly.

    good_latex_01.jpg    undamaged; any detector firing is a false positive
    fold_cotton_02.jpg   a fold crease, on cotton
    tear_nitrile_05.jpg  a fingertip tear, on nitrile

Undamaged photos are what make precision meaningful: without them a
detector that fires on everything scores perfect recall and nothing
contradicts it.

Usage:
    python evaluate.py                    # gloves/ , labels from filenames
    python evaluate.py --by-material      # also break results down per material
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

# Filename keyword -> detector key (must match a DefectSpec.key in
# detectors/__init__.py). None marks an undamaged glove.
#
# Read independently of the material, so a defect photographed on two
# materials scores correctly on both. An earlier version mapped material ->
# defect, which silently mislabelled every cross-material photo.
DEFECT_KEYWORDS: Dict[str, Optional[str]] = {
    "fold": "damage_by_fold",
    "dirty": "dirty",
    "tear": "tearing_at_finger",
    "good": None,
}

MATERIALS = ("cotton", "latex", "nitrile")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_from_filename(path: Path) -> Tuple[Optional[str], Optional[Set[str]]]:
    """(material, expected defect names) parsed from a filename.

    Returns ``(material, None)`` when the name carries no defect keyword, so
    the caller can skip it rather than silently scoring it as undamaged.
    An empty *set* is different: it means the photo IS labelled undamaged,
    and every detector firing on it counts as a false positive.

    Matching is by whole word between underscores, so a material name can
    never be mistaken for a defect keyword or vice versa.
    """
    parts = set(re.split(r"[_\-. ]+", path.stem.lower()))
    material = next((m for m in MATERIALS if m in parts), None)

    labelled = parts & set(DEFECT_KEYWORDS)
    if not labelled:
        return material, None
    expected = {DEFECT_KEYWORDS[word] for word in labelled}
    expected.discard(None)      # "good" contributes no expected defect
    return material, expected


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
    parser.add_argument(
        "--by-material", action="store_true",
        help="also break each detector down per material, which is the "
             "evidence that a detector works on more than one of them")
    args = parser.parse_args()

    folder = Path(args.input)
    images = sorted(p for p in folder.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"No images in {folder}", file=sys.stderr)
        return 1

    inspector = GloveInspector()
    evaluated: List[Tuple[InspectionReport, Set[str]]] = []
    materials: List[Optional[str]] = []
    unlabelled: List[str] = []

    for path in images:
        material, expected = label_from_filename(path)
        if expected is None:
            unlabelled.append(path.name)
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        evaluated.append((inspector.inspect(image, path.name), expected))
        materials.append(material)

    if unlabelled:
        print(f"Skipped {len(unlabelled)} file(s) with no defect keyword "
              f"({'/'.join(DEFECT_KEYWORDS)}) in the name: "
              f"{', '.join(unlabelled[:5])}")
    if not evaluated:
        print("Nothing labelled to evaluate. Name photos like "
              "fold_cotton_01.jpg or good_latex_02.jpg.", file=sys.stderr)
        return 1

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

        if args.by_material and subset is evaluated:
            print("\n  per material (recall on photos that should show the "
                  "defect; '-' means none were shot)")
            present = [m for m in MATERIALS if m in materials]
            print("    " + f"{'detector':18}" +
                  "".join(f"{m:>12}" for m in present))
            for detector in inspector.detector_names:
                cells = []
                for mat in present:
                    rows = [(r, e) for (r, e), m in zip(subset, materials)
                            if m == mat]
                    tp, fp, fn, _ = score_counts(rows, detector)
                    cells.append(f"{ratio(tp, tp + fn):>9} " if tp + fn
                                 else f"{'-':>9} ")
                    cells[-1] = cells[-1].rjust(12)
                print("    " + f"{detector:18}" + "".join(cells))

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
