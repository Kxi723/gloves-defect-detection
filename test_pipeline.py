"""
Quick batch test: run the glove inspection pipeline over a folder of photos.

Usage:
    python test_pipeline.py                          # gloves -> output/
    python test_pipeline.py -i my_photos -o results

For every image it saves an annotated copy (defect boxes + verdict) and the
segmentation mask to the output folder, then prints a per-image summary
table and overall totals.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2

from gdd.pipeline import GloveInspector, InspectionReport

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def print_summary(reports: List[InspectionReport]) -> None:
    """Fixed-width per-image table plus totals."""
    name_w = max(20, max((len(r.image_name) for r in reports), default=20))
    header = (f"{'Image':<{name_w}}  {'Verdict':<9}  {'Defects found':<40}  "
              f"{'Time (s)':>8}")
    print("\n" + header)
    print("-" * len(header))
    for r in reports:
        defects = ", ".join(r.defect_types_found) or "-"
        print(f"{r.image_name:<{name_w}}  {r.verdict():<9}  {defects:<40}  "
              f"{r.elapsed_seconds:>8.2f}")

    counts = {"PASS": 0, "DEFECT": 0, "SEG-FAIL": 0}
    for r in reports:
        counts[r.verdict()] = counts.get(r.verdict(), 0) + 1
    print("-" * len(header))
    print(f"Total: {len(reports)}   " +
          "   ".join(f"{k}: {v}" for k, v in counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", default="gloves",
                        help="folder of glove photos (default: gloves)")
    parser.add_argument("-o", "--output", default="output",
                        help="folder for annotated results (default: output)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        return 1
    images = find_images(input_dir)
    if not images:
        print(f"No images in {input_dir} (looked for {sorted(IMAGE_EXTENSIONS)})",
              file=sys.stderr)
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    inspector = GloveInspector()
    print(f"Inspecting {len(images)} image(s) with detectors: "
          f"{', '.join(inspector.detector_names)}")

    reports: List[InspectionReport] = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"  ! could not read {path.name}, skipped", file=sys.stderr)
            continue

        report = inspector.inspect(image, image_name=path.name)
        reports.append(report)

        # The combined figure (original | mask | detection + legend) is the
        # one to paste into the report; the other two are for debugging.
        cv2.imwrite(str(output_dir / f"{path.stem}_report.png"),
                    inspector.render_report(report, original=image))
        cv2.imwrite(str(output_dir / f"{path.stem}_annotated.png"),
                    inspector.annotate(report))
        if report.segmentation is not None:
            cv2.imwrite(str(output_dir / f"{path.stem}_mask.png"),
                        report.segmentation.mask_raw)

        # Per-detector evidence, useful while calibrating config.py.
        print(f"  {path.name}: {report.verdict()}")
        for warning in report.warnings:
            print(f"    !! {warning}")
        for name, result in report.results.items():
            flag = "!" if result.defect_found else " "
            print(f"    [{flag}] {name}: {result.details}")

    print_summary(reports)
    print(f"\nAnnotated images written to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
