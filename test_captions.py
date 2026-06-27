"""Test caption detection reliability.

Usage:
    python test_captions.py path/to/paper.pdf

Shows each detected image + caption and flags potential issues.
"""

import re
import sys
from pathlib import Path

from src.extraction import extract_images_with_captions


def analyze_captions(pdf_path: str):
    results = extract_images_with_captions(pdf_path)

    total = len(results)
    empty = 0
    has_figure_ref = 0
    partial = 0

    print(f"\n{'='*70}")
    print(f"PDF: {pdf_path}")
    print(f"Total images detected: {total}")
    print(f"{'='*70}\n")

    for img in results:
        page = img["numero_page"]
        idx = img["index_image"]
        caption = img["legende_detectee"]

        issues = []

        if caption == "Aucune légende trouvée":
            issues.append("EMPTY")
            empty += 1
        else:
            has_fig = bool(re.search(r'(?i)\b(?:fig|figure|table|tab)\b', caption))
            if has_fig:
                has_figure_ref += 1
            else:
                issues.append("NO FIGURE/TABLE REFERENCE")
                partial += 1

            has_number = bool(re.search(r'\d+', caption))
            if not has_number:
                issues.append("NO NUMBER")

        status = " ".join(f"[{i}]" for i in issues) if issues else "[OK]"
        print(f"  p.{page} img#{idx} {status}")
        print(f"    Caption: {caption[:120]}")
        print()

    print(f"{'='*70}")
    print(f"Summary:")
    print(f"  Total images:       {total}")
    print(f"  Empty captions:     {empty} ({empty/total*100:.0f}%)" if total else "  Empty captions:     0")
    print(f"  With fig/table ref: {has_figure_ref} ({has_figure_ref/total*100:.0f}%)" if total else "  With fig/table ref: 0")
    print(f"  Partial/no ref:     {partial} ({partial/total*100:.0f}%)" if total else "  Partial/no ref:     0")
    print(f"{'='*70}")

    if empty > 0:
        print("\n⚠️  Some captions are empty — the caption zone may not overlap the text.")
    if partial > 0:
        print("\n⚠️  Some captions have no 'Figure'/'Table' keyword — may be partial text.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pdf = sys.argv[1]
    if not Path(pdf).exists():
        print(f"File not found: {pdf}")
        sys.exit(1)

    analyze_captions(pdf)
