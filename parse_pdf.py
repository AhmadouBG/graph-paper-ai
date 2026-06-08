import argparse
import logging
import sys
from pathlib import Path

from src.ingestion.parser import parse_paper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Parse a research paper PDF into Markdown with images")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("output_dir", type=str, nargs="?", default="output", help="Output directory (default: output/)")
    parser.add_argument("--marker", action="store_true", help="Use Marker for higher-quality conversion (may download models)")
    parser.add_argument("--marker-timeout", type=int, default=120, help="Timeout in seconds for Marker conversion (default: 120)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    output_dir = Path(args.output_dir)

    if not pdf_path.exists():
        print(f"Error: File '{pdf_path}' does not exist.")
        return

    method = "Marker" if args.marker else "PyMuPDF"
    print(f"Parsing PDF: {pdf_path} (method: {method}) ...")
    if args.marker:
        print("(Marker model download may take several minutes on first run)")
    try:
        result = parse_paper(
            pdf_path,
            output_dir=output_dir,
            use_marker=args.marker,
            marker_timeout=args.marker_timeout,
        )

        print("\n=== METADATA ===")
        for key, value in result.metadata.items():
            print(f"  {key}: {value}")

        print(f"\n=== EXTRACTED IMAGES ===")
        print(f"  Images extracted to: {output_dir / 'images'}")
        print(f"  Number of images extracted: {len(result.images)}")
        for img in result.images:
            print(f"  - ID: {img.node_id}, Page: {img.page}, Path: {img.path}")

        markdown_file = output_dir / "extracted_paper.md"
        markdown_file.write_text(result.markdown, encoding="utf-8")
        print(f"\nSaved full UTF-8 Markdown to: {markdown_file}")

        print("\n=== MARKDOWN PREVIEW (first 3000 chars) ===")
        safe_markdown = result.markdown[:3000].encode(
            sys.stdout.encoding, errors="replace"
        ).decode(sys.stdout.encoding)
        print(safe_markdown)
        print("...")

    except Exception as e:
        import traceback
        print(f"Failed to parse PDF: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
