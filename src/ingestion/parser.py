from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import fitz

from src.exceptions import IngestionError
from src.ingestion import ImageInfo, ProcessingResult

logger = logging.getLogger(__name__)


FIGURE_LABELS = re.compile(
    r"\b(?:Fig(?:ure)?|Table|Equation|Section)\s+(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


def _validate_pdf(pdf_path: Path) -> fitz.Document:
    if not pdf_path.exists():
        raise IngestionError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise IngestionError(f"File is not a PDF: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise IngestionError(f"Failed to open PDF: {e}") from e
    if doc.page_count == 0:
        doc.close()
        raise IngestionError(f"PDF has no pages: {pdf_path}")
    return doc


def _find_refs_on_page(page: fitz.Page) -> List[Tuple[str, str]]:
    text = page.get_text("text")
    refs: List[Tuple[str, str]] = []
    for match in re.finditer(r"\b(Fig(?:ure)?|Table)\s+(\d+(?:\.\d+)?)", text, re.IGNORECASE):
        kind = "table" if match.group(1).lower().startswith("t") else "fig"
        refs.append((kind, match.group(2)))
    return refs


def _extract_images(
    doc: fitz.Document,
    images_dir: Path,
) -> List[ImageInfo]:
    images: List[ImageInfo] = []
    raster_found = False

    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block.get("type") != 1:
                continue
            bbox = block["bbox"]
            x0, y0, x1, y1 = bbox
            if round(x1 - x0) < 50 or round(y1 - y0) < 50:
                continue
            try:
                image = block.get("image")
                if isinstance(image, bytes):
                    raster_found = True
                    pix = fitz.Pixmap(image)
                else:
                    xref = block.get("number", 0)
                    if xref:
                        raster_found = True
                        pix_data = doc.extract_image(xref)["image"]
                        pix = fitz.Pixmap(pix_data)
                    else:
                        continue

                node_id = f"fig_{page_num + 1}"
                filename = f"{node_id}.png"
                image_path = images_dir / filename
                pix.save(str(image_path))
                pix = None

                images.append(ImageInfo(
                    node_id=node_id,
                    path=image_path,
                    page=page_num + 1,
                    bbox=bbox,
                ))
            except Exception as e:
                logger.warning("Failed to extract image on page %d: %s", page_num + 1, e)

    if not raster_found:
        logger.info("No embedded raster images found; rendering pages with figure/table references")
        seen_refs: set[str] = set()
        for page_num in range(doc.page_count):
            page = doc[page_num]
            refs = _find_refs_on_page(page)
            if not refs:
                continue
            for kind, num in refs:
                node_id = f"{kind}_{num}"
                if node_id in seen_refs:
                    continue
                seen_refs.add(node_id)
                try:
                    pix = page.get_pixmap(dpi=150)
                    filename = f"{node_id}.png"
                    image_path = images_dir / filename
                    pix.save(str(image_path))
                    pix = None
                    images.append(ImageInfo(
                        node_id=node_id,
                        path=image_path,
                        page=page_num + 1,
                        bbox=(0, 0, page.rect.width, page.rect.height),
                    ))
                    logger.info("Rendered %s -> %s", node_id, image_path)
                except Exception as e:
                    logger.warning("Failed to render page %d: %s", page_num + 1, e)

    return images


def _extract_text_fitz(doc: fitz.Document) -> str:
    pages: List[str] = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text")
        pages.append(f"## Page {page_num + 1}\n\n{text}")
    return "\n\n".join(pages)


def _process_math_formulas(markdown: str) -> str:
    if not markdown:
        return markdown
    markdown = markdown.replace(r"\[", "$$").replace(r"\]", "$$")
    markdown = markdown.replace(r"\(", "$").replace(r"\)", "$")
    return markdown


def _try_marker_subprocess(pdf_path: Path, timeout_sec: int = 120) -> Optional[str]:
    import subprocess, sys as _sys
    from pathlib import Path as _Path

    worker = _Path(__file__).parent / "_marker_worker.py"
    if not worker.exists():
        return None

    try:
        proc = subprocess.run(
            [_sys.executable, str(worker), str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout
        return None
    except Exception:
        return None


def parse_paper(
    pdf_path: str | Path,
    output_dir: str | Path = "output",
    use_marker: bool = False,
    marker_timeout: int = 120,
) -> ProcessingResult:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    doc = _validate_pdf(pdf_path)
    page_count = doc.page_count

    images = _extract_images(doc, images_dir)

    try:
        if use_marker:
            logger.info("Attempting Marker conversion (timeout=%ds)...", marker_timeout)
            marker_md = _try_marker_subprocess(pdf_path, marker_timeout)
            if marker_md is not None:
                logger.info("Marker conversion succeeded")
                markdown = marker_md
            else:
                logger.info("Marker failed, falling back to PyMuPDF")
                markdown = _extract_text_fitz(doc)
        else:
            logger.info("Extracting text via PyMuPDF...")
            markdown = _extract_text_fitz(doc)

        markdown = _process_math_formulas(markdown)
    finally:
        doc.close()

    return ProcessingResult(
        markdown=markdown,
        images=images,
        metadata={
            "source": pdf_path.name,
            "page_count": page_count,
            "image_count": len(images),
        },
    )
