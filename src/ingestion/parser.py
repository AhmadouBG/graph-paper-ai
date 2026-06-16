from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import fitz

from src.exceptions import IngestionError
from src.ingestion.utils_class import ImageInfo, ProcessingResult

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
    import subprocess
    import sys as _sys
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


async def _parse_with_llamaparse(pdf_path: Path, api_key: str):
    from llama_cloud import AsyncLlamaCloud
    client = AsyncLlamaCloud(api_key=api_key)
    file_obj = await client.files.create(file=pdf_path, purpose="parse")
    result = await client.parsing.parse(
        file_id=file_obj.id,
        tier="agentic",
        expand=["items"],
        version="latest"
    )
    return result


def _add_text_to_last_node(stack, text):
    last_node = stack[-1]["node"]
    clean_text = text.strip()
    if last_node["summary"]:
        last_node["summary"] += " " + clean_text
    else:
        last_node["summary"] = clean_text


def _build_complete_hierarchical_tree(raw_result) -> List[dict]:
    if not hasattr(raw_result, "items") or not getattr(raw_result.items, "pages", None):
        logger.warning("No pages found in LlamaParse response.")
        return []
    root_nodes = []
    node_counter = 0
    active_stack = []
    for idx, page in enumerate(raw_result.items.pages):
        page_num = idx + 1
        current_text_fragments = []
        current_heading = None
        current_level = None
        if not hasattr(page, "items") or not page.items:
            continue
        for item in page.items:
            if item.type == "heading":
                if current_heading and current_text_fragments:
                    text_summary = " ".join(current_text_fragments)
                    _add_text_to_last_node(active_stack, text_summary)
                    current_text_fragments = []
                current_heading = (
                    item.value if hasattr(item, "value") else item.md.replace("#", "").strip()
                )
                current_level = getattr(item, "level", 1)
                node_id_str = f"{node_counter:04d}"
                node_counter += 1
                new_node = {
                    "node_id": node_id_str,
                    "title": current_heading,
                    "page_index": page_num,
                    "summary": "",
                    "visuals": [],
                    "tables": [],
                    "nodes": [],
                }
                while active_stack and active_stack[-1]["level"] >= current_level:
                    active_stack.pop()
                if not active_stack:
                    root_nodes.append(new_node)
                else:
                    active_stack[-1]["node"]["nodes"].append(new_node)
                active_stack.append({"level": current_level, "node": new_node})
            elif item.type == "text" and hasattr(item, "value") and item.value:
                current_text_fragments.append(item.value)
            elif item.type in ["image", "figure"] and active_stack:
                visual_text = getattr(item, "md", "") or getattr(item, "value", "")
                if visual_text:
                    current_text_fragments.append(visual_text)
                visual_info = {
                    "type": item.type,
                    "image_id": getattr(item, "name", f"fig_{node_counter}"),
                    "page": page_num,
                    "path": getattr(item, "path", None) or getattr(item, "image_path", ""),
                    "bbox": [
                        (b.__dict__ if hasattr(b, "__dict__") else b)
                        for b in item.bbox
                    ] if getattr(item, "bbox", None) else [],
                }
                active_stack[-1]["node"]["visuals"].append(visual_info)
            elif item.type == "table" and active_stack:
                table_info = {
                    "page": page_num,
                    "markdown_content": getattr(item, "md", "") or getattr(item, "value", ""),
                    "bbox": [
                        (b.__dict__ if hasattr(b, "__dict__") else b)
                        for b in item.bbox
                    ] if getattr(item, "bbox", None) else [],
                }
                active_stack[-1]["node"]["tables"].append(table_info)
        if current_text_fragments and active_stack:
            text_summary = " ".join(current_text_fragments)
            _add_text_to_last_node(active_stack, text_summary)
    return root_nodes


def _llamaparse_tree_to_markdown(tree: List[dict], level: int = 1) -> str:
    md_parts = []
    for node in tree:
        title = node.get("title", "").strip()
        md_parts.append(f"{'#' * level} {title}\n")

        summary = node.get("summary", "").strip()
        if summary:
            md_parts.append(summary + "\n")

        for table in node.get("tables", []):
            content = table.get("markdown_content", "").strip()
            if content:
                md_parts.append(content + "\n")

        for visual in node.get("visuals", []):
            img_id = visual.get("image_id", "")
            img_path = visual.get("path", "")
            if img_id:
                md_parts.append(f"![{img_id}]({img_path})\n")

        child_nodes = node.get("nodes", [])
        if child_nodes:
            md_parts.append(_llamaparse_tree_to_markdown(child_nodes, level + 1))

    return "\n".join(md_parts)


def _extract_images_from_llamaparse_tree(tree: List[dict], images_dir: Path) -> List[ImageInfo]:
    images: List[ImageInfo] = []

    def _traverse(nodes):
        for node in nodes:
            for v in node.get("visuals", []):
                node_id = v.get("image_id", "")
                path_str = v.get("path") or v.get("image_path") or ""
                path = Path(path_str) if path_str else images_dir / f"{node_id}.png"
                page = v.get("page", 1)

                bbox_list = v.get("bbox", [])
                if bbox_list and len(bbox_list) >= 1:
                    b = bbox_list[0]
                    if isinstance(b, dict):
                        x0 = b.get("x0", b.get("left", b.get("x", 0.0)))
                        y0 = b.get("y0", b.get("top", b.get("y", 0.0)))
                        x1 = b.get("x1", b.get("right", x0 + b.get("width", 0.0)))
                        y1 = b.get("y1", b.get("bottom", y0 + b.get("height", 0.0)))
                    else:
                        x0, y0, x1, y1 = 0.0, 0.0, 0.0, 0.0
                    bbox = (float(x0), float(y0), float(x1), float(y1))
                else:
                    bbox = (0.0, 0.0, 0.0, 0.0)

                images.append(ImageInfo(
                    node_id=node_id,
                    path=path,
                    page=page,
                    bbox=bbox,
                ))
            _traverse(node.get("nodes", []))

    _traverse(tree)
    return images


def parse_paper(
    pdf_path: str | Path,
    output_dir: str | Path = "output",
    use_marker: bool = False,
    marker_timeout: int = 120,
    use_llamaparse: bool = False,
) -> ProcessingResult:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    doc = _validate_pdf(pdf_path)
    page_count = doc.page_count

    try:
        if use_llamaparse:
            logger.info("Attempting LlamaParse conversion...")
            api_key = os.environ.get("LLAMACLOUD_API_KEY") or os.environ.get("LLAMA_CLOUD_API_KEY")
            if not api_key:
                raise IngestionError(
                    "LLAMACLOUD_API_KEY environment variable is not set, "
                    "but is required for LlamaParse."
                )

            raw_result = asyncio.run(_parse_with_llamaparse(pdf_path, api_key))
            complete_tree = _build_complete_hierarchical_tree(raw_result)
            markdown = _llamaparse_tree_to_markdown(complete_tree)
            images = _extract_images_from_llamaparse_tree(complete_tree, images_dir)
            edges = []

            return ProcessingResult(
                markdown=markdown,
                images=images,
                edges=edges,
                metadata={
                    "source": pdf_path.name,
                    "page_count": page_count,
                    "image_count": len(images),
                    "parser": "llamaparse",
                    "llamaparse_tree": complete_tree,
                },
            )

        images = _extract_images(doc, images_dir)

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
        edges = []

    finally:
        doc.close()

    return ProcessingResult(
        markdown=markdown,
        images=images,
        edges=edges,
        metadata={
            "source": pdf_path.name,
            "page_count": page_count,
            "image_count": len(images),
        },
    )
