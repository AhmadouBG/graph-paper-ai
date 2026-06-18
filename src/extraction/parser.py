from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.exceptions import IngestionError

logger = logging.getLogger(__name__)
load_dotenv()


async def _parse_with_llamaparse(pdf_path: Path, api_key: str):
    from llama_cloud import AsyncLlamaCloud

    client = AsyncLlamaCloud(api_key=api_key)
    file_obj = await client.files.create(file=pdf_path, purpose="parse")
    result = await client.parsing.parse(
        file_id=file_obj.id,
        tier="agentic",
        expand=["items"],
        version="latest",
    )
    return result


def _add_text_to_last_node(stack: list[dict], text: str):
    last_node = stack[-1]["node"]
    clean_text = text.strip()
    if last_node["summary"]:
        last_node["summary"] += " " + clean_text
    else:
        last_node["summary"] = clean_text
    last_node["text"] = last_node["summary"]


def _download_image(url: str, pdf_stem: str, page_num: int, img_idx: int) -> str:
    import httpx
    local_dir = Path("_images") / pdf_stem
    local_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(url.split("?")[0]).suffix or ".png"
    local_path = local_dir / f"p{page_num}_{img_idx}{ext}"
    if local_path.exists():
        return str(local_path)
    try:
        resp = httpx.get(url, timeout=30)
        resp.raise_for_status()
        local_path.write_bytes(resp.content)
        return str(local_path)
    except Exception as exc:
        logger.warning(f"Failed to download image {url}: {exc}")
        return ""


def _build_complete_hierarchical_tree(raw_result, pdf_path: Path) -> list[dict]:
    if not hasattr(raw_result, "items") or not getattr(raw_result.items, "pages", None):
        logger.warning("No pages found in LlamaParse response.")
        return []
    root_nodes: list[dict] = []
    node_counter = 0
    active_stack: list[dict] = []
    for idx, page in enumerate(raw_result.items.pages):
        page_num = idx + 1
        current_text_fragments: list[str] = []
        current_heading = None
        if not hasattr(page, "items") or not page.items:
            continue
        for item in page.items:
            if item.type == "heading":
                if current_heading and current_text_fragments:
                    text_summary = " ".join(current_text_fragments)
                    _add_text_to_last_node(active_stack, text_summary)
                    current_text_fragments = []
                elif not current_heading and current_text_fragments and active_stack:
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
            elif item.type == "image" and active_stack:
                md = item.md if hasattr(item, "md") and item.md else ""
                cap = item.caption if hasattr(item, "caption") and item.caption else ""
                img_url = item.url if hasattr(item, "url") else ""
                visual_text = md or cap
                if visual_text:
                    current_text_fragments.append(visual_text)
                print(f"  [IMAGE] caption={cap!r}  url={img_url!r}")
                local_path = (
                    _download_image(img_url, pdf_path.stem, page_num, node_counter)
                    if img_url else ""
                )
                image_id = cap or f"fig_{node_counter}"
                visual_info = {
                    "type": "image",
                    "image_id": image_id,
                    "page": page_num,
                    "path": local_path,
                    "url": img_url,
                    "bbox": [
                        (b.__dict__ if hasattr(b, "__dict__") else b)
                        for b in item.bbox
                    ]
                    if hasattr(item, "bbox") and item.bbox
                    else [],
                }
                active_stack[-1]["node"]["visuals"].append(visual_info)
            elif item.type == "table" and active_stack:
                table_info = {
                    "page": page_num,
                    "markdown_content": getattr(item, "md", "") or getattr(item, "value", ""),
                    "bbox": [
                        (b.__dict__ if hasattr(b, "__dict__") else b)
                        for b in item.bbox
                    ]
                    if getattr(item, "bbox", None)
                    else [],
                }
                active_stack[-1]["node"]["tables"].append(table_info)
        if current_text_fragments and active_stack:
            text_summary = " ".join(current_text_fragments)
            _add_text_to_last_node(active_stack, text_summary)
    return root_nodes


def _extract_images_from_llamaparse_tree(tree: list[dict]) -> list[dict]:
    images: list[dict] = []

    def _traverse(nodes):
        for node in nodes:
            for v in node.get("visuals", []):
                image_id = v.get("image_id", "")
                path_str = v.get("path") or v.get("image_path") or ""
                images.append({
                    "image_id": image_id,
                    "path": path_str,
                    "page": v.get("page", 1),
                })
            _traverse(node.get("nodes", []))

    _traverse(tree)
    return images


def display_document_structure(nodes: list[dict], indent_level: int = 0):
    if indent_level == 0:
        print("Full Document Structure:\n")
    for node in nodes:
        spacing = "  " * indent_level
        suffix = ""
        v = len(node.get("visuals", []))
        t = len(node.get("tables", []))
        if v or t:
            parts = [f"{v} visual(s)" if v else "", f"{t} table(s)" if t else ""]
            suffix = f"  [{' '.join(parts)}]"
        print(f"{spacing}[{node['node_id']}] {node['title']}  (p.{node['page_index']}){suffix}")
        if "nodes" in node and node["nodes"]:
            display_document_structure(node["nodes"], indent_level + 1)


def parse_paper(pdf_path: str | Path) -> dict[str, Any]:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise IngestionError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise IngestionError(f"File is not a PDF: {pdf_path}")

    api_key = os.environ.get("LLAMACLOUD_API_KEY")
    if not api_key:
        raise IngestionError(
            "LLAMACLOUD_API_KEY environment variable is required."
        )

    raw_result = asyncio.run(_parse_with_llamaparse(pdf_path, api_key))
    complete_tree = _build_complete_hierarchical_tree(raw_result, pdf_path)
    display_document_structure(complete_tree)
    print(json.dumps(complete_tree, indent=2, ensure_ascii=False))
    images = _extract_images_from_llamaparse_tree(complete_tree)
    markdown = raw_result.markdown_full if hasattr(raw_result, "markdown_full") else ""

    return {
        "markdown": markdown,
        "images": images,
        "metadata": {
            "source": pdf_path.name,
            "page_count": len(raw_result.items.pages) if hasattr(raw_result, "items") else 0,
            "image_count": len(images),
            "parser": "llamaparse",
            "llamaparse_tree": complete_tree,
        },
    }
