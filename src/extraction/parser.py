from __future__ import annotations


import logging
from pathlib import Path
import re
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

from llama_cloud import LlamaCloud

def _parse_with_llamacloud(pdf_path: str, api_key: str) -> list[dict]:
    """
    Returns a list of page dicts with keys: 'page' (int), 'md' (str).
    Drop-in replacement for the deprecated LlamaParse approach.
    """

    # 1. Initialize client (reads LLAMA_CLOUD_API_KEY from environment)
    client = LlamaCloud(api_key=api_key)

    # 2. Upload the file
    file = client.files.create(file=pdf_path, purpose="parse")

    # 3. Parse and explicitly ask for "json" in expand
    result = client.parsing.parse(
        file_id=file.id,
        tier="agentic",
        version="latest",
        expand=["markdown"],  # <--- CRITICAL: Ensures the JSON structure is loaded
    )

    if result.markdown.pages:
        p = result.markdown.pages[0]
        print("page attrs:", [a for a in dir(p) if not a.startswith("_")])
        print("page dict:", p.__dict__ if hasattr(p, "__dict__") else vars(p))
    
    return [
        {
            "page": p.page_number,
            "md": p.markdown or "",
        }
        for p in result.markdown.pages
        if p.success
    ]

def _build_pure_text_tree(markdown_text: str, page_image_map: dict[int, list[dict]]) -> list[dict]:
    text_with_page_tags = re.sub(r'---\s*Page\s*(\d+)\s*---', r'[[PAGE_\1]]', markdown_text)
    lines = text_with_page_tags.split("\n")

    root_nodes = []
    stack = []
    current_page = 1
    node_counter = 0

    def make_node(node_id: str, title: str, page: int) -> dict:
        """Helper to build a consistent node dict for any page."""
        page_imgs = page_image_map.get(page, [])
        return {
            "node_id": node_id,
            "title": title,
            "page_start": page,
            "page_end": page,
            "content_lines": [],
            "base64_images": [img["base64"] for img in page_imgs],
            "image_captions": [img["caption"] for img in page_imgs],
            "nodes": [],  # always present
        }

    intro_node = make_node(f"{node_counter:04d}", "Document Header / Introduction", 1)
    node_counter += 1
    root_nodes.append(intro_node)
    stack.append({"level": 0, "node": intro_node})

    for line in lines:
        page_match = re.search(r'\[\[PAGE_(\d+)\]\]', line)
        if page_match:
            current_page = int(page_match.group(1))
            if stack:
                stack[-1]["node"]["page_end"] = current_page
            continue

        heading_match = re.match(r'^(#{1,6})\s+(.*)$', line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            new_node = make_node(f"{node_counter:04d}", title, current_page)
            node_counter += 1

            while stack and stack[-1]["level"] >= level:
                stack.pop()

            if not stack:
                root_nodes.append(new_node)
            else:
                stack[-1]["node"]["nodes"].append(new_node)

            stack.append({"level": level, "node": new_node})
        else:
            if stack and line.strip():
                stack[-1]["node"]["content_lines"].append(line)

    def finalize_tree(nodes: list[dict], next_start: int | None = None) -> None:
        for i, n in enumerate(nodes):
            n["content"] = "\n".join(n["content_lines"])
            del n["content_lines"]
            if i + 1 < len(nodes):
                n["page_end"] = max(n["page_start"], nodes[i + 1]["page_start"])
            elif next_start:
                n["page_end"] = max(n["page_start"], next_start)
            if n["nodes"]:
                finalize_tree(n["nodes"], n["page_end"])

    finalize_tree(root_nodes)
    return root_nodes

 


