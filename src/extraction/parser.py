from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any
import re
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

async def _get_images_as_base64_map(raw_result, api_key: str) -> dict[int, list[str]]:
    """Fetches images from LlamaCloud and converts them straight to Base64 strings in RAM."""
    from llama_cloud import AsyncLlamaCloud
    import base64
    
    client = AsyncLlamaCloud(api_key=api_key)
    page_image_map = {}
    
    pages = getattr(raw_result.items, "pages", []) if hasattr(raw_result, "items") else []
    
    for idx, page in enumerate(pages):
        page_num = idx + 1
        page_image_map[page_num] = []
        
        for item in getattr(page, "items", []):
            if getattr(item, "type", "") == "image":
                image_id = getattr(item, "id", None) or getattr(item, "name", None)
                if not image_id:
                    continue
                
                try:
                    # 1. Fetch image bytes into RAM
                    image_bytes = await client.parsing.get_image(
                        file_id=raw_result.id, 
                        image_id=image_id
                    )
                    # 2. Encode to Base64 string directly
                    base64_str = base64.b64encode(image_bytes).decode('utf-8')
                    page_image_map[page_num].append(base64_str)
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch image {image_id} on page {page_num}: {e}")
                    
    return page_image_map


def _build_pure_text_tree(markdown_text: str, page_image_map: dict[int, list[str]]) -> list[dict]:
    text_with_page_tags = re.sub(r'---\s*Page\s*(\d+)\s*---', r'[[PAGE_\1]]', markdown_text)
    lines = text_with_page_tags.split("\n")
    
    root_nodes = []
    stack = []
    current_page = 1
    node_counter = 0
    
    intro_node = {
        "node_id": f"{node_counter:04d}",
        "title": "Document Header / Introduction",
        "page_start": 1,
        "page_end": 1,
        "content_lines": [],
        "base64_images": page_image_map.get(1, []), # ✨ In-memory Base64 strings
        "nodes": []
    }
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
            
            new_node = {
                "node_id": f"{node_counter:04d}",
                "title": title,
                "page_start": current_page,
                "page_end": current_page,
                "content_lines": [],
                "base64_images": page_image_map.get(current_page, []), # ✨ In-memory Base64 strings
                "nodes": []
            }
            node_counter += 1
            
            while stack and stack[-1]["level"] >= level:
                stack.pop()
                
            if not stack:
                root_nodes.append(new_node)
                stack.append({"level": level, "node": new_node})
            else:
                stack[-1]["node"]["nodes"].append(new_node)
                stack.append({"level": level, "node": new_node})
        else:
            if stack and line.strip():
                stack[-1]["node"]["content_lines"].append(line)

    def finalize_tree(nodes, next_start=None):
        for i, n in enumerate(nodes):
            n["content"] = "\n".join(n["content_lines"])
            del n["content_lines"]
            if i + 1 < len(nodes):
                n["page_end"] = max(n["page_start"], nodes[i+1]["page_start"])
            elif next_start:
                n["page_end"] = max(n["page_start"], next_start)
            if n["nodes"]:
                finalize_tree(n["nodes"], n["page_end"])
                
    finalize_tree(root_nodes)
    return root_nodes

 


