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


def _build_pure_text_tree(markdown_text: str) -> list[dict]:
    # Normalisation des marqueurs de page de LlamaParse
    text_with_page_tags = re.sub(r'---\s*Page\s*(\d+)\s*---', r'[[PAGE_\1]]', markdown_text)
    lines = text_with_page_tags.split("\n")
    
    root_nodes = []
    stack = []
    current_page = 1
    node_counter = 0
    
    # Nœud racine par défaut pour l'en-tête du document
    intro_node = {
        "node_id": f"{node_counter:04d}",
        "title": "Document Header / Introduction",
        "page_start": 1,
        "page_end": 1,
        "content_lines": [],
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
 


