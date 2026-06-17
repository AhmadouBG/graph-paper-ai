from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx

from src.ingestion.graph import _parse_sections

PAGE_MARKER_RE = re.compile(r"^## Page (\d+)", re.MULTILINE)

PAGE_SECTION_RE = re.compile(r"^Page\s+(\d+)$", re.IGNORECASE)



@dataclass
class SectionNode:
    node_id: str
    title: str
    page_index: Optional[int]
    text: str
    nodes: List[SectionNode] = field(default_factory=list)


def _find_section_page(markdown: str, section_start: int) -> int:
    before = markdown[:section_start]
    matches = list(PAGE_MARKER_RE.finditer(before))
    if matches:
        return int(matches[-1].group(1))
    return 1


def build_page_index(markdown: str) -> str:
    sections = _parse_sections(markdown)
    lines: List[str] = []
    idx = 0
    for sec in sections:
        title = sec["title"]
        if PAGE_SECTION_RE.match(title):
            continue
        page = _find_section_page(markdown, sec["start"])
        lines.append(f"[{idx:04d}] {title} (p.{page})")
        idx += 1
    if idx == 0:
        lines.append("[0000] Document (p.1)")
    return "\n".join(lines)


def build_node_id_map(graph: nx.DiGraph) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for n, attr in graph.nodes(data=True):
        if attr.get("node_type") == "section":
            idx = attr.get("_section_idx")
            if idx is not None:
                mapping[f"{idx:04d}"] = n
    return mapping



def print_tree(nodes: List[SectionNode], indent: str = "") -> str:
    lines: List[str] = []
    for i, node in enumerate(nodes):
        is_last = i == len(nodes) - 1
        connector = "└── " if is_last else "├── "
        page_str = f" (page {node.page_index})" if node.page_index is not None else ""
        lines.append(f"{indent}{connector}{node.title}{page_str} [{node.node_id}]")
        child_indent = indent + ("    " if is_last else "│   ")
        lines.append(print_tree(node.nodes, indent=child_indent))
    return "\n".join(lines).rstrip("\n")
