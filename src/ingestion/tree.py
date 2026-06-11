from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import networkx as nx

from src.ingestion.graph import _parse_sections

PAGE_MARKER_RE = re.compile(r"^## Page (\d+)", re.MULTILINE)
PAGE_SECTION_RE = re.compile(r"^Page \d+$")


@dataclass
class SectionNode:
    node_id: str
    title: str
    page: Optional[int]
    children: List[SectionNode] = field(default_factory=list)


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


def build_section_tree(graph: nx.DiGraph) -> List[SectionNode]:
    is_page_marker: set[str] = set()
    for n, attr in graph.nodes(data=True):
        if attr.get("node_type") == "section" and PAGE_SECTION_RE.match(attr.get("title", "")):
            is_page_marker.add(n)

    content_sections = [
        n for n, attr in graph.nodes(data=True)
        if attr.get("node_type") == "section" and n not in is_page_marker
    ]

    has_non_marker_parent = {
        v for u, v, d in graph.out_edges(data=True)
        if d.get("edge_type") == "contains"
        and u not in is_page_marker
        and v in content_sections
    }

    roots = sorted(
        [n for n in content_sections if n not in has_non_marker_parent],
        key=lambda n: _section_order(graph, n),
    )

    def _build(node_id: str) -> SectionNode:
        attr = graph.nodes[node_id]
        children: List[SectionNode] = []
        for _, v, d in graph.out_edges(node_id, data=True):
            if d.get("edge_type") != "contains":
                continue
            if v in is_page_marker:
                for _, gv, gd in graph.out_edges(v, data=True):
                    if gd.get("edge_type") == "contains" and gv in content_sections:
                        children.append(_build(gv))
            elif v in content_sections:
                children.append(_build(v))
        children.sort(key=lambda c: _section_order(graph, c.node_id))
        return SectionNode(
            node_id=node_id,
            title=attr.get("title", node_id),
            page=attr.get("page"),
            children=children,
        )

    return [_build(r) for r in roots]


def _section_order(graph: nx.DiGraph, node_id: str) -> int:
    return graph.nodes[node_id].get("_section_idx", 0)


def print_tree(nodes: List[SectionNode], show_ids: bool = True) -> str:
    lines: List[str] = []
    for i, node in enumerate(nodes):
        is_last = i == len(nodes) - 1
        connector = "└── " if is_last else "├── "
        page_str = f" (page {node.page})" if node.page is not None else ""
        id_str = f" (node: {node.node_id})" if show_ids else ""
        line = f"{connector}{node.title}{page_str}{id_str}"
        lines.append(line)
        child_lines = print_tree(node.children, show_ids=show_ids)
        if child_lines:
            indent = "    " if is_last else "│   "
            for line in child_lines.split("\n"):
                lines.append(f"{indent}{line}")
    return "\n".join(lines)
