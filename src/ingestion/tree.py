from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx

from src.ingestion.graph import _parse_sections

PAGE_MARKER_RE = re.compile(r"^## Page (\d+)", re.MULTILINE)
PAGE_SECTION_RE = re.compile(r"^Page \d+$")


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
        key=lambda n: graph.nodes[n].get("_section_idx", 0),
    )

    ordered: List[str] = []
    def _collect(node_id: str) -> None:
        ordered.append(node_id)
        for _, v, d in graph.out_edges(node_id, data=True):
            if d.get("edge_type") != "contains":
                continue
            if v in is_page_marker:
                for _, gv, gd in graph.out_edges(v, data=True):
                    if gd.get("edge_type") == "contains" and gv in content_sections:
                        _collect(gv)
            elif v in content_sections:
                _collect(v)

    for r in roots:
        _collect(r)

    gid_to_seq = {gid: f"{i:04d}" for i, gid in enumerate(ordered)}

    def _build(graph_node_id: str) -> SectionNode:
        attr = graph.nodes[graph_node_id]
        child_pairs: List[tuple[int, SectionNode]] = []
        for _, v, d in graph.out_edges(graph_node_id, data=True):
            if d.get("edge_type") != "contains":
                continue
            if v in is_page_marker:
                for _, gv, gd in graph.out_edges(v, data=True):
                    if gd.get("edge_type") == "contains" and gv in content_sections:
                        child = _build(gv)
                        child_pairs.append((
                            graph.nodes[gv].get("_section_idx", 0), child,
                        ))
            elif v in content_sections:
                child = _build(v)
                child_pairs.append((
                    graph.nodes[v].get("_section_idx", 0), child,
                ))
        child_pairs.sort(key=lambda p: p[0])
        children = [p[1] for p in child_pairs]

        return SectionNode(
            node_id=gid_to_seq.get(graph_node_id, "0000"),
            title=attr.get("title", graph_node_id),
            page_index=attr.get("page"),
            text=attr.get("content", ""),
            nodes=children,
        )

    return [_build(r) for r in roots]


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
