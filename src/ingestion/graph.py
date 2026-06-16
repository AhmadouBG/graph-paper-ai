from __future__ import annotations

import re
from typing import Dict, List, Optional

import networkx as nx

from src.ingestion.utils_class import CrossReference, ProcessingResult

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
FORMULA_DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
FORMULA_INLINE_RE = re.compile(r"\$(.+?)\$")


def _parse_sections(markdown: str) -> List[Dict]:
    sections: List[Dict] = []
    heading_spans = [
        (m.start(), m.end(), len(m.group(1)), m.group(2).strip())
        for m in HEADING_RE.finditer(markdown)
    ]
    heading_spans.sort(key=lambda x: x[0])

    for i, (start, end, level, title) in enumerate(heading_spans):
        next_start = heading_spans[i + 1][0] if i + 1 < len(heading_spans) else len(markdown)
        content = markdown[end:next_start].strip()
        sections.append({
            "level": level,
            "title": title,
            "content": content,
            "start": start,
            "end": next_start,
        })

    if not sections:
        sections.append({
            "level": 1,
            "title": "Document",
            "content": markdown.strip(),
            "start": 0,
            "end": len(markdown),
        })

    return sections


def _section_node_id(title: str, level: int, used_ids: set[str]) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", title.lower()).strip("_")
    base = f"section_{safe}" if safe else f"section_level_{level}"
    node_id = base
    counter = 1
    while node_id in used_ids:
        node_id = f"{base}_{counter}"
        counter += 1
    return node_id


def _find_formulas(markdown: str) -> List[Dict]:
    formulas: List[Dict] = []
    for i, m in enumerate(FORMULA_DISPLAY_RE.finditer(markdown)):
        formulas.append({
            "node_id": f"formula_display_{i + 1}",
            "content": m.group(1).strip(),
            "start": m.start(),
        })
    for i, m in enumerate(FORMULA_INLINE_RE.finditer(markdown)):
        formulas.append({
            "node_id": f"formula_inline_{i + 1}",
            "content": m.group(1).strip(),
            "start": m.start(),
        })
    formulas.sort(key=lambda f: f["start"])
    for idx, f in enumerate(formulas):
        f["node_id"] = f"formula_{idx + 1}"
    return formulas


def _find_section_containing(sections: List[Dict], position: int) -> Optional[int]:
    for i, sec in enumerate(sections):
        if sec["start"] <= position < sec["end"]:
            return i
    return None


def _find_caption_for_image(markdown: str, img) -> Optional[str]:
    label_pattern = re.compile(
        rf"\b(?:Fig(?:ure)?s?\.?\s*{img.node_id.split('_')[-1]}|"
        rf"{img.node_id})\b",
        re.IGNORECASE,
    )
    match = label_pattern.search(markdown)
    if not match:
        return None
    start = max(0, match.start() - 100)
    end = min(len(markdown), match.end() + 100)
    caption = markdown[start:end].strip()
    return caption[:300]


def build_graph_from_tree(tree_nodes: List[Dict]) -> nx.DiGraph:
    graph = nx.DiGraph()
    def _add(parent, nodes):
        for node in nodes:
            nid = "tree_" + node["node_id"]
            sidx = int(node["node_id"])
            graph.add_node(nid, node_type="section", title=node["title"],
                content=node.get("summary", ""), page=node.get("page_index", 1), _section_idx=sidx)
            if parent:
                graph.add_edge(parent, nid, edge_type="contains")
            for v in node.get("visuals", []):
                vid = v.get("image_id", "fig_" + nid)
                graph.add_node(vid, node_type="figure", title=v.get("image_id", ""),
                    content=v.get("path", ""), page=v.get("page", node.get("page_index", 1)))
                graph.add_edge(nid, vid, edge_type="contains")
            for ti, t in enumerate(node.get("tables", [])):
                tid = "tbl_" + node["node_id"] + "_" + str(ti)
                graph.add_node(tid, node_type="text_block", title="",
                    content=t.get("markdown_content", ""), page=t.get("page", node.get("page_index", 1)))
                graph.add_edge(nid, tid, edge_type="contains")
            _add(nid, node.get("nodes", []))
    _add(None, tree_nodes)
    return graph



