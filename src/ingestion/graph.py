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


def _build_graph_from_llamaparse_tree(tree_nodes: List[Dict]) -> nx.DiGraph:
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


def build_adjacency_map(
    parsed: ProcessingResult,
    refs: Optional[List[CrossReference]] = None,
) -> nx.DiGraph:
    if parsed.metadata.get("parser") == "llamaparse" and "llamaparse_tree" in parsed.metadata:
        graph = _build_graph_from_llamaparse_tree(parsed.metadata["llamaparse_tree"])

        for n, attr in list(graph.nodes(data=True)):
            if attr.get("node_type") == "section":
                content = attr.get("content", "")
                formulas = _find_formulas(content)
                for f in formulas:
                    graph.add_node(
                        f["node_id"],
                        node_type="formula",
                        content=f["content"],
                        page=attr.get("page"),
                        bbox=None,
                    )
                    graph.add_edge(n, f["node_id"], edge_type="formula")

        if refs:
            for ref in refs:
                target_id = ref.target_node_id
                if not graph.has_node(target_id):
                    graph.add_node(
                        target_id,
                        node_type="figure",
                        content="",
                        page=ref.page,
                        bbox=None,
                    )
                ref_node_id = f"ref_{target_id}_{abs(hash(ref.context)) % 10000}"
                graph.add_node(
                    ref_node_id,
                    node_type="text_block",
                    content=ref.context,
                    page=ref.page,
                    bbox=None,
                )
                graph.add_edge(ref_node_id, target_id, edge_type="references")

                containing_sec_id = None
                for n, attr in graph.nodes(data=True):
                    if attr.get("node_type") == "section":
                        if ref.context[:50] in attr.get("content", ""):
                            containing_sec_id = n
                            break
                if containing_sec_id:
                    graph.add_edge(containing_sec_id, ref_node_id, edge_type="contains")

        for edge in parsed.edges:
            if not graph.has_node(edge.source_id):
                graph.add_node(
                    edge.source_id,
                    node_type="figure",
                    content="",
                    page=edge.page,
                    bbox=None,
                )
            if not graph.has_node(edge.target_id):
                graph.add_node(
                    edge.target_id,
                    node_type="text_block",
                    content="",
                    page=edge.page,
                    bbox=None,
                )
            graph.add_edge(edge.source_id, edge.target_id, edge_type=edge.edge_type)

        return graph

    graph = nx.DiGraph()
    used_section_ids: set[str] = set()

    sections = _parse_sections(parsed.markdown)
    parent_stack: List[str] = []

    for idx, sec in enumerate(sections):
        node_id = _section_node_id(sec["title"], sec["level"], used_section_ids)
        used_section_ids.add(node_id)
        sec["_node_id"] = node_id
        graph.add_node(
            node_id,
            node_type="section",
            title=sec["title"],
            content=sec["content"],
            page=None,
            bbox=None,
            _section_idx=idx,
        )

        while parent_stack:
            parent_id = parent_stack[-1]
            parent_sec = next(s for s in sections if s.get("_node_id") == parent_id)
            if parent_sec["level"] >= sec["level"]:
                parent_stack.pop()
            else:
                break

        if parent_stack:
            graph.add_edge(parent_stack[-1], node_id, edge_type="contains")

        parent_stack.append(node_id)

    for sec in sections:
        formulas = _find_formulas(sec["content"])
        for f in formulas:
            graph.add_node(
                f["node_id"],
                node_type="formula",
                content=f["content"],
                page=None,
                bbox=None,
            )
            graph.add_edge(sec["_node_id"], f["node_id"], edge_type="formula")

    for img in parsed.images:
        graph.add_node(
            img.node_id,
            node_type="figure",
            content=str(img.path),
            page=img.page,
            bbox=img.bbox,
        )

        caption = _find_caption_for_image(parsed.markdown, img)
        if caption:
            caption_id = f"caption_{img.node_id}"
            graph.add_node(
                caption_id,
                node_type="text_block",
                content=caption,
                page=img.page,
                bbox=None,
            )
            graph.add_edge(img.node_id, caption_id, edge_type="captions")

        image_ref_pos = parsed.markdown.lower().find(img.node_id)
        if image_ref_pos >= 0:
            sec_idx = _find_section_containing(sections, image_ref_pos)
            if sec_idx is not None:
                graph.add_edge(sections[sec_idx]["_node_id"], img.node_id, edge_type="contains")

    if refs:
        for ref in refs:
            target_id = ref.target_node_id
            if not graph.has_node(target_id):
                graph.add_node(
                    target_id,
                    node_type="figure",
                    content="",
                    page=ref.page,
                    bbox=None,
                )
            ref_node_id = f"ref_{target_id}_{abs(hash(ref.context)) % 10000}"
            graph.add_node(
                ref_node_id,
                node_type="text_block",
                content=ref.context,
                page=ref.page,
                bbox=None,
            )
            graph.add_edge(ref_node_id, target_id, edge_type="references")

            ref_pos = parsed.markdown.find(ref.context[:50])
            if ref_pos >= 0:
                sec_idx = _find_section_containing(sections, ref_pos)
                if sec_idx is not None:
                    graph.add_edge(
                        sections[sec_idx]["_node_id"], ref_node_id, edge_type="contains",
                    )

    for edge in parsed.edges:
        if not graph.has_node(edge.source_id):
            graph.add_node(
                edge.source_id,
                node_type="figure",
                content="",
                page=edge.page,
                bbox=None,
            )
        if not graph.has_node(edge.target_id):
            graph.add_node(
                edge.target_id,
                node_type="text_block",
                content="",
                page=edge.page,
                bbox=None,
            )
        graph.add_edge(edge.source_id, edge.target_id, edge_type=edge.edge_type)

    return graph
