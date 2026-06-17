from __future__ import annotations

from src.query.anchor_extraction import AnchorType, extract_query_anchors


def find_nodes_by_ids(tree: list[dict], target_ids: list[str]) -> list[dict]:
    found: list[dict] = []
    for node in tree:
        if node["node_id"] in target_ids:
            found.append(node)
        if node.get("nodes"):
            found.extend(find_nodes_by_ids(node["nodes"], target_ids))
    return found


def _find_nodes_containing_anchor(tree: list[dict], anchor_id: str) -> list[dict]:
    matched: list[dict] = []
    for node in tree:
        for v in node.get("visuals", []):
            if v.get("image_id") == anchor_id:
                matched.append(node)
                break
        else:
            for t in node.get("tables", []):
                table_page = t.get("page", "")
                tid = f"table_{table_page}"
                if tid == anchor_id:
                    matched.append(node)
                    break
        if node.get("nodes"):
            matched.extend(_find_nodes_containing_anchor(node["nodes"], anchor_id))
    return matched


def retrieve_via_anchors(query: str, tree: list[dict]) -> list[dict]:
    parsed = extract_query_anchors(query)
    if not parsed.has_anchors:
        return []
    section_ids = [
        a.node_id for a in parsed.anchors
        if a.anchor_type in (AnchorType.SECTION,)
    ]
    visual_ids = [
        a.node_id for a in parsed.anchors
        if a.anchor_type in (AnchorType.FIGURE, AnchorType.TABLE, AnchorType.EQUATION)
    ]
    matched: list[dict] = []
    if section_ids:
        matched.extend(find_nodes_by_ids(tree, section_ids))
    for vid in visual_ids:
        matched.extend(_find_nodes_containing_anchor(tree, vid))
    seen = set()
    deduped = []
    for n in matched:
        if id(n) not in seen:
            seen.add(id(n))
            deduped.append(n)
    return deduped
