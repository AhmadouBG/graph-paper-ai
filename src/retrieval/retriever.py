from __future__ import annotations


def find_nodes_by_ids(tree: list[dict], target_ids: list[str]) -> list[dict]:
    found: list[dict] = []
    for node in tree:
        if node["node_id"] in target_ids:
            found.append(node)
        if node.get("nodes"):
            found.extend(find_nodes_by_ids(node["nodes"], target_ids))
    return found
