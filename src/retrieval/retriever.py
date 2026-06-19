from __future__ import annotations


def find_nodes_by_ids(tree: list[dict], target_ids: list[str]) -> list[dict]:
    found = []
    for n in tree:
        if n.get("node_id") in target_ids or n.get("id") in target_ids:
            found.append(n)
        if n.get("nodes"):
            found.extend(find_nodes_by_ids(n["nodes"], target_ids))
    return found
