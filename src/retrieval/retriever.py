from __future__ import annotations


def retrieve_nodes(selected_ids: list[str], tree: list[dict]) -> list[dict]:
    """
    Recursively walks the tree and returns the full node dicts
    matching the selected IDs. Falls back to the root node if nothing is found.
    """
    def find_nodes(nodes: list[dict], target_ids: list[str]) -> list[dict]:
        found = []
        for n in nodes:
            if n["node_id"] in target_ids:
                found.append(n)
            if n.get("nodes"):
                found.extend(find_nodes(n["nodes"], target_ids))
        return found

    retrieved = find_nodes(tree, selected_ids)

    if not retrieved:
        print("⚠️ No valid nodes found by LLM. Using fallback section.")
        retrieved = [tree[0]] if tree else []

    node_ids = [n["node_id"] for n in retrieved]
    section_titles = [n["title"] for n in retrieved]
    print(f"🎯 Retrieved node IDs: {node_ids}")
    print(f"📄 Sections found: {section_titles}")
    print("="*60 + "\n")

    return retrieved