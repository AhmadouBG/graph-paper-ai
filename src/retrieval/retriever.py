from __future__ import annotations


def get_node_full_text(node: dict) -> str:
    """Returns the content of a node, aggregating child sub-nodes if content is empty."""
    content = node.get("content", "").strip()
    if content:
        return content

    # If node content is empty, aggregate content from child nodes
    child_texts = []
    def collect(ns):
        for child in ns:
            c = child.get("content", "").strip()
            if c:
                child_texts.append(f"### {child.get('title', '')}\n{c}")
            if child.get("nodes"):
                collect(child["nodes"])
    if node.get("nodes"):
        collect(node["nodes"])

    return "\n\n".join(child_texts)

def retrieve_nodes(selected_ids: list[str], tree: list[dict]) -> list[dict]:
    """
    Parcourt l'arbre de manière récursive, extrait les nœuds et les RE-TRIE
    selon leur ordre d'apparition original (chronologique) dans le document.
    Si un sous-nœud est sélectionné, inclut également sa section parente
    pour garantir un contexte comparatif complet (tables + texte).
    """
    # 1. Indexation et cartographie des parents
    all_nodes_map = {}
    parent_map = {}

    def index_tree(nodes: list[dict], parent: dict | None = None):
        for n in nodes:
            nid = n["node_id"]
            all_nodes_map[nid] = n
            if parent:
                parent_map[nid] = parent["node_id"]
            if n.get("nodes"):
                index_tree(n["nodes"], parent=n)

    index_tree(tree)

    # 2. Extension des IDs aux nœuds parents pour éviter le morcellement du contexte
    expanded_set = set()
    for sid in selected_ids:
        expanded_set.add(sid)
        if sid in parent_map:
            expanded_set.add(parent_map[sid])
        # Si sid a un parent_id explicite dans le nœud
        node_obj = all_nodes_map.get(sid)
        if node_obj and node_obj.get("parent_id"):
            expanded_set.add(node_obj["parent_id"])

    # 3. Récupération dans l'ordre chronologique natif
    def find_nodes(nodes: list[dict], target_ids_set: set[str]) -> list[dict]:
        found = []
        for n in nodes:
            if n["node_id"] in target_ids_set:
                n_copy = dict(n)
                n_copy["text"] = get_node_full_text(n)
                found.append(n_copy)
            if n.get("nodes"):
                found.extend(find_nodes(n["nodes"], target_ids_set))
        return found

    retrieved = find_nodes(tree, expanded_set)

    if not retrieved:
        print("⚠️ No valid nodes found by LLM. Using fallback section.")
        if tree:
            fallback = dict(tree[0])
            fallback["text"] = get_node_full_text(tree[0])
            retrieved = [fallback]
        else:
            retrieved = []

    node_ids = [n["node_id"] for n in retrieved]
    section_titles = [n["title"] for n in retrieved]
    print(f"🎯 Retrieved node IDs (with parent expansion): {node_ids}")
    print(f"📄 Sections found: {section_titles}")
    print("="*60 + "\n")

    return retrieved

