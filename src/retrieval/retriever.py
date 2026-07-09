from __future__ import annotations

from deepeval.tracing import observe, update_current_span


@observe(type="retriever")
def retrieve_nodes(selected_ids: list[str], tree: list[dict]) -> list[dict]:
    """
    Parcourt l'arbre de manière récursive, extrait les nœuds et les RE-TRIE 
    impérativement selon leur ordre d'apparition original (chronologique) 
    dans le document. Élimine également les doublons et les nœuds vides.
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

    # The retrieved node contents are the context passed to the generator.
    update_current_span(
        input=selected_ids,
        output=section_titles,
        retrieval_context=[n.get("content", "") for n in retrieved],
        metadata={"retrieved_node_ids": node_ids},
    )

    return retrieved
