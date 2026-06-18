from __future__ import annotations

from src.query.generator import generate_answer
from src.retrieval.retriever import find_nodes_by_ids
from src.retrieval.tree_search import llm_tree_search_ollama


def vectorless_rag(
    query: str,
    tree: list[dict],
    model: str,
    verbose: bool = True,
) -> str:
    """
    Full end-to-end PageIndex RAG pipeline:

    Step 1: LLM Tree Search  → finds relevant node_ids
    Step 2: Node Retrieval   → fetches section content
    Step 3: Answer Generation → produces cited answer
    """
    if verbose:
        print(f"{'='*55}")
        print(f"🔍 Query: {query}")
        print(f"{'='*55}")

    # Step 1: Tree Search
    search_result  = llm_tree_search_ollama(query, tree, model)
    node_ids       = search_result.get("node_list", [])

    if verbose:
        print(f"\nRaw response: {search_result.get('raw_response', '')[:300]}...")
        print(f"🎯 Retrieved node IDs: {node_ids}")

    # Step 2: Retrieve nodes
    nodes = find_nodes_by_ids(tree, node_ids)

    if verbose:
        print(f"📄 Sections found: {[n['title'] for n in nodes]}")
        for n in nodes:
            v = len(n.get("visuals", []))
            t = len(n.get("tables", []))
            print(f"   [{n['node_id']}] {n['title']}: {v} visuals, {t} tables")

    # Step 3: Generate answer
    answer = generate_answer(query, nodes, model)

    if verbose:
        print(f"\n📝 Answer:\n{answer}")

    return answer
