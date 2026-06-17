from __future__ import annotations

from src.llm.ollama_client import OllamaClient
from src.query.generator import generate_answer
from src.retrieval.retriever import find_nodes_by_ids, retrieve_via_anchors
from src.retrieval.tree_search import llm_tree_search


def vectorless_rag(
    query: str,
    tree: list[dict],
    llm: OllamaClient,
    verbose: bool = True,
) -> str:
    if verbose:
        print(f"{'='*55}")
        print(f"Query: {query}")
        print(f"{'='*55}")

    anchor_nodes = retrieve_via_anchors(query, tree)
    if anchor_nodes:
        if verbose:
            print(f"Anchor-matched sections: {[n['title'] for n in anchor_nodes]}")
        nodes = anchor_nodes
    else:
        search_result = llm_tree_search(query, tree, llm)
        node_ids = search_result.get("node_list", [])
        if verbose:
            reasoning = str(search_result.get("thinking", ""))[:300]
            print(f"Tree search reasoning: {reasoning}...")
            print(f"Retrieved node IDs: {node_ids}")
        nodes = find_nodes_by_ids(tree, node_ids)

    if verbose:
        print(f"Sections found: {[n['title'] for n in nodes]}")

    answer = generate_answer(query, nodes, llm)

    if verbose:
        print(f"\nAnswer:\n{answer}")

    return answer
