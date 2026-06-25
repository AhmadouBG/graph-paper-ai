from __future__ import annotations

from src.retrieval.tree_search import llm_tree_search_ollama
from src.retrieval.retriever import retrieve_nodes
from src.query.generator import generate_answer


def print_tree(nodes: list[dict], indent: int = 0) -> None:
    """Recursively print tree titles for a visual overview."""
    for node in nodes:
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        page = node.get("page_index") or node.get("page_start") or "?"
        print(f"{prefix}[{node['node_id']}] {node['title']}  (p.{page})")
        if node.get("nodes"):
            print_tree(node["nodes"], indent + 1)


def vectorless_rag_no_loss(query: str, tree: list[dict], model: str) -> dict:
    """
    Full RAG pipeline:
      1. Tree Search  — select relevant node IDs with an LLM
      2. Retriever    — fetch the full node content from the tree
      3. Generator    — build context and call Ollama for the final answer
    """
    # 1. Tree Search
    print("🔍 Executing LLM Tree Search...")
    selected_ids = llm_tree_search_ollama(query, tree, model)

    # 2. Retriever
    retrieved_nodes = retrieve_nodes(selected_ids, tree)

    # 3. Generator
    return generate_answer(query, retrieved_nodes, model)
