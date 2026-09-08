from __future__ import annotations

from src.query.generator import generate_answer
from src.retrieval.retriever import retrieve_nodes
from src.retrieval.tree_search import llm_tree_search_ollama


def print_tree(nodes: list[dict], indent: int = 0) -> None:
    """display recursively the titles of the tree for a visual preview."""
    for node in nodes:
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        # Use of the official key defined in the parser
        page = node.get("page_start") or "?"
        print(f"{prefix}[{node['node_id']}] {node['title']}  (p.{page})")
        if node.get("nodes"):
            print_tree(node["nodes"], indent + 1)


def get_total_pages(nodes: list[dict]) -> int:
    """Parcourt l'arbre complet et retourne la page de fin la plus élevée."""
    max_page = 1

    def walk(ns):
        nonlocal max_page
        for n in ns:
            max_page = max(max_page, n.get("page_end", 1), n.get("page_start", 1))
            if n.get("nodes"):
                walk(n["nodes"])

    walk(nodes)
    return max_page


def vectorless_rag_no_loss(
    query: str,
    tree: list[dict],
    model: str = "qwen2.5:3b",
) -> dict:
    """
    Complete hierarchical Vectorless RAG pipeline :
      1. Tree Search  — Selects node IDs via Ollama (JSON Output)
      2. Retriever    — Retrieves and deduplicates full node content
      3. Generator    — Synthesizes context and generates anchored response via Ollama
    """

    # 1. Tree Search (Semantic routing on the compressed tree)
    print("🔍 Executing LLM Tree Search (Ollama)...")
    search_result = llm_tree_search_ollama(
        query=query,
        tree=tree,
    )

    # Extraction of the list of IDs from the JSON dictionary returned by Qwen
    node_ids = search_result.get("node_list", [])
    thinking = search_result.get("thinking", "Pas de raisonnement fourni.")

    print(f"💡 Thinking of the routeur : {thinking}")

    print("📄 Retrieving the full content of the nodes...")
    retrieved_nodes = retrieve_nodes(node_ids, tree)

    print("🧠 Generation of the answer (Ollama)...")
    answer = generate_answer(query=query, nodes=retrieved_nodes, model=model)

    retrieved_sections = []
    retrieved_texts = []
    for n in retrieved_nodes:
        p_start = n.get("page_start", "?")
        p_end = n.get("page_end", "?")
        page_str = f"p. {p_start}" if p_start == p_end else f"p. {p_start}-{p_end}"
        retrieved_sections.append(f"{n['title']} ({page_str})")
        retrieved_texts.append(n.get("text") or n.get("content", ""))

    # Practise for Streamlit : return the answer and the reasoning of the choice of the nodes
    return {
        "answer": answer,
        "thinking": thinking,
        "retrieved_sections": retrieved_sections,
        "retrieved_texts": retrieved_texts,
    }
