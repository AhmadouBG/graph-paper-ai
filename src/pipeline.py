from __future__ import annotations

# Importation de vos modules corrigés
from src.retrieval.tree_search import llm_tree_search_ollama
from src.retrieval.retriever import retrieve_nodes
from src.query.generator import generate_answer  # ✨ Changé pour la version Ollama

def print_tree(nodes: list[dict], indent: int = 0) -> None:
    """Affiche récursivement les titres de l'arbre pour un aperçu visuel."""
    for node in nodes:
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        # Utilisation de la clé officielle définie dans le parseur
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
    model: str = "qwen2.5:3b"  # 100% local par défaut
) -> dict:  # Retourne un dictionnaire avec la réponse et le raisonnement pour l'UI Streamlit
    """
    Pipeline Vectorless RAG hiérarchique complet :
      1. Tree Search  — Sélectionne les IDs des nœuds via Ollama (Sortie JSON)
      2. Retriever    — Récupère et déduplique le contenu complet des nœuds
      3. Generator    — Synthétise le contexte et génère la réponse ancrée via Ollama
    """
    
    # 1. Tree Search (Aiguillage sémantique sur l'arbre compressé)
    print("🔍 Execution du LLM Tree Search (Ollama)...")
    search_result = llm_tree_search_ollama(
        query=query, 
        tree=tree, 
    )
    
    # Extraction de la liste d'IDs depuis le dictionnaire JSON renvoyé par Qwen
    node_ids = search_result.get("node_list", [])
    thinking = search_result.get("thinking", "Pas de raisonnement fourni.")
    
    print(f"💡 Raisonnement du routeur : {thinking}")

    # 2. Retriever (Extraction récursive et sécurisée des nœuds)
    print("📄 Récupération du contenu complet des nœuds...")
    retrieved_nodes = retrieve_nodes(node_ids, tree)

    # 3. Generator (Génération de la réponse finale avec citations)
    print("🧠 Génération de la réponse ancrée (Ollama)...")
    answer = generate_answer(
         query=query,
         nodes=retrieved_nodes,
         model=model
    )
    
    retrieved_sections = []
    for n in retrieved_nodes:
        p_start = n.get("page_start", "?")
        p_end = n.get("page_end", "?")
        page_str = f"p. {p_start}" if p_start == p_end else f"p. {p_start}-{p_end}"
        retrieved_sections.append(f"{n['title']} ({page_str})")

    # Pratique pour Streamlit : On renvoie la réponse ET le raisonnement du choix des nœuds
    return {
        "answer": answer,
        "thinking": thinking,
        "retrieved_sections": retrieved_sections
    }

