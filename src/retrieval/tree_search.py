import re
from rank_bm25 import BM25Okapi
from deepeval.tracing import observe, update_current_span, update_retriever_span

def tokenize(text: str) -> list[str]:
    """Nettoie et découpe le texte en tokens pour le moteur lexical BM25."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

def find_visual_element_by_regex(query: str, flattened_nodes: list) -> list[str]:
    """
    Scanne les requêtes pour détecter les mentions de tableaux/figures.
    Gère les chiffres romains (III, IV) et arabes, insensible aux balises HTML collées.
    """
    cleaned_query = query.lower().strip()
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query)
    
    # Capture le type de composant et son identifiant (ex: table, iii, 3)
    pattern = r'(?:table|fig\.?|figure)\s+([a-zA-Z0-9_]+)'
    match = re.search(pattern, cleaned_query)
    
    if not match:
        return []
        
    target_number = match.group(1)
    matched_node_ids = []
    
    for n in flattened_nodes:
        # Analyse sur une copie en minuscules. Le contenu original du nœud reste INTACT.
        content_to_scan = f"{n['title']} {n.get('content', '')}".lower()
        
        # Le pattern accepte que le numéro soit suivi de ponctuation (ex: "table iii.") 
        # ou directement d'une balise HTML (ex: "table iii<table>")
        visual_pattern = rf'(?:table|fig\.?|figure)\s+{re.escape(target_number)}(?:\b|[^a-z0-9_]|<)'
        
        if re.search(visual_pattern, content_to_scan):
            matched_node_ids.append(n["node_id"])
            
    return matched_node_ids

@observe(type="retriever")
def llm_tree_search_ollama(query: str, tree: list, model: str = "qwen2.5:3b", top_k: int = 2) -> dict:
    """
    Routeur hybride pour Vectorless RAG :
    - Passe A : Détection déterministe par Regex (idéal pour Table III, Fig 5) en < 1ms.
    - Passe B : Recherche lexicale BM25 pour les concepts sémantiques.
    """
    # 1. Mise à plat de l'arbre
    flattened_nodes = []
    def flatten(nodes):
        for n in nodes:
            flattened_nodes.append(n)
            if n.get("nodes"):
                flatten(n["nodes"])
    flatten(tree)

    if not flattened_nodes:
        return {"thinking": "L'arbre documentaire est vide.", "node_list": ["0000"]}

    # 2. Exécution de la Passe A (Regex pour composants visuels)
    regex_matched_ids = find_visual_element_by_regex(query, flattened_nodes)
    
    if regex_matched_ids:
        final_ids = regex_matched_ids[:top_k]
        titles = [n["title"] for n in flattened_nodes if n["node_id"] in final_ids]
        update_retriever_span(top_k=top_k)
        update_current_span(
            input=query,
            output=titles,
            metadata={"method": "regex", "top_k": top_k, "selected_ids": final_ids},
        )
        return {
            "thinking": f"🎯 [Regex] Composant détecté dans la structure brute de : {', '.join(titles)}",
            "node_list": final_ids
        }

    # 3. Exécution de la Passe B (BM25 pour questions génériques)
    corpus_tokens = []
    for n in flattened_nodes:
        node_text = f"{n['title']} {n.get('content', '')}"
        corpus_tokens.append(tokenize(node_text))

    bm25 = BM25Okapi(corpus_tokens)
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    node_scores = []
    for idx, score in enumerate(scores):
        node_scores.append({"node": flattened_nodes[idx], "score": score})
        
    node_scores.sort(key=lambda x: x["score"], reverse=True)

    selected_ids = []
    selected_titles = []
    for item in node_scores[:top_k]:
        if item["score"] > 0:
            selected_ids.append(item["node"]["node_id"])
            selected_titles.append(item["node"]["title"])

    if not selected_ids:
        selected_ids = [flattened_nodes[0]["node_id"]]
        selected_titles = [flattened_nodes[0]["title"]]

    update_retriever_span(top_k=top_k)
    update_current_span(
        input=query,
        output=selected_titles,
        metadata={"method": "bm25", "top_k": top_k, "selected_ids": selected_ids},
    )
    return {
        "thinking": f"🔍 [BM25] Correspondance lexicale dans : {', '.join(selected_titles)}",
        "node_list": selected_ids
    }
