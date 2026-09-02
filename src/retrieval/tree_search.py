import re
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]:
    """Nettoie et découpe le texte en tokens pour le moteur lexical BM25."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()
def find_visual_element_by_regex(query: str, flattened_nodes: list, top_k: int = 3) -> list[str]:
    """
    Analyse la requête et applique un routage contextuel intelligent :
    - Si FIGURE : Récupère le nœud de la légende + le nœud PRÉCÉDENT (données au-dessus).
    - Si TABLE : Récupère le nœud de la légende + le nœud SUIVANT (données en-dessous).
    """
    cleaned_query = query.lower().strip()
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query)
    
    # Détection du type et du numéro (gère "table iii", "fig. 5", "figure 2:")
    pattern = r'(?:(table)|(fig\.?|figure))\s+([a-zA-Z0-9_]+)'
    match = re.search(pattern, cleaned_query)
    
    if not match:
        return []
        
    is_table = bool(match.group(1))   # True si c'est un tableau
    is_figure = bool(match.group(2))  # True si c'est une figure
    target_number = match.group(3)     # Ex: "iii" ou "5"
    
    matched_indices = []
    
    # 1. Trouver les index des nœuds qui contiennent la légende
    for idx, n in enumerate(flattened_nodes):
        content_to_scan = f"{n['title']} {n.get('content', '')}".lower()
        
        # Regex tolérante aux caractères collés (ex: "figure 2:", "table iii.")
        visual_pattern = rf'(?:table|fig\.?|figure)\s+{re.escape(target_number)}(?:\b|[^a-z0-9_]|<)'
        
        if re.search(visual_pattern, content_to_scan):
            matched_indices.append(idx)
            
    if not matched_indices:
        return []

    # 2. Application de la règle de voisinage structurel scientifique
    final_node_ids = []
    seen_ids = set()
    
    for idx in matched_indices:
        current_node = flattened_nodes[idx]
        
        if is_figure:
            # RÈGLE FIGURE : L'image est au-dessus. On prend le nœud précédent si disponible.
            if idx > 0:
                prev_node = flattened_nodes[idx - 1]
                if prev_node["node_id"] not in seen_ids:
                    seen_ids.add(prev_node["node_id"])
                    final_node_ids.append(prev_node["node_id"])
            
            # On ajoute le nœud de la légende lui-même
            if current_node["node_id"] not in seen_ids:
                seen_ids.add(current_node["node_id"])
                final_node_ids.append(current_node["node_id"])
                
        elif is_table:
            # RÈGLE TABLEAU : La légende est en haut. On ajoute d'abord le nœud de la légende.
            if current_node["node_id"] not in seen_ids:
                seen_ids.add(current_node["node_id"])
                final_node_ids.append(current_node["node_id"])
                
            # Les données HTML sont en-dessous. On prend le nœud suivant si disponible.
            if idx < len(flattened_nodes) - 1:
                next_node = flattened_nodes[idx + 1]
                if next_node["node_id"] not in seen_ids:
                    seen_ids.add(next_node["node_id"])
                    final_node_ids.append(next_node["node_id"])

    return final_node_ids[:top_k]

def llm_tree_search_ollama(query: str, tree: list, model: str = "qwen2.5:3b", top_k: int = 3) -> dict:
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
        titles = [n["title"] for n in flattened_nodes if n["node_id"] in regex_matched_ids]
        return {
            "thinking": f"🎯 [Regex] Composant détecté dans la structure brute de : {', '.join(titles)}",
            "node_list": regex_matched_ids
        }

    # 3. Exécution de la Passe B (BM25 pour questions génériques avec détection d'intention comparative)
    corpus_tokens = []
    for n in flattened_nodes:
        # Ponderer le titre pour donner plus de poids à la sémantique de la section
        node_text = f"{n['title']} {n['title']} {n.get('content', '')}"
        corpus_tokens.append(tokenize(node_text))

    bm25 = BM25Okapi(corpus_tokens)
    query_tokens = tokenize(query)
    scores = list(bm25.get_scores(query_tokens))

    # 4. Passe B.1: Détection d'intention comparative / de synthèse
    COMPARATIVE_KEYWORDS = {
        "best", "worst", "compare", "comparison", "overall", "performance",
        "conclusion", "versus", "vs", "summary", "highest", "lowest", "rank",
        "ranking", "evaluate", "evaluation", "which"
    }
    SYNTHESIS_SECTION_KEYWORDS = {
        "performance", "analysis", "conclusion", "discussion", "results",
        "result", "comparison", "summary", "evaluation", "overview"
    }

    is_comparative = any(kw in query_tokens for kw in COMPARATIVE_KEYWORDS)

    node_scores = []
    for idx, score in enumerate(scores):
        n = flattened_nodes[idx]
        title_lower = n["title"].lower()

        # Si la requête est comparative, booster les nœuds de synthèse / conclusion / analyse de performance
        if is_comparative:
            if any(s_kw in title_lower for s_kw in SYNTHESIS_SECTION_KEYWORDS):
                # Triple le score si le titre indique une section de synthèse
                score *= 3.0 if score > 0 else 5.0

        node_scores.append({"node": n, "score": score})

    node_scores.sort(key=lambda x: x["score"], reverse=True)

    # Pour les requêtes comparatives, augmenter top_k pour capturer à la fois le tableau et l'analyse
    effective_top_k = top_k + 1 if is_comparative else top_k

    selected_ids = []
    selected_titles = []
    for item in node_scores[:effective_top_k]:
        if item["score"] > 0:
            selected_ids.append(item["node"]["node_id"])
            selected_titles.append(item["node"]["title"])

    if not selected_ids:
        selected_ids = [flattened_nodes[0]["node_id"]]
        selected_titles = [flattened_nodes[0]["title"]]

    tag = "📊 [BM25 + Comparative Boost]" if is_comparative else "🔍 [BM25]"
    return {
        "thinking": f"{tag} Correspondance lexicale dans : {', '.join(selected_titles)}",
        "node_list": selected_ids
    }

