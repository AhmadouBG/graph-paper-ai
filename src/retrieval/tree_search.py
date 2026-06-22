# from __future__ import annotations

# import ollama
# import json

# def llm_tree_search_ollama(query: str, tree: list[dict], model: str) -> dict:
#     # Compress tree to save tokens — only send titles + short summaries
#     def flatten_and_compress(nodes) -> list[dict]:
#         flat_list = []
#         for n in nodes:
#             # Récupération sécurisée du texte (on teste toutes les clés possibles de votre structure)
#             text_content = n.get("text", "") or n.get("summary", "") or ""
            
#             flat_list.append({
#                 "node_id": n["node_id"],
#                 "title":   n["title"],
#                 "page":    n.get("page_index", "?"),
#                 "summary": text_content[:250] # Les 250 premiers caractères suffisent pour le choix
#             })
            
#             # Si le nœud a des enfants (clé "nodes"), on les extrait et on les met à plat aussi
#             if n.get("nodes"):
#                 flat_list.extend(flatten_and_compress(n["nodes"]))
                
#         return flat_list

#     # Production de la liste plate
#     compressed_flat_tree = flatten_and_compress(tree)
    
    
#     prompt = f"""You are given a query and a document's tree structure (like a Table of Contents).
# Your task: identify which node IDs most likely contain the answer to the query.
# Think step-by-step about which sections are relevant.

# Query: {query}

# Document Tree:
# {json.dumps(compressed_flat_tree, indent=2)}

# Reply ONLY in this exact JSON format:
# {{
#   "thinking": "<your step-by-step reasoning>",
#   "node_list": ["node_id1", "node_id2"]
# }}"""

#      # 2. Appel Ollama avec gestion des plantages du modèle 3B sur document long
#     try:
#         response = ollama.chat(
#             model=model,
#             messages=[{"role": "user", "content": prompt}],
#             format="json", # Force Ollama à structurer sa sortie
#             options={"num_ctx": 16384} # ✨ CORRECTION 2 : Augmente la fenêtre de contexte pour 22 pages
#         )
#         content = response["message"]["content"]
#         return json.loads(content)
        
#     except (json.JSONDecodeError, KeyError, Exception) as e:
#         # En cas de coupure ou JSON invalide par le modèle 3B, on applique un Fallback
#         print(f"⚠️ Ollama JSON Error or Timeout: {e}. Extraction manuelle des IDs par Regex.")
        
#         # Tentative d'extraction des IDs par Regex si le JSON brut a coupé
#         found_ids = re.findall(r'"node_id":\s*"(\d+)"', json.dumps(compressed_tree))
#         # Fallback de secours : On renvoie les 3 premiers IDs de l'arbre pour éviter que le RAG renvoie None
#         fallback_list = found_ids[:3] if found_ids else []
        
#         return {
#             "thinking": "Fallback activated due to model generation error.",
#             "node_list": fallback_list,
#             "raw_response": content if 'content' in locals() else ""
#         }
