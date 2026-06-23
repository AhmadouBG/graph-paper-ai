import json
import ollama
import re

def print_tree(nodes, indent=0):
    """Affiche récursivement les titres de l'arbre pour un aperçu visuel."""
    for node in nodes:
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        # Gère à la fois page_index et page_start selon votre générateur d'arbre
        page = node.get("page_index") or node.get("page_start") or "?"
        print(f"{prefix}[{node['node_id']}] {node['title']}  (p.{page})")
        if node.get("nodes"):
            print_tree(node["nodes"], indent + 1)

def llm_tree_search_ollama(query: str, tree: list[dict], model: str) -> list[str]:
    """
    Utilise Ollama pour analyser la structure de l'arbre et sélectionner les meilleurs IDs.
    Optimisé pour CPU (Zéro texte brut envoyé, uniquement titres + métadonnées).
    """
    # 1. Aplatir et compresser l'arbre au strict minimum pour économiser la RAM/CPU
    def compress_and_flatten(nodes):
        flat_list = []
        for n in nodes:
            has_visuals = "Yes" if n.get("base64_images") else "No"
            flat_list.append({
                "node_id": n["node_id"],
                "title": n["title"],
                "pages": f"{n.get('page_start', '?')}-{n.get('page_end', '?')}",
                "contains_images_or_figures": has_visuals
            })
            if n.get("nodes"):
                flat_list.extend(compress_and_flatten(n["nodes"]))
        return flat_list

    compressed_tree = compress_and_flatten(tree)

    # 2. Prompt ultra-direct forçant le modèle à cibler les figures si demandées
    prompt = f"""You are a document navigation assistant. Analyze the user query and the document structure.
Select up to 2 Node IDs that are the most relevant to answer the query.

CRITICAL RULE: If the query asks for a specific Figure (e.g., 'fig 4' or 'figure 4'), prioritize sections where 'contains_images_or_figures' is 'Yes' and matches the logical flow.

Query: {query}

Document Structure:
{json.dumps(compressed_tree, indent=2)}

Reply ONLY in this exact JSON format:
{{
  "thinking": "<short reasoning>",
  "node_list": ["node_id1", "node_id2"]
}}"""

    try:
        # Premier appel LLM (léger car le prompt est très court)
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"num_ctx": 4096}
        )
        content = response["message"]["content"]
        result = json.loads(content)
        
        # Affichage du raisonnement du LLM dans la console
        print("\n" + "="*60)
        print(f"🧠 LLM Tree Search Reasoning: {result.get('thinking', 'N/A')}")
        return result.get("node_list", [])
        
    except Exception as e:
        print(f"⚠️ Ollama Tree Search Error: {e}. Fallback to first nodes.")
        # Sécurité CPU : si Ollama échoue ou coupe, on renvoie le premier ID valide
        return [compressed_tree[0]["node_id"]] if compressed_tree else []

def vectorless_rag_no_loss(query: str, tree: list[dict], model: str) -> str:
    # 1. Affichage de l'arbre complet (Pretty-print) au début de la question
    print("📚 Full tree:\n")
    print(json.dumps(tree, indent=2))
    print("\n" + "="*60)
    
    
    print("📚 Full Document Structure:\n")
    print_tree(tree)
    print("\n" + "="*60)

   # ── ÉTAPE 1 : Recherche sémantique de l'arbre via Ollama ──────────────────
    print("🔍 Executing LLM Tree Search...")
    selected_ids = llm_tree_search_ollama(query, tree, model)
    
    # ── ÉTAPE 2 : Extraction récursive des nœuds complets sélectionnés ────────
    def find_nodes(nodes, target_ids):
        found = []
        for n in nodes:
            if n["node_id"] in target_ids:
                found.append(n)
            if n.get("nodes"):
                found.extend(find_nodes(n["nodes"], target_ids))
        return found

    retrieved_nodes = find_nodes(tree, selected_ids)
    
    # Sécurité si le LLM n'a rien sélectionné ou s'est trompé d'ID
    if not retrieved_nodes:
        print("⚠️ No valid nodes found by LLM. Using fallback section.")
        retrieved_nodes = [tree[0]]

    # Extraction des données pour l'affichage final
    node_ids = [n["node_id"] for n in retrieved_nodes]
    section_titles = [n["title"] for n in retrieved_nodes]
    
    print(f"🎯 Retrieved node IDs: {node_ids}")
    print(f"📄 Sections found: {section_titles}")
    print("="*60 + "\n")

    # ── ÉTAPE 3 : Collecte du contexte réel et des images Base64 ──────────────
    context_list = []
    source_citations = []
    ollama_images = []
    
    for sec in retrieved_nodes:
        # Tronquer le contenu textuel pour ne pas saturer le deuxième appel CPU
        truncated_content = sec.get("content", "")[:2500]
        pages_range = f"{sec.get('page_start', '?')}-{sec.get('page_end', '?')}"
        
        context_list.append(f"[Pages: {pages_range} | Section: {sec['title']}]\n{truncated_content}")
        source_citations.append(f"Section: '{sec['title']}', Page {pages_range}")
        
        # Récupération des images Base64 stockées en RAM
        for b64_str in sec.get("base64_images", []):
            ollama_images.append(b64_str)
            
    context = "\n\n".join(context_list)

    # ── ÉTAPE 4 : Génération de la réponse finale avec le contexte complet ────
    generation_prompt = f"""You are an advanced Vision-Language AI. 
Answer the query using ONLY the verified document context below.
If images are provided, analyze them carefully to build your response.

Query: {query}

Context:
{context}"""

    message_payload = {
        "role": "user",
        "content": generation_prompt
    }
    
    if ollama_images:
        print(f"I'll upload {len(ollama_images)} image(s) from memory to Ollama content.")
        message_payload["images"] = ollama_images

    try:
        # Deuxième appel LLM : Génération finale
        final_res = ollama.chat(
            model=model, 
            messages=[message_payload],
            options={
                "num_ctx": 4096,
                "num_predict": 256
            }
        )
        answer = final_res["message"]["content"].strip()
    except Exception as e:
        answer = f"Error during final response generation: {str(e)}"

    return {
        "answer": answer,
        "sources": source_citations
    }
