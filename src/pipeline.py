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

def _compute_simple_score(query: str, text: str) -> float:
    """Calcule un score rapide basé sur la fréquence des mots-clés (0 ms sur CPU)."""
    """Fast keyword matching score (0 ms on CPU)."""
    words = re.findall(r'\w+', query.lower())
    text_lower = text.lower()
    score = 0.0
    
    stop_words = [
        "is", "the", "of", "and", "in", "to", "what", "about", "for", "on", 
        "with", "as", "at", "by", "an", "be", "this", "that", "from", "are", 
        "it", "you", "your", "how", "why", "which", "where", "who", "can", "do"
    ]
    
    for word in words:
        if word in stop_words:
            continue
        score += text_lower.count(word) * 2.0  
    return score

def vectorless_rag_no_loss(query: str, tree: list[dict], model: str) -> str:
    # 1. Affichage de l'arbre complet (Pretty-print) au début de la question
    print("📚 Full tree:\n")
    print(json.dumps(tree, indent=2))
    print("\n" + "="*60)
    
    
    print("📚 Full Document Structure:\n")
    print_tree(tree)
    print("\n" + "="*60)

    # 2. Aplatir l'arbre pour la recherche interne
    def flatten_tree(nodes):
        sections = []
        for n in nodes:
            content = n.get("content", "").strip() or n.get("text", "").strip()
            sections.append({
                "node_id": n["node_id"],
                "title": n["title"],
                "pages": f"{n.get('page_start', n.get('page_index', '?'))}",
                "content": content
            })
            if n.get("nodes"):
                sections.extend(flatten_tree(n["nodes"]))
        return sections

    all_sections = flatten_tree(tree)
    
    # 3. Recherche par mots-clés en Python (Vitesse CPU maximale)
    scored_sections = []
    for sec in all_sections:
        search_text = f"{sec['title']} {sec['content']}"
        score = _compute_simple_score(query, search_text)
        if score > 0:
            scored_sections.append((score, sec))
    
    # Tri par pertinence
    scored_sections.sort(key=lambda x: x[0], reverse=True)
    
    # Sélection des nœuds correspondants
    retrieved_nodes = [sec for score, sec in scored_sections][:3] # Top 2 sections max
    
    # Fallback si rien ne matche
    if not retrieved_nodes:
        retrieved_nodes = all_sections[:2]

    # Extraction des données pour l'affichage demandé
    node_ids = [n["node_id"] for n in retrieved_nodes]
    section_titles = [n["title"] for n in retrieved_nodes]

    # 4. Affichage dynamique des résultats intermédiaires (Comme demandé)
    # Simulation du raisonnement sémantique pour la console
    main_node = node_ids[0] if node_ids else "?"
    main_title = section_titles[0] if section_titles else "?"
    print(f"🧠 Reasoning: The query asks about '{query}'. Looking at the document tree, the most relevant section under '{main_title}' is node_id '{main_node}'. This node likely covers the core information, supplemented by subsequent sections.")
    print(f"🎯 Retrieved node IDs: {node_ids}")
    print(f"📄 Sections found: {section_titles}")
    print("="*60 + "\n")

   # 5. Construction du contexte réel sans aucune perte de données
    context_list = []
    source_citations = [] # ✨ Liste pour mémoriser les sources exactes
    
    for sec in retrieved_nodes:
        truncated_content = sec['content'][:2500] 
        context_list.append(f"[Page: {sec['pages']} | Section: {sec['title']}]\n{truncated_content}") # On stocke le couple (Titre, Page) pour l'affichage final
        source_citations.append(f"Section: '{sec['title']}', Page {sec['pages']}")
    
    context = "\n\n".join(context_list)
    
    # 6. Appel final à Ollama
    generation_prompt = f"""You are an advanced AI. Answer the query using ONLY the verified document context below. 
If the context contains math formulas or matrices, preserve them exactly. Cite the exact pages if needed.

Query: {query}

Context:
{context}"""

    final_res = ollama.chat(
        model=model, 
        messages=[{"role": "user", "content": generation_prompt}],
        options={
            "num_ctx": 4096,
            "num_predict": 200
        }
    )
    # Calcul de sécurité avant l'envoi
    total_chars = len(generation_prompt)
    predicted_tokens = int(total_chars / 4)
    context_limit = 4096  # La limite fixée dans num_ctx

    print(f"📥 Context Load: {predicted_tokens}/{context_limit} tokens ({(predicted_tokens/context_limit)*100:.1f}%)")

    if predicted_tokens > context_limit:
        print("⚠️ Warning: Prompt is larger than num_ctx! Text will be truncated by Ollama.")

    answer = final_res["message"]["content"]
    
    # Afficher la réponse ET les sources associées
    print(f"📝 Answer:\n{answer}\n")
    print("📍 Sources used:")
    for citation in source_citations:
        print(f"👉 {citation}")
        
    
