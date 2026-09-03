import ollama

def generate_answer(query: str, nodes: list, model: str = "qwen2.5:3b") -> str:
    """
    Génère une réponse ancrée dans le contexte fourni en utilisant Ollama (Qwen 2.5).
    Force le modèle à citer explicitement les titres de sections et les pages.
    """
    if not nodes:
        return "⚠️ Aucune section pertinente n'a été trouvée dans le document."
    
    # 1. Reconstruction précise du contexte avec vos vraies clés (content, page_start, page_end)
    context_parts = []
    for node in nodes:
        # Formater l'intervalle de pages de manière propre
        p_start = node.get("page_start", "?")
        p_end = node.get("page_end", "?")
        page_info = f"Page {p_start}" if p_start == p_end else f"Pages {p_start}-{p_end}"
        
        context_parts.append(
            f"[Section: '{node['title']}' | {page_info}]\n"
            f"{node.get('text') or node.get('content', 'Contenu non disponible.')}"
        )
    context = "\n\n---\n\n".join(context_parts)
    
    print("="*60 + "\n" + "CONTEXT : " + context + "\n" + "="*60 + "\n")

    user_prompt = f"""You are an expert document analyst.
Answer the question using ONLY the provided context.
For every claim you make, cite the section title and page number in parentheses.
Be concise and precise.

Question: {query}

Context:
{context}

Answer:"""
    
    try:
        # 3. Appel à l'instance locale d'Ollama
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.0,  # Température basse pour garantir la fidélité au texte source
                "num_ctx": 2048,
                "num_predict": 512,
                "keep_alive": "10m"
                    # Fenêtre étendue pour accueillir tout le contenu des nœuds extraits
            }
        )
        
        return response['message']['content']

    except Exception as e:
        return f"⚠️ Erreur lors de la génération de la réponse avec Ollama : {str(e)}"
