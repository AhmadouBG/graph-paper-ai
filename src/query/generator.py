import ollama


def generate_answer(query: str, nodes: list, model: str = "qwen2.5:3b") -> str:
    """
    Generate an answer anchored in the provided context using Ollama (Qwen 2.5).
    Force the model to explicitly cite section titles and pages.
    """
    if not nodes:
        return "⚠️ No relevant sections were found in the document."

    context_parts = []
    for node in nodes:
        p_start = node.get("page_start", "?")
        p_end = node.get("page_end", "?")
        page_info = f"Page {p_start}" if p_start == p_end else f"Pages {p_start}-{p_end}"

        # We take the full and fluid content for the DeepEval judge
        context_parts.append(
            f"[Section: '{node['title']}' | {page_info}]\n"
            f"{node.get('text') or node.get('content', 'Content not available.')}"
        )
    context = "\n\n---\n\n".join(context_parts)

    print("=" * 60 + "\n" + "CONTEXT : " + context + "\n" + "=" * 60 + "\n")

    user_prompt = f"""You are an expert document analyst.
Answer the question using ONLY the provided context.
For every claim you make, cite the section title and page number in parentheses.
Be concise and precise.

Question: {query}

Context:
{context}

Answer:"""

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            options={
                "temperature": 0.0,
                "num_ctx": 2048,
                "num_predict": 512,
                "keep_alive": "10m",
            },
        )

        return response["message"]["content"]

    except Exception as e:
        return f"⚠️ Error during response generation with Ollama : {str(e)}"
