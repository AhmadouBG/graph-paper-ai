from __future__ import annotations

import ollama


def generate_answer(query: str, retrieved_nodes: list[dict], model: str) -> dict:
    """
    Builds a context string and an image list from the retrieved nodes,
    then calls Ollama (text-only or multimodal) to produce the final answer.

    Returns:
        dict with keys "answer" (str) and "sources" (list[str])
    """
    context_list: list[str] = []
    source_citations: list[str] = []
    ollama_images: list[str] = []

    for sec in retrieved_nodes:
        truncated_content = sec.get("content", "")[:2500]
        pages_range = f"{sec.get('page_start', '?')}-{sec.get('page_end', '?')}"

        context_list.append(f"[Pages: {pages_range} | Section: {sec['title']}]\n{truncated_content}")
        source_citations.append(f"Section: '{sec['title']}', Page {pages_range}")

        for b64_str in sec.get("base64_images", []):
            ollama_images.append(b64_str)

    context = "\n\n".join(context_list)

    generation_prompt = f"""You are an advanced, highly precise AI assistant. 
Answer the user query based strictly on the verified text context and any attached visual data provided below.

INSTRUCTIONS:
1. Focus directly on answering the specific query.
2. If the query is text-based, use the text context (including tables and formulas) to answer accurately.
3. If the query refers to a chart, diagram, or Figure (and visual data is attached), carefully analyze the image(s) to formulate your answer.
4. Always maintain precision and cite specific data or page numbers from the context when applicable.

Query: {query}

Context:
{context}"""

    message_payload: dict = {
        "role": "user",
        "content": generation_prompt
    }

    if ollama_images:
        print(f"🖼️ Attaching {len(ollama_images)} image(s) from memory to Ollama multimodal context.")
        message_payload["images"] = ollama_images

    try:
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
