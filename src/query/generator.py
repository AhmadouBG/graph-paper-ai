import re
import ollama


def _extract_figure_number(query: str) -> str | None:
    """Extract '5' from 'figure 5', 'fig. 5', 'fig5', etc."""
    m = re.search(r'fig(?:ure)?\.?\s*(\d+)', query.lower())
    return m.group(1) if m else None


def _find_best_image_for_figure(
    target_fig_num: str,
    node: dict,
) -> str | None:
    """
    Scans the node's content for a [Visual Component] Caption line
    that mentions the target figure number, then returns the base64
    image at the matching index (first image = index 0, etc.).
    Falls back to the first image if no caption matches.
    """
    content = node.get("content", "")
    images = node.get("base64_images", [])
    
    if not images:
        return None

    # Find all caption lines in the node content (injected by run_pipeline)
    caption_lines = re.findall(
        r'\[Visual Component\] Caption:\s*(.*?)(?:\n|$)', content
    )

    # Try to match the target figure number against each caption
    for i, caption in enumerate(caption_lines):
        # Normalize: remove spaces/dots, lowercase → "fig1", "fig2", etc.
        normalized = re.sub(r'[.\s]', '', caption.lower())
        # Match "fig5", "figure5", "fig.5"
        if re.search(rf'fig(?:ure)?{target_fig_num}\b', normalized):
            # Return the image at the same index if available, else first
            return images[i] if i < len(images) else images[0]

    # No caption match — fall back to first image in the node
    return images[0]


def generate_answer(query: str, retrieved_nodes: list[dict], model: str) -> dict:
    context_list = []
    source_citations = []
    ollama_images = []

    is_visual_query = bool(
        re.search(r'fig(?:ure)?|chart|image|graph|plot|table|show', query.lower())
    )
    
    target_fig = _extract_figure_number(query)  # e.g. "5" or None

    for sec in retrieved_nodes:
        truncated_content = sec.get("content", "")[:2500]
        pages_range = f"{sec.get('page_start', '?')}-{sec.get('page_end', '?')}"
        context_list.append(
            f"[Pages: {pages_range} | Section: {sec['title']}]\n{truncated_content}"
        )
        source_citations.append(f"Section: '{sec['title']}', Page {pages_range}")

        if is_visual_query:
            images = sec.get("base64_images", [])
            if not images:
                continue

            if target_fig:
                matched = _find_best_image_for_figure(target_fig, sec)
                if matched:
                    ollama_images.append(matched)
                    print(f"🎯 Matched Figure {target_fig} image from node '{sec['title']}'")
            else:
                # Generic visual query — send at most 1 image to stay fast
                ollama_images.append(images[0])
                print(f"🖼️ Generic visual query — attaching 1 image from '{sec['title']}'")

    context = "\n\n".join(context_list)

    generation_prompt = f"""You are an advanced AI assistant running locally.
Answer the user query based strictly on the context and any attached visual data below.

Query: {query}

Context:
{context}"""

    message_payload = {"role": "user", "content": generation_prompt}

    if ollama_images:
        print(f"🖼️ Attaching {len(ollama_images)} image(s) to multimodal context.")
        message_payload["images"] = ollama_images
    else:
        print("📝 Text query — no images attached.")

    response = ollama.chat(
        model=model,
        messages=[message_payload],
        options={"num_ctx": 4096, "num_predict": 1024, "temperature": 0.1}
    )

    return {
        "answer": response["message"]["content"],
        "sources": source_citations,
    }