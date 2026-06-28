import re
import ollama

# In generator.py

def _find_figure_globally(
    target_fig_num: str,
    tree: list[dict],
    page_image_map: dict[int, list[dict]]
) -> str | None:
    """
    Two-pass search:
    Pass 1: match against image_captions (PyMuPDF) — index-aligned with base64_images
    Pass 2: match against content captions (LlamaParse) — use page_image_map directly
    """
    # ── Pass 1: image_captions (fast, index-aligned) ──────────────────────
    def walk_image_captions(nodes):
        for n in nodes:
            images = n.get("base64_images", [])
            captions = n.get("image_captions", [])
            for i, caption in enumerate(captions):
                normalized = re.sub(r'[.\s]', '', caption.lower())
                if re.search(rf'fig(?:ure)?{re.escape(target_fig_num)}\b', normalized):
                    img = images[i] if i < len(images) else (images[0] if images else None)
                    if img:
                        print(f"✅ Pass 1 match: Fig {target_fig_num} in node '{n['node_id']}'")
                        return img
            if n.get("nodes"):
                found = walk_image_captions(n["nodes"])
                if found:
                    return found
        return None

    result = walk_image_captions(tree)
    if result:
        return result

    # ── Pass 2: content captions (LlamaParse) → page_image_map lookup ────
    def walk_content_captions(nodes):
        for n in nodes:
            content = n.get("content", "")
            content_caps = re.findall(
                r'\[Visual Component\] Caption:\s*(.*?)(?:\n|$)', content
            )
            for caption in content_caps:
                normalized = re.sub(r'[.\s]', '', caption.lower())
                if re.search(rf'fig(?:ure)?{re.escape(target_fig_num)}\b', normalized):
                    # Found the caption in content — now find the image by page
                    # Search page_image_map for this figure's page
                    for page_num, imgs in page_image_map.items():
                        for img_dict in imgs:
                            img_label = re.sub(r'[.\s]', '', img_dict.get("label", "").lower())
                            img_cap = re.sub(r'[.\s]', '', img_dict.get("caption", "").lower())
                            if re.search(rf'fig(?:ure)?{re.escape(target_fig_num)}\b', img_label) or \
                               re.search(rf'fig(?:ure)?{re.escape(target_fig_num)}\b', img_cap):
                                print(f"✅ Pass 2 match: Fig {target_fig_num} on page {page_num}")
                                return img_dict["base64"]
                    # Caption found in content but no page_image_map match
                    # — figure might be a table rendered without extractable image
                    print(f"⚠️ Fig {target_fig_num} caption found in content but no image in page_image_map")
            if n.get("nodes"):
                found = walk_content_captions(n["nodes"])
                if found:
                    return found
        return None

    return walk_content_captions(tree)
    
def _extract_figure_number(query: str) -> str | None:
    """Extract '5' from 'figure 5', 'fig. 5', 'fig5', etc."""
    m = re.search(r'fig(?:ure)?\.?\s*(\d+)', query.lower())
    return m.group(1) if m else None

def _find_figure_in_tree(target_fig_num: str, tree: list[dict]) -> str | None:
    """
    Walk every node and match target figure number against
    image_captions (PyMuPDF, reliable) rather than content captions.
    """
    def walk(nodes):
        for n in nodes:
            images = n.get("base64_images", [])
            captions = n.get("image_captions", [])
            for i, caption in enumerate(captions):
                normalized = re.sub(r'[.\s]', '', caption.lower())
                if re.search(rf'fig(?:ure)?{re.escape(target_fig_num)}\b', normalized):
                    img = images[i] if i < len(images) else (images[0] if images else None)
                    if img:
                        print(f"🎯 Found Fig {target_fig_num} in node '{n['node_id']}' | caption: {caption}")
                        return img
            if n.get("nodes"):
                found = walk(n["nodes"])
                if found:
                    return found
        return None
    return walk(tree)

def _find_best_image_for_figure(target_fig_num: str, node: dict) -> str | None:
    images = node.get("base64_images", [])
    captions = node.get("image_captions", [])  # PyMuPDF captions — reliable index alignment

    if not images:
        return None

    for i, caption in enumerate(captions):
        normalized = re.sub(r'[.\s]', '', caption.lower())
        if re.search(rf'fig(?:ure)?{re.escape(target_fig_num)}\b', normalized):
            return images[i] if i < len(images) else images[0]

    return None  # no match — do NOT fall back


def generate_answer(query: str, retrieved_nodes: list[dict], model: str, full_tree: list[dict] = None) -> dict:
    context_list = []
    source_citations = []
    ollama_images = []

    is_visual_query = bool(
        re.search(r'fig(?:ure)?|chart|image|graph|plot|table|show', query.lower())
    )
    
    target_fig = _extract_figure_number(query)  # e.g. "5" or None

    for sec in retrieved_nodes:
        print(f"\n🔍 DEBUG GENERATOR NODE: {sec['node_id']} | {sec['title']}")
        print(f"   image_captions: {sec.get('image_captions', [])}")
        print(f"   base64_images count: {len(sec.get('base64_images', []))}")
        content_caps = re.findall(r'\[Visual Component\] Caption:\s*(.*?)(?:\n|$)', sec.get('content',''))
        print(f"   content captions: {content_caps}")
        print(f"   target_fig: {target_fig}")
        print(f"   is_visual_query: {is_visual_query}")
        truncated_content = sec.get("content", "")[:2500]
        pages_range = f"{sec.get('page_start', '?')}-{sec.get('page_end', '?')}"
        context_list.append(
            f"[Pages: {pages_range} | Section: {sec['title']}]\n{truncated_content}"
        )
        source_citations.append(f"Section: '{sec['title']}', Page {pages_range}")

        if is_visual_query and target_fig:
            # First try retrieved nodes
            for sec in retrieved_nodes:
                matched = _find_best_image_for_figure(target_fig, sec)
                if matched:
                    ollama_images.append(matched)
                    print(f"🎯 Matched Figure {target_fig} in node '{sec['title']}'")
                    break

            # If not found and we have the full tree, search everything
            if not ollama_images and full_tree:
                print(f"⚠️ Searching full tree for Figure {target_fig}...")
                matched = _find_figure_in_tree(target_fig, full_tree)
                if matched:
                    ollama_images.append(matched)

        elif is_visual_query and not target_fig:
            for sec in retrieved_nodes:
                images = sec.get("base64_images", [])
                if images:
                    ollama_images.append(images[0])
                    break

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
        options={"num_ctx": 4096, "num_predict": 256, "temperature": 0.0}
    )

    return {
        "answer": response["message"]["content"],
        "sources": source_citations,
    }