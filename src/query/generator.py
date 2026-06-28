import re
import ollama

# In generator.py
def _find_figure_globally(
    target_fig_num: str,
    tree: list[dict],
    page_image_map: dict[int, list[dict]]
) -> str | None:

    normalized_target = f"fig{target_fig_num}"  # e.g. "fig4"

    # ── Pass 1: image_captions on nodes (PyMuPDF, index-aligned) ─────────
    def walk_image_captions(nodes):
        for n in nodes:
            images = n.get("base64_images", [])
            captions = n.get("image_captions", [])
            for i, caption in enumerate(captions):
                normalized = re.sub(r'[.\s]', '', caption.lower())
                if re.search(rf'fig(?:ure)?{re.escape(target_fig_num)}\b', normalized):
                    img = images[i] if i < len(images) else (images[0] if images else None)
                    if img:
                        print(f"✅ Pass 1: Fig {target_fig_num} in node '{n['node_id']}'")
                        return img
            if n.get("nodes"):
                found = walk_image_captions(n["nodes"])
                if found:
                    return found
        return None

    result = walk_image_captions(tree)
    if result:
        return result

    # ── Pass 2: search page_image_map directly by label ──────────────────
    print(f"🔍 Pass 2: scanning page_image_map for '{normalized_target}'...")
    for page_num in sorted(page_image_map.keys()):
        for img_dict in page_image_map[page_num]:
            label_norm = re.sub(r'[.\s]', '', img_dict.get("label", "").lower())
            cap_norm = re.sub(r'[.\s]', '', img_dict.get("caption", "").lower())
            if re.search(rf'fig(?:ure)?{re.escape(target_fig_num)}\b', label_norm) or \
               re.search(rf'fig(?:ure)?{re.escape(target_fig_num)}\b', cap_norm):
                print(f"✅ Pass 2: Fig {target_fig_num} found on page {page_num} | label='{img_dict['label']}'")
                return img_dict["base64"]

    # ── Pass 3: content captions → find page → get image from page_image_map
    print(f"🔍 Pass 3: scanning content captions in tree...")
    def walk_content(nodes):
        for n in nodes:
            content_caps = re.findall(
                r'\[Visual Component\] Caption:\s*(.*?)(?:\n|$)',
                n.get("content", "")
            )
            for caption in content_caps:
                normalized = re.sub(r'[.\s]', '', caption.lower())
                if re.search(rf'fig(?:ure)?{re.escape(target_fig_num)}\b', normalized):
                    # Found in content — get image from the node's page range
                    page_start = n.get("page_start", 0)
                    page_end = n.get("page_end", page_start)
                    for pnum in range(page_start, page_end + 1):
                        if pnum in page_image_map:
                            for img_dict in page_image_map[pnum]:
                                if img_dict.get("base64"):
                                    print(f"✅ Pass 3: Fig {target_fig_num} via content caption, page {pnum}")
                                    return img_dict["base64"]
            if n.get("nodes"):
                found = walk_content(n["nodes"])
                if found:
                    return found
        return None

    result = walk_content(tree)
    if result:
        return result

    print(f"❌ Fig {target_fig_num} not found in any pass.")
    return None

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


def generate_answer(query: str, retrieved_nodes: list[dict], model: str, full_tree: list[dict] = None,
                    page_image_map: dict = None) -> dict:
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
                    print(f"🎯 Matched Fig {target_fig} in retrieved node '{sec['title']}'")
                    break

            # Full tree search with page_image_map fallback
            if not ollama_images and full_tree:
                print(f"⚠️ Searching full tree for Figure {target_fig}...")
                matched = _find_figure_globally(target_fig, full_tree, page_image_map or {})
                if matched:
                    ollama_images.append(matched)

        elif is_visual_query:
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
        options={"num_ctx": 2048, "num_predict": 256, "temperature": 0.0}
    )

    return {
        "answer": response["message"]["content"],
        "sources": source_citations,
    }