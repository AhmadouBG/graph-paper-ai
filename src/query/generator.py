import re
import ollama

def _roman_to_int(s: str) -> int | None:
    roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    s = s.upper().strip()
    if not s or not all(c in roman for c in s):
        return None
    result = 0
    for i in range(len(s)):
        if i + 1 < len(s) and roman[s[i]] < roman[s[i + 1]]:
            result -= roman[s[i]]
        else:
            result += roman[s[i]]
    return result

def _extract_component_metadata(query: str) -> tuple[str | None, str | None, str | None]:
    query_lower = query.lower()
    
    m_table = re.search(r'table\s*\.?\s*([ivxlcdm]+|\d+)', query_lower)
    if m_table:
        raw_num = m_table.group(1)
        if raw_num.isdigit():
            return "table", raw_num, raw_num
        roman_val = _roman_to_int(raw_num.upper())
        return "table", (str(roman_val) if roman_val else raw_num), raw_num
            
    m_fig = re.search(r'fig(?:ure)?\.?\s*([ivxlcdm]+|\d+)', query_lower)
    if m_fig:
        raw_num = m_fig.group(1)
        if raw_num.isdigit():
            return "fig", raw_num, raw_num
        roman_val = _roman_to_int(raw_num.upper())
        return "fig", (str(roman_val) if roman_val else raw_num), raw_num
            
    return None, None, None

def generate_answer(query: str, retrieved_nodes: list[dict], model: str, full_tree: list[dict] = None,
                    page_image_map: dict = None) -> dict:
    context_list = []
    source_citations = []
    ollama_images = []

    is_visual_query = bool(
        re.search(r'fig(?:ure)?|chart|image|graph|plot|table|show', query.lower())
    )

    target_type, target_num, target_raw = _extract_component_metadata(query)

    # ── Build context from all retrieved nodes ────────────────────────────
    for node in retrieved_nodes:
        content_caps = re.findall(r'\[Visual Component\] Caption:\s*(.*?)(?:\n|$)', node.get('content', ''))
        print(f"   content captions: {content_caps}")
        print(f"   target_{target_type}: {target_num}")
        print(f"   is_visual_query: {is_visual_query}")
        
        full_node_content = node.get("content", "")
        table_context_injection = ""
        
        # Injection de sécurité uniquement pour préserver les structures de tableaux volumineuses
        if target_type == "table":
            all_tables = re.findall(r'(<table\b[^>]*>.*?</table>)', full_node_content, re.DOTALL | re.IGNORECASE)
            if all_tables:
                table_context_injection = "\n\n[DETECTED TABLES]:\n" + "\n\n".join(all_tables)

        truncated_content = full_node_content[:2500]
        pages_range = f"{node.get('page_start', '?')}-{node.get('page_end', '?')}"
        final_chunk_content = truncated_content + table_context_injection
        
        context_list.append(
            f"[Pages: {pages_range} | Section: {node['title']}]\n{final_chunk_content}"
        )
        source_citations.append(f"Section: '{node['title']}', Page {pages_range}")

    # ── Find the target image ─────────────────────────────────────────────
    image_found = False
    if is_visual_query and target_num and target_type:
        # STRATÉGIE HYBRIDE STRICTE :
        # Si c'est une Figure ("fig"), on VEUT absolument trouver son image Base64.
        if target_type == "fig":
            # Pass 1: Recherche locale dans le nœud courant
            for node in retrieved_nodes:
                matched = _find_best_image_for_component(target_type, target_num, node)
                if matched:
                    ollama_images.append(matched)
                    image_found = True
                    print(f"🎯 Matched Figure {target_num} in retrieved node '{node['title']}'")
                    break

            # Pass 2: Recherche globale dans tout l'arbre ou page_image_map
            if not ollama_images and full_tree:
                print(f"⚠️ Searching full tree for Figure {target_num}...")
                matched = _find_component_globally(target_type, target_num, target_raw, full_tree, page_image_map or {})
                if matched:
                    ollama_images.append(matched)
                    image_found = True

        # Si c'est un tableau ("table"), on court-circuite pour utiliser uniquement le HTML textuel
        elif target_type == "table":
            print(f"ℹ️ Table query detected. Skipping image lookup to analyze text/HTML context.")
            image_found = False

    raw_context = "\n\n".join(context_list)

    # ── Build the prompt ──────────────────────────────────────────────────
    # Mode vision actif uniquement si on a récupéré l'image Base64 d'une FIGURE
    if target_type == "fig" and ollama_images:
        caption_hint = ""
        for page_imgs in (page_image_map or {}).values():
            for img in page_imgs:
                lbl_norm = re.sub(r'[\s.]', '', img.get("label", "").lower())
                if "fig" in lbl_norm and target_num in lbl_norm:
                    cap = img.get("caption", "")
                    if cap and cap != "Aucune légende trouvée":
                        caption_hint = f"\nCaption: {cap}"
                    break
            if caption_hint: break

        generation_prompt = (
            f"You are analysing a figure from a research paper.\n"
            f"The attached image is Figure {target_num}.{caption_hint}\n\n"
            f"User question: {query}\n\n"
            f"Answer based only on what you see in the image. "
            f"Be specific: describe axes, values, trends, and conclusions visible in Figure {target_num}."
        )
        print(f"🔬 Vision-only prompt for Figure {target_num}")
    else:
        # Mode texte : Idéal pour les Tableaux HTML et les requêtes textuelles classiques
        context = raw_context
        table_instruction = ""
        if target_type == "table":
            table_instruction = f"Look carefully for any markdown or HTML <table>...</table> structure within the context that corresponds to Table {target_num} (which might be written as Table {target_raw.upper()} in Roman numerals).\n"

        generation_prompt = f"""You are an advanced AI assistant running locally.
Answer the user query based strictly on the context and any attached data below.
{table_instruction}
FORMATTING RULES:
- When listing items, always put each item on its own line with a dash: "- Item"
- Never run list items together separated only by commas or colons
- Keep answers concise and well-structured
- Use **bold** for key terms or section headers

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
        options={"num_ctx": 4096, "num_predict": 512, "temperature": 0.0}
    )

    return {
        "answer": response["message"]["content"],
        "sources": source_citations,
    }
