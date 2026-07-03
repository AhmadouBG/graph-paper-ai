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

def _find_component_globally(target_type: str, target_num: str, target_raw: str, tree: list[dict], page_image_map: dict[int, list[dict]]) -> str | None:
    # On normalise les patterns selon le type (fig ou table)
    if target_type == "fig":
        patterns = [
            rf'fig(?:ure)?\.?\s*{re.escape(target_num)}\b',
            rf'fig(?:ure)?\.?\s*{re.escape(target_raw)}\b'
        ]
    else:
        patterns = [
            rf'table\.?\s*{re.escape(target_num)}\b',
            rf'table\.?\s*{re.escape(target_raw)}\b'
        ]

    target_clean_arabic = re.sub(r'[.\s]', '', f"{target_type}{target_num}")
    target_clean_raw = re.sub(r'[.\s]', '', f"{target_type}{target_raw}")

    # Pass 1: Vérification dans les nœuds de l'arbre
    def walk_image_captions(nodes):
        for n in nodes:
            images = n.get("base64_images", [])
            captions = n.get("image_captions", [])
            for i, caption in enumerate(captions):
                normalized = re.sub(r'[.\s]', '', caption.lower())
                # Match via label normalisé ou via regex
                if target_clean_arabic in normalized or target_clean_raw in normalized:
                    img = images[i] if i < len(images) else (images[0] if images else None)
                    if img: return img
                for pattern in patterns:
                    if re.search(pattern, caption.lower()):
                        img = images[i] if i < len(images) else (images[0] if images else None)
                        if img: return img
            if n.get("nodes"):
                found = walk_image_captions(n["nodes"])
                if found: return found
        return None

    result = walk_image_captions(tree)
    if result: return result

    # Pass 2: Recherche directe par balayage dans la map globale des pages
    for page_num in sorted(page_image_map.keys()):
        for img_dict in page_image_map[page_num]:
            label_norm = re.sub(r'[.\s]', '', img_dict.get("label", "").lower())
            cap_norm = re.sub(r'[.\s]', '', img_dict.get("caption", "").lower())
            
            # Match tolérant pour cibler la bonne image
            if (target_num in label_norm or target_raw in label_norm or 
                target_num in cap_norm or target_raw in cap_norm):
                # S'assurer que le préfixe correspond au type recherché
                if target_type in label_norm or target_type in cap_norm or target_type == "fig" and "fig" in label_norm:
                    return img_dict["base64"]
    return None

def _find_best_image_for_component(target_type: str, target_num: str, node: dict) -> str | None:
    images = node.get("base64_images", [])
    captions = node.get("image_captions", [])
    if not images: return None
    
    # Rendre la regex flexible selon fig ou table
    prefix_pattern = r'fig(?:ure)?' if target_type == "fig" else r'table'
    pattern = rf'{prefix_pattern}\.?\s*{re.escape(target_num)}\b'
    
    for i, caption in enumerate(captions):
        if re.search(pattern, caption.lower()):
            return images[i] if i < len(images) else images[0]
    return None

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
