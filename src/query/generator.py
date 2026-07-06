import re
import os
import ollama
# Initialisation du client ollama (S'assure de lire OLLAMA_HOST dans votre .env)


_ROMAN_MAP = {
    1: "I", 2: "II", 3: "III", 4: "IV", 5: "V",
    6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"
}

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

def _extract_content_for_target(
    content: str,
    target_type: str,
    target_num: str,
    caption_index: dict[str, str],
) -> str:
    """
    Isole le contenu d'une figure ou d'un tableau en ignorant STRICTEMENT 
    toutes les citations croisées textuelles.
    """
    arabic = target_num
    roman_map = {1:"I", 2:"II", 3:"III", 4:"IV", 5:"V",
                 6:"VI", 7:"VII", 8:"VIII", 9:"IX", 10:"X"}
    try:
        roman = roman_map[int(arabic)]
    except (ValueError, KeyError):
        roman = ""

    negative_lookbehind = r'(?<!see\s)(?<!in\s)(?<!and\s)(?<!from\s)(?<!with\s)(?<!to\s)(?<!of\s)(?<!showed\s)(?<!shows\s)'
    boundary_prefix = r'(?:\n|^|Title:\s*|Caption:\s*)'

    if target_type == "fig":
        target_patterns = [rf'{boundary_prefix}{negative_lookbehind}Fig(?:ure)?\.?\s*{re.escape(arabic)}\b']
    else:
        target_patterns = [
            rf'{boundary_prefix}{negative_lookbehind}TABLE\s+{re.escape(roman)}\b' if roman else None,
            rf'{boundary_prefix}{negative_lookbehind}Table\s+{re.escape(arabic)}\b',
            rf'{boundary_prefix}{negative_lookbehind}TABLE\s+{re.escape(arabic)}\b',
        ]
        target_patterns = [p for p in target_patterns if p]

    any_caption = r'(?:\n|^)(?:Fig(?:ure)?\.?\s*\d+[a-z]?|TABLE\s+(?:[IVXLCDM]+|\d+)|Table\s+\d+[a-z]?)\b'

    target_pos = None
    target_end = None
    target_key = f"{target_type}{target_num}"
    specific_caption = caption_index.get(target_key, "") if caption_index else ""

    if specific_caption:
        idx = content.find(specific_caption)
        if idx != -1:
            target_pos = idx
            target_end = idx + len(specific_caption)

    if target_pos is None:
        for pattern in target_patterns:
            m = re.search(pattern, content, re.IGNORECASE)
            if m:
                remaining_text = content[m.end():m.end()+30].lower()
                if not any(v in remaining_text for v in [" is ", " are ", " shows ", " presents ", " illustrates "]):
                    target_pos = m.start()
                    target_end = m.end()
                    break

    if target_pos is None:
        return ""

    if target_type == "fig":
        prev_end = 0
        for m in re.finditer(any_caption, content, re.IGNORECASE):
            if m.start() >= target_pos:
                break
            prev_end = m.end()
        caption_line_end = content.find('\n', target_end)
        return content[prev_end:caption_line_end if caption_line_end != -1 else len(content)].strip()
    else:
        next_pos = None
        for m in re.finditer(any_caption, content, re.IGNORECASE):
            if m.start() > target_pos + 10:
                next_pos = m.start()
                break
        end = next_pos if next_pos else min(len(content), target_pos + 4000)
        return content[target_pos:end].strip()


def generate_answer(query: str, nodes: list, model: str = "gpt-4o", caption_index: dict[str, str] | None = None) -> str:
    """
    Prend les nœuds extraits, applique le découpage intelligent autour des composants visuels 
    pour éviter les distractions, et génère une réponse sourcée via gpt-4o.
    """
    if not nodes:
        return "⚠️ No relevant sections found in the document."
    
    # Extraction des métadonnées de la cible visuelle
    target_type, target_num, target_raw = _extract_component_metadata(query)

    context_parts = []
    for node in nodes:
        raw_content = node.get("content", node.get("text", "Content not available."))
        pages = f"{node.get('page_start', '?')}-{node.get('page_end', '?')}"
        
        # ✨ ISOLATION INTELLIGENTE : Applique le découpeur de tranches si l'utilisateur cible un tableau ou une figure
        if target_type and target_num:
            content = _extract_content_for_target(
                raw_content, target_type, target_num, caption_index or {}
            )
            # Sécurité anti-tranche vide (Glisse sur les 3000 premiers caractères par défaut)
            if len(content) < 80:
                content = raw_content[:3000]
        else:
            content = raw_content[:3000]

        context_parts.append(
            f"[Section: '{node['title']}' | Page {pages}]\n"
            f"{content}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    
    # En-tête conditionnelle pour le prompt GPT-4o
    target_label_hint = ""
    if target_type and target_num:
        roman_hint = f" / TABLE {_ROMAN_MAP[int(target_num)]}" if target_type == "table" and target_num.isdigit() and int(target_num) in _ROMAN_MAP else ""
        target_label_hint = f"Answer ONLY and strictly about {target_type.capitalize()} {target_num}{roman_hint}. Ignore information about other figures/tables.\n"

    prompt = f"""You are an expert document analyst.
{target_label_hint}Answer the question using ONLY the provided context.
For every claim you make, cite the section title and page number in parentheses.
Be concise and precise.

Question: {query}

Context:
{context}

Answer:"""
    
    # Appel ollama propre et direct
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]
