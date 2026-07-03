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

def generate_answer(query: str, retrieved_nodes: list[dict], model: str) -> dict:
    context_list = []
    source_citations = []

    for node in retrieved_nodes:
        # Include full content up to 3000 chars — tables can be long
        truncated_content = node.get("content", "")[:3000]
        pages_range = f"{node.get('page_start', '?')}-{node.get('page_end', '?')}"
        context_list.append(
            f"[Pages: {pages_range} | Section: {node['title']}]\n{truncated_content}"
        )
        source_citations.append(f"Section: '{node['title']}', Page {pages_range}")

    context = "\n\n".join(context_list)

    target_type, target_num, target_raw = _extract_component_metadata(query)

    # Build Roman numeral hint for table queries
    roman_hint = ""
    if target_type == "table" and target_num:
        roman_map = {
            1:"I", 2:"II", 3:"III", 4:"IV", 5:"V",
            6:"VI", 7:"VII", 8:"VIII", 9:"IX", 10:"X"
        }
        try:
            roman_hint = f" (also written as TABLE {roman_map[int(target_num)]})"
        except (ValueError, KeyError):
            pass

    if target_type and target_num:
        generation_prompt = f"""You are an advanced AI assistant analyzing a research paper.
The user wants information about {target_type.capitalize()} {target_num}{roman_hint}.

INSTRUCTIONS:
- Find the {target_type.capitalize()} {target_num}{roman_hint} in the context below
- It may appear as HTML <table>...</table>, a markdown table, a mermaid diagram, or descriptive text
- Extract and explain its contents clearly: describe headers, values, and conclusions
- If you cannot find {target_type.capitalize()} {target_num} specifically, say so — do NOT describe a different {target_type}

FORMATTING:
- Use **bold** for column/header names
- Present data row by row when relevant
- Keep the answer concise and accurate

Query: {query}

Context:
{context}"""

    else:
        generation_prompt = f"""You are an advanced AI assistant analyzing a research paper.
Answer the user query based strictly on the context below.

FORMATTING RULES:
- When listing items, put each on its own line with a dash: "- Item"
- Use **bold** for key terms or section headers
- Keep answers concise and well-structured

Query: {query}

Context:
{context}"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": generation_prompt}],
        options={"num_ctx": 8192, "num_predict": 1024, "temperature": 0.1}
    )

    return {
        "answer": response["message"]["content"],
        "sources": source_citations,
    }