import re
import json
from rank_bm25 import BM25Okapi
import ollama
import threading
# Initialisation du client OpenAI (assurez-vous d'avoir OPENAI_API_KEY dans votre .env)
def _call_ollama_with_timeout(model: str, prompt: str, timeout_seconds: int = 15) -> str | None:
    """
    Exécute l'appel Ollama dans un thread séparé pour forcer un Timeout strict.
    """
    result = [None]
    error = [None]

    def _call():
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                format="json", # 🚀 Force Ollama à répondre en JSON structuré
                options={
                    "num_ctx": 4096,     # Augmenté à 4096 pour supporter les structures d'arbres
                    "num_predict": 256,
                    "temperature": 0.0,  # Déterministe pour la navigation
                }
            )
            result[0] = response["message"]["content"].strip()
        except Exception as e:
            error[0] = e

    # Lancement du thread en arrière-plan
    thread = threading.Thread(target=_call, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        print(f"⏱️ Tree search timed out after {timeout_seconds}s.")
        return None
    if error[0]:
        print(f"⚠️ Ollama error: {error[0]}")
        return None
    return result[0]

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r'[a-z](?:-[a-z0-9]+)+|[a-z0-9]+', text)
    stop_words = {
        "what", "is", "the", "a", "an", "of", "and", "in", "to", "about",
        "for", "on", "with", "how", "are", "does", "do", "can", "this",
        "that", "it", "be", "was", "were", "has", "have", "had",
    }
    return [t for t in tokens if t not in stop_words and (len(t) > 2 or '-' in t)]

def _bm25_fallback(query: str, compressed: list[dict]) -> list[str]:
    """BM25 de secours pour les requêtes textuelles globales."""
    if not compressed:
        return []

    corpus = []
    for item in compressed:
        title_tokens = _tokenize(item["title"]) * 3
        body_tokens = _tokenize(item["_search"])
        corpus.append(title_tokens + body_tokens)

    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize(query)

    if not query_tokens:
        return [compressed[0]["node_id"]]

    scores = bm25.get_scores(query_tokens)
    ranked = sorted(zip(scores, compressed), key=lambda x: x[0], reverse=True)

    if ranked[0][0] <= 0:
        return [compressed[0]["node_id"]]

    result = [ranked[0][1]["node_id"]]
    if len(ranked) > 1 and ranked[1][0] >= ranked[0][0] * 0.5:
        result.append(ranked[1][1]["node_id"])

    print(f"🎯 BM25 Fallback → {result} (top score: {ranked[0][0]:.2f})")
    return result

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

def _extract_visual_target(query: str) -> tuple[str | None, str | None]:
    q = query.lower()
    fig_match = re.search(r'\bfig(?:ure)?s?\.?\s*([ivxlcdm]+|\d+[a-z]?)', q)
    if fig_match:
        raw = fig_match.group(1).strip('.')
        if raw.isdigit() or (len(raw) > 1 and raw[-1].isalpha() and raw[:-1].isdigit()):
            return raw, None
        num = _roman_to_int(raw.upper())
        return (str(num) if num else raw), None

    table_match = re.search(r'\btable\s*\.?\s*([ivxlcdm]+|\d+[a-z]?)', q)
    if table_match:
        raw = table_match.group(1).strip('.')
        if raw.isdigit():
            return None, raw
        num = _roman_to_int(raw.upper())
        return None, (str(num) if num else raw)
    return None, None


def llm_tree_search_ollama(query: str,
                            tree: list[dict],
                            model: str,
                            caption_index: dict[str, str]) -> list[str]:
    """
    Pipeline de recherche hybride :
    1. Raccourcis Regex ultra-stricts (Sniper) pour bloquer les fausses citations.
    2. Navigation intelligente par LLM (OpenAI GPT-4o) avec structure compressée.
    3. Repli algorithmique BM25 en dernier recours.
    """
    
    # ── ÉTAPE 0 : Aplatissement de l'arbre pour les moteurs de recherche textuels ──
    def flatten_tree(nodes: list[dict], accumulator: list[dict]) -> None:
        for n in nodes:
            has_visuals = "Yes" if n.get("base64_images") else "No"
            content_text = n.get("content", "")
            captions = re.findall(r'\[Visual Component\] Caption:\s*(.*?)(?:\n|$)', content_text)
            
            entry = {
                "node_id": n["node_id"],
                "title": n["title"],
                "page_start": n.get("page_start", "?"),
                "page_end": n.get("page_end", "?"),
                "has_figures": has_visuals,
                "figure_captions": [re.sub(r'["\\\x00-\x1f]', '', c) for c in captions],
                "_search": content_text.lower(),
            }
            accumulator.append(entry)
            if n.get("nodes"):
                flatten_tree(n["nodes"], accumulator)

    compressed_search_list = []
    flatten_tree(tree, compressed_search_list)

    # ── ÉTAPE 1 : RECHERCHE PAR RACCOURCIS DIRECTS (ANTI-CITATIONS) ──
    fig_num, table_num = _extract_visual_target(query)
    target_num = fig_num or table_num
    is_fig = fig_num is not None

    if target_num:
        prefix = "fig" if is_fig else "table"
        roman_map = {1:"I",2:"II",3:"III",4:"IV",5:"V",6:"VI",7:"VII",8:"VIII",9:"IX",10:"X"}
        try:
            roman_form = roman_map.get(int(target_num), "")
        except ValueError:
            roman_form = ""

        # Pass A : Correspondance exacte via caption_index
        if caption_index:
            target_keys = [f"{prefix}{target_num}"]
            if roman_form:
                target_keys.append(f"{prefix}{roman_form.lower()}")

            matched_caption = None
            for key in target_keys:
                if key in caption_index:
                    matched_caption = caption_index[key]
                    break

            if matched_caption:
                cap_fragment = matched_caption[:50].lower()
                for item in compressed_search_list:
                    if cap_fragment in item["_search"]:
                        print(f"🎯 Pass A (caption_index) → node {item['node_id']}")
                        return [item["node_id"]]

        # Pass B : Recherche brute par Regex isolées (Anti-citations)
        print(f"⚠️ {prefix.title()} {target_num} non trouvé dans l'index. Recherche brute structurelle...")
        negative_lookbehind = r'(?<!see\s)(?<!in\s)(?<!and\s)(?<!from\s)(?<!with\s)(?<!to\s)(?<!of\s)(?<!showed\s)(?<!shows\s)'
        boundary_prefix = r'(?:\n|^|Title:\s*|Caption:\s*)'

        strict_patterns = [rf'{boundary_prefix}{negative_lookbehind}{prefix}(?:ure)?s?\.?\s*{re.escape(target_num)}\b']
        if roman_form:
            strict_patterns.append(rf'{boundary_prefix}{negative_lookbehind}{prefix}(?:ure)?s?\.?\s*{re.escape(roman_form)}\b')

        for item in compressed_search_list:
            for pattern in strict_patterns:
                m = re.search(pattern, item["_search"], re.IGNORECASE)
                if m:
                    remaining_text = item["_search"][m.end():m.end()+30]
                    if not any(v in remaining_text for v in [" is ", " are ", " shows ", " presents ", " illustrates "]):
                        print(f"🔍 Pass B → VRAIE légende validée pour node {item['node_id']} '{item['title']}'")
                        return [item["node_id"]]

        print(f"❌ Raccourcis directs échoués pour {prefix} {target_num}. Passage à l'analyse LLM.")

    # ── ÉTAPE 2 : NAVIGATION INTELLIGENTE PAR LLM (OLLAMA LOCAL) ──
    print(f"🧠 Appel de Ollama ({model}) pour analyser la structure de l'arbre...")
    
    def compress_tree_for_llm(nodes):
        out = []
        for n in nodes:
            entry = {
                "node_id": n["node_id"],
                "title":   n["title"],
                "page":    f"{n.get('page_start', '?')}-{n.get('page_end', '?')}",
                "summary": n.get("content", "")[:150]
            }
            if n.get("nodes"):
                entry["children"] = compress_tree_for_llm(n["nodes"])
            out.append(entry)
        return out
    
    compressed_tree = compress_tree_for_llm(tree)
    
    prompt = f"""You are given a query and a document's tree structure (like a Table of Contents).
Your task: identify which node IDs most likely contain the answer to the query.
Think step-by-step about which sections are relevant.

Query: {query}

Document Tree:
{json.dumps(compressed_tree, indent=2)}

Reply ONLY in this exact JSON format:
{{
  "thinking": "<your step-by-step reasoning>",
  "node_list": ["node_id1", "node_id2"]
}}"""

    # Appel d'Ollama via notre fonction sécurisée par Thread
    content = _call_ollama_with_timeout(model, prompt, timeout_seconds=15)

    if content:
        try:
            # Nettoyage et validation du format JSON reçu d'Ollama
            result = _safe_parse_json(content)
            valid_ids = {item["node_id"] for item in compressed_search_list}
            node_list = [nid for nid in result.get("node_list", []) if nid in valid_ids]
            
            if node_list:
                print(f"🧠 Ollama Navigation : {result.get('thinking', 'N/A')}")
                print(f"✅ Selected via Ollama: {node_list}")
                return node_list
            print("⚠️ Ollama n'a renvoyé aucun identifiant de nœud valide.")
        except Exception as e:
            print(f"⚠️ Erreur de parsing JSON Ollama: {e}")

    # ── ÉTAPE 3 : REPLI DE SÉCURITÉ ALGORITHMIQUE (BM25) ──
    print("↩️ Ollama en échec ou Timeout dépassé (15s). Repli automatique sur le moteur BM25.")
    return _bm25_fallback(query, compressed_search_list)