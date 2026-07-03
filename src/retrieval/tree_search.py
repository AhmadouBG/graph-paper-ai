from __future__ import annotations
import json
import re
import threading
import ollama
from rank_bm25 import BM25Okapi

TREE_SEARCH_MODEL = "qwen2.5:3b"


def _safe_parse_json(raw: str) -> dict:
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in response.")
    raw = raw[start:end + 1]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    repaired = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', raw)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    ids = re.findall(r'"(\d{4})"', raw)
    if ids:
        return {"thinking": "regex-extracted", "node_list": ids}

    raise ValueError("Could not parse JSON.")


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r'[a-z](?:-[a-z0-9]+)+|[a-z0-9]+', text)
    stop_words = {
        "what", "is", "the", "a", "an", "of", "and", "in", "to", "about",
        "for", "on", "with", "how", "are", "does", "do", "can", "this",
        "that", "it", "be", "was", "were", "has", "have", "had",
    }
    return [t for t in tokens if t not in stop_words and (len(t) > 2 or '-' in t)]


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
    """
    Returns (fig_num, table_num) as arabic numeral strings.
    e.g. "explain figure 5"  → ("5", None)
         "give me table III" → (None, "3")
         "show table 4"      → (None, "4")
    """
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


def _bm25_fallback(query: str, compressed: list[dict]) -> list[str]:
    """BM25 ranked retrieval — only for text queries, not figure/table queries."""
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

    print(f"🎯 BM25 → {result} (top score: {ranked[0][0]:.2f})")
    return result


def _call_ollama_with_timeout(model: str, prompt: str, timeout_seconds: int = 15) -> str | None:
    result = [None]
    error = [None]

    def _call():
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={
                    "num_ctx": 2048,
                    "num_predict": 256,
                    "temperature": 0.0,
                }
            )
            result[0] = response["message"]["content"].strip()
        except Exception as e:
            error[0] = e

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


def llm_tree_search_ollama(query: str, tree: list[dict]) -> list[str]:

    def compress(nodes):
        out = []
        for n in nodes:
            has_visuals = "Yes" if n.get("base64_images") else "No"
            content_text = n.get("content", "")
            captions = re.findall(
                r'\[Visual Component\] Caption:\s*(.*?)(?:\n|$)', content_text
            )
            safe_title = re.sub(r'["\\\x00-\x1f]', '', n["title"])
            preview = " ".join(content_text[:300].split())
            preview = re.sub(r'[\x00-\x1f"\\]', '', preview)

            entry = {
                "node_id": n["node_id"],
                "title": safe_title,
                "pages": f"{n.get('page_start', '?')}-{n.get('page_end', '?')}",
                "has_figures": has_visuals,
                "figure_captions": [re.sub(r'["\\\x00-\x1f]', '', c) for c in captions],
                "preview": preview,
                "_search": content_text.lower(),
            }
            if n.get("nodes"):
                entry["children"] = [
                    {"node_id": c["node_id"], "title": re.sub(r'["\\\x00-\x1f]', '', c["title"])}
                    for c in n["nodes"]
                ]
            out.append(entry)
            if n.get("nodes"):
                out.extend(compress(n["nodes"]))
            print(f"🔍 DEBUG compress()")
        # At the end of compress(), before return:
        for item in out:
            if item["figure_captions"]:
                print(f"  node {item['node_id']} '{item['title']}' → captions: {item['figure_captions']}")
        return out

    compressed = compress(tree)

    # ── 1. Figure / Table shortcut ────────────────────────────────────────
    fig_num, table_num = _extract_visual_target(query)
    print(f"🔍 DEBUG _extract_visual_target: fig_num={fig_num}, table_num={table_num}")
    target_num = fig_num or table_num
    is_fig = fig_num is not None

    if target_num:
        prefix = "fig" if is_fig else "table"
        roman_val = _roman_to_int(target_num.upper())
        arabic = str(roman_val) if roman_val else target_num

        # Pass A: match against indexed captions (strict)
        for item in compressed:
            for caption in item["figure_captions"]:
                normalized = re.sub(r'[\s.\-,:]', '', caption.lower())
                # Must have prefix immediately followed by number (no gap)
                if re.search(rf'{prefix}(?:ure)?s?{re.escape(arabic)}(?!\d)', normalized):
                    print(f"🎯 Pass A → node {item['node_id']} (caption: {caption[:60]})")
                    return [item["node_id"]]

        # Pass B: strict body text — prefix must be ADJACENT to number
        # Pass B: strict body search including Roman numeral form
        print(f"⚠️ {prefix.title()} {arabic} not in captions. Strict body search...")

        roman_map = {
            1:"I", 2:"II", 3:"III", 4:"IV", 5:"V",
            6:"VI", 7:"VII", 8:"VIII", 9:"IX", 10:"X"
        }
        try:
            arabic_int = int(arabic)
            roman_form = roman_map.get(arabic_int, "")
        except ValueError:
            roman_form = ""

        strict_patterns = [
            rf'\b{prefix}(?:ure)?s?\.?\s*{re.escape(arabic)}\b',
        ]
        if roman_form:
            strict_patterns.append(
                rf'\b{prefix}(?:ure)?s?\.?\s*{re.escape(roman_form)}\b'
            )

        for item in compressed:
            body = item["_search"]
            for pattern in strict_patterns:
                if re.search(pattern, body, re.IGNORECASE):
                    print(f"🔍 Pass B → node {item['node_id']} '{item['title']}'")
                    return [item["node_id"]]

        print(f"❌ {prefix.title()} {arabic} not found. Falling back to BM25.")
        return _bm25_fallback(query, compressed)


    # ── 2. LLM tree search ────────────────────────────────────────────────
    tree_for_prompt = [
        {k: v for k, v in item.items() if k != "_search"}
        for item in compressed
    ]

    prompt = f"""You are a document navigation assistant. Find which sections contain the answer to the query.

Think step-by-step:
1. Read the query carefully.
2. Scan each node's title, preview text, and figure_captions.
3. Select 1-2 node IDs most likely to contain the answer.

Query: {query}

Document structure:
{json.dumps(tree_for_prompt, indent=2)}

Reply ONLY in this exact JSON format, no markdown, no extra text:
{{
  "thinking": "<your reasoning>",
  "node_list": ["node_id1", "node_id2"]
}}"""

    content = _call_ollama_with_timeout(TREE_SEARCH_MODEL, prompt, timeout_seconds=15)

    if content:
        try:
            result = _safe_parse_json(content)
            valid_ids = {item["node_id"] for item in compressed}
            node_list = [nid for nid in result.get("node_list", []) if nid in valid_ids]
            if node_list:
                print(f"\n{'='*60}")
                print(f"🧠 LLM: {result.get('thinking', 'N/A')}")
                print(f"✅ Selected: {node_list}")
                return node_list
            print("⚠️ LLM returned no valid node IDs.")
        except Exception as e:
            print(f"⚠️ Parse error: {e}")

    # ── 3. BM25 fallback ─────────────────────────────────────────────────
    print("↩️ Falling back to BM25.")
    return _bm25_fallback(query, compressed)