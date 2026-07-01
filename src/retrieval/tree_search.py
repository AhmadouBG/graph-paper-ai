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
    """Simple tokenizer that preserves hyphenated compounds."""
    text = text.lower()
    tokens = re.findall(r'[a-z](?:-[a-z0-9]+)+|[a-z0-9]+', text)
    stop_words = {
        "what", "is", "the", "a", "an", "of", "and", "in", "to", "about",
        "for", "on", "with", "how", "are", "does", "do", "can", "this",
        "that", "it", "be", "was", "were", "has", "have", "had",
    }
    return [t for t in tokens if t not in stop_words and (len(t) > 2 or '-' in t)]


def _bm25_fallback(query: str, compressed: list[dict]) -> list[str]:
    """
    BM25-ranked fallback. Better than raw keyword matching because it
    normalizes for node length and weights rare terms (e.g. 'z-score') higher.
    """
    if not compressed:
        return []

    # Build corpus: title gets repeated 3x to weight it like before
    corpus = []
    for item in compressed:
        title_tokens = _tokenize(item["title"]) * 3   # title weight
        body_tokens = _tokenize(item["_search"])
        corpus.append(title_tokens + body_tokens)

    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize(query)

    if not query_tokens:
        return [compressed[0]["node_id"]]

    scores = bm25.get_scores(query_tokens)
    ranked = sorted(
        zip(scores, compressed), key=lambda x: x[0], reverse=True
    )

    if ranked[0][0] <= 0:
        # No real match — fall back to first node
        return [compressed[0]["node_id"]]

    result = [ranked[0][1]["node_id"]]
    # Include second node only if it scores at least half the top score
    if len(ranked) > 1 and ranked[1][0] >= ranked[0][0] * 0.5:
        result.append(ranked[1][1]["node_id"])

    print(f"🎯 BM25 fallback → {result} (top score: {ranked[0][0]:.2f})")
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
        return out

    compressed = compress(tree)

    # ── 1. Figure shortcut: skip LLM, match caption directly ─────────────
    fig_match = re.search(r'fig(?:ure)?\.?\s*(\d+)', query.lower())
    if fig_match:
        target_num = fig_match.group(1)
        for item in compressed:
            for caption in item["figure_captions"]:
                normalized = re.sub(r'[.\s]', '', caption.lower())
                if re.search(rf'fig(?:ure)?{re.escape(target_num)}\b', normalized):
                    print(f"🎯 Figure shortcut → node {item['node_id']} (caption: {caption})")
                    return [item["node_id"]]
        print(f"⚠️ No caption match for Figure {target_num}, trying LLM.")

    # ── 2. LLM tree search (with timeout) ────────────────────────────────
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

    content = _call_ollama_with_timeout(TREE_SEARCH_MODEL, prompt)

    if content:
        try:
            result = _safe_parse_json(content)
            valid_ids = {item["node_id"] for item in compressed}
            node_list = [nid for nid in result.get("node_list", []) if nid in valid_ids]
            if node_list:
                print(f"\n{'='*60}")
                print(f"🧠 LLM reasoning: {result.get('thinking', 'N/A')}")
                print(f"✅ Selected nodes: {node_list}")
                return node_list
            print("⚠️ LLM returned no valid node IDs.")
        except Exception as e:
            print(f"⚠️ Parse error: {e}")

    # ── 3. Keyword fallback ───────────────────────────────────────────────
    print("↩️ Falling back to keyword scorer.")
    return _bm25_fallback(query, compressed)

