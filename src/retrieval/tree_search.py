from __future__ import annotations
import json
import re
import threading
import ollama

TREE_SEARCH_MODEL = "qwen2.5vl:3b"


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
                    "num_ctx": 4096,
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


def llm_tree_search_ollama(query: str, tree: list[dict], model: str) -> list[str]:

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

    content = _call_ollama_with_timeout(TREE_SEARCH_MODEL, prompt, timeout_seconds=15)

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
    return _keyword_fallback(query, compressed)


def _keyword_fallback(query: str, compressed: list[dict]) -> list[str]:
    query_lower = query.lower()

    # Figure query — also handled here in case shortcut missed
    fig_match = re.search(r'fig(?:ure)?\.?\s*(\d+)', query_lower)
    if fig_match:
        target_num = fig_match.group(1)
        for item in compressed:
            for cap in item["figure_captions"]:
                if re.search(rf'fig(?:ure)?\.?\s*{re.escape(target_num)}\b', cap.lower()):
                    print(f"🎯 Figure fallback → {item['node_id']}")
                    return [item["node_id"]]

    # Text query
    stop_words = {
        "what", "is", "the", "a", "an", "of", "and", "in", "to", "about",
        "for", "on", "with", "how", "are", "does", "do", "can", "this",
        "that", "it", "be", "was", "were", "has", "have", "had",
    }
    tokens = re.findall(r'[a-z](?:-[a-z0-9]+)+|[a-z0-9]+', query_lower)
    query_words = [t for t in tokens if t not in stop_words and (len(t) > 2 or '-' in t)]

    if not query_words:
        return [compressed[0]["node_id"]] if compressed else []

    scores = []
    for item in compressed:
        scope = f"{item['title'].lower()} {item['_search']}"
        score = 0.0
        for w in query_words:
            pat = re.escape(w).replace(r'\-', r'[\s\-]') if '-' in w else re.escape(w)
            hits = len(re.findall(rf'\b{pat}\b', scope))
            score += hits * (4 if '-' in w else 1)
            title_hits = len(re.findall(rf'\b{pat}\b', item['title'].lower()))
            score += title_hits * (6 if '-' in w else 3)
        if score > 0:
            scores.append((score, item["node_id"]))

    scores.sort(reverse=True)
    if scores:
        result = [scores[0][1]]
        if len(scores) > 1 and scores[1][0] >= scores[0][0] * 0.5:
            result.append(scores[1][1])
        print(f"🎯 Keyword fallback → {result}")
        return result

    return [compressed[0]["node_id"]] if compressed else []