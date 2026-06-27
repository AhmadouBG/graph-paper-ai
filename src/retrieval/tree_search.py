from __future__ import annotations
import json
import re
import ollama

TREE_SEARCH_MODEL = "qwen2.5vl:3b"  # ← change to whatever text model you have


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


def llm_tree_search_ollama(query: str, tree: list[dict], model: str) -> list[str]:
    """
    Tree search using a dedicated text model for reliable JSON output.
    `model` is the vision model used for generation — tree search uses
    TREE_SEARCH_MODEL instead for reliable instruction following.
    """
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

            # Also flatten children for search
            if n.get("nodes"):
                out.extend(compress(n["nodes"]))
        return out

    compressed = compress(tree)

    # Build the tree structure for the prompt (without _search key)
    tree_for_prompt = [
        {k: v for k, v in item.items() if k != "_search"}
        for item in compressed
    ]

    prompt = f"""You are a document navigation assistant. Your task is to find which sections of a document contain the answer to a user query.

Think step-by-step:
1. Read the query carefully.
2. Scan each node's title, preview text, and figure captions.
3. Select the 1-2 node IDs most likely to contain the answer.

Query: {query}

Document structure:
{json.dumps(tree_for_prompt, indent=2)}

Reply ONLY in this exact JSON format, no markdown, no extra text:
{{
  "thinking": "<your reasoning>",
  "node_list": ["node_id1", "node_id2"]
}}"""

    # ── Attempt with dedicated text model ────────────────────────────────
    try:
        response = ollama.chat(
            model=TREE_SEARCH_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={
                "num_ctx": 8192,   # text models handle more context
                "num_predict": 300,
                "temperature": 0.0,
            }
        )
        content = response["message"]["content"].strip()
        result = _safe_parse_json(content)

        # Validate IDs exist in tree
        valid_ids = {item["node_id"] for item in compressed}
        node_list = [nid for nid in result.get("node_list", []) if nid in valid_ids]

        if node_list:
            print(f"\n{'='*60}")
            print(f"🧠 LLM reasoning: {result.get('thinking', 'N/A')}")
            print(f"✅ Selected nodes: {node_list}")
            return node_list

        print("⚠️ LLM returned no valid node IDs.")

    except Exception as e:
        print(f"⚠️ Tree search error: {e}")

    # ── Python keyword fallback (only if LLM fails) ───────────────────────
    print("↩️ Falling back to keyword scorer.")
    return _keyword_fallback(query, compressed)


def _keyword_fallback(query: str, compressed: list[dict]) -> list[str]:
    """Keyword scorer as safety net only."""
    query_lower = query.lower()

    # Figure query
    fig_match = re.search(r'fig(?:ure)?\.?\s*(\d+)', query_lower)
    if fig_match:
        target_num = fig_match.group(1)
        for item in compressed:
            for cap in item["figure_captions"]:
                if re.search(rf'fig(?:ure)?\.?\s*{target_num}\b', cap.lower()):
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
            weight = 4 if '-' in w else 1
            score += hits * weight
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