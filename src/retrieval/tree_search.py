from __future__ import annotations

import json
import re

import ollama


def llm_tree_search_ollama(query: str, tree: list[dict], model: str) -> list[str]:
    """
    Uses Ollama to analyze tree structures and select the most relevant node IDs.
    Hardened for CPU: strictly sanitizes text previews to prevent JSON failures.
    """
    def compress_and_flatten(nodes):
        flat_list = []
        for n in nodes:
            has_visuals = "Yes" if n.get("base64_images") else "No"
            content_text = n.get("content", "")

            # Only extract clean captions added by PyMuPDF to prevent broken quotation marks
            captions_found = re.findall(r'\[Visual Component\] Caption:\s*(.*?)(?:\n|$)', content_text)

            # Create a safe text preview devoid of JSON-breaking characters
            safe_title = re.sub(r'["\\\x00-\x1f]', '', n["title"])

            flat_list.append({
                "node_id": n["node_id"],
                "title": safe_title,
                "pages": f"{n.get('page_start', '?')}-{n.get('page_end', '?')}",
                "contains_images_or_figures": has_visuals,
                "detected_captions": [re.sub(r'["\\\x00-\x1f]', '', c) for c in captions_found]
            })
            if n.get("nodes"):
                flat_list.extend(compress_and_flatten(n["nodes"]))
        return flat_list

    compressed_tree = compress_and_flatten(tree)

    prompt = f"""You are a document navigation assistant. Analyze the user query and the document structure layout.
Select up to 3 Node IDs that are the most relevant to answer the query.

CRITICAL RULE: If the query asks for a specific Figure (e.g., 'fig 4'), prioritize sections where 'contains_images_or_figures' is 'Yes' and matches the target caption number.

Query: {query}

Document Structure:
{json.dumps(compressed_tree, indent=2)}

Reply ONLY in this exact JSON format, do not write markdown blocks or trailing text:
{{
  "thinking": "<short reasoning>",
  "node_list": ["node_id1", "node_id2"]
}}"""

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={
                "num_ctx": 4096,
            }
        )
        content = response["message"]["content"].strip()

        # Clean potential LLM conversational garbage wraps
        if not content.startswith("{"):
            content = content[content.find("{"):content.rfind("}")+1]

        result = json.loads(content)
        print("\n" + "="*60)
        print(f"🧠 LLM Tree Search Reasoning: {result.get('thinking', 'N/A')}")
        return result.get("node_list", [])

    except Exception as e:
        print(f"⚠️ Ollama Tree Search Error: {e}. Activating Smart Keyword Fallback Strategy.")

        # If Ollama crashes on JSON, Python scans the sanitized maps instantly
        fig_match = re.search(r'fig(?:ure)?\.?\s*(\d+)', query.lower())
        if fig_match:
            target_fig = fig_match.group(0)       # e.g., "fig 4"
            clean_target = re.sub(r'[.\s]+', '', target_fig)  # e.g., "fig4"

            for item in compressed_tree:
                combined_text = (item["title"] + " " + " ".join(item["detected_captions"])).lower()
                clean_combined = re.sub(r'[.\s]+', '', combined_text)
                if clean_target in clean_combined:
                    print(f"🎯 Smart Fallback matched Node ID: {item['node_id']}")
                    return [item["node_id"]]

        # Ultimate safe fallback
        return [tree[0]["node_id"]] if tree else []
