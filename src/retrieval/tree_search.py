from __future__ import annotations

import json
from typing import Any

import ollama


def compress_tree(tree: list[dict]) -> list[dict]:
    out: list[dict] = []
    for n in tree:
        entry: dict[str, Any] = {
            "node_id": n["node_id"],
            "title": n["title"],
            "page": n.get("page_index", "?"),
            "summary": (n.get("text") or n.get("summary", ""))[:2000],
        }
        visuals = n.get("visuals")
        if visuals:
            entry["visuals"] = [
                {"type": v.get("type"), "image_id": v.get("image_id"), "page": v.get("page")}
                for v in visuals
            ]
        tables = n.get("tables")
        if tables:
            entry["tables"] = [
                {"page": t.get("page"), "content_preview": t.get("markdown_content", "")[:200]}
                for t in tables
            ]
        if n.get("nodes"):
            entry["children"] = compress_tree(n["nodes"])
        out.append(entry)
    return out


def llm_tree_search_ollama(query: str, tree: list[dict], model: str) -> dict:
    compressed_tree = compress_tree(tree)
    prompt = (
        f"""You are given a query and a document's tree structure (like a Table of Contents).
Your task: identify which node IDs most likely contain the answer to the query.
Think step-by-step about which sections are relevant.

Query: {query}

Document Tree:
{json.dumps(compressed_tree, indent=2, ensure_ascii=False)}

Reply ONLY in this exact JSON format:
{{
  "thinking": "<your step-by-step reasoning>",
  "node_list": ["node_id1", "node_id2"]
}}"""
    )
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    content = response["message"]["content"]
    try:
        parsed = json.loads(content)
        node_list = parsed.get("node_list", [])
    except json.JSONDecodeError:
        node_list = [n.strip() for n in content.replace("\n", ",").split(",") if n.strip()]
    return {"node_list": node_list, "raw_response": content}
