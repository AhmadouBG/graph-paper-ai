from __future__ import annotations

import json
from typing import Any

from src.llm.ollama_client import OllamaClient


def compress_tree(tree: list[dict]) -> list[dict]:
    out: list[dict] = []
    for n in tree:
        entry: dict[str, Any] = {
            "node_id": n["node_id"],
            "title": n["title"],
            "page": n.get("page_index", "?"),
            "summary": (n.get("text") or n.get("summary", ""))[:1000],
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


def llm_tree_search(query: str, tree: list[dict], llm: OllamaClient) -> dict:
    compressed = compress_tree(tree)
    prompt = (
        "You are given a query and a document's tree structure (like a Table of Contents).\n"
        "Your task: identify which node IDs most likely contain the answer to the query.\n"
        "Think step-by-step about which sections are relevant.\n\n"
        f"Query: {query}\n\n"
        f"Document Tree:\n{json.dumps(compressed, indent=2, ensure_ascii=False)}\n\n"
        "Reply ONLY in this exact JSON format:\n"
        '{\n'
        '  "thinking": "<your step-by-step reasoning>",\n'
        '  "node_list": ["node_id1", "node_id2"]\n'
        '}'
    )
    response = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    try:
        return json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return {"thinking": "Failed to parse JSON", "node_list": []}
