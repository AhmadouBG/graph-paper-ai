from __future__ import annotations

import base64
import re
from pathlib import Path

import ollama


def _collect_images(nodes: list[dict]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        for v in node.get("visuals", []):
            p = v.get("path") or ""
            if p and Path(p).exists() and p not in seen:
                seen.add(p)
                paths.append(p)
    return paths


def generate_answer(
    query: str,
    nodes: list[dict],
    model: str,
) -> str:
    if not nodes:
        return "No relevant sections found in the document."

    context_parts = []
    figures_info = []

    for node in nodes:
        figure_mentions = re.findall(r"(Figure \d+: [^\n.]+)", node.get("summary", ""))
        for fig_desc in figure_mentions:
            figures_info.append(
                f"- {fig_desc} (Section: '{node['title']}' | Page {node.get('page_index', '?')})"
            )

        context_parts.append(
            f"[Section: '{node['title']}' | Page {node.get('page_index', '?')}]\n"
            f"{node.get('text', 'Content not available.')}"
        )

        for v in node.get("visuals", []):
            img_id = v.get("image_id", "")
            if img_id:
                figures_info.append(
                    f"- {img_id} (Section: '{node['title']}' | Page "
                    f"{v.get('page', node.get('page_index', '?'))})"
                )

        for t in node.get("tables", []):
            md = t.get("markdown_content", "")
            if md:
                context_parts.append(
                    f"[Table on page {t.get('page', node.get('page_index', '?'))}]\n{md}"
                )

    context = "\n\n---\n\n".join(context_parts)

    if figures_info:
        figures_section = "\n\n---\n\n" + "Figures Mentioned:\n" + "\n".join(figures_info)
        context += figures_section

    prompt = f"""You are an expert document analyst.
Answer the question using ONLY the provided context.
For every claim you make, cite the section title and page number in parentheses.
If the context mentions figures, graphs, or images, ensure you reference them appropriately.
Be concise and precise.

Question: {query}

Context:
{context}

Answer:"""

    image_paths = _collect_images(nodes)
    images_b64 = []
    for p in image_paths:
        images_b64.append(base64.b64encode(Path(p).read_bytes()).decode("utf-8"))

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": images_b64}]
    )
    return response["message"]["content"]
