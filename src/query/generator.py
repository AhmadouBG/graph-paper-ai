from __future__ import annotations

from pathlib import Path

from src.llm.ollama_client import OllamaClient


def _build_context_from_nodes(nodes: list[dict]) -> tuple[str, list[str]]:
    context_parts: list[str] = []
    image_paths: list[str] = []

    for node in nodes:
        title = node.get("title", "Unknown")
        page = node.get("page_index", "?")
        content = node.get("text") or node.get("summary", "")

        context_parts.append(
            f"[Section: '{title}' | Page {page}]\n{content}"
        )

        for visual in node.get("visuals", []):
            img_path = visual.get("path") or ""
            if img_path and Path(img_path).exists():
                image_paths.append(str(img_path))
            caption = visual.get("image_id", "")
            if caption:
                context_parts.append(
                    f"[Figure: {caption} on page {visual.get('page', page)}]"
                )

        for table in node.get("tables", []):
            table_md = table.get("markdown_content", "")
            if table_md:
                context_parts.append(
                    f"[Table on page {table.get('page', page)}]\n{table_md}"
                )

    context = "\n\n---\n\n".join(context_parts)
    return context, image_paths


def generate_answer(
    query: str,
    nodes: list[dict],
    llm: OllamaClient,
) -> str:
    if not nodes:
        return "No relevant sections found in the document."

    context, image_paths = _build_context_from_nodes(nodes)

    prompt = (
        "You are an expert document analyst.\n"
        "Answer the question using ONLY the provided context.\n"
        "For every claim you make, cite the section title and page number in parentheses.\n"
        "If images are provided with the message, examine them carefully and incorporate "
        "their visual information into your answer.\n"
        "If the context mentions figures, graphs, tables, or images, reference them by name.\n"
        "Be concise and precise.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )

    return llm.chat(
        messages=[{"role": "user", "content": prompt}],
        images=image_paths if image_paths else None,
    )
