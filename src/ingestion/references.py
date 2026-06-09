from __future__ import annotations

import re
from typing import List

from src.ingestion.utils_class import CrossReference

SINGLE_PATTERN = re.compile(
    r"\b(?:Fig(?:ure)?s?\.?|Table|Section|Equation)\s+(\d+(?:\.\d+)*)([a-z])?\b",
    re.IGNORECASE,
)

LIST_CONTINUATION = re.compile(r"(?:and|,)\s*(\d+(?:\.\d+)*)\b")

NODE_PREFIX: dict[str, str] = {
    "figure": "fig",
    "table": "table",
    "section": "section",
    "equation": "equation",
}


def _resolve_type(label: str) -> str:
    key = label.lower().rstrip(".")
    if key.startswith("fig"):
        return "figure"
    if key.startswith("tab"):
        return "table"
    if key.startswith("section"):
        return "section"
    if key.startswith("equat"):
        return "equation"
    return "unknown"


def _extract_context(markdown: str, match: re.Match, window: int = 100) -> str:
    start = max(0, match.start() - window)
    end = min(len(markdown), match.end() + window)
    context = markdown[start:end].strip()
    return context[:200]


def _extract_list_numbers(
    markdown: str, match: re.Match, seen: set[str], prefix: str
) -> list[CrossReference]:
    refs: list[CrossReference] = []
    after = markdown[match.end():]
    for lm in LIST_CONTINUATION.finditer(after):
        pos = match.end() + lm.start()
        if pos - match.end() > 80:
            break
        num_str = lm.group(1)
        node_id = f"{prefix}_{num_str}".replace(".", "_")
        if node_id in seen:
            continue
        seen.add(node_id)
        ctx_start = max(0, match.start() - 50)
        ctx_end = min(len(markdown), pos + 50)
        context = markdown[ctx_start:ctx_end].strip()[:200]
        refs.append(CrossReference(
            target_node_id=node_id,
            reference_type="figure",
            context=context,
            page=None,
        ))
    return refs


def extract_cross_references(markdown: str) -> List[CrossReference]:
    if not markdown:
        return []

    refs: List[CrossReference] = []
    seen: set[str] = set()

    for match in SINGLE_PATTERN.finditer(markdown):
        label = match.group(0).split()[0]
        num_str = match.group(1)
        suffix = match.group(2) or ""
        ref_type = _resolve_type(label)
        prefix = NODE_PREFIX.get(ref_type, ref_type)
        node_id = f"{prefix}_{num_str}{suffix}".replace(".", "_")

        if node_id in seen:
            continue
        seen.add(node_id)

        context = _extract_context(markdown, match)
        refs.append(CrossReference(
            target_node_id=node_id,
            reference_type=ref_type,
            context=context,
            page=None,
        ))

        if ref_type == "figure":
            refs.extend(_extract_list_numbers(markdown, match, seen, prefix))

    return refs
