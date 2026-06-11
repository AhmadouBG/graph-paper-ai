from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import networkx as nx

from src.ingestion.tree import build_section_tree
from src.ingestion.tree import print_tree as _render_tree
from src.llm.ollama_client import OllamaClient, OllamaMessage

TREE_SEARCH_PROMPT = (
    "You are a document retrieval system. Given a user query and a document's"
    " section tree, identify the most relevant sections by their node IDs.\n"
    "\n"
    "Return ONLY a comma-separated list of the most relevant node IDs,"
    " nothing else. Example: section_introduction,section_background\n"
    "\n"
    "Document tree:\n"
    "{tree}\n"
    "\n"
    "User query: {query}\n"
    "\n"
    "Relevant node IDs:"
)

ANSWER_PROMPT = (
    "Answer the user's question based on the provided document context.\n"
    "\n"
    "Context:\n"
    "{context}\n"
    "\n"
    "Query: {query}\n"
    "\n"
    "Answer:"
)

NODE_ID_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_-]*")


@dataclass
class TreeSearchResult:
    selected_ids: List[str] = field(default_factory=list)
    context: str = ""


def _parse_selected_ids(response: str, max_ids: int = 20) -> List[str]:
    section_ids: List[str] = []
    for m in NODE_ID_PATTERN.finditer(response):
        node_id = m.group(0)
        if node_id.startswith("section_") and node_id not in section_ids:
            section_ids.append(node_id)
    return section_ids[:max_ids]


def _fetch_section_content(
    graph: nx.DiGraph,
    selected_ids: List[str],
) -> str:
    parts: List[str] = []
    for node_id in selected_ids:
        if not graph.has_node(node_id):
            continue
        attr = graph.nodes[node_id]
        title = attr.get("title", node_id)
        content = attr.get("content", "")
        parts.append(f"[{node_id}] {title}\n{content}")
    return "\n\n".join(parts)


def tree_search(
    query: str,
    graph: nx.DiGraph,
    llm_client: OllamaClient,
    tree: Optional[str] = None,
    model: Optional[str] = None,
) -> TreeSearchResult:
    if tree is None:
        section_nodes = build_section_tree(graph)
        tree = _render_tree(section_nodes, show_ids=True)
    prompt = TREE_SEARCH_PROMPT.format(tree=tree, query=query)
    messages = [OllamaMessage(role="user", content=prompt)]
    response = llm_client.chat(messages, model=model)
    selected_ids = _parse_selected_ids(response)
    context = _fetch_section_content(graph, selected_ids)
    return TreeSearchResult(selected_ids=selected_ids, context=context)


def answer_query(
    query: str,
    context: str,
    llm_client: OllamaClient,
    model: Optional[str] = None,
) -> str:
    prompt = ANSWER_PROMPT.format(context=context, query=query)
    messages = [OllamaMessage(role="user", content=prompt)]
    return llm_client.chat(messages, model=model)
