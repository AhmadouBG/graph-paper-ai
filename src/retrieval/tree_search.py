from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx

from src.ingestion.tree import build_node_id_map, build_section_tree, print_tree as _render_tree
from src.llm.ollama_client import OllamaClient, OllamaMessage

TREE_SEARCH_PROMPT = (
    "You are a document retrieval system. Given a user query and a document's"
    " section tree, identify the most relevant sections by their [XXXX] IDs.\n"
    "\n"
    "Return ONLY a comma-separated list of the most relevant section IDs,"
    " nothing else. Example: 0000,0003,0005\n"
    "\n"
    "Document tree:\n"
    "{tree}\n"
    "\n"
    "User query: {query}\n"
    "\n"
    "Relevant section IDs:"
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

ID_PATTERN = re.compile(r"\b(\d{4})\b")


@dataclass
class TreeSearchResult:
    selected_ids: List[str] = field(default_factory=list)
    context: str = ""


def _parse_selected_ids(response: str, max_ids: int = 20) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for m in ID_PATTERN.finditer(response):
        seq = m.group(1)
        if seq not in seen:
            seen.add(seq)
            result.append(seq)
    return result[:max_ids]


def _fetch_section_content(
    graph: nx.DiGraph,
    selected_ids: List[str],
    node_id_map: Dict[str, str],
) -> str:
    parts: List[str] = []
    for seq_id in selected_ids:
        graph_node_id = node_id_map.get(seq_id)
        if not graph_node_id or not graph.has_node(graph_node_id):
            continue
        attr = graph.nodes[graph_node_id]
        title = attr.get("title", graph_node_id)
        content = attr.get("content", "")
        parts.append(f"[{seq_id}] {title}\n{content}")
    return "\n\n".join(parts)


def tree_search(
    query: str,
    graph: nx.DiGraph,
    llm_client: OllamaClient,
    tree: Optional[str] = None,
    node_id_map: Optional[Dict[str, str]] = None,
    model: Optional[str] = None,
) -> TreeSearchResult:
    if tree is None:
        section_nodes = build_section_tree(graph)
        tree = _render_tree(section_nodes)
    if node_id_map is None:
        node_id_map = build_node_id_map(graph)
    prompt = TREE_SEARCH_PROMPT.format(tree=tree, query=query)
    messages = [OllamaMessage(role="user", content=prompt)]
    response = llm_client.chat(messages, model=model)
    selected_ids = _parse_selected_ids(response)
    context = _fetch_section_content(graph, selected_ids, node_id_map)
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
