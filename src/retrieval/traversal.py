from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx

# A ContextNode is a node in the traversal that represents a piece of the paper that is relevant to the query.
# as a retrieval system, it is important to know the node_type and node_id to ensure that we are retrieving the correct node.

@dataclass
class ContextNode:
    node_id: str
    node_type: str
    content: str
    page: int | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# stores the result of the traversal
@dataclass
class ContextResult:
    nodes: List[ContextNode] = field(default_factory=list)
    total_tokens: int = 0
    truncated: bool = False

# count tokens
def _count_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)

# performs a breadth-first search on the graph to collect relevant nodes
# BFS is used because the retrieval problem requires proximity-preserving context collection
# 1. Semantic relevance decays with distance — a node 1 hop away (direct neighbor) is more likely relevant to the query anchor than a node 5 hops away. BFS collects all depth-1 nodes before depth-2, ensuring closest relations are prioritized.
# 2. Token limit guarantees — with max_tokens=8000, BFS naturally fills the context with the nearest nodes first. DFS could dive deep into a single branch and exhaust the budget on distant relations while missing closer ones.
# 3. Breadth > depth for multi-modal papers — a figure anchor's immediate neighbors (caption text, containing section, sibling figures) are more useful than chasing a chain of references deep into unrelated sections.
#BFS is a level-by-level traversal, ensuring that the closest nodes to the query anchor are visited first. This is important because the most relevant nodes are likely to be the closest to the query anchor.
def bfs_traverse(
    graph: nx.DiGraph,
    start_nodes: List[str],
    depth: int = 1,
    max_tokens: int = 8000,
    edge_types: Optional[List[str]] = None,
) -> ContextResult:
    result = ContextResult()
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque()

    for node_id in start_nodes:
        if node_id not in visited:
            visited.add(node_id)
            queue.append((node_id, 0))

    def _neighbors(node: str) -> List[str]:
        candidates: List[str] = []
        for n in graph.successors(node):
            if edge_types is None or graph.edges[node, n].get("edge_type") in edge_types:
                candidates.append(n)
        for n in graph.predecessors(node):
            if edge_types is None or graph.edges[n, node].get("edge_type") in edge_types:
                candidates.append(n)
        return candidates

    while queue:
        current_id, current_depth = queue.popleft()

        if not graph.has_node(current_id):
            continue

        node_data = graph.nodes[current_id]
        node = ContextNode(
            node_id=current_id,
            node_type=node_data.get("node_type", "unknown"),
            content=node_data.get("content", ""),
            page=node_data.get("page"),
            metadata={k: v for k, v in node_data.items()
                      if k not in ("node_type", "content", "page")},
        )
        result.nodes.append(node)
        result.total_tokens += _count_tokens(node.content)

        if current_depth < depth and result.total_tokens < max_tokens:
            for neighbor in _neighbors(current_id):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_depth + 1))

            for neighbor in graph.predecessors(current_id):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_depth + 1))

    if result.total_tokens > max_tokens:
        accumulated = 0
        trimmed: List[ContextNode] = []
        for node in result.nodes:
            tokens = _count_tokens(node.content)
            if accumulated + tokens <= max_tokens:
                trimmed.append(node)
                accumulated += tokens
            else:
                break
        result.nodes = trimmed
        result.total_tokens = accumulated
        result.truncated = True

    return result
# The pipeline is: query → anchors → BFS context → LLM answer.
# BFS casts the net; the LLM keeps what matters
