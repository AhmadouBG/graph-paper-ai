"""
tests/test_retriever.py
=======================
Unit tests for src/retrieval/retriever.py

Tests cover:
  - get_node_full_text : content aggregation for leaf/empty nodes
  - retrieve_nodes     : tree traversal, parent expansion, ordering, fallback
"""
from __future__ import annotations

import pytest

from src.retrieval.retriever import get_node_full_text, retrieve_nodes


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_tree() -> list[dict]:
    """
    Builds a representative 3-level tree:

    0000 Introduction (content)
      └── 0001 Background (content)
      └── 0002 Prior Work (empty content, has children)
            └── 0003 Dataset (content)
    0004 Methods (content)
      └── 0005 Experiment Setup (content)
    0006 Results (content)
    """
    return [
        {
            "node_id": "0000", "title": "Introduction",
            "page_start": 1, "page_end": 2,
            "content": "This paper addresses a key problem.",
            "nodes": [
                {
                    "node_id": "0001", "title": "Background",
                    "page_start": 1, "page_end": 1,
                    "content": "Background context on prior work.",
                    "parent_id": "0000", "nodes": [],
                },
                {
                    "node_id": "0002", "title": "Prior Work",
                    "page_start": 2, "page_end": 2,
                    "content": "",                    # empty — should aggregate from children
                    "parent_id": "0000",
                    "nodes": [
                        {
                            "node_id": "0003", "title": "Dataset",
                            "page_start": 2, "page_end": 2,
                            "content": "We used the MNIST dataset.",
                            "parent_id": "0002", "nodes": [],
                        },
                    ],
                },
            ],
        },
        {
            "node_id": "0004", "title": "Methods",
            "page_start": 3, "page_end": 4,
            "content": "We applied a novel vectorless RAG approach.",
            "nodes": [
                {
                    "node_id": "0005", "title": "Experiment Setup",
                    "page_start": 3, "page_end": 3,
                    "content": "Experiments ran on GPU cluster.",
                    "parent_id": "0004", "nodes": [],
                },
            ],
        },
        {
            "node_id": "0006", "title": "Results",
            "page_start": 5, "page_end": 6,
            "content": "Accuracy reached 98% on benchmark.",
            "nodes": [],
        },
    ]


# ── get_node_full_text ────────────────────────────────────────────────────────

def test_full_text_returns_own_content_when_present():
    node = {"content": "Direct content.", "nodes": []}
    assert get_node_full_text(node) == "Direct content."


def test_full_text_aggregates_children_when_empty():
    node = {
        "content": "",
        "nodes": [
            {"title": "Child A", "content": "Child A content.", "nodes": []},
            {"title": "Child B", "content": "Child B content.", "nodes": []},
        ],
    }
    result = get_node_full_text(node)
    assert "Child A content." in result
    assert "Child B content." in result


def test_full_text_aggregates_recursively():
    node = {
        "content": "",
        "nodes": [
            {
                "title": "Child",
                "content": "",
                "nodes": [
                    {"title": "Grandchild", "content": "Deep content.", "nodes": []},
                ],
            },
        ],
    }
    result = get_node_full_text(node)
    assert "Deep content." in result


def test_full_text_empty_node_no_children():
    node = {"content": "", "nodes": []}
    result = get_node_full_text(node)
    assert result == ""


def test_full_text_strips_whitespace():
    node = {"content": "   lots of spaces   ", "nodes": []}
    assert get_node_full_text(node) == "lots of spaces"


# ── retrieve_nodes ────────────────────────────────────────────────────────────

def test_retrieve_known_root_node(tmp_path):
    tree = _make_tree()
    result = retrieve_nodes(["0006"], tree)
    ids = [n["node_id"] for n in result]
    assert "0006" in ids


def test_retrieve_adds_text_field():
    tree = _make_tree()
    result = retrieve_nodes(["0006"], tree)
    for n in result:
        assert "text" in n, "retrieve_nodes must inject a 'text' key"


def test_retrieve_child_also_returns_parent():
    """Selecting a child node should expand to include its parent."""
    tree = _make_tree()
    result = retrieve_nodes(["0001"], tree)    # 0001 is a child of 0000
    ids = {n["node_id"] for n in result}
    assert "0001" in ids
    assert "0000" in ids, "Parent node should be included via parent expansion"


def test_retrieve_empty_ids_uses_fallback():
    tree = _make_tree()
    result = retrieve_nodes([], tree)
    assert len(result) == 1
    assert result[0]["node_id"] == tree[0]["node_id"]


def test_retrieve_invalid_ids_uses_fallback():
    tree = _make_tree()
    result = retrieve_nodes(["nonexistent_id"], tree)
    # Fallback → first root node
    assert len(result) == 1
    assert result[0]["node_id"] == tree[0]["node_id"]


def test_retrieve_preserves_document_order():
    """Nodes must be returned in their original document order, not selection order."""
    tree = _make_tree()
    # Select 0006 (Results) then 0004 (Methods) — reversed order
    result = retrieve_nodes(["0006", "0004"], tree)
    ids = [n["node_id"] for n in result]
    assert ids.index("0004") < ids.index("0006")


def test_retrieve_no_duplicates():
    tree = _make_tree()
    # 0001 expands to include 0000; if we also explicitly ask 0000 → no dup
    result = retrieve_nodes(["0000", "0001"], tree)
    ids = [n["node_id"] for n in result]
    assert len(ids) == len(set(ids)), "No duplicate nodes in result"


def test_retrieve_empty_tree_returns_empty():
    result = retrieve_nodes(["0000"], [])
    assert result == []


def test_retrieve_deep_nested_node():
    """Node 0003 is nested 2 levels deep — retriever must find it."""
    tree = _make_tree()
    result = retrieve_nodes(["0003"], tree)
    ids = {n["node_id"] for n in result}
    assert "0003" in ids


def test_retrieve_text_field_aggregated_for_empty_content():
    """Node 0002 has empty content — text should aggregate from child 0003."""
    tree = _make_tree()
    result = retrieve_nodes(["0002"], tree)
    node_0002 = next(n for n in result if n["node_id"] == "0002")
    assert "MNIST" in node_0002["text"]
