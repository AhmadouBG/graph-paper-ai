"""
tests/test_parser.py
====================
Unit tests for src/extraction/parser.py

Tests cover:
  - _count_tokens      : rough token estimator
  - _split_long_content: paragraph-level chunking
  - _build_pure_text_tree: full tree construction from markdown
  - _make_sub_nodes    : long-node splitting into atomic sub-nodes
"""

from __future__ import annotations

from src.extraction.parser import (
    _build_pure_text_tree,
    _count_tokens,
    _make_sub_nodes,
    _split_long_content,
)

# ── _count_tokens ─────────────────────────────────────────────────────────────


def test_count_tokens_empty():
    assert _count_tokens("") == 1  # max(1, 0)


def test_count_tokens_basic():
    # 40 chars → 10 tokens
    text = "a" * 40
    assert _count_tokens(text) == 10


def test_count_tokens_proportional():
    assert _count_tokens("a" * 400) == 100


# ── _split_long_content ───────────────────────────────────────────────────────


def test_split_short_content_not_split():
    short = "Short text."
    result = _split_long_content(short, max_tokens=100)
    assert result == [short]


def test_split_long_content_splits_on_paragraphs():
    # Build a text with 3 long paragraphs, each > 100 tokens (~400+ chars each)
    para = "word " * 120  # ~600 chars = 150 tokens
    text = f"{para}\n\n{para}\n\n{para}"
    result = _split_long_content(text, max_tokens=200)
    assert len(result) > 1


def test_split_preserves_all_content():
    para = "word " * 120
    text = f"{para}\n\n{para}\n\n{para}"
    result = _split_long_content(text, max_tokens=200)
    # Round-trip: all words should still be there
    joined = " ".join(result)
    assert joined.count("word") == text.count("word")


# ── _build_pure_text_tree ─────────────────────────────────────────────────────

SIMPLE_MARKDOWN = """
--- Page 1 ---
# Introduction

This is the introduction paragraph.

## Background

Some background information here.

--- Page 2 ---
## Methods

We used a novel approach.

# Results

Key findings are summarised below.
"""


def test_tree_returns_list():
    tree = _build_pure_text_tree(SIMPLE_MARKDOWN)
    assert isinstance(tree, list)


def test_tree_is_not_empty():
    tree = _build_pure_text_tree(SIMPLE_MARKDOWN)
    assert len(tree) > 0


def test_tree_root_nodes_have_required_keys():
    tree = _build_pure_text_tree(SIMPLE_MARKDOWN)
    required = {"node_id", "title", "page_start", "page_end", "content", "nodes"}
    for node in tree:
        assert required.issubset(node.keys()), f"Node missing keys: {node.keys()}"


def test_tree_introduction_node_exists():
    tree = _build_pure_text_tree(SIMPLE_MARKDOWN)
    all_titles = [n["title"] for n in tree]
    assert any("Introduction" in t for t in all_titles)


def test_tree_page_numbers_tracked():
    tree = _build_pure_text_tree(SIMPLE_MARKDOWN)

    def max_page(nodes):
        m = 0
        for n in nodes:
            m = max(m, n.get("page_end", 0))
            if n.get("nodes"):
                m = max(m, max_page(n["nodes"]))
        return m

    assert max_page(tree) >= 2  # document has 2 pages


def test_tree_nested_children():
    tree = _build_pure_text_tree(SIMPLE_MARKDOWN)

    def find(nodes, title_fragment):
        for n in nodes:
            if title_fragment in n["title"]:
                return n
            found = find(n.get("nodes", []), title_fragment)
            if found:
                return found
        return None

    # "Background" is a ## under # Introduction — should be nested
    bg = find(tree, "Background")
    assert bg is not None, "Background node not found in tree"


def test_tree_deduplicate_h1_titles():
    md = "# Introduction\n\nFirst.\n\n# Introduction\n\nDuplicate.\n"
    tree = _build_pure_text_tree(md)
    intro_nodes = [n for n in tree if "Introduction" in n["title"]]
    assert len(intro_nodes) == 1, "Duplicate H1 title should be deduplicated"


def test_empty_markdown_returns_header_node():
    tree = _build_pure_text_tree("")
    assert len(tree) == 1
    assert tree[0]["title"] == "Document Header / Abstract"


def test_node_ids_are_unique():
    tree = _build_pure_text_tree(SIMPLE_MARKDOWN)

    def collect_ids(nodes, acc):
        for n in nodes:
            acc.append(n["node_id"])
            collect_ids(n.get("nodes", []), acc)

    ids = []
    collect_ids(tree, ids)
    assert len(ids) == len(set(ids)), "All node_ids must be unique"


def test_content_attached_to_correct_node():
    tree = _build_pure_text_tree(SIMPLE_MARKDOWN)

    def find(nodes, title_fragment):
        for n in nodes:
            if title_fragment in n["title"]:
                return n
            found = find(n.get("nodes", []), title_fragment)
            if found:
                return found
        return None

    methods_node = find(tree, "Methods")
    assert methods_node is not None
    assert "novel approach" in methods_node["content"]


# ── _make_sub_nodes ───────────────────────────────────────────────────────────


def _make_dummy_parent(content: str) -> dict:
    return {
        "node_id": "0000",
        "parent_id": None,
        "title": "Test Section",
        "page_start": 1,
        "page_end": 2,
        "content": content,
        "nodes": [],
    }


def test_make_sub_nodes_short_content_no_split():
    parent = _make_dummy_parent("Short content that does not need splitting.")
    sub_nodes, _ = _make_sub_nodes(parent, node_counter=10)
    assert sub_nodes == []


def test_make_sub_nodes_empty_content():
    parent = _make_dummy_parent("")
    sub_nodes, counter = _make_sub_nodes(parent, node_counter=10)
    assert sub_nodes == []
    assert counter == 10


def test_make_sub_nodes_long_content_splits():
    # Create two very long paragraphs that will exceed MAX_TOKENS_PER_NODE (800)
    para = "word " * 500  # ~2500 chars = ~625 tokens each → total 1250 tokens
    content = f"{para}\n\n{para}"
    parent = _make_dummy_parent(content)
    sub_nodes, _ = _make_sub_nodes(parent, node_counter=10)
    assert len(sub_nodes) >= 2


def test_make_sub_nodes_inherit_parent_id():
    para = "word " * 500
    content = f"{para}\n\n{para}"
    parent = _make_dummy_parent(content)
    sub_nodes, _ = _make_sub_nodes(parent, node_counter=10)
    for sub in sub_nodes:
        assert sub["parent_id"] == "0000"


def test_make_sub_nodes_inherit_page_range():
    para = "word " * 500
    content = f"{para}\n\n{para}"
    parent = _make_dummy_parent(content)
    sub_nodes, _ = _make_sub_nodes(parent, node_counter=10)
    for sub in sub_nodes:
        assert sub["page_start"] == 1
        assert sub["page_end"] == 2


def test_make_sub_nodes_table_boundary():
    """A TABLE caption should act as a split boundary."""
    text_before = "word " * 300  # ~375 tokens
    table_block = "TABLE I. Results of the experiment\n" + "data " * 200
    text_after = "word " * 300
    content = f"{text_before}\n{table_block}\n{text_after}"
    parent = _make_dummy_parent(content)
    sub_nodes, _ = _make_sub_nodes(parent, node_counter=5)
    # The TABLE caption boundary should force a split
    assert len(sub_nodes) >= 2
