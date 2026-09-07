"""
tests/test_pipeline.py
======================
Unit tests for src/pipeline.py

Tests cover:
  - print_tree     : outputs correct indentation / node info (no crash)
  - get_total_pages: finds the highest page number across the whole tree
  - vectorless_rag_no_loss: full end-to-end pipeline with all Ollama calls mocked
"""
from __future__ import annotations

import io
import sys
from unittest.mock import patch

from src.pipeline import get_total_pages, print_tree, vectorless_rag_no_loss

# ── Fixture ───────────────────────────────────────────────────────────────────

def _make_tree() -> list[dict]:
    return [
        {
            "node_id": "0000", "title": "Introduction",
            "page_start": 1, "page_end": 3,
            "content": "Intro text.", "text": "Intro text.",
            "nodes": [
                {
                    "node_id": "0001", "title": "Background",
                    "page_start": 2, "page_end": 3,
                    "content": "Background details.", "text": "Background details.",
                    "nodes": [],
                },
            ],
        },
        {
            "node_id": "0002", "title": "Methods",
            "page_start": 4, "page_end": 7,
            "content": "We propose a novel approach.", "text": "We propose a novel approach.",
            "nodes": [],
        },
        {
            "node_id": "0003", "title": "Results",
            "page_start": 8, "page_end": 10,
            "content": "Accuracy: 98%.", "text": "Accuracy: 98%.",
            "nodes": [],
        },
    ]


# ── print_tree ────────────────────────────────────────────────────────────────

def _capture_print_tree(tree) -> str:
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    print_tree(tree)
    sys.stdout = old
    return buf.getvalue()


def test_print_tree_runs_without_error():
    _capture_print_tree(_make_tree())


def test_print_tree_includes_all_titles():
    output = _capture_print_tree(_make_tree())
    assert "Introduction" in output
    assert "Methods" in output
    assert "Results" in output


def test_print_tree_includes_nested_title():
    output = _capture_print_tree(_make_tree())
    assert "Background" in output


def test_print_tree_shows_node_ids():
    output = _capture_print_tree(_make_tree())
    assert "0000" in output


def test_print_tree_empty_tree_no_crash():
    output = _capture_print_tree([])
    assert output == ""


# ── get_total_pages ───────────────────────────────────────────────────────────

def test_get_total_pages_flat_tree():
    tree = _make_tree()
    assert get_total_pages(tree) == 10  # Results ends on page 10


def test_get_total_pages_with_nested_deeper():
    tree = [
        {
            "node_id": "0000", "title": "A",
            "page_start": 1, "page_end": 2, "content": "", "nodes": [
                {
                    "node_id": "0001", "title": "B",
                    "page_start": 1, "page_end": 5, "content": "", "nodes": [
                        {
                            "node_id": "0002", "title": "C",
                            "page_start": 3, "page_end": 15, "content": "", "nodes": [],
                        }
                    ],
                }
            ],
        }
    ]
    assert get_total_pages(tree) == 15


def test_get_total_pages_single_page():
    tree = [{"node_id": "x", "title": "X", "page_start": 1, "page_end": 1,
             "content": "", "nodes": []}]
    assert get_total_pages(tree) == 1


def test_get_total_pages_empty_returns_one():
    # No nodes → default max_page = 1
    assert get_total_pages([]) == 1


# ── vectorless_rag_no_loss ────────────────────────────────────────────────────

def _mock_tree_search_result(node_ids: list[str]) -> dict:
    return {"node_list": node_ids, "thinking": "Mocked routing decision."}


def _mock_ollama_chat_answer(content: str = "Mocked answer."):
    return {"message": {"content": content}}


@patch("src.pipeline.llm_tree_search_ollama")
@patch("src.pipeline.generate_answer")
def test_rag_returns_expected_keys(mock_gen, mock_search):
    tree = _make_tree()
    mock_search.return_value = _mock_tree_search_result(["0002"])
    mock_gen.return_value = "Methods use a novel approach."

    result = vectorless_rag_no_loss("What methods were used?", tree)

    assert "answer" in result
    assert "thinking" in result
    assert "retrieved_sections" in result
    assert "retrieved_texts" in result


@patch("src.pipeline.llm_tree_search_ollama")
@patch("src.pipeline.generate_answer")
def test_rag_answer_comes_from_generator(mock_gen, mock_search):
    tree = _make_tree()
    mock_search.return_value = _mock_tree_search_result(["0003"])
    mock_gen.return_value = "Accuracy was 98%."

    result = vectorless_rag_no_loss("What were the results?", tree)
    assert result["answer"] == "Accuracy was 98%."


@patch("src.pipeline.llm_tree_search_ollama")
@patch("src.pipeline.generate_answer")
def test_rag_thinking_propagated(mock_gen, mock_search):
    tree = _make_tree()
    mock_search.return_value = {
        "node_list": ["0000"],
        "thinking": "Introduction is the most relevant node.",
    }
    mock_gen.return_value = "Some answer."

    result = vectorless_rag_no_loss("Describe the introduction.", tree)
    assert "Introduction is the most relevant node." in result["thinking"]


@patch("src.pipeline.llm_tree_search_ollama")
@patch("src.pipeline.generate_answer")
def test_rag_retrieved_sections_include_titles(mock_gen, mock_search):
    tree = _make_tree()
    mock_search.return_value = _mock_tree_search_result(["0002"])
    mock_gen.return_value = "Answer."

    result = vectorless_rag_no_loss("What methods were used?", tree)
    assert any("Methods" in s for s in result["retrieved_sections"])


@patch("src.pipeline.llm_tree_search_ollama")
@patch("src.pipeline.generate_answer")
def test_rag_retrieved_sections_include_pages(mock_gen, mock_search):
    tree = _make_tree()
    mock_search.return_value = _mock_tree_search_result(["0002"])
    mock_gen.return_value = "Answer."

    result = vectorless_rag_no_loss("What methods were used?", tree)
    # Should format page range as "p. 4-7"
    assert any("p." in s for s in result["retrieved_sections"])


@patch("src.pipeline.llm_tree_search_ollama")
@patch("src.pipeline.generate_answer")
def test_rag_retrieved_texts_not_empty(mock_gen, mock_search):
    tree = _make_tree()
    mock_search.return_value = _mock_tree_search_result(["0003"])
    mock_gen.return_value = "Answer."

    result = vectorless_rag_no_loss("Results?", tree)
    assert len(result["retrieved_texts"]) > 0


@patch("src.pipeline.llm_tree_search_ollama")
@patch("src.pipeline.generate_answer")
def test_rag_fallback_on_invalid_node_ids(mock_gen, mock_search):
    """When LLM returns non-existent IDs, retriever falls back to root node."""
    tree = _make_tree()
    mock_search.return_value = _mock_tree_search_result(["DOES_NOT_EXIST"])
    mock_gen.return_value = "Fallback answer."

    result = vectorless_rag_no_loss("Anything?", tree)
    # Should not crash and still return a valid structure
    assert result["answer"] == "Fallback answer."
    assert isinstance(result["retrieved_sections"], list)


@patch("src.pipeline.llm_tree_search_ollama")
@patch("src.pipeline.generate_answer")
def test_rag_same_page_start_end_formats_single_page(mock_gen, mock_search):
    """When page_start == page_end, section label should be 'p. X' not 'p. X-X'."""
    tree = [
        {
            "node_id": "n1", "title": "Single Page Section",
            "page_start": 3, "page_end": 3,
            "content": "Content here.", "text": "Content here.",
            "nodes": [],
        }
    ]
    mock_search.return_value = _mock_tree_search_result(["n1"])
    mock_gen.return_value = "Answer."

    result = vectorless_rag_no_loss("What?", tree)
    section_label = result["retrieved_sections"][0]
    assert "p. 3" in section_label
    assert "p. 3-3" not in section_label
