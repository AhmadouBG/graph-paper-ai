"""
tests/test_tree_search.py
=========================
Unit tests for src/retrieval/tree_search.py (FastTreeRetriever)

Tests cover:
  - Flattening       : tree → flat list
  - tokenize()       : lowercasing and punctuation removal
  - Regex pass (A)   : figure / table detection
  - BM25 pass (B)    : candidate pre-filtering and comparative boosting
  - SLM pass (C)     : mocked LLM call + anti-hallucination filter + fallback
  - search()         : full routing pipeline (regex → BM25 → SLM)
  - Edge cases       : empty tree, no candidates
"""

from __future__ import annotations

from unittest.mock import patch

from src.retrieval.tree_search import FastTreeRetriever

# ── Fixture ───────────────────────────────────────────────────────────────────


def _make_tree() -> list[dict]:
    return [
        {
            "node_id": "intro",
            "title": "Introduction",
            "content": "This paper introduces a vectorless RAG pipeline.",
            "page_start": 1,
            "page_end": 2,
            "nodes": [],
        },
        {
            "node_id": "methods",
            "title": "Methods",
            "content": "We apply BM25 and SLM re-ranking. See Figure 1.",
            "page_start": 3,
            "page_end": 4,
            "nodes": [],
        },
        {
            "node_id": "results",
            "title": "Results",
            "content": "TABLE I. Overall accuracy results across benchmarks.",
            "page_start": 5,
            "page_end": 6,
            "nodes": [],
        },
        {
            "node_id": "conclusion",
            "title": "Conclusion",
            "content": "Best performance is achieved by the proposed pipeline.",
            "page_start": 7,
            "page_end": 7,
            "nodes": [],
        },
        {
            "node_id": "fig_section",
            "title": "Figure 1 Analysis",
            "content": "Figure 1 shows the pipeline overview diagram.",
            "page_start": 4,
            "page_end": 4,
            "nodes": [],
        },
    ]


# ── Flattening ────────────────────────────────────────────────────────────────


def test_flatten_top_level():
    tree = _make_tree()
    r = FastTreeRetriever(tree)
    assert len(r.flattened_nodes) == len(tree)


def test_flatten_nested_nodes():
    tree = [
        {
            "node_id": "root",
            "title": "Root",
            "content": "",
            "page_start": 1,
            "page_end": 2,
            "nodes": [
                {
                    "node_id": "child",
                    "title": "Child",
                    "content": "text",
                    "page_start": 1,
                    "page_end": 1,
                    "nodes": [],
                },
            ],
        }
    ]
    r = FastTreeRetriever(tree)
    ids = [n["node_id"] for n in r.flattened_nodes]
    assert "root" in ids and "child" in ids


# ── tokenize ─────────────────────────────────────────────────────────────────


def test_tokenize_lowercases():
    result = FastTreeRetriever.tokenize("Hello World")
    assert result == ["hello", "world"]


def test_tokenize_removes_punctuation():
    result = FastTreeRetriever.tokenize("Hello, World!")
    assert "," not in result and "!" not in result


def test_tokenize_empty_string():
    assert FastTreeRetriever.tokenize("") == []


# ── Regex Pass (A) ────────────────────────────────────────────────────────────


def test_regex_detects_figure_query():
    tree = _make_tree()
    r = FastTreeRetriever(tree)
    ids = r.find_visual_element_by_regex("What does Figure 1 show?")
    assert len(ids) > 0


def test_regex_returns_correct_node_for_figure():
    tree = _make_tree()
    r = FastTreeRetriever(tree)
    ids = r.find_visual_element_by_regex("Explain Figure 1")
    assert "fig_section" in ids or "methods" in ids  # methods contains "Figure 1"


def test_regex_detects_table_query():
    tree = _make_tree()
    r = FastTreeRetriever(tree)
    ids = r.find_visual_element_by_regex("Show me Table 1")
    # "results" node contains "TABLE I"
    assert len(ids) > 0


def test_regex_returns_empty_for_plain_query():
    tree = _make_tree()
    r = FastTreeRetriever(tree)
    ids = r.find_visual_element_by_regex("What is the main contribution?")
    assert ids == []


# ── BM25 Pass (B) ─────────────────────────────────────────────────────────────


def test_bm25_returns_candidates():
    tree = _make_tree()
    r = FastTreeRetriever(tree)
    candidates = r._bm25_candidate_filter("vectorless RAG pipeline", max_candidates=3)
    assert len(candidates) <= 3
    assert len(candidates) > 0


def test_bm25_intro_tops_relevant_query():
    tree = _make_tree()
    r = FastTreeRetriever(tree)
    candidates = r._bm25_candidate_filter("vectorless RAG pipeline", max_candidates=5)
    top_id = candidates[0]["node_id"]
    assert top_id == "intro"


def test_bm25_boosts_conclusion_on_comparative_query():
    tree = _make_tree()
    r = FastTreeRetriever(tree)
    # "best performance comparison evaluation" → comparative intent
    candidates = r._bm25_candidate_filter(
        "What is the best performance comparison evaluation overall?", max_candidates=5
    )
    ids = [n["node_id"] for n in candidates]
    # Conclusion should be boosted and appear in top candidates
    assert "conclusion" in ids


def test_bm25_fallback_on_no_matches():
    tree = _make_tree()
    r = FastTreeRetriever(tree)
    # Completely unrelated query → scores all 0 → fallback to first N nodes
    candidates = r._bm25_candidate_filter("xyzzy nonsense gibberish", max_candidates=3)
    assert len(candidates) > 0


# ── SLM Pass (C) — mocked ────────────────────────────────────────────────────


def _make_retriever_with_mock_slm(mock_response_content: str) -> FastTreeRetriever:
    tree = _make_tree()
    r = FastTreeRetriever(tree)
    mock_resp = {"message": {"content": mock_response_content}}
    with patch("src.retrieval.tree_search.ollama.chat", return_value=mock_resp):
        result = r._slm_rerank(
            query="What is the main contribution?",
            candidates=r.flattened_nodes[:3],
            top_k=2,
        )
    return result


def test_slm_returns_valid_ids_from_llm():
    slm_json = '{"thinking": "intro is most relevant", "node_list": ["intro", "methods"]}'
    result = _make_retriever_with_mock_slm(slm_json)
    assert "intro" in result["node_list"]


def test_slm_filters_hallucinated_ids():
    # LLM returns a node_id that does not exist in the candidate list
    slm_json = '{"thinking": "chosen", "node_list": ["intro", "FAKE_NODE_9999"]}'
    result = _make_retriever_with_mock_slm(slm_json)
    assert "FAKE_NODE_9999" not in result["node_list"]


def test_slm_includes_thinking():
    slm_json = '{"thinking": "intro is best", "node_list": ["intro"]}'
    result = _make_retriever_with_mock_slm(slm_json)
    assert "thinking" in result
    assert len(result["thinking"]) > 0


def test_slm_falls_back_on_ollama_exception():
    tree = _make_tree()
    r = FastTreeRetriever(tree)
    candidates = r.flattened_nodes[:3]
    with patch("src.retrieval.tree_search.ollama.chat", side_effect=Exception("Ollama down")):
        result = r._slm_rerank("What is RAG?", candidates, top_k=2)
    # Fallback must return BM25 candidates without crashing
    assert "node_list" in result
    assert len(result["node_list"]) > 0
    assert "BM25 Fallback" in result["thinking"]


def test_slm_falls_back_on_bad_json():
    tree = _make_tree()
    r = FastTreeRetriever(tree)
    candidates = r.flattened_nodes[:3]
    with patch(
        "src.retrieval.tree_search.ollama.chat",
        return_value={"message": {"content": "not valid json at all"}},
    ):
        result = r._slm_rerank("What is RAG?", candidates, top_k=2)
    assert "node_list" in result
    assert "BM25 Fallback" in result["thinking"]


# ── search() — full routing ───────────────────────────────────────────────────


def test_search_regex_short_circuits_for_figure():
    """If Regex pass matches, SLM is never called."""
    tree = _make_tree()
    r = FastTreeRetriever(tree)
    with patch.object(r, "_slm_rerank") as mock_slm:
        result = r.search("What is shown in Figure 1?")
    mock_slm.assert_not_called()
    assert "node_list" in result
    assert "Regex" in result["thinking"]


def test_search_falls_through_to_bm25_and_slm():
    """A plain semantic query skips Regex and goes through BM25 → SLM."""
    tree = _make_tree()
    r = FastTreeRetriever(tree)
    slm_json = '{"thinking": "intro is relevant", "node_list": ["intro"]}'
    with patch(
        "src.retrieval.tree_search.ollama.chat", return_value={"message": {"content": slm_json}}
    ):
        result = r.search("What is the main contribution of this paper?")
    assert "node_list" in result
    assert len(result["node_list"]) > 0


def test_search_returns_valid_structure():
    tree = _make_tree()
    r = FastTreeRetriever(tree)
    slm_json = '{"thinking": "ok", "node_list": ["results"]}'
    with patch(
        "src.retrieval.tree_search.ollama.chat", return_value={"message": {"content": slm_json}}
    ):
        result = r.search("Show me the results")
    assert "thinking" in result
    assert "node_list" in result
    assert isinstance(result["node_list"], list)


def test_search_empty_tree_returns_default():
    r = FastTreeRetriever([])
    result = r.search("anything")
    assert result["node_list"] == ["0000"]
