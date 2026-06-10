from __future__ import annotations

from src.query import Anchor, AnchorType, QueryParseResult
from src.query.anchor_extraction import (
    QueryType,
    _to_node_id,
    extract_query_anchors,
)


class TestExtractQueryAnchors:
    def test_returns_queryparseresult(self):
        result = extract_query_anchors("Show Figure 3")
        assert isinstance(result, QueryParseResult)

    def test_none_input_returns_no_anchors(self):
        result = extract_query_anchors(None)
        assert not result.has_anchors
        assert len(result.anchors) == 0

    def test_extract_figure_anchor(self):
        result = extract_query_anchors("Show Figure 3")
        assert result.has_anchors
        assert result.query_type == QueryType.STRUCTURAL
        assert len(result.anchors) == 1
        assert result.anchors[0].node_id == "fig_3"
        assert result.anchors[0].anchor_type == AnchorType.FIGURE

    def test_extract_table_anchor(self):
        result = extract_query_anchors("What does Table 1 show?")
        assert result.has_anchors
        assert result.anchors[0].node_id == "table_1"
        assert result.anchors[0].anchor_type == AnchorType.TABLE

    def test_extract_section_anchor(self):
        result = extract_query_anchors("Summarize Section 4")
        assert result.has_anchors
        assert result.anchors[0].node_id == "section_4"
        assert result.anchors[0].anchor_type == AnchorType.SECTION
        assert result.anchors[0].requested_children is True

    def test_extract_equation_anchor(self):
        result = extract_query_anchors("Explain Equation 5")
        assert result.has_anchors
        assert result.anchors[0].node_id == "equation_5"
        assert result.anchors[0].anchor_type == AnchorType.EQUATION

    def test_multi_anchor_extraction(self):
        query = "what does Table 3 say about the method in Figure 2?"
        result = extract_query_anchors(query)
        assert result.has_anchors
        assert len(result.anchors) == 2
        node_ids = {a.node_id for a in result.anchors}
        assert node_ids == {"table_3", "fig_2"}

    def test_no_anchor_semantic_fallback(self):
        result = extract_query_anchors("what is the main result?")
        assert not result.has_anchors
        assert result.query_type == QueryType.SEMANTIC
        assert len(result.anchors) == 0

    def test_ambiguous_reference_no_error(self):
        result = extract_query_anchors("what about the foo bar?")
        assert not result.has_anchors
        assert len(result.anchors) == 0

    def test_fig_abbreviation(self):
        result = extract_query_anchors("What is shown in Fig. 2?")
        assert result.has_anchors
        assert result.anchors[0].node_id == "fig_2"

    def test_section_with_subsection(self):
        result = extract_query_anchors("Explain Section 4.1")
        assert result.has_anchors
        assert result.anchors[0].node_id == "section_4_1"
        assert result.anchors[0].requested_children is True

    def test_figure_with_letter_suffix(self):
        result = extract_query_anchors("What is Figure 3a?")
        assert result.has_anchors
        assert result.anchors[0].node_id == "fig_3a"

    def test_hierarchical_figure_number(self):
        result = extract_query_anchors("See Figure 1.2")
        assert result.has_anchors
        assert result.anchors[0].node_id == "fig_1_2"

    def test_hierarchical_table_number(self):
        result = extract_query_anchors("Check Table 2.1")
        assert result.has_anchors
        assert result.anchors[0].node_id == "table_2_1"

    def test_sec_abbreviation(self):
        result = extract_query_anchors("See Sec. 4")
        assert result.has_anchors
        assert result.anchors[0].node_id == "section_4"

    def test_sect_abbreviation(self):
        result = extract_query_anchors("See Sect. 4")
        assert result.has_anchors
        assert result.anchors[0].node_id == "section_4"

    def test_empty_string_returns_no_anchors(self):
        result = extract_query_anchors("")
        assert not result.has_anchors
        assert len(result.anchors) == 0


class TestNodeIdMapping:
    def test_figure_node_id(self):
        assert _to_node_id(AnchorType.FIGURE, "1") == "fig_1"

    def test_table_node_id(self):
        assert _to_node_id(AnchorType.TABLE, "3") == "table_3"

    def test_section_node_id(self):
        assert _to_node_id(AnchorType.SECTION, "4") == "section_4"

    def test_equation_node_id(self):
        assert _to_node_id(AnchorType.EQUATION, "5") == "equation_5"


class TestAnchorDataclass:
    def test_default_requested_children_is_false(self):
        anchor = Anchor(node_id="fig_1", anchor_type=AnchorType.FIGURE, label="1")
        assert anchor.requested_children is False

    def test_section_sets_requested_children(self):
        anchor = Anchor(
            node_id="section_4",
            anchor_type=AnchorType.SECTION,
            label="4",
            requested_children=True,
        )
        assert anchor.requested_children is True


class TestQueryParseResultDataclass:
    def test_defaults_no_anchors_semantic(self):
        result = QueryParseResult()
        assert len(result.anchors) == 0
        assert result.has_anchors is False
        assert result.query_type == QueryType.SEMANTIC

    def test_structural_query_type(self):
        result = QueryParseResult(
            anchors=[Anchor(node_id="fig_1", anchor_type=AnchorType.FIGURE, label="1")],
            has_anchors=True,
            query_type=QueryType.STRUCTURAL,
        )
        assert result.query_type == QueryType.STRUCTURAL
