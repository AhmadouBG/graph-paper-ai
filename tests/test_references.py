from __future__ import annotations

from src.ingestion import extract_cross_references


def test_figure_reference_extraction():
    md = "As shown in Figure 2, the results demonstrate significant improvement."
    result = extract_cross_references(md)
    assert len(result) == 1
    ref = result[0]
    assert ref.target_node_id == "fig_2"
    assert ref.reference_type == "figure"
    assert "Figure 2" in ref.context


def test_multi_type_references():
    md = (
        "Table 3 shows the metrics. "
        "See Section 4.1 for details. "
        "Equation 7 defines the loss. "
        "Fig. 5 illustrates the architecture."
    )
    result = extract_cross_references(md)
    ids = {r.target_node_id for r in result}
    assert "table_3" in ids
    assert "section_4_1" in ids
    assert "equation_7" in ids
    assert "fig_5" in ids
    assert len(result) == 4


def test_no_references_found():
    md = "This paper presents a novel approach to natural language processing."
    result = extract_cross_references(md)
    assert result == []


def test_generic_usage_filtered():
    md = "The figure below illustrates the concept. Table discussions follow."
    result = extract_cross_references(md)
    assert result == []


def test_empty_string():
    assert extract_cross_references("") == []


def test_figure_plural_not_matched():
    md = "Figures 2 and 3 show different results."
    result = extract_cross_references(md)
    assert len(result) == 2
    assert result[0].target_node_id == "fig_2"
    assert result[1].target_node_id == "fig_3"


def test_fig_abbreviation():
    md = "The results (Fig. 3) confirm the hypothesis."
    result = extract_cross_references(md)
    assert len(result) == 1
    assert result[0].target_node_id == "fig_3"


def test_section_with_dot():
    md = "Refer to Section 4.2.1 for implementation details."
    result = extract_cross_references(md)
    assert len(result) == 1
    assert result[0].target_node_id == "section_4_2_1"


def test_context_captures_surrounding_text():
    md = (
        "The training loss decreased steadily. "
        "As shown in Figure 3, the model converges after 50 epochs. "
        "This confirms our hypothesis."
    )
    result = extract_cross_references(md)
    assert len(result) == 1
    assert "Figure 3" in result[0].context
    assert "model converges" in result[0].context


def test_multiple_references_same_line():
    md = "Figure 2 and Table 1 both support the claim in Section 3."
    result = extract_cross_references(md)
    ids = {r.target_node_id for r in result}
    assert "fig_2" in ids
    assert "table_1" in ids
    assert "section_3" in ids
    assert len(result) == 3


def test_equation_reference():
    md = "The objective function is defined in Equation 5."
    result = extract_cross_references(md)
    assert len(result) == 1
    assert result[0].target_node_id == "equation_5"
    assert result[0].reference_type == "equation"


def test_lowercase_references():
    md = "see figure 2 and table 1 for details."
    result = extract_cross_references(md)
    ids = {r.target_node_id for r in result}
    assert "fig_2" in ids
    assert "table_1" in ids
