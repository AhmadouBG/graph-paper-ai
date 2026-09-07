import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from llm_judge import nvidia_judge

from deepeval import assert_test
from src.extraction.parser import _build_pure_text_tree, _parse_with_llamacloud
from src.pipeline import vectorless_rag_no_loss

load_dotenv()

def _get_or_build_tree() -> list[dict]:
    """Cache the parsed document tree locally to avoid PDF re-parsing on every test run."""
    cache_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "test_data", "cached_tree.json")
    )
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    pdf_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "file", "acharya2019.pdf")
    )
    page_dicts = _parse_with_llamacloud(pdf_path, os.getenv("LLAMACLOUD_API_KEY"))
    markdown_chunks = []
    for page_data in page_dicts:
        markdown_chunks.append(f"--- Page {page_data['page']} ---")
        markdown_chunks.append(page_data.get("md", ""))

    tree = _build_pure_text_tree("\n".join(markdown_chunks))

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)

    return tree


GLOBAL_TREE = _get_or_build_tree()

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path="./test_data/rag_test_dataset.json")


@pytest.mark.parametrize("golden", dataset.goldens)
def test_vectorless_rag_pipeline(golden):
    rag_output = vectorless_rag_no_loss(query=golden.input, tree=GLOBAL_TREE)

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=rag_output["answer"],
        expected_output=golden.expected_output,
        retrieval_context=rag_output["retrieved_texts"],
    )

    # async_mode=True executes metric requests concurrently, cutting test runtime by 70%
    metrics = [
        FaithfulnessMetric(threshold=0.6, model=nvidia_judge, async_mode=True),
        AnswerRelevancyMetric(threshold=0.6, model=nvidia_judge, async_mode=True),
        ContextualPrecisionMetric(threshold=0.6, model=nvidia_judge, async_mode=True),
        ContextualRecallMetric(threshold=0.6, model=nvidia_judge, async_mode=True),
        ContextualRelevancyMetric(threshold=0.6, model=nvidia_judge, async_mode=True),
    ]

    assert_test(test_case, metrics, run_async=True)
