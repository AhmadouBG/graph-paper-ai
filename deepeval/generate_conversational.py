import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import FiltrationConfig
from llm_judge import nvidia_judge

from src.extraction.parser import _build_pure_text_tree, _parse_with_llamacloud

load_dotenv()


def extract_node_texts(nodes: list[dict]) -> list[str]:
    """Extract content from all tree nodes recursively."""
    texts = []
    for node in nodes:
        content = node.get("content", "").strip()
        if content:
            full_chunk = f"# {node.get('title', '')}\n\n{content}"
            texts.append(full_chunk)
        if node.get("nodes"):
            texts.extend(extract_node_texts(node["nodes"]))
    return texts


def generate_test_suite():
    print("⏳ Extracting PDF via LlamaCloud...")
    pdf_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "file", "acharya2019.pdf")
    )
    page_dicts = _parse_with_llamacloud(pdf_path, os.getenv("LLAMACLOUD_API_KEY"))
    # Identical linear reconstruction
    markdown_chunks = []
    for page_data in page_dicts:
        page_num = page_data["page"]
        markdown_chunks.append(f"--- Page {page_num} ---")
        markdown_chunks.append(page_data.get("md", ""))
    markdown_content = "\n".join(markdown_chunks)

    # Identical tree structuration
    tree = _build_pure_text_tree(markdown_content)

    # Extract exact texts from tree nodes
    node_texts = extract_node_texts(tree)
    contexts = [[text] for text in node_texts if len(text) > 50]

    print("✅ Done ! " + str(len(contexts)) + " documental nodes ready.")

    if not contexts:
        print("❌ No text extracted.")
        return

    filtration_config = FiltrationConfig(critic_model=nvidia_judge)

    synthesizer = Synthesizer(
        model=nvidia_judge, filtration_config=filtration_config, async_mode=False, max_concurrent=1
    )

    print("🧠 Generating Q&A pairs via NVIDIA (openai/gpt-oss-120b)...")

    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts[:5],
        max_goldens_per_context=1,
    )

    if not goldens:
        print("❌ List of Goldens is empty.")
        return

    dataset = EvaluationDataset(goldens=goldens)
    dataset.save_as(file_type="json", directory="./test_data", file_name="rag_test_dataset")
    print("🎉 Done ! Dataset saved in ./test_data/rag_test_dataset.json")


if __name__ == "__main__":
    generate_test_suite()
