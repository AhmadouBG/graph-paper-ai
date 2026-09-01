import os
import sys
from dotenv import load_dotenv

# Permet d'importer depuis le dossier src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer.config import FiltrationConfig
from src.extraction.parser import _parse_with_llamacloud, _build_pure_text_tree
from llm_judge import bedrock_judge

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
    print("⏳ Extraction du PDF via LlamaCloud...")
    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "file", "acharya2019.pdf"))
    page_dicts = _parse_with_llamacloud(pdf_path, os.getenv("LLAMACLOUD_API_KEY"))
    
    # Reconstruction linéaire identique à app.py
    markdown_chunks = []
    for page_data in page_dicts:
        page_num = page_data["page"]
        markdown_chunks.append(f"--- Page {page_num} ---")
        markdown_chunks.append(page_data.get("md", ""))
    markdown_content = "\n".join(markdown_chunks)

    # 1. Structuration identique de l'arbre documentaire
    tree = _build_pure_text_tree(markdown_content)

    # 2. Extraction des textes exacts des nœuds de l'arbre
    node_texts = extract_node_texts(tree)
    contexts = [[text] for text in node_texts if len(text) > 50]

    print(f"✅ Extraction réussie ! {len(contexts)} nœuds documentaires prêts.")

    if not contexts:
        print("❌ Aucun texte extrait.")
        return

    filtration_config = FiltrationConfig(critic_model=bedrock_judge)

    synthesizer = Synthesizer(
        model=bedrock_judge,
        filtration_config=filtration_config,
        async_mode=False,
        max_concurrent=1
    )

    print("🧠 Génération des paires Q&A via NVIDIA (openai/gpt-oss-120b)...")

    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts[:5],
        max_goldens_per_context=1,
    )

    if not goldens:
        print("❌ Alerte : Liste de Goldens vide.")
        return

    dataset = EvaluationDataset(goldens=goldens)
    dataset.save_as(
        file_type="json",
        directory="./test_data",
        file_name="rag_test_dataset"
    )
    print(f"🎉 Succès ! Jeu de données sauvegardé dans ./test_data/rag_test_dataset.json")


if __name__ == "__main__":
    generate_test_suite()
