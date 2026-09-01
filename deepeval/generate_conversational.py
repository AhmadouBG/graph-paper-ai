import os
import sys
from dotenv import load_dotenv

# Permet d'importer depuis le dossier src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer.config import FiltrationConfig
from src.extraction.parser import _parse_with_llamacloud
from llm_judge import bedrock_judge

load_dotenv()

def generate_test_suite():
    print("⏳ Extraction du PDF via LlamaCloud...")
    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "file", "acharya2019.pdf"))
    page_dicts = _parse_with_llamacloud(pdf_path, os.getenv("LLAMACLOUD_API_KEY"))
    
    # Format attendu par DeepEval : [["Texte page 1"], ["Texte page 2"], ...]
    contexts = []
    for page_data in page_dicts:
        text_content = page_data.get("md", "").strip()
        if text_content:
            contexts.append([text_content])
            
    print(f"✅ Extraction réussie ! {len(contexts)} pages textuelles prêtes.")
    
    if not contexts:
        print("❌ Aucun texte extrait.")
        return

    # 1. Configure NVIDIA judge as the filtration critic
    filtration_config = FiltrationConfig(critic_model=bedrock_judge)

    # 2. Pass filtration_config to the Synthesizer constructor
    synthesizer = Synthesizer(
        model=bedrock_judge,
        filtration_config=filtration_config,
        async_mode=False,       # Disable uncontrolled parallel generation
        max_concurrent=1
    )

    print("🧠 Génération des paires Q&A via NVIDIA (openai/gpt-oss-120b)...")
    
    # 3. CORRECTION : Nettoyage des arguments de la méthode d'action
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts[:5], # Restreint aux 5 premières pages pour valider le test rapidement
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
