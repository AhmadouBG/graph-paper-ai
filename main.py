import argparse
import asyncio  # ✨ Requis pour exécuter votre fonction asynchrone
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import ollama

# Importation de vos fonctions personnalisées
from src.pipeline import vectorless_rag_no_loss
# Assurez-vous d'importer vos deux nouvelles fonctions depuis votre dossier source
from src.extraction.parser import _parse_with_llamaparse, _build_pure_text_tree, _get_images_as_base64_map

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _url_to_host(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname

async def run_pipeline_orchestration(pdf_path: Path, api_key: str) -> list[dict]:
    """
    Fonction orchestratrice asynchrone pour chaîner LlamaParse, 
    l'extraction d'images en RAM et la construction de l'arbre.
    """
    print(f"Parsing: {pdf_path} with LlamaParse (Agentic Tier)...")
    raw_result = await _parse_with_llamaparse(pdf_path, api_key)
    
    # 1. Extraction du Markdown textuel
    markdown_content = getattr(raw_result, "markdown", "")
    if not markdown_content and isinstance(raw_result, dict):
        markdown_content = raw_result.get("markdown", "")
        
    # Reconstitution de secours si .markdown n'est pas fourni directement
    if not markdown_content and hasattr(raw_result, "items"):
        pages_md = []
        for idx, page in enumerate(getattr(raw_result.items, "pages", [])):
            pages_md.append(f"--- Page {idx + 1} ---")
            for item in getattr(page, "items", []):
                if hasattr(item, "md") and item.md:
                    pages_md.append(item.md)
                elif hasattr(item, "value") and item.value:
                    pages_md.append(item.value)
        markdown_content = "\n".join(pages_md)

    if not markdown_content:
        print("Error: Could not extract markdown text from LlamaParse response.")
        sys.exit(1)

    # 2. Chaînage propre : Extraction des images directement en Base64 dans la RAM
    print("🖼️ Fetching document images and converting to Base64 in RAM...")
    page_image_map = await _get_images_as_base64_map(raw_result, api_key)
    
    # 3. Construction de l'arbre final combinant Texte + Images Base64
    tree = _build_pure_text_tree(markdown_content, page_image_map)
    return tree

def main():
    parser = argparse.ArgumentParser(
        description="Parse a PDF with LlamaParse and answer questions using vectorless Graph-RAG"
    )
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--model", type=str, default="qwen2.5vl:3b",
                        help="Ollama model (default: qwen2.5vl:3b)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="Ollama server URL (default: http://localhost:11434)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--query", "-q", type=str, help="Single question to answer")
    group.add_argument("--interactive", "-i", action="store_true",
                       help="Interactive Q&A session")

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: File '{pdf_path}' does not exist.")
        sys.exit(1)

    api_key = os.environ.get("LLAMACLOUD_API_KEY")
    if not api_key:
        print("Error: LLAMACLOUD_API_KEY environment variable is not set.")
        sys.exit(1)

    os.environ["OLLAMA_HOST"] = _url_to_host(args.ollama_url)

    # ── Check Ollama availability ─────────────────────────────
    try:
        ollama.list()
    except Exception:
        print(f"\nError: Ollama not available at {args.ollama_url}")
        print("  Start the server:  ollama serve")
        print(f"  Then pull the model:  ollama pull {args.model}")
        sys.exit(1)
    print(f"  Ollama: {args.model} connected")

    # ── Orchestration Loop (Async to Sync) ─────────────────────
    # On exécute la fonction de chaînage asynchrone de manière sécurisée
    tree = asyncio.run(run_pipeline_orchestration(pdf_path, api_key))
    print(f"  Tree successfully built: {len(tree)} root sections identified with embedded visuals.")

    # ── Answer ─────────────────────────────────────────────────
    if args.query:
        print(f"\nQuery: {args.query}\n")
        result_data = vectorless_rag_no_loss(args.query, tree, args.model)
        
        print(f"📝 Answer:\n{result_data['answer']}\n")
        print("📍 Sources used:")
        for citation in result_data["sources"]:
            print(f"👉 {citation}")

    else:
        print("\nInteractive mode. Type questions (Ctrl+C to quit).\n")
        try:
            while True:
                try:
                    q = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not q:
                    continue
                if q.lower() in ("exit", "quit", "/q"):
                    break
                
                result_data = vectorless_rag_no_loss(q, tree, args.model)
                
                print(f"📝 Answer:\n{result_data['answer']}\n")
                print("📍 Sources used:")
                for citation in result_data["sources"]:
                    print(f"👉 {citation}")
                print()
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()