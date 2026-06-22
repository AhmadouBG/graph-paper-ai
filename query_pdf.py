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
from src.extraction.parser import _parse_with_llamaparse, _build_pure_text_tree 

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _url_to_host(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname


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

    # Récupération de la clé API LlamaCloud depuis l'environnement
    api_key = os.environ.get("LLAMACLOUD_API_KEY")
    if not api_key:
        print("Error: LLAMA_CLOUD_API_KEY environment variable is not set.")
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

    # ── Parse PDF with LlamaParse & Build Tree ──────────────────
    print(f"Parsing: {pdf_path} with LlamaParse (Agentic Tier)...")
    
    # Exécution synchrone de la fonction asynchrone _parse_with_llamaparse
    raw_result = asyncio.run(_parse_with_llamaparse(pdf_path, api_key))
    
    # Extraction du texte brut au format Markdown (généralement dans result.markdown ou obtenu via l'API)
    # Note : Si votre instance renvoie directement l'objet de parsing LlamaCloud,
    # vérifiez la façon dont vous accédez au texte brut (ex: raw_result.markdown ou via get_text_by_page())
    markdown_content = getattr(raw_result, "markdown", "")
    if not markdown_content and isinstance(raw_result, dict):
        markdown_content = raw_result.get("markdown", "")
        
    # Si LlamaCloud renvoie un objet complexe d'items textuels sans chaîne brute globale, 
    # vous pouvez reconstruire la chaîne Markdown en joignant les pages :
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

    # Génération de l'arbre textuel navigable sans perte
    tree = _build_pure_text_tree(markdown_content)
    print(f"  Tree successfully built: {len(tree)} root sections identified.")

    # ── Answer ─────────────────────────────────────────────────
    if args.query:
        print(f"\nQuery: {args.query}\n")
        answer = vectorless_rag_no_loss(args.query, tree, args.model)

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
                print()
                answer = vectorless_rag_no_loss(q, tree, args.model)
                print(f"📝 Answer:\n{answer}")
                print()
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
