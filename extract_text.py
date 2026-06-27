"""Usage: python extract_text.py path/to/paper.pdf"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from llama_parse import LlamaParse

from src.extraction import extract_images_with_captions

load_dotenv()


def extract_text(pdf_path: str) -> str:
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY") or os.environ.get("LLAMACLOUD_API_KEY")
    if not api_key:
        print("Error: LLAMA_CLOUD_API_KEY not set in .env")
        sys.exit(1)

    print("Extracting images and captions...")
    images = extract_images_with_captions(pdf_path)

    page_captions = {}
    for img in images:
        p = img["numero_page"]
        if p not in page_captions:
            page_captions[p] = []
        if img["legende_detectee"] and img["legende_detectee"] != "Aucune légende trouvée":
            page_captions[p].append(f"[Visual Component] Caption: {img['legende_detectee']}")

    print("Parsing with LlamaParse...")
    parser = LlamaParse(api_key=api_key, result_type="markdown", verbose=True)
    json_objs = parser.get_json_result(pdf_path)
    json_list = json_objs[0]["pages"] if isinstance(json_objs, list) else json_objs["pages"]

    chunks = []
    for page in json_list:
        pn = page["page"]
        chunks.append(f"--- Page {pn} ---")
        if pn in page_captions:
            chunks.append("\n".join(page_captions[pn]))
        chunks.append(page.get("md", page.get("text", "")))

    return "\n".join(chunks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pdf = sys.argv[1]
    if not Path(pdf).exists():
        print(f"File not found: {pdf}")
        sys.exit(1)

    text = extract_text(pdf)

    out_path = Path(pdf).with_suffix(".txt")
    out_path.write_text(text, encoding="utf-8")
    print(f"\nSaved to {out_path}")
    print(f"Total characters: {len(text)}")
