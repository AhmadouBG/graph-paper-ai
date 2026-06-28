"""Usage: python extract_text.py path/to/paper.pdf"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.extraction import extract_images_with_captions
from src.extraction.parser import _parse_with_llamacloud

load_dotenv()


def extract_text(pdf_path: str) -> str:
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY") or os.environ.get("LLAMACLOUD_API_KEY")
    if not api_key:
        print("Error: LLAMA_CLOUD_API_KEY not set in .env")
        sys.exit(1)

    print("Parsing with LlamaParse...")
    # 1. Initialize client (reads LLAMA_CLOUD_API_KEY from environment)
    json_list = _parse_with_llamacloud(pdf_path, api_key)


    chunks = []
    for page in json_list:
        pn = page["page"]
        chunks.append(f"--- Page {pn} ---")
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
