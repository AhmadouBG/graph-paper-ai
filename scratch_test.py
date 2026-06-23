import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from src.extraction.parser import _parse_with_llamaparse

load_dotenv()

async def test_parse():
    api_key = os.environ.get("LLAMACLOUD_API_KEY")
    if not api_key:
        print("Error: LLAMACLOUD_API_KEY not set")
        return
    
    pdf_path = Path("file/plosone.pdf")
    print(f"Parsing {pdf_path}...")
    result = await _parse_with_llamaparse(pdf_path, api_key)
    
    print("\nResult properties:")
    for prop in dir(result):
        if not prop.startswith("_"):
            print(f"- {prop}")
            
    # Check items
    if hasattr(result, "items"):
        print(f"\nResult has 'items'. Type of items: {type(result.items)}")
        pages = getattr(result.items, "pages", [])
        print(f"Number of pages: {len(pages)}")
        
        # Look for images/figures across pages
        images_found = 0
        total_items = 0
        item_types = set()
        
        for idx, page in enumerate(pages):
            page_items = getattr(page, "items", [])
            total_items += len(page_items)
            for item in page_items:
                itype = getattr(item, "type", None)
                if itype:
                    item_types.add(itype)
                if itype == "image":
                    images_found += 1
                    print(f"Page {idx+1} has image: ID={getattr(item, 'id', None)}, Name={getattr(item, 'name', None)}")
        
        print(f"\nTotal items: {total_items}")
        print(f"Item types encountered: {item_types}")
        print(f"Images found: {images_found}")
    else:
        print("\nResult has NO 'items' attribute.")

if __name__ == "__main__":
    asyncio.run(test_parse())
