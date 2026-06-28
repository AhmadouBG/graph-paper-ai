import re

def build_caption_map_from_markdown(markdown_text: str) -> dict[int, list[dict]]:
    """
    Extract figure/table captions from LlamaParse markdown by page.
    Returns {page_num: [{"label": "Fig 4", "caption": "full caption text"}]}
    """
    # Split by page markers
    pages = re.split(r'---\s*Page\s*(\d+)\s*---', markdown_text)
    
    caption_map: dict[int, list[dict]] = {}
    
    # pages alternates: [text_before, page_num, text, page_num, text, ...]
    for i in range(1, len(pages), 2):
        page_num = int(pages[i])
        page_text = pages[i + 1] if i + 1 < len(pages) else ""
        
        # Match patterns like:
        # "Fig 1. Some description."
        # "Figure 2: Some description."  
        # "Table 3. Some description."
        captions = re.findall(
            r'((?:Fig(?:ure)?|Table)\.?\s*\d+[a-z]?[.:]\s*[^\n]{5,120})',
            page_text,
            re.IGNORECASE
        )
        
        if captions:
            caption_map[page_num] = []
            for cap in captions:
                cap = cap.strip()
                # Extract the label: "Fig 4", "Table 3", etc.
                label_match = re.match(
                    r'((?:Fig(?:ure)?|Table)\.?\s*\d+[a-z]?)', cap, re.IGNORECASE
                )
                label = label_match.group(1).strip() if label_match else cap[:10]
                caption_map[page_num].append({
                    "label": label,
                    "full_caption": cap,
                    "normalized": re.sub(r'[.\s]', '', label.lower()),  # "fig4", "table3"
                })
    
    return caption_map


def match_images_to_captions(
    page_image_map: dict[int, list[dict]],
    caption_map: dict[int, list[dict]],
    lookahead_pages: int = 1
) -> dict[int, list[dict]]:
    """
    For each image, find its caption by checking:
    1. Same page as image
    2. Page immediately after (caption sometimes follows figure)
    
    Returns enriched page_image_map with 'caption' filled from LlamaParse.
    """
    enriched: dict[int, list[dict]] = {}
    
    for page_num, images in page_image_map.items():
        enriched[page_num] = []
        
        # Collect captions from this page and next N pages
        candidate_captions = []
        for offset in range(lookahead_pages + 1):
            check_page = page_num + offset
            if check_page in caption_map:
                candidate_captions.extend(caption_map[check_page])
        
        for i, img in enumerate(images):
            # Try to assign the i-th caption to the i-th image on this page
            if i < len(candidate_captions):
                enriched[page_num].append({
                    "base64": img["base64"],
                    "extension": img["extension"],
                    "caption": candidate_captions[i]["full_caption"],
                    "label": candidate_captions[i]["label"],
                    "normalized_label": candidate_captions[i]["normalized"],
                })
            else:
                # No caption found — keep image but mark it
                enriched[page_num].append({
                    "base64": img["base64"],
                    "extension": img["extension"],
                    "caption": "Aucune légende trouvée",
                    "label": "",
                    "normalized_label": "",
                })
    
    return enriched