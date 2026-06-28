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
    enriched: dict[int, list[dict]] = {}

    for page_num, images in page_image_map.items():
        enriched[page_num] = []

        # Collect captions from this page and next N pages
        candidate_captions = []
        for offset in range(lookahead_pages + 1):
            check_page = page_num + offset
            if check_page in caption_map:
                candidate_captions.extend(caption_map[check_page])

        # ✨ KEY FIX: separate fig captions from table captions
        # Images are figures/charts — prefer Fig/Figure captions over Table captions
        fig_captions = [
            c for c in candidate_captions
            if re.match(r'fig(?:ure)?', c["normalized"], re.IGNORECASE)
        ]
        table_captions = [
            c for c in candidate_captions
            if re.match(r'table', c["normalized"], re.IGNORECASE)
        ]

        for i, img in enumerate(images):
            # Try to assign a figure caption first
            if i < len(fig_captions):
                cap = fig_captions[i]
            elif i < len(table_captions):
                # Only use table caption if no figure caption available
                cap = table_captions[i]
            else:
                cap = None

            if cap:
                enriched[page_num].append({
                    "base64": img["base64"],
                    "extension": img["extension"],
                    "caption": cap["full_caption"],
                    "label": cap["label"],
                    "normalized_label": cap["normalized"],
                })
            else:
                enriched[page_num].append({
                    "base64": img["base64"],
                    "extension": img["extension"],
                    "caption": "Aucune légende trouvée",
                    "label": "",
                    "normalized_label": "",
                })

    return enriched