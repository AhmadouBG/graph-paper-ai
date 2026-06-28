from __future__ import annotations

import base64

import fitz  # PyMuPDF
# OLD extract_images_with_captions — you can strip it down to just this:
def extract_images_with_captions(pdf_path: str, top_margin: int = 70, bottom_margin: int = 70) -> list[dict]:
    """Extract images from PDF. Captions are now sourced from LlamaParse markdown."""
    doc = fitz.open(pdf_path)
    extracted_images = []

    for num_page in range(len(doc)):
        page = doc[num_page]
        images = page.get_images(full=True)
        page_height = page.rect.height
        upper_limit = top_margin
        lower_limit = page_height - bottom_margin

        for index_img, img in enumerate(images):
            xref = img[0]
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            rect = rects[0]
            y0 = rect.y0 if hasattr(rect, "y0") else rect[1]
            y1 = rect.y1 if hasattr(rect, "y1") else rect[3]

            # Still skip headers/footers
            if y0 < upper_limit or y1 > lower_limit:
                continue

            base_image = doc.extract_image(xref)
            extracted_images.append({
                "numero_page": num_page + 1,
                "index_image": index_img + 1,
                "extension": base_image["ext"],
                "base64": base64.b64encode(base_image["image"]).decode("utf-8"),
                # No legende_detectee needed anymore
            })

    return extracted_images


# OLD build_page_image_map — simplify to match new structure:
def build_page_image_map(extracted_images: list[dict]) -> dict[int, list[dict]]:
    """Group images by page. Captions will be added later by match_images_to_captions."""
    page_map: dict[int, list[dict]] = {}
    for img in extracted_images:
        page = img["numero_page"]
        if page not in page_map:
            page_map[page] = []
        page_map[page].append({
            "base64": img["base64"],
            "extension": img["extension"],
            "caption": "Aucune légende trouvée",  # placeholder until matched
            "label": "",
            "normalized_label": "",
        })
    return page_map