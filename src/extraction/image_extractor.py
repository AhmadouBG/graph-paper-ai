from __future__ import annotations

import base64

import fitz  # PyMuPDF


def extract_images_with_captions(
    pdf_path: str,
    top_margin: int = 70,
    bottom_margin: int = 70,
    caption_distance: int = 50,
) -> list[dict]:
    """
    Extract content-area images with their captions from a PDF using PyMuPDF.

    Args:
        pdf_path:         Path to the PDF file.
        top_margin:       Pixels from the top to exclude (removes headers).
        bottom_margin:    Pixels from the bottom to exclude (removes footers).
        caption_distance: Pixel distance below/above the image to search for a caption.

    Returns:
        List of dicts with keys:
            - "numero_page"     (int)   1-based page number
            - "index_image"     (int)   1-based image index on that page
            - "extension"       (str)   e.g. "png", "jpeg"
            - "legende_detectee" (str)  detected caption or "Aucune légende trouvée"
            - "base64"          (str)   Base64-encoded image bytes
    """
    doc = fitz.open(pdf_path)
    extracted_images: list[dict] = []

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

            # Robust coordinate extraction (Rect object or plain tuple)
            y0 = rect.y0 if hasattr(rect, "y0") else rect[1]
            y1 = rect.y1 if hasattr(rect, "y1") else rect[3]
            x0 = rect.x0 if hasattr(rect, "x0") else rect[0]
            x1 = rect.x1 if hasattr(rect, "x1") else rect[2]

            # Skip headers and footers
            if y0 < upper_limit or y1 > lower_limit:
                continue

            # Search for caption below the image
            caption_zone = fitz.Rect(x0 - 20, y1, x1 + 20, y1 + caption_distance)
            caption_text = page.get_text("text", clip=caption_zone).strip()

            # Fallback: search above the image (some tables have captions on top)
            if not caption_text:
                caption_zone_above = fitz.Rect(x0 - 20, y0 - caption_distance, x1 + 20, y0)
                caption_text = page.get_text("text", clip=caption_zone_above).strip()

            # Collapse whitespace
            caption_text = " ".join(caption_text.split())

            # Extract raw image bytes and encode to Base64
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            base64_str = base64.b64encode(image_bytes).decode("utf-8")

            extracted_images.append({
                "numero_page": num_page + 1,
                "index_image": index_img + 1,
                "extension": image_ext,
                "legende_detectee": caption_text if caption_text else "Aucune légende trouvée",
                "base64": base64_str,
            })

    return extracted_images
