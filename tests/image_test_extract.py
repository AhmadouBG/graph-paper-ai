import fitz  # PyMuPDF
import base64

def extract_images_with_captions(pdf_path, top_margin=70, bottom_margin=70, caption_distance=50):
    """
    Extract images with their captions from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        top_margin: Top margin to ignore
        bottom_margin: Bottom margin to ignore
        caption_distance: Distance to look for captions below images
        
    Returns:
        List of dictionaries containing image information
    """
    doc = fitz.open(pdf_path)
    extracted_images = []
    
    for num_page in range(len(doc)):
        page = doc[num_page]
        images = page.get_images(full=True)
        hauteur_page = page.rect.height
        
        limite_haute = top_margin
        limite_basse = hauteur_page - bottom_margin
        
        for index_img, img in enumerate(images):
            xref = img[0]
            
            rects_image = page.get_image_rects(xref)
            if not rects_image:
                continue
                
            rect = rects_image[0]
            
            # 1. Sécurité structure / Filtrage marges
            y0 = rect.y0 if hasattr(rect, "y0") else rect[1]
            y1 = rect.y1 if hasattr(rect, "y1") else rect[3]
            x0 = rect.x0 if hasattr(rect, "x0") else rect[0]
            x1 = rect.x1 if hasattr(rect, "x1") else rect[2]
            
            if y0 < limite_haute or y1 > limite_basse:
                continue
            
            # 2. RECHERCHE DE LA LÉGENDE (Zone sous l'image)
            # On définit un rectangle juste en dessous de l'image pour chercher du texte
            zone_legende = fitz.Rect(x0 - 20, y1, x1 + 20, y1 + caption_distance)
            texte_legende = page.get_text("text", clip=zone_legende).strip()
            
            # Optionnel : Si rien en dessous, chercher juste au-dessus (ex: certains tableaux/figures)
            if not texte_legende:
                zone_legende_haut = fitz.Rect(x0 - 20, y0 - caption_distance, x1 + 20, y0)
                texte_legende = page.get_text("text", clip=zone_legende_haut).strip()
            
            # Nettoyer les sauts de ligne inutiles dans la légende détectée
            texte_legende = " ".join(texte_legende.split())
            
            # 3. Extraction standard de l'image
            base_image = doc.extract_image(xref)
            octets_image = base_image["image"]
            extension_image = base_image["ext"]
            chaine_base64 = base64.b64encode(octets_image).decode('utf-8')
            
            extracted_images.append({
                "numero_page": num_page + 1,
                "index_image": index_img + 1,
                "extension": extension_image,
                "legende_detectee": texte_legende if texte_legende else "Aucune légende trouvée",
                "base64": chaine_base64
            })
            
    return extracted_images
