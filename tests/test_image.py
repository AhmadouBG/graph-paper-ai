import base64
from io import BytesIO
from PIL import Image
from main import run_pipeline_orchestration
import asyncio
from pathlib import Path
import os
from dotenv import load_dotenv

# ── Configuration ──────────────────────────────
load_dotenv()
API_KEY = os.getenv("LLAMACLOUD_API_KEY")
PDF_PATH = Path("file/plosone.pdf")

def verify_and_show_extracted_image(tree: list[dict]):
    """
    Parcourt l'arbre à la recherche de la première image Base64 disponible
    et l'affiche à l'écran pour vérification visuelle.
    """
    def find_first_image(nodes):
        for n in nodes:
            if n.get("base64_images"):
                return n["base64_images"][0], n["title"]
            if n.get("nodes"):
                img, title = find_first_image(n["nodes"])
                if img:
                    return img, title
        return None, None

    print("🔎 Searching for an extracted image in the tree...")
    b64_string, section_title = find_first_image(tree)

    if not b64_string:
        print("❌ No images found in the tree. LlamaParse did not capture any visuals for this document.")
        return

    print(f"✅ Image found in section: '{section_title}'!")
    print(f"📦 Base64 String length: {len(b64_string)} characters.")

    try:
        # 1. Décoder la chaîne Base64 en octets binaires
        image_bytes = base64.b64decode(b64_string)
        
        # 2. Charger les octets en tant qu'image PIL en mémoire RAM
        image = Image.open(BytesIO(image_bytes))
        
        # 3. Afficher l'image à l'écran via le lecteur natif de votre OS
        print("🖼️ Opening image viewer... Check your desktop windows.")
        image.show()
        
    except Exception as e:
        print(f"❌ Failed to decode or display the image: {e}")

# Exemple d'intégration dans votre flux après la ligne :
tree = asyncio.run(run_pipeline_orchestration(PDF_PATH, API_KEY))
verify_and_show_extracted_image(tree)
