import os
import re
from llama_cloud import LlamaCloud

# Initialize LlamaCloud Client
client = LlamaCloud(api_key=os.environ.get("LLAMACLOUD_API_KEY"))

file = client.files.create(file="file/plosone.pdf", purpose="parse")

# Execute the parse job with image expansion
result = client.parsing.parse(
    file_id=file.id,
    tier="agentic",
    version="latest",
    output_options={
        "markdown": {"tables": {"output_tables_as_markdown": True}},
        "images_to_save": ["screenshot"],  # Captures the visual regions
    },
    expand=["markdown", "images_content_metadata"],  # Pulls text structure & image tokens
)

markdown_text = result.markdown
images_metadata = result.images_content_metadata or []

print("--- EXTRACTING IMAGES AND CAPTIONS FROM MARKDOWN ---")

# Step 1: Find all markdown image tags like ![screenshot_p0_idx0.png](...)
# This regex extracts the image reference tag name
image_tags = re.findall(r"\!\[(.*?)\]\(.*?\)", markdown_text)

# Step 2: Split the document by the image markers to capture the surrounding content/captions
segments = re.split(r"\!\[.*?\]\(.*?\)", markdown_text)

# Step 3: Match the extracted images with their adjacent text blocks (captions)
for index, tag_name in enumerate(image_tags):
    # The text following the image tag typically contains the figure caption
    following_text = segments[index + 1].strip() if (index + 1) < len(segments) else ""
    
    # Grab the first 2-3 lines of text right below the image as the caption
    caption_lines = [line.strip() for line in following_text.split("\n") if line.strip()]
    extracted_caption = " ".join(caption_lines[:2]) if caption_lines else "No caption detected."
    
    print(f"\n🖼️ Found Image Tag: {tag_name}")
    print(f"📝 Associated Caption text: \"{extracted_caption}\"")
    
    # Step 4: Map back to the temporary download URL from your expanded metadata
    matching_meta = next((img for img in images_metadata if img.name in tag_name or tag_name in img.name), None)
    if matching_meta:
        print(f"🔗 Download URL: {matching_meta.download_url}")
