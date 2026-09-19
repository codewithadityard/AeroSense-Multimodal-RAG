import os
from unstructured.partition.pdf import partition_pdf

# 1. Setup Input and Output paths
pdf_path = "data/induction_exhaust_systems.pdf"
output_dir = "output_elements"
image_output_dir = os.path.join(output_dir, "images")

os.makedirs(image_output_dir, exist_ok=True)

print(f"Starting extraction on {pdf_path} using 'hi_res' strategy...")

# 2. Run the Partitioning Function
# The 'hi_res' strategy uses a vision model (Detectron2/YOLOX) to identify layouts.
elements = partition_pdf(
    filename=pdf_path,
    strategy="hi_res",
    infer_table_structure=True, # Critical: Forces the model to extract table grid logic
    extract_images_in_pdf=True, # Critical: Crops and saves images/diagrams
    extract_image_block_output_dir=image_output_dir,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000, # Where to save the cropped images
)

print(f"Extraction complete. Found {len(elements)} elements.")

# 3. Categorize and Save the Output
texts = []
tables = []

for element in elements:
    # Elements are categorized by Unstructured's vision model
    if element.category in ["CompositeElement", "NarrativeText", "Title", "ListItem"]:
        texts.append(element.text)
        
    elif  element.category == "Table":
        tables.append({
            "text": element.text,
            "html": getattr(element.metadata, "text_as_html", element.text)
        })

# Save texts and tables for the next step
import json

with open(os.path.join(output_dir, "texts.json"), "w") as f:
    json.dump(texts, f,indent=4)
    
with open(os.path.join(output_dir, "tables.json"), "w") as f:
    json.dump(tables, f,indent=4)

print(f"Saved {len(texts)} text chunks and {len(tables)} tables to {output_dir}/")
print(f"Images are saved in {image_output_dir}/")

print(texts)
print(tables)