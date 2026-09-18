import json
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. Setup paths
output_dir = "output_elements"
db_dir = "chroma_db"

texts_path = os.path.join(output_dir, "texts.json")
tables_path = os.path.join(output_dir, "tables.json")
captions_path = os.path.join(output_dir, "image_captions.json")

# 2. Initialize the Local Embedding Model
# This is a tiny, fast model that works great on CPUs
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Load the data from JSON
with open(texts_path, "r") as f:
    raw_texts = json.load(f)

with open(tables_path, "r") as f:
    raw_tables = json.load(f)

# Note: Only load captions if you have successfully run caption_images.py
if os.path.exists(captions_path):
    with open(captions_path, "r") as f:
        raw_captions = json.load(f)
else:
    raw_captions = []

# 4. Convert everything into LangChain Documents
documents = []

# Process Text Chunks
print("Processing texts...")
for text in raw_texts:
    # We skip very short chunks (like rogue page numbers)
    if len(text.strip()) > 20: 
        doc = Document(
            page_content=text,
            metadata={"source": "faa_handbook", "type": "text"}
        )
        documents.append(doc)

# Process Tables
print("Processing tables...")
for table in raw_tables:
    # We embed the HTML version because it preserves structure!
    if "html" in table and table["html"]:
        doc = Document(
            page_content=table["html"],
            metadata={"source": "faa_handbook", "type": "table"}
        )
        documents.append(doc)

# Process Image Captions
print("Processing images...")
for item in raw_captions:
    doc = Document(
        # The content we embed is the VLM's description of the image
        page_content=item["caption"],
        # The metadata holds the path to the physical image file!
        metadata={
            "source": "faa_handbook", 
            "type": "image",
            "filepath": item["filepath"]
        }
    )
    documents.append(doc)

print(f"Total documents prepared: {len(documents)}")

# 5. Build and Save the Vector Database
print("Embedding documents and building ChromaDB. This may take a minute...")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=db_dir
)

print(f"Success! Vector database saved to {db_dir}/")