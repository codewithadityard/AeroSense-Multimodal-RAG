import os
import glob
import json
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader



image_loader=ImageLoader()

print("Initializing CLIP Embedding Function...")
clip_ef = OpenCLIPEmbeddingFunction()

print("Connecting to new Multimodal ChromaDB...\n")
client = chromadb.PersistentClient(path="chroma_multimodal_db")

# Create a Joint Space collection
collection = client.get_or_create_collection(
    name="aerosense_joint_space", 
    embedding_function=clip_ef,
    data_loader=image_loader
)

# --- 1. Embed the Raw Images ---
image_dir = os.path.join("output_elements", "images")
image_files = glob.glob(f"{image_dir}/*.jpg")

if not image_files:
    print(f"No images found in {image_dir}.")
else:
    print(f"Found {len(image_files)} images. Slicing pixels into vectors...")
    
    img_ids = [f"img_{i}" for i in range(len(image_files))]
    img_metadatas = [{"type": "image", "filepath": path} for path in image_files]
    
    collection.add(
        ids=img_ids,
        uris=image_files, 
        metadatas=img_metadatas
    )
    print("All images successfully embedded into the Joint Space!")

# --- 2. Embed the Actual Texts from Phase 1 ---
text_json_path = os.path.join("output_elements", "texts.json")

if not os.path.exists(text_json_path):
    print(f"\nCould not find {text_json_path}. Did Phase 1 save it here?")
else:
    print(f"\nLoading real text chunks from {text_json_path}...")
    with open(text_json_path, "r", encoding="utf-8") as f:
        text_data = json.load(f)
    
    # Unstructured.io usually saves elements as dictionaries. 
    # We will extract the actual string from the "text" key if it's a dict.
    real_texts = []
    for item in text_data:
        if isinstance(item, dict) and "text" in item:
            real_texts.append(item["text"])
        elif isinstance(item, str):
            real_texts.append(item)

    if real_texts:
        print(f"Found {len(real_texts)} text chunks. Embedding into Joint Space (Note: CLIP truncates to 77 tokens)...")
        
        # We process in batches to avoid overwhelming the CPU RAM
        batch_size = 100
        for i in range(0, len(real_texts), batch_size):
            batch_texts = real_texts[i:i + batch_size]
            batch_ids = [f"text_{j}" for j in range(i, i + len(batch_texts))]
            batch_metadatas = [{"type": "text"} for _ in batch_texts]
            
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            print(f"  -> Embedded batch {i} to {i + len(batch_texts)}")
            
        print("All real text chunks successfully embedded!")

print("\n" + "="*50)
print("Phase 2 Database Build Complete! The Joint Space is ready.")
print("="*50 + "\n")