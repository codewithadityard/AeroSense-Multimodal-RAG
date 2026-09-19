import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from PIL import Image


print("Loading CLIP Model...")
clip_ef = OpenCLIPEmbeddingFunction()



print("Connecting to the Joint Space Database...")
client = chromadb.PersistentClient(path="chroma_multimodal_db")
img_loader=ImageLoader()

# Fetch our existing collection
collection = client.get_collection(
    name="aerosense_joint_space", 
    embedding_function=clip_ef,
    data_loader=img_loader
)

def find_diagram(query):
    print(f"\nSearching for pixels matching: '{query}'...")
    
    # We query using text, but filter the database to ONLY return images
    results = collection.query(
        query_texts=[query],
        n_results=2, 
        where={"type": "image"},
        include=["uris","distances"] # This tells Chroma to ignore the text chunks
    )
    
    # Extract the file paths (uris) and the mathematical distances
    uris = results.get('uris', [[]])[0]
    distances = results.get('distances', [[]])[0]
    
    if not uris:
        print("No images found in the database.")
        return

    print("\nTop Image Matches:")
    for i, (uri, dist) in enumerate(zip(uris, distances)):
        # Lower distance means a closer match
        print(f"  {i+1}. {uri} (Distance: {dist:.4f})")
        
    # Automatically pop open the best match using Pillow
    best_image_path = uris[0]
    try:
        print(f" Opening {best_image_path} to verify...")
        img = Image.open(best_image_path)
        img.show()
    except Exception as e:
        print(f"Could not open image automatically. You can view it here: {best_image_path}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("AeroSense Phase 2: Multimodal Search Engine")
    print("="*50)
    
    while True:
        user_query = input("\nWhat diagram do you want to find? (or type 'quit'): ")
        if user_query.lower() == 'quit':
            break
        find_diagram(user_query)