import os
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader # Added this!
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from PIL import Image

print("Initializing AeroSense Dual-Retrieval Engine...\n")

# --- Loading Text Embeddings ---
text_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Fixed typo: persist_directory (not presistent)
text_db = Chroma(persist_directory="chroma_db", embedding_function=text_embeddings)

# --- Loading Image Embeddings ---
img_embeddings = OpenCLIPEmbeddingFunction()
# Fixed logic: Connect to client first, THEN get the collection
client = chromadb.PersistentClient(path="chroma_multimodal_db")
img_collection = client.get_collection(
    name="aerosense_joint_space", 
    embedding_function=img_embeddings,
    data_loader=ImageLoader() # Remember this from the ValueError!
)

# --- Loading model ---
llm = Ollama(model="llama3.1")

def search(query, top_k=3):
    # 1. Fix LangChain search syntax
    raw_docs = text_db.similarity_search(query, k=top_k)
    # Extract the actual text from the Document objects
    txt_context = "\n".join([doc.page_content for doc in raw_docs])

    # 2. Fix ChromaDB search syntax (.query instead of .add)
    img_result = img_collection.query(
        query_texts=[query],
        n_results=1, # We only need the top 1 image match
        where={"type": "image"},
        include=["uris", "distances"] # We must request the URIs!
    )
    
    # 3. Fix Prompt Logic - Llama only reads the text to answer the query
    prompt = f"""You are an expert aerospace engineer.
    Answer the user's query using ONLY the provided context.
    
    "CONTEXT": {txt_context}

    "QUERY": {query}
    """

    print("Generating answer...")
    result = llm.invoke(prompt)
    print("\n--- AEROSENSE ANSWER ---")
    print(result)

    # 4. Safely extract and show the image that CLIP found
    if img_result.get("uris") and img_result["uris"][0]:
        best_img_path = img_result["uris"][0][0]
        
        print(f"\n--- ASSOCIATED DIAGRAM ---")
        print(best_img_path)
        
        if os.path.exists(best_img_path):
            try:
                img = Image.open(best_img_path)
                img.show()
            except Exception as e:
                print("Could not open image.")

# To actually run your function:
if __name__ == "__main__":
    test_query = "Show me a diagram of the exhaust collector ring."
    search(test_query)