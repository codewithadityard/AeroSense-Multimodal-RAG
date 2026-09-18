from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Initialize the exact same embedding model we used to build the DB
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Connect to our existing local database
print("Connecting to ChromaDB...\n")
db_dir = "chroma_db"
vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)

def search_aerosense(query, top_k=3):
    print(f"--- SEARCHING FOR: '{query}' ---")
    
    # 3. Perform the Similarity Search
    # This converts the query to a vector and finds the closest matches
    results = vectorstore.similarity_search(query, k=top_k)
    
    if not results:
        print("No relevant information found.")
        return

    # 4. Display the results
    for i, doc in enumerate(results):
        print(f"\nRESULT {i + 1} (Type: {doc.metadata.get('type', 'Unknown')}):")
        
        # If the result is an image, we want to show the filepath!
        if doc.metadata.get('type') == 'image':
            print(f"  IMAGE FOUND! Filepath: {doc.metadata.get('filepath')}")
            print(f"  Caption Match: {doc.page_content}")
        
        # If it's a table, we can render or print the HTML
        elif doc.metadata.get('type') == 'table':
            print("  TABLE FOUND!")
            print(doc.page_content[:200] + "... [Table Truncated for viewing]")
            
        # If it's normal text, print a snippet
        else:
            print("TEXT FOUND!")
            print(doc.page_content[:200] + "...")
            
    print("\n" + "="*50 + "\n")

# --- Let's test it out! ---
if __name__ == "__main__":
    # Test 1: Searching for a concept (Should return text/tables)
    search_aerosense("What causes induction system icing?")
    
    # Test 2: Searching for a visual (Should trigger our Florence-2 captions)
    search_aerosense("Show me a diagram of the exhaust collector ring.")
    
    # Test 3: Interactive prompt for you to try
    while True:
        user_query = input("Ask AeroSense a question (or type 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        search_aerosense(user_query)