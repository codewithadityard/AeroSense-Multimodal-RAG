from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# 1. Load the Vector Database
print("Loading Database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# 2. Load Llama 3.1 via Ollama (Runs efficiently on CPU)
print("Connecting to Local Ollama (Llama 3.1)...")
llm = Ollama(model="llama3.1")

def ask_aerosense(query):
    print(f"\nThinking about: '{query}'...")
    
    # 3. Retrieve Context from ChromaDB
    results = vectorstore.similarity_search(query, k=3)
    
    context_texts = []
    image_paths = []
    
    # Separate the text/captions from the image filepaths
    for doc in results:
        context_texts.append(doc.page_content)
        if doc.metadata.get('type') == 'image':
            image_paths.append(doc.metadata.get('filepath'))
            
    context_string = "\n---\n".join(context_texts)
    
    # 4. Format the Prompt
    prompt = f"""You are AeroSense, an aviation engineering assistant. 
Answer the user's question using ONLY the provided context below. Be concise and professional.
If the context does not contain the answer, say "I cannot find the answer in the manuals."

Context:
{context_string}

Question:
{query}
"""
    
    # 5. Generate the Answer using Llama 3.1
    answer = llm.invoke(prompt)
    
    # 6. Display Result 
    print("\n" + "="*60)
    print("AeroSense Answer (Llama 3.1):")
    print(answer.strip())
    
    if image_paths:
        print("\nAssociated Diagrams:")
        for path in set(image_paths): 
            print(f" -> {path} (Ready to render in UI)")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Interactive loop
    while True:
        user_query = input("Ask AeroSense (or type 'quit'): ")
        if user_query.lower() == 'quit':
            break
        ask_aerosense(user_query)