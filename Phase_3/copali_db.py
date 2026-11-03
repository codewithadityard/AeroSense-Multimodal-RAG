import torch
import fitz  # PyMuPDF for PDF to Image conversion
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
from qdrant_client import QdrantClient, models
import base64
from io import BytesIO
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage




# ==========================================
# 1. INITIALIZATION: MODELS & DATABASE
# ==========================================

print("Loading ColPali Model and Processor...")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model_name = "vidore/colpali-v1.3"

# Load the model in bfloat16 to save memory
model = ColPali.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16, 
    device_map=device
).eval()
processor = ColPaliProcessor.from_pretrained(model_name)

print("Initializing Local Qdrant Database...")
# Initialize a local disk-based Qdrant client relative to project root
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
qdrant_db_path = os.path.join(project_dir, "qdrant_colpali_db")
client = QdrantClient(path=qdrant_db_path)
collection_name = "aerosense_visual_pages"

# Create the collection if it doesn't exist
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "colpali_vector": models.VectorParams(
                size=128,  # ColPali embedding dimension
                distance=models.Distance.COSINE, #
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM # Native MaxSim
                ),
                # Disable HNSW indexing to save RAM when doing raw multivector scoring
                hnsw_config=models.HnswConfigDiff(m=0) 
            )
        }
    )
    print(f"Created new collection: {collection_name}")

# ==========================================
# 2. INDEXING: PDF -> IMAGES -> QDRANT
# ==========================================

def index_pdf(pdf_path: str):
    print(f"\nProcessing PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        print(f"Embedding page {page_num + 1}/{len(doc)}...")
        
        # 1. Convert PDF page to PIL Image
        page = doc[page_num]
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # 2. Process image through ColPali
        with torch.no_grad():
            processed_images = processor.process_images([img]).to(device)
            # Returns shape: (batch_size, num_patches, embed_dim)
            image_embeddings = model(**processed_images) 
        
        # Extract the multi-vector matrix for this specific page
        page_matrix = image_embeddings[0].cpu().float().tolist()
        
        # 3. Upsert into Qdrant
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=page_num,  # Using page number as ID
                    vector={"colpali_vector": page_matrix},
                    payload={
                        "source": pdf_path,
                        "page_number": page_num + 1
                    }
                )
            ]
        )
    print("Indexing complete!")

# ==========================================
# 3. RETRIEVAL: QUERY -> MAX_SIM -> PAGE
# ==========================================

def diagnose_anomaly(sensor_readings, engine_unit_nr, current_cycle, query=None):
    """
    Triggered when the watchdog detects an anomaly. 
    Retrieves the manual and asks the Vision LLM for a diagnosis.
    """
    print(f"\n[ALERT] Anomaly triggered on Engine {engine_unit_nr} at cycle {current_cycle}")
    
    # 1. Format live sensor data
    sensor_text = ", ".join([f"s_{i+1}: {val:.2f}" for i, val in enumerate(sensor_readings[3:])])

    # 2. Define the retrieval query and get the image
    if query is None:
        query = "Troubleshooting high sensor readings indicating impending engine failure, HPC, Fan Degradation"
    print(f"Retrieving manual from ColPali for: '{query}'...")
    img_b64 = search_visual_index(query, top_k=1)
    
    if not img_b64:
        print("[WARNING] Could not retrieve visual page from ColPali index. Visual diagnosis cannot proceed.")
        return

    # 3. Setup LangChain and prompt
    llm = ChatGroq(
        model_name="qwen/qwen3.8-27b",
        temperature=0.0,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )
    
    sys_instruction = f"""
    You are AeroSense, an expert aerospace predictive maintenance AI.
    
    A telemetry watchdog has detected an impending failure on an engine.
    Current Sensor Readings: {sensor_text}
    
    Attached is the relevant page from the engine maintenance manual.
    
    Based on the provided sensor readings and the visual information in the manual page:
    1. Identify the likely fault mode (e.g., HPC Degradation, Fan Degradation, or other).
    2. Explain your reasoning linking the sensor data to the manual.
    3. Recommend immediate maintenance actions.
    """
    
    # Inject ONLY the instruction (with sensor data) and the retrieved image
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": sys_instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}} 
            ]
        )
    ]
    
    print("\nAnalyzing page image with Groq Vision (Llama 3.2 11B Vision)...")
    response = llm.invoke(messages)
    
    print("\n--- AeroSense Extraction ---")
    print(response.content)
    print("----------------------------")



    
def search_visual_index(query: str, top_k: int = 1):
    """
    Searches Qdrant using ColPali and returns the base64 image of the top result.
    """
    print(f"\nSearching Qdrant for: '{query}'")
    
    if not client.collection_exists(collection_name):
        print(f"[WARNING] Qdrant collection '{collection_name}' does not exist yet. Please run index_pdf() first!")
        return None
        
    with torch.no_grad():
        processed_queries = processor.process_queries([query]).to(device)
        query_embeddings = model(**processed_queries)[0].cpu().float().tolist()
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_embeddings,
        using="colpali_vector",
        limit=top_k,
        with_payload=True
    )
    
    if not results.points:
        print(f"[WARNING] No visual pages found in index matching the query.")
        return None

    # Extract the top result
    top_result = results.points[0]
    page_num = top_result.payload['page_number']
    pdf_source = top_result.payload['source']
    
    if not os.path.exists(pdf_source):
        # Fallback to local project data folder
        fallback_pdf = os.path.join(project_dir, "data", os.path.basename(pdf_source))
        if os.path.exists(fallback_pdf):
            pdf_source = fallback_pdf
        else:
            print(f"[ERROR] Source PDF not found at '{pdf_source}' or '{fallback_pdf}'.")
            return None
    
    print(f"Top Match - Page {page_num} from {pdf_source} (Score: {top_result.score:.4f})")
    
    # Open the PDF and convert the specific page to an image
    doc = fitz.open(pdf_source)
    page = doc[page_num - 1] 
    pix = page.get_pixmap(dpi=150)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Convert PIL Image to Base64 for Groq Vision
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_b64

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    pdf_file = os.path.join(project_dir, "data", "induction_exhaust_systems.pdf")
    
    # Uncomment the next line to index a PDF for the first time
    #if os.path.exists(pdf_file):
    #    index_pdf(pdf_file)
    #else:
    #    print(f"[NOTE] Place your manual at '{pdf_file}' to index.")
    
    # Test a query
    test_query = "What is the torque specification for bolt MS21250-5?"
    search_visual_index(test_query)



    client.close()