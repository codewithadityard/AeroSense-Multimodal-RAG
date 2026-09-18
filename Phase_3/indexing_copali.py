import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from PIL import Image
from extraction_copali import pdf_to_images

# 1. Load Model and Processor
model_name = "vidore/colpali-v1.2"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device
)
processor = ColPaliProcessor.from_pretrained(model_name)

# 2. Process Page Images
images = pdf_to_images("financial_report_tables.pdf")

# Batch process page images
dataloader = torch.utils.data.DataLoader(images, batch_size=2)
page_embeddings = []

with torch.no_grad():
    for batch in dataloader:
        inputs = processor.process_images(batch).to(device)
        embeddings = model(**inputs)  # Shape: (batch_size, num_patches, embed_dim)
        page_embeddings.extend(embeddings.cpu())