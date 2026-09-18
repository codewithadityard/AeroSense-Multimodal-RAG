import fitz  # PyMuPDF
from PIL import Image

def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[Image.Image]:
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images