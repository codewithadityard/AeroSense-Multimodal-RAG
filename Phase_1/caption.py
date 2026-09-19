import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

# 1. Setup paths
images_dir = "output_elements/images"
output_file = "output_elements/image_captions.json"
model_id = "microsoft/Florence-2-base"

# Determine device
device = "cpu" if not torch.cuda.is_available() else "cuda"

# 2. Create a function that intercepts the dependency check
def bypass_flash_attn_check(filename):
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

# 3. Load the Florence-2 Model (Single, patched load!)
print(f"Loading Florence-2-base model on {device} (bypassing flash_attn)...")
with patch("transformers.dynamic_module_utils.get_imports", bypass_flash_attn_check):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True
    ).to(device).eval()

processor = AutoProcessor.from_pretrained(
    model_id, 
    trust_remote_code=True
)

print(f"Model loaded successfully on {device}!")

# 4. Define the captioning function
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        # <MORE_DETAILED_CAPTION> forces it to describe components
        prompt = "<MORE_DETAILED_CAPTION>"
        
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        
        # Generate the text description
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            do_sample=False,
            num_beams=3
        )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Parse out the actual caption from the model output format
        parsed_answer = processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        return parsed_answer[prompt]
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# 5. Process all images
image_captions = []
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

print(f"Found {len(image_files)} images to process. Starting captioning...")

for img_file in tqdm(image_files):
    img_path = os.path.join(images_dir, img_file)
    caption = generate_caption(img_path)
    
    if caption:
        image_captions.append({
            "image_filename": img_file,
            "filepath": img_path,
            "caption": caption
        })

# 6. Save the results
with open(output_file, "w") as f:
    json.dump(image_captions, f, indent=4)

print(f"\nFinished! Saved {len(image_captions)} captions to {output_file}")


