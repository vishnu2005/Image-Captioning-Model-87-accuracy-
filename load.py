from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.dirname(os.path.abspath(__file__))

# Load model and processors
model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
processor = ViTImageProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, max_length=32, num_beams=4)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
