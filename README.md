# Image-Captioning-Model-87-accuracy-
# 🧠 Image Captioning Model (87% Accuracy)

This repository contains a fully trained image captioning model based on [`nlpconnect/vit-gpt2-image-captioning`](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning). The model was **fine-tuned on a 40,000-image subset of the MS COCO dataset** and achieved an impressive **87% captioning accuracy** on validation data.

## Project Highlights

- ✅ Model: Vision Transformer (ViT) + GPT2
- ✅ Dataset: MS COCO (40k subset)
- ✅ Accuracy: 87%
- ✅ Framework: PyTorch
- ✅ Use Case: Generate meaningful and coherent captions for input images
- ✅ Supports further enhancement via multilingual TTS, hazardous object detection (optional)

---

## Model Architecture Overview

### Vision Transformer (ViT)
- Used for encoding image features.
- Breaks an image into patches and processes them like tokens.
- Trained to learn **visual context representations**.

### GPT-2 Decoder
- Pretrained autoregressive language model.
- Generates natural language captions **word by word** based on encoded image context.
- Learns grammar, structure, and contextual correctness.

### ⛓️ Combined Encoder-Decoder Pipeline
- The `nlpconnect/vit-gpt2-image-captioning` model leverages a **VisionEncoderDecoderModel**.
- Image → ViT Encoder → Contextual Embeddings → GPT2 Decoder → Caption

---

## 📊 Training Details

| Detail                   | Description                          |
|-------------------------|--------------------------------------|
| Dataset                 | MS COCO (subset: 40,000 images)      |
| Epochs                  | 4 (configurable)                     |
| Evaluation Metric       | BLEU / Accuracy                      |
| Final Accuracy Achieved | **87%**                              |
| Loss Function           | CrossEntropy                         |
| Optimizer               | AdamW                                |
| Framework               | PyTorch                              |

---

## 🧪 How to Use

Clone the repo and run inference:

```bash
git clone https://github.com/vishnu2005/Image-Captioning-Model-87-accuracy-.git
cd Image-Captioning-Model-87-accuracy-
python run_inference.py --image path/to/your/image.jpg
```

## 🧱 Download Trained Model

Due to GitHub's 100MB file size limit, the trained model file `pytorch_model.bin` (912MB) is hosted externally.

👉 [Click here to download the model](https://drive.google.com/uc?export=download&id=1plxo7TuJjjTnylXzuyc_h9Ds7pOQyrIx)

After downloading, place the file inside your project directory like this:
final_trained_model/
├── pytorch_model.bin 👈 put it here
├── run_inference.py
├── caption_generator.py
...

