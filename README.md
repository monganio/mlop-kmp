git add README.md
git commit -m "Fix Hugging Face configuration in README"
git push origin main

# High-Throughput Image Classification API (EfficientNet-B0)

This project provides an optimized Image Classification API using FastAPI and a dynamically quantized EfficientNet-B0 ONNX model.

## Setup Locally
1. Run `python prepare_model.py` to generate the quantized model.
2. Start server: `uvicorn main:app --reload`

## 🚀 How to Test the API (Cloud)

Use the following cURL command to test the endpoint on Hugging Face Spaces:
```bash
curl -X POST "https://monganio-mlop.hf.space/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg"