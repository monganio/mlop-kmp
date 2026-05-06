import os
import io
import asyncio
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field
import onnxruntime as ort
import numpy as np
from PIL import Image, UnidentifiedImageError

app = FastAPI(title="MLOps Image Classification API (EfficientNet-B0)")

# Setup Path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model_quant.onnx"

# โหลด Model เตรียมไว้
try:
    session = ort.InferenceSession(str(MODEL_PATH))
    input_name = session.get_inputs()[0].name
except Exception as e:
    print(f"Failed to load model: {e}")

# ProcessPoolExecutor สำหรับงาน CPU-bound
executor = ProcessPoolExecutor(max_workers=os.cpu_count() or 2)

class PredictionResponse(BaseModel):
    predicted_class: int = Field(..., description="The predicted class ID")
    confidence: float = Field(..., description="Confidence score of the prediction")

def run_inference(image_bytes: bytes) -> dict:
    try:
        # เช็คไฟล์รูปเสีย
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise ValueError("Corrupted image file")

        # Preprocessing สำหรับ EfficientNet-B0
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        img_data = np.array(image).astype(np.float32) / 255.0
        
        # Normalization ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std
        
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)

        # Inference
        outputs = session.run(None, {input_name: img_data})
        logits = outputs[0]
        
        predicted_class = int(np.argmax(logits))
        exp_logits = np.exp(logits - np.max(logits))
        confidence = float(np.max(exp_logits / np.sum(exp_logits)))

        return {"predicted_class": predicted_class, "confidence": confidence}
    except Exception as e:
        raise ValueError(str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Invalid file type. Only JPEG and PNG are allowed."
        )

    image_bytes = await file.read()
    
    if len(image_bytes) > 5 * 1024 * 1024: # 5MB Limit
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="File size exceeds the 5MB limit."
        )

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, run_inference, image_bytes)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error.")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "API is running"}