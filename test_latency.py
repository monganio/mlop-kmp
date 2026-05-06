import torch
import onnxruntime as ort
import time
import numpy as np
from transformers import AutoModelForImageClassification

# เตรียมข้อมูลจำลอง (Dummy Input)
input_data = torch.randn(1, 3, 224, 224)
input_numpy = input_data.numpy()

# 1. ทดสอบความเร็ว Original (PyTorch) [cite: 6]
model_pt = AutoModelForImageClassification.from_pretrained("google/efficientnet-b0")
start = time.time()
with torch.no_grad():
    for _ in range(10): # รัน 10 รอบเพื่อหาค่าเฉลี่ย
        _ = model_pt(input_data)
latency_pt = (time.time() - start) / 10
print(f"Original Latency: {latency_pt:.4f} s")

# 2. ทดสอบความเร็ว ONNX
session_onnx = ort.InferenceSession("models/model.onnx")
start = time.time()
for _ in range(10):
    _ = session_onnx.run(None, {"pixel_values": input_numpy})
latency_onnx = (time.time() - start) / 10
print(f"ONNX Latency: {latency_onnx:.4f} s")

# 3. ทดสอบความเร็ว Quantized 
session_quant = ort.InferenceSession("models/model_quant.onnx")
start = time.time()
for _ in range(10):
    _ = session_quant.run(None, {"pixel_values": input_numpy})
latency_quant = (time.time() - start) / 10
print(f"Quantized Latency: {latency_quant:.4f} s")