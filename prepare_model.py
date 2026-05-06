import torch
import os
import time
from transformers import AutoImageProcessor, AutoModelForImageClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

os.makedirs("models", exist_ok=True)

model_id = "google/efficientnet-b0" 
onnx_path = "models/model.onnx"
quant_path = "models/model_quant.onnx"

print(f"1. Loading Model ({model_id}) from Hugging Face...")
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)
model.eval()

# EfficientNet-B0 ใช้ Input ขนาด 224x224
dummy_input = torch.randn(1, 3, 224, 224)

print("2. Converting to ONNX...")
start_time = time.time()
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path, 
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['pixel_values'], 
    output_names=['logits']
)
print(f"ONNX Conversion Done in {time.time() - start_time:.2f}s. Size: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")

print("3. Applying Dynamic Quantization...")
start_time = time.time()
quantize_dynamic(
    model_input=onnx_path,
    model_output=quant_path,
    weight_type=QuantType.QUInt8
)
print(f"Quantization Done in {time.time() - start_time:.2f}s. Size: {os.path.getsize(quant_path) / (1024*1024):.2f} MB")