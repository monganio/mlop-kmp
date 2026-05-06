import pytest
from fastapi.testclient import TestClient
from main import app
import io
from PIL import Image

client = TestClient(app)

def create_valid_image():
    file = io.BytesIO()
    image = Image.new('RGB', (224, 224), color='green')
    image.save(file, 'jpeg')
    file.seek(0)
    return file

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_endpoint_success():
    img = create_valid_image()
    response = client.post("/predict", files={"file": ("test.jpg", img, "image/jpeg")})
    assert response.status_code == 200
    json_data = response.json()
    assert "predicted_class" in json_data
    assert "confidence" in json_data

def test_predict_endpoint_invalid_type():
    response = client.post("/predict", files={"file": ("test.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

def test_predict_endpoint_corrupted_image():
    response = client.post("/predict", files={"file": ("bad.jpg", b"fake_bytes", "image/jpeg")})
    assert response.status_code == 400
    assert "Corrupted" in response.json()["detail"]