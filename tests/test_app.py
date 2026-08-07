import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_read_root():
    """Test the root endpoint status and feature list."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "service" in data
    assert "required_features" in data
    assert len(data["required_features"]) > 0

def test_predict_endpoint_valid():
    """Test predict endpoint with valid feature inputs."""
    payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert "label" in data
    assert data["label"] in ["Good Quality", "Poor Quality"]

def test_predict_endpoint_missing_feature():
    """Test predict endpoint with missing required feature (should return 422 Unprocessable Entity)."""
    payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        # citric_acid is missing
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_endpoint_invalid_type():
    """Test predict endpoint with invalid feature types (should return 422 Unprocessable Entity)."""
    payload = {
        "fixed_acidity": "invalid_type",  # should be a float
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
