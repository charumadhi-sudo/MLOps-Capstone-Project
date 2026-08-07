import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import predict_single, load_artifacts, get_feature_names

app = FastAPI(
    title="Wine Quality Prediction API",
    description="A FastAPI prediction service for classifying red wine quality based on chemical features.",
    version="1.0.0"
)

# Input request schema using standard snake_case fields
class WineInput(BaseModel):
    fixed_acidity: float = Field(..., description="Fixed acidity (g/L)", example=7.4)
    volatile_acidity: float = Field(..., description="Volatile acidity (g/L)", example=0.7)
    citric_acid: float = Field(..., description="Citric acid (g/L)", example=0.0)
    residual_sugar: float = Field(..., description="Residual sugar (g/L)", example=1.9)
    chlorides: float = Field(..., description="Chlorides (g/L)", example=0.076)
    free_sulfur_dioxide: float = Field(..., description="Free sulfur dioxide (mg/L)", example=11.0)
    total_sulfur_dioxide: float = Field(..., description="Total sulfur dioxide (mg/L)", example=34.0)
    density: float = Field(..., description="Density (g/mL)", example=0.9978)
    pH: float = Field(..., description="pH value", example=3.51)
    sulphates: float = Field(..., description="Sulphates (g/L)", example=0.56)
    alcohol: float = Field(..., description="Alcohol content (% vol)", example=9.4)

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Binary prediction: 1 = Good, 0 = Poor")
    probability: float = Field(..., description="Probability of being high quality")
    label: str = Field(..., description="Human-readable prediction label")

@app.on_event("startup")
def startup_event():
    """Load model and scaler during service startup."""
    try:
        load_artifacts()
        print("Model and scaler artifacts successfully loaded on startup.")
    except Exception as e:
        print(f"Error loading artifacts on startup: {e}")

@app.get("/")
def read_root():
    """Root endpoint returning service status and list of required features."""
    try:
        features = get_feature_names()
    except Exception:
        features = ["Could not load feature names. Check if model is trained."]
    return {
        "status": "online",
        "service": "Wine Quality Prediction API",
        "required_features": features
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(features: WineInput):
    """Endpoint for wine quality classification."""
    try:
        # Convert Pydantic model to dict
        input_data = features.dict()
        
        # Run prediction
        result = predict_single(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
