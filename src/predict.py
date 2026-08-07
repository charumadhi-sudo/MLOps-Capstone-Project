import os
import pickle
import numpy as np

# Global variables for model, scaler, and features
_MODEL = None
_SCALER = None
_FEATURE_NAMES = None

def load_artifacts():
    """Loads the model and scaler from the models directory."""
    global _MODEL, _SCALER, _FEATURE_NAMES
    
    if _MODEL is not None and _SCALER is not None:
        return
        
    model_path = os.path.join("models", "best_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    feature_path = os.path.join("models", "feature_names.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Serialized model/scaler not found. Please run training first. "
            f"Expected paths: {model_path}, {scaler_path}"
        )
        
    with open(model_path, "rb") as f:
        _MODEL = pickle.load(f)
    with open(scaler_path, "rb") as f:
        _SCALER = pickle.load(f)
    with open(feature_path, "rb") as f:
        _FEATURE_NAMES = pickle.load(f)

def predict_single(features_dict):
    """
    Makes a prediction on a single sample represented as a dictionary.
    Keys must match the feature names.
    """
    load_artifacts()
    
    # Extract features in the correct order
    try:
        features_list = [features_dict[name] for name in _FEATURE_NAMES]
    except KeyError as e:
        # Fallback to key matching by replacing space with underscore etc.
        # e.g., 'fixed_acidity' vs 'fixed acidity'
        normalized_dict = {k.replace("_", " ").lower(): v for k, v in features_dict.items()}
        try:
            features_list = [normalized_dict[name.lower()] for name in _FEATURE_NAMES]
        except KeyError as e_inner:
            raise ValueError(f"Missing required feature: {e_inner}. Expected features: {_FEATURE_NAMES}")
            
    # Reshape and scale
    features_arr = np.array(features_list).reshape(1, -1)
    features_scaled = _SCALER.transform(features_arr)
    
    # Predict
    prediction = int(_MODEL.predict(features_scaled)[0])
    
    # Predict probability if the model supports it
    probability = None
    if hasattr(_MODEL, "predict_proba"):
        prob_arr = _MODEL.predict_proba(features_scaled)[0]
        # prob_arr contains [prob(class 0), prob(class 1)]
        probability = float(prob_arr[1])
        
    return {
        "prediction": prediction,
        "probability": probability,
        "label": "Good Quality" if prediction == 1 else "Poor Quality"
    }

def get_feature_names():
    load_artifacts()
    return _FEATURE_NAMES
