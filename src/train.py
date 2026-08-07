import os
import pickle
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils import load_and_preprocess_data

def train_and_track():
    # 1. Set MLflow tracking URI to support registry locally
    db_path = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(db_path)
    mlflow.set_experiment("Wine_Quality_Classification")
    
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data()
    
    # Define the 3 models we want to train
    models_config = [
        {
            "name": "LogisticRegression",
            "model_class": LogisticRegression,
            "params": {"C": 1.0, "max_iter": 1000, "random_state": 42}
        },
        {
            "name": "RandomForest",
            "model_class": RandomForestClassifier,
            "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        },
        {
            "name": "SupportVectorClassifier",
            "model_class": SVC,
            "params": {"C": 1.0, "kernel": "rbf", "probability": True, "random_state": 42}
        }
    ]
    
    best_f1 = -1.0
    best_model_run_id = None
    best_model_name = None
    best_model_obj = None
    
    for config in models_config:
        model_name = config["name"]
        model_params = config["params"]
        
        print(f"\nTraining {model_name}...")
        
        # Start MLflow run
        with mlflow.start_run(run_name=model_name) as run:
            # Instantiate model
            model = config["model_class"](**model_params)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            print(f"{model_name} Results -> Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
            
            # Log hyperparameters
            mlflow.log_params(model_params)
            
            # Log metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })
            
            # Log model artifact with input example/signature if possible
            mlflow.sklearn.log_model(model, artifact_path="model")
            
            # Check if this is the best model based on F1-score
            if f1 > best_f1:
                best_f1 = f1
                best_model_run_id = run.info.run_id
                best_model_name = model_name
                best_model_obj = model
    
    print(f"\nBest Model: {best_model_name} with F1-score: {best_f1:.4f}")
    
    # 2. Register the best model in the MLflow Model Registry
    model_uri = f"runs:/{best_model_run_id}/model"
    print(f"Registering the best model in MLflow Model Registry...")
    try:
        model_details = mlflow.register_model(model_uri=model_uri, name="wine_quality_model")
        print(f"Successfully registered model: {model_details.name}, version: {model_details.version}")
    except Exception as e:
        print(f"Warning: Model registration failed (likely sqlite registry setup issue): {e}")
    
    # 3. Save model & scaler locally to 'models/' directory for FastAPI to load without MLflow dependency
    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model_obj, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
        
    print("Saved best model, scaler, and feature names to 'models/' directory.")

if __name__ == "__main__":
    train_and_track()
