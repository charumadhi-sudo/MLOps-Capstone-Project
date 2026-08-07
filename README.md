# MLOps Pipeline Capstone Project: Red Wine Quality Classifier

This project implements a complete, production-ready MLOps pipeline for classifying red wine quality. Given the chemical properties of a wine sample, the model classifies it as either **Good Quality** (score >= 6) or **Poor Quality** (score < 6).

The pipeline demonstrates:
1. **Data Versioning**: Tracking the dataset using DVC.
2. **Machine Learning Pipeline**: Data loading, preprocessing, model training, and evaluation for 3 different classifiers (Logistic Regression, Random Forest, Support Vector Classifier).
3. **Experiment Tracking & Model Registry**: Logging training runs and registering the best model with MLflow.
4. **Prediction API**: Exposing inference endpoints using FastAPI.
5. **Docker Containerization**: Packaging the prediction service for containerized environments.
6. **CI/CD Workflow**: Automating unit tests and Docker image builds via GitHub Actions.

---

## Project Structure

```
project/
|-- .github/
|   |-- workflows/
|       |-- ci.yml           # GitHub Actions workflow
|-- data/
|   |-- winequality-red.csv  # Versioned raw dataset (tracked by DVC)
|-- models/
|   |-- best_model.pkl       # Trained classification model
|   |-- scaler.pkl           # Feature scaler parameters
|   |-- feature_names.pkl    # Serialized feature names
|-- src/
|   |-- __init__.py
|   |-- app.py               # FastAPI application
|   |-- predict.py           # Prediction helper service
|   |-- train.py             # Model training & MLflow tracking script
|   |-- utils.py             # Data loading and scaling utilities
|-- tests/
|   |-- test_app.py          # Pytest unit tests for the API
|-- Dockerfile               # Container configurations
|-- requirements.txt         # Python package dependencies
|-- dvc.yaml                 # DVC pipeline stages
|-- README.md                # Documentation
|-- .gitignore               # Ignored files (venv, mlflow local files, etc.)
```

---

## Local Setup

### 1. Prerequisites
- Python 3.11.x
- Docker
- Git & DVC

### 2. Create Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

### 1. Data Versioning (DVC)
The dataset is tracked by DVC in `data/winequality-red.csv.dvc`. To retrieve it:
```bash
dvc pull
```

### 2. Train Models and Track with MLflow
Run the training script to train Logistic Regression, Random Forest, and SVC models. It will track all metrics in local MLflow and register the best performing model:
```bash
python -m src.train
```

To view the MLflow Dashboard and compare experiments:
```bash
# Starts MLflow server using sqlite backend
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Run & Query the FastAPI Service

### 1. Run Locally
```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```
- Access Interactive API documentation (Swagger) at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Access service health check at [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### 2. Run with Docker
```bash
# Build Docker image
docker build -t wine-quality-api:latest .

# Run Docker container
docker run -d -p 8000:8000 --name wine-api wine-quality-api:latest
```
Test using:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

---

## Running Tests
Run the unit test suite using `pytest`:
```bash
python -m pytest
```
