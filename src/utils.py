import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

def download_dataset(url=DATA_URL, output_path="data/winequality-red.csv"):
    """Downloads the Red Wine Quality dataset if it doesn't exist."""
    if os.path.exists(output_path):
        print(f"Dataset already exists at {output_path}")
        return output_path

    print(f"Downloading dataset from {url}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Dataset saved to {output_path}")
    else:
        raise Exception(f"Failed to download dataset. Status code: {response.status_code}")
    return output_path

def load_and_preprocess_data(file_path="data/winequality-red.csv", test_size=0.2, random_state=42):
    """
    Loads dataset, converts quality into a binary target (quality >= 6),
    splits into train/test, and scales features.
    """
    if not os.path.exists(file_path):
        download_dataset(output_path=file_path)

    # Note: Wine Quality CSV uses semicolon ';' as separator
    df = pd.read_csv(file_path, sep=';')
    
    # Check if data loaded correctly
    if df.empty:
        raise ValueError(f"Loaded dataset from {file_path} is empty")
        
    # Feature engineering: create a binary target (1 = Good Wine, 0 = Bad Wine)
    # quality >= 6 is considered good wine
    df['target'] = (df['quality'] >= 6).astype(int)
    
    X = df.drop(columns=['quality', 'target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Keep column names for DataFrame construction if needed
    feature_names = X.columns.tolist()
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
