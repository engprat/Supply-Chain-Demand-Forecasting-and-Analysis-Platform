"""Evaluate the trained XGBoost model with key metrics.

This script loads the saved model and test data, then calculates and prints
important evaluation metrics including R², MAE, and RMSE.

Usage:
    python evaluate_model.py --model_path models/new_supply_chain_model.pkl --test_data_path training_data/kc_demand_history_complete.csv
"""

import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same preprocessing as in training."""
    # Convert date columns to timestamps
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "object"]).filter(like="date").columns
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f"{col}_ts"] = df[col].astype("int64") // 10**9
    
    # Drop original datetime columns
    df = df.drop(columns=datetime_cols, errors='ignore')
    
    # Handle categorical columns (same as in training)
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.difference(["transaction_id"])
    low_card_cols = [c for c in cat_cols if df[c].nunique() <= 100]
    high_card_cols = [c for c in cat_cols if c not in low_card_cols]
    
    # One-hot encode low-cardinality columns
    if low_card_cols:
        df = pd.get_dummies(df, columns=low_card_cols, drop_first=True)
    
    # Label-encode high-cardinality columns
    for col in high_card_cols:
        df[col] = df[col].astype("category").cat.codes
    
    # Fill NaNs in numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(-1)
    
    return df

def load_model_and_data(model_path: Path, test_data_path: Path) -> tuple:
    """Load the trained model and prepare test data."""
    print(f"Loading model from {model_path}...")
    model_data = joblib.load(model_path)
    model = model_data["model"]
    feature_names = model_data["features"]
    
    print(f"Loading and preprocessing test data from {test_data_path}...")
    test_data = pd.read_csv(test_data_path, low_memory=False)
    
    # Store target and drop from features
    y_test = test_data["total_demand_qty"]
    X_test = test_data.drop(columns=["total_demand_qty"])
    
    # Apply same preprocessing as in training
    X_test = preprocess_data(X_test)
    
    # Ensure only features used in training are present (in same order)
    X_test = X_test.reindex(columns=feature_names, fill_value=0)
    
    return model, X_test, y_test

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate and return a dictionary of evaluation metrics."""
    metrics = {
        "r2_score": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,  # as percentage
        "explained_variance": np.var(y_pred) / np.var(y_true) if np.var(y_true) > 0 else 0,
        "mean_actual": float(np.mean(y_true)),
        "mean_predicted": float(np.mean(y_pred)),
    }
    return metrics

def print_metrics(metrics: dict):
    """Print the evaluation metrics in a readable format."""
    print("\n=== Model Evaluation Metrics ===")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"Explained Variance: {metrics['explained_variance']:.4f}")
    print(f"Mean Actual: {metrics['mean_actual']:.2f}")
    print(f"Mean Predicted: {metrics['mean_predicted']:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate XGBoost model performance")
    parser.add_argument("--model_path", type=Path, default="models/new_supply_chain_model.pkl",
                       help="Path to the trained model file")
    parser.add_argument("--test_data_path", type=Path, 
                       default="training_data/kc_demand_history_complete.csv",
                       help="Path to the test data CSV")
    args = parser.parse_args()
    
    # Load model and data
    model, X_test, y_test = load_model_and_data(args.model_path, args.test_data_path)
    
    # Make predictions
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics)
    
    # Save metrics to file
    metrics_path = args.model_path.parent / "model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
