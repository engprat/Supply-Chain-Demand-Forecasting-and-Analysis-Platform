'''src/models/train_model.py'''
import os
import sys
import logging
import joblib
import json
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Add root to sys path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import constants (ensure DATA_PATH is imported correctly)
from src.utils.constants import DATA_PATH, ENCODING, MODEL_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """Load and prepare the preprocessed dataset"""
    logging.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Convert order_date to datetime
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['order_date'])
    
    logging.info(f"Dataset loaded with shape: {df.shape}")
    logging.info(f"Date range: {df['order_date'].min()} to {df['order_date'].max()}")
    
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add additional time-based features"""
    logging.info("Adding time-based features...")
    
    # Sort by date for time series features
    df = df.sort_values(['order_date']).reset_index(drop=True)
    
    # Additional time features
    df['quarter'] = df['order_date'].dt.quarter
    df['is_month_start'] = df['order_date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['order_date'].dt.is_month_end.astype(int)
    df['is_weekend'] = (df['order_day_of_week'] >= 5).astype(int)
    
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling window features"""
    logging.info("Adding lag and rolling features...")
    
    # Sort by date and customer/product for proper lag calculation
    df = df.sort_values(['customer_id', 'product_id', 'order_date']).reset_index(drop=True)
    
    # Revenue-based features
    df['revenue_lag_1'] = df.groupby(['customer_id', 'product_id'])['revenue'].shift(1)
    df['revenue_lag_7'] = df.groupby(['customer_id', 'product_id'])['revenue'].shift(7)
    
    # Rolling features (7-day window)
    df['revenue_rolling_mean_7'] = df.groupby(['customer_id', 'product_id'])['revenue'] \
        .rolling(window=7, min_periods=1).mean().reset_index(level=[0, 1], drop=True)  # Reset only the group levels
    
    df['revenue_rolling_std_7'] = df.groupby(['customer_id', 'product_id'])['revenue'] \
        .rolling(window=7, min_periods=1).std().reset_index(level=[0, 1], drop=True)  # Reset only the group levels
    
    # Quantity-based features
    df['qty_lag_1'] = df.groupby(['customer_id', 'product_id'])['order_qty'].shift(1)
    df['qty_rolling_mean_7'] = df.groupby(['customer_id', 'product_id'])['order_qty'] \
        .rolling(window=7, min_periods=1).mean().reset_index(level=[0, 1], drop=True)  # Reset only the group levels
    
    # Fill NaN values with median for lag features
    numeric_cols = ['revenue_lag_1', 'revenue_lag_7', 'revenue_rolling_std_7', 'qty_lag_1']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features for model training"""
    logging.info("Preparing features for training...")
    
    # Define feature columns based on your preprocessed dataset
    numeric_features = [
        'order_qty', 'order_year', 'order_month', 'order_day', 'order_day_of_week',
        'quarter', 'is_month_start', 'is_month_end', 'is_weekend',
        'city_encoded', 'category_encoded',
        'revenue_lag_1', 'revenue_lag_7', 'revenue_rolling_mean_7', 'revenue_rolling_std_7',
        'qty_lag_1', 'qty_rolling_mean_7'
    ]
    
    # Binary features (already encoded)
    binary_features = [
        'is_late', 'channel_Retail', 'customer_type_Enterprise', 'country_USA'
    ]
    
    # Delivery status features (one-hot encoded)
    delivery_features = [col for col in df.columns if col.startswith('delivery_status_')]
    
    # Combine all features
    all_features = numeric_features + binary_features + delivery_features
    
    # Filter features that actually exist in the dataset
    available_features = [f for f in all_features if f in df.columns]
    missing_features = [f for f in all_features if f not in df.columns]
    
    if missing_features:
        logging.warning(f"Missing features: {missing_features}")
    
    logging.info(f"Using {len(available_features)} features for training")
    
    X = df[available_features].copy()
    y = df['revenue'].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(X.median())
    
    return X, y, available_features

def train_model(X: pd.DataFrame, y: pd.Series, features: list):
    """Train the LightGBM model"""
    logging.info(f"Training model with {len(features)} features on {len(X)} samples")
    
    # Split data chronologically for time series
    # Use the last 20% of data (by date) as test set
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logging.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Train LightGBM model
    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate model
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Log results
    logging.info("=== Model Performance ===")
    logging.info(f"Training - MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}, R²: {train_r2:.3f}")
    logging.info(f"Testing  - MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, R²: {test_r2:.3f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    logging.info("\n=== Top 10 Feature Importances ===")
    logging.info(importance_df.head(10).to_string(index=False))
    
    return model, {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'feature_importance': importance_df
    }

def save_model_and_metrics(model, metrics: dict, model_dir: str = "models"):
    """Save the trained model and metrics"""
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, "supply_chain_model.pkl")
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_to_save = {
        'train_mae': round(metrics['train_mae'], 2),
        'test_mae': round(metrics['test_mae'], 2),
        'train_rmse': round(metrics['train_rmse'], 2),
        'test_rmse': round(metrics['test_rmse'], 2),
        'train_r2': round(metrics['train_r2'], 3),
        'test_r2': round(metrics['test_r2'], 3)
    }
    
    with open("logs/model_metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=4)
    
    # Save feature importance
    metrics['feature_importance'].to_csv("logs/feature_importance.csv", index=False)
    
    logging.info("Metrics and feature importance saved to logs/")

def plot_results(model, X_test, y_test, save_dir: str = "logs"):
    """Plot model results"""
    os.makedirs(save_dir, exist_ok=True)
    
    y_pred = model.predict(X_test)
    
    # Prediction vs Actual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Revenue')
    plt.ylabel('Predicted Revenue')
    plt.title('Actual vs Predicted Revenue')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Residuals plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Revenue')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'residuals_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Plots saved to {save_dir}/")

def main():
    """Main training pipeline"""
    data_path = DATA_PATH
    
    if not os.path.exists(data_path):
        logging.error(f"❌ Dataset not found at {data_path}")
        sys.exit(1)
    
    try:
        # Load and prepare data
        df = load_and_prepare_data(data_path)
        
        # Add features
        df = add_time_features(df)
        df = add_lag_features(df)
        
        # Prepare features
        X, y, features = prepare_features(df)
        
        # Train model
        model, metrics = train_model(X, y, features)
        
        # Save results
        save_model_and_metrics(model, metrics)
        
        # Create plots (using last 20% as test set)
        split_idx = int(len(X) * 0.8)
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        plot_results(model, X_test, y_test)
        
        logging.info("✅ Model training completed successfully!")
        
    except Exception as e:
        logging.error(f"❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    data_path = DATA_PATH
    if not os.path.exists(data_path):
        logging.error(f"❌ Dataset not found at {data_path}")
        sys.exit(1)
    main()


