'''src/utils/validation.py'''
import json
import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# === Logging Setup ===
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/data_validation.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# === Utility Functions ===


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """Detects outliers using IQR method for a given column. Multiplier is configurable."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    return df[(df[column] < q1 - multiplier * iqr) | (df[column] > q3 + multiplier * iqr)]

def remove_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """Removes outliers using IQR method for a given column. Multiplier is configurable."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    mask = (df[column] >= q1 - multiplier * iqr) & (df[column] <= q3 + multiplier * iqr)
    return df[mask]


def summarize_outliers(df: pd.DataFrame, numeric_cols: list) -> dict:
    """Returns a dict {col: n_outliers} for each numeric column."""
    outlier_counts = {}
    for col in numeric_cols:
        n = detect_outliers_iqr(df, col).shape[0]
        outlier_counts[col] = int(n)
    return outlier_counts



def write_local_alert(message: str) -> None:
    """Writes anomaly alerts to a local alert log file."""
    alert_path = "logs/anomaly_alerts.log"
    with open(alert_path, "a") as f:
        f.write(message + "\n")
    logger.warning(f"⚠️ Alert written to {alert_path}")


def save_json(data: Dict[str, Any], path: str) -> None:
    """Saves a Python dictionary as a formatted JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def encode_categorical_columns(
    df: pd.DataFrame, columns: list
) -> Dict[str, Dict[str, int]]:
    """Encodes categorical columns using LabelEncoder and returns the mapping."""
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = {
            str(cls): int(code)
            for cls, code in zip(le.classes_, le.transform(le.classes_))
        }
    return encoders


def scale_numeric_columns(
    df: pd.DataFrame, columns: list, method: str = "MinMax"
) -> pd.DataFrame:
    """Scales numeric columns using the specified method ('MinMax' or 'Standard')."""
    df_scaled = df.copy()
    if method == "MinMax":
        scaler = MinMaxScaler()
    elif method == "Standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Unsupported scaling method. Use 'MinMax' or 'Standard'.")

    scaled_cols = [col + f"_{method}" for col in columns]
    df_scaled[scaled_cols] = scaler.fit_transform(df[columns])
    return df_scaled


def check_column_types(df, columns_schema):
    """
    Checks each column in df against the expected type in columns_schema.
    Returns a dict: {column: (expected_type, actual_dtype)} for mismatches.
    """
    mismatches = {}
    for col, props in columns_schema.items():
        if col in df.columns:
            typ = props['type']
            dtype = df[col].dtype
            if typ == "float":
                if not (pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col])):
                    mismatches[col] = (typ, str(dtype))
            elif typ == "int":
                if not pd.api.types.is_integer_dtype(df[col]):
                    mismatches[col] = (typ, str(dtype))
            elif typ == "datetime":
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    mismatches[col] = (typ, str(dtype))
            elif typ == "string":
                if not (dtype == object or pd.api.types.is_string_dtype(df[col])):
                    mismatches[col] = (typ, str(dtype))
    return mismatches


def check_required_nulls(df, columns_schema):
    """
    Checks for nulls in all required columns.
    Returns a dict: {column: null_count} for required columns with nulls.
    """
    nulls = {}
    for col, props in columns_schema.items():
        if props.get('required') and col in df.columns:
            n_null = df[col].isnull().sum()
            if n_null > 0:
                nulls[col] = int(n_null)
    return nulls
