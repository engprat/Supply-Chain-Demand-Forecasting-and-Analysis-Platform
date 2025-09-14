'''src/utils/io_utils.py'''
import json
import logging
import os
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)

def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def write_local_alert(message: str) -> None:
    os.makedirs("logs", exist_ok=True)
    alert_path = "logs/anomaly_alerts.log"
    with open(alert_path, "a") as f:
        f.write(message + "\n")
    logger.warning(f"⚠️ Alert written to {alert_path}")

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    return df[(df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)]
