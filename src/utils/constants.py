'''src/utils/constants.py'''
import os
from datetime import datetime



def get_run_timestamp():
    # Returns timestamp in ISO format with local time and UTC
    now = datetime.now()
    now_utc = datetime.utcnow()
    return {
        "local": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

CRITICAL_COLS = [
    "order_id", "order_date", "product_id", "order_qty", "revenue"
]

NUMERIC_COLS = [
    "order_qty", "revenue"
]

CATEGORICAL_COLS = [
    "city", "category", "channel_Retail", "delivery_status_Late delivery", 
    "delivery_status_Late or partial", "delivery_status_On-time full", 
    "delivery_status_Shipping canceled", "delivery_status_Shipping on time", 
    "customer_type_Enterprise", "country_USA"
]



# ID columns for reference/grouping
ID_COLS = [
    "order_id",
    "customer_id", 
    "product_id"
]

# Date columns
DATE_COLS = [
    "order_date"
]

DATA_PATH = "data/processed/dataset.csv"
ENCODING = "utf-8"  # Changed from ISO-8859-1 as most modern CSVs use UTF-8
MODEL_PATH = "models/"

