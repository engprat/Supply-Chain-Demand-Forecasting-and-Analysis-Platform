import pandas as pd
import numpy as np
import uuid
from datetime import timedelta

def analyze_and_filter(df: pd.DataFrame, key_fields=None):
    if key_fields is None:
        key_fields = ['order_id', 'product_id', 'order_date', 'order_qty', 'city']

    print("\n🔎 Missing Value Summary (before filtering):")
    print(df[key_fields + ['customer_id', 'revenue']].isna().sum())

    # --- Impute order_id ---
    def fill_order_id(val):
        if pd.notnull(val) and str(val).strip().lower() not in ('', 'none', 'nan', '0'):
            return str(val)
        return str(uuid.uuid4())

    df['order_id'] = df['order_id'].apply(fill_order_id).astype(str)

    # --- Impute product_id ---
    def fill_product_id(val):
        if pd.notnull(val) and str(val).strip().lower() not in ('', 'none', 'nan', '0'):
            return str(val)
        return "unknown_sku_" + str(uuid.uuid4())[:8]

    df['product_id'] = df['product_id'].apply(fill_product_id).astype(str)

    # --- Impute order_date ---
    if 'order_date' in df.columns:
        median_date = df['order_date'].dropna().median()
        df['order_date'] = df['order_date'].fillna(median_date).astype('datetime64[ns]')

    # --- Impute city ---
    if 'city' in df.columns:
        mode_city = df['city'].mode()[0] if not df['city'].mode().empty else 'Unknown'
        df['city'] = df['city'].fillna(mode_city).astype(str)

    # --- Impute customer_id ---
    if 'customer_id' in df.columns:
        df['customer_id'] = df['customer_id'].fillna('anonymous').astype(str)

    # --- Impute order_qty ---
    if 'order_qty' in df.columns:
        df['order_qty'] = pd.to_numeric(df['order_qty'], errors='coerce').fillna(0.0)

    # --- Impute revenue ---
    if 'revenue' in df.columns:
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0.0)

    # --- Drop duplicates and ensure clean data ---
    df = df.drop_duplicates()

    # --- Final key field diagnostics ---
    print("\n🔍 Post-imputation Key Field Check:")
    print(df[key_fields + ['customer_id', 'revenue']].isna().sum())
    print("\n🔍 Sample key field values:")
    print(df[key_fields + ['customer_id', 'revenue']].head(10))

    return df
