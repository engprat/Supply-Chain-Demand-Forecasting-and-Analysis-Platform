import pandas as pd
import numpy as np
import glob
import os
import sys
from sklearn.preprocessing import LabelEncoder

# Add the project root to Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import from the correct location
from src.data.data_filtering import analyze_and_filter

# Set paths relative to project root
raw_folder = os.path.join(project_root, "data", "raw")
processed_folder = os.path.join(project_root, "data", "processed")

# Create processed directory if it doesn't exist
os.makedirs(processed_folder, exist_ok=True)

print(f"Project root: {project_root}")
print(f"Looking for CSV files in: {raw_folder}")
print(f"Will save processed files to: {processed_folder}")

# Search for CSV files in raw folder and subfolders
csv_files = []
csv_files.extend(glob.glob(os.path.join(raw_folder, "*.csv")))  # Direct files
csv_files.extend(glob.glob(os.path.join(raw_folder, "*", "*.csv")))  # Files in subfolders

# Filter out .DS_Store files
csv_files = [f for f in csv_files if not f.endswith('.DS_Store')]

print(f"Loading and concatenating the following CSV files from {raw_folder} and subfolders:")
for f in csv_files:
    print(f" - {f}")

if not csv_files:
    print(f"❌ No CSV files found in {raw_folder} or its subfolders!")
    print(f"Directory structure:")
    if os.path.exists(raw_folder):
        for root, dirs, files in os.walk(raw_folder):
            print(f"  {root}: {files}")
    else:
        print(f"  {raw_folder} does not exist!")
    exit(1)

synthetic_columns = [
    'order_id', 'order_date', 'customer_id', 'city', 'product_id', 'product_name', 'category',
    'order_qty', 'revenue', 'order_year', 'order_month', 'order_day', 'order_day_of_week',
    'is_late', 'channel_Retail', 'delivery_status_Late delivery', 'delivery_status_Late or partial',
    'delivery_status_On-time full', 'delivery_status_Shipping canceled', 'delivery_status_Shipping on time',
    'customer_type_Enterprise', 'country_USA', 'city_encoded', 'category_encoded',
    'forecast_date', 'base_demand_qty', 'promotional_demand_qty', 'is_promotional',
    'promo_type', 'discount_percent', 'price', 'temperature', 'humidity', 'inflation', 'location'
]

# Updated mapping to include AtliQ dataset columns
map_dataco_to_synthetic = {
    # Original mappings
    'Order Id': 'order_id', 'order_id': 'order_id', 'transaction_id': 'order_id',
    'order date (DateOrders)': 'order_date', 'Order Date': 'order_date', 'order_date': 'order_date',
    'transaction_date': 'order_date', 'date': 'order_date', 'forecast_date': 'order_date',
    'weather_date': 'order_date', 'promo_start_date': 'order_date', 'alert_date': 'order_date',
    'created_date': 'order_date', 'last_updated': 'order_date', 'start_date': 'order_date',
    'Order Customer Id': 'customer_id', 'customer_id': 'customer_id', 'customer': 'customer_id',
    'Order City': 'city', 'city': 'city', 'location': 'city',
    'Product Card Id': 'product_id', 'product_id': 'product_id', 'sku_id': 'product_id',
    'Product Name': 'product_name', 'product_name': 'product_name',
    'Category Name': 'category', 'category': 'category',
    'Order Item Quantity': 'order_qty', 'order_qty': 'order_qty', 'total_demand_qty': 'order_qty',
    'Sales': 'revenue', 'revenue': 'revenue', 'pos_sales_value': 'revenue', 'total_revenue': 'revenue',
    'Delivery Status': 'delivery_status',
    'Order Country': 'country', 'country': 'country',
    
    # AtliQ dataset mappings
    'customer_id': 'customer_id',
    'customer_name': 'customer_id',
    'customer_code': 'customer_id',
    'city_code': 'city',
    'city': 'city',
    'product_code': 'product_id',
    'product_name': 'product_name',
    'variant': 'category',
    'category': 'category',
    'order_quantity': 'order_qty',
    'qty_ordered': 'order_qty',
    'quantity': 'order_qty',
    'sales_amount': 'revenue',
    'net_sales': 'revenue',
    'gross_sales': 'revenue',
    'order_date': 'order_date',
    'date': 'order_date',
    'order_placement_date': 'order_date',
    'delivery_date': 'order_date',
    
    # Date dimension mappings
    'order_id': 'order_id',
    'order_placement_date': 'order_date',
    'agreed_delivery_date': 'order_date',
    'actual_delivery_date': 'order_date',
    'delivery_time_days': 'is_late',  # Can be converted to boolean
    
    # Additional possible mappings
    'month': 'order_month',
    'year': 'order_year',
    'week_no': 'order_day_of_week'
}

all_dfs = []
key_fields = ['order_id', 'product_id', 'order_date', 'order_qty', 'city']

for f in csv_files:
    try:
        print(f"\n📂 Processing: {os.path.basename(f)}")
        try:
            df = pd.read_csv(f, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            print(f"  ⚠️ utf-8 decode failed for {f}, trying latin1...")
            df = pd.read_csv(f, encoding='latin1')

        print(f"  📊 Original shape: {df.shape}")
        print(f"  📋 Original columns: {list(df.columns)}")

        # Rename columns
        df = df.rename(columns=map_dataco_to_synthetic)
        df = df.loc[:, ~df.columns.duplicated()]

        print(f"  🔄 After column mapping: {list(df.columns)}")

        # Try to find a date column
        date_candidates = ['order_date', 'date', 'order_placement_date', 'agreed_delivery_date', 'actual_delivery_date']
        date_col_found = None
        
        for date_col in date_candidates:
            if date_col in df.columns:
                date_col_found = date_col
                break
        
        if date_col_found and date_col_found != 'order_date':
            df['order_date'] = df[date_col_found]
            print(f"  📅 Using '{date_col_found}' as order_date")

        if 'order_date' not in df.columns:
            print(f"⚠️ Skipping {f}: 'order_date' column not found after mapping")
            print(f"  Available columns: {list(df.columns)}")
            continue

        # Convert order_date
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        if df['order_date'].isnull().all():
            print(f"⚠️ Skipping {f}: all values in 'order_date' are null")
            continue

        # Add missing synthetic columns
        for col in synthetic_columns:
            if col not in df.columns:
                if col in ['order_qty', 'revenue', 'order_year', 'order_month', 'order_day', 'order_day_of_week',
                        'city_encoded', 'category_encoded', 'base_demand_qty', 'promotional_demand_qty',
                        'discount_percent', 'price', 'temperature', 'humidity', 'inflation']:
                    df[col] = 0.0
                elif col in ['is_late', 'is_promotional']:
                    df[col] = False
                elif col == 'forecast_date':
                    df[col] = df['order_date'] if 'order_date' in df.columns else pd.NaT
                elif col == 'location':
                    df[col] = df.get('city', 'Unknown')
                else:
                    df[col] = 'None'

        # Generate date components
        dt = df['order_date']
        df['order_year'] = dt.dt.year.fillna(0).astype(int)
        df['order_month'] = dt.dt.month.fillna(0).astype(int)
        df['order_day'] = dt.dt.day.fillna(0).astype(int)
        df['order_day_of_week'] = dt.dt.dayofweek.fillna(0).astype(int)

        # Keep only synthetic columns
        df = df[synthetic_columns]

        print(f"✅ Keeping {len(df)} rows from {os.path.basename(f)} after mapping and fallback.")
        print(f"  Non-null key fields in {os.path.basename(f)}:")
        available_key_fields = [field for field in key_fields if field in df.columns]
        if available_key_fields:
            print(df[available_key_fields].notnull().sum())

        all_dfs.append(df)

    except Exception as e:
        print(f"❌ Error processing {f}: {e}")
        continue

if not all_dfs:
    raise RuntimeError("No CSVs loaded successfully!")

print(f"\n🔗 Combining {len(all_dfs)} DataFrames...")
combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df = combined_df.dropna(subset=['order_date'])

print(f"📊 Combined dataset shape: {combined_df.shape}")

def is_valid_transaction(row):
    return (
        pd.notnull(row['order_id']) and str(row['order_id']).strip().lower() not in ('', 'none', 'nan', '0')
        and pd.notnull(row['product_id']) and str(row['product_id']).strip().lower() not in ('', 'none', 'nan', '0')
        and pd.notnull(row['order_date'])
        and pd.notnull(row['order_qty']) and row['order_qty'] not in (0, '0', '', 'none', 'nan', None)
        and pd.notnull(row['city']) and str(row['city']).strip().lower() not in ('', 'none', 'nan')
    )

# 🔄 Run filtering and inference logic on combined_df
print("\n🔎 Running data filtering and imputation...")
try:
    filtered_df = analyze_and_filter(combined_df)
except Exception as e:
    print(f"⚠️ Data filtering failed: {e}")
    print("Using original combined data without filtering...")
    filtered_df = combined_df

# Fill missing values
cat_cols = ['delivery_status', 'channel', 'product_name', 'customer_type', 'category', 'city', 'promo_type']
for col in cat_cols:
    if col in filtered_df.columns:
        filtered_df[col] = filtered_df[col].fillna(filtered_df[col].mode()[0] if not filtered_df[col].mode().empty else 'Unknown')

num_cols = ['revenue', 'order_qty', 'base_demand_qty', 'promotional_demand_qty', 'discount_percent',
            'price', 'temperature', 'humidity', 'inflation']
for col in num_cols:
    if col in filtered_df.columns:
        filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0.0)

# Encode
for col, encoder in [('city', 'city_encoded'), ('category', 'category_encoded')]:
    if col in filtered_df.columns:
        le = LabelEncoder()
        filtered_df[encoder] = le.fit_transform(filtered_df[col].astype(str))

filtered_df = filtered_df.drop_duplicates()

print("🔍 Key fields NA counts after filtering:")
available_key_fields = [field for field in key_fields if field in filtered_df.columns]
if available_key_fields:
    print(filtered_df[available_key_fields].isna().sum())
    print("🔍 Key fields sample values after filtering:")
    print(filtered_df[available_key_fields].head(10))

# Save the processed dataset
output_path = os.path.join(processed_folder, "dataset.csv")
filtered_df.to_csv(output_path, index=False)

print(f"\n✅ Filtered transactional dataset saved to: {output_path}")
print(f"📊 Filtered dataset shape: {filtered_df.shape}")
print(f"📋 Columns: {list(filtered_df.columns)}")
print(filtered_df.head())

# Reference data processing
reference_files = [
    "kc_customer_master_data_complete.csv",
    "kc_sku_master_data_complete.csv",
    "kc_weather_data_complete.csv",
    "kc_external_factors_complete.csv",
    "kc_promotional_calendar_complete.csv",
    "kc_inventory_snapshots_complete.csv",
    "kc_alerts_exceptions_complete.csv",
    "kc_forecast_scenarios_complete.csv"
]

print("\n🔄 Processing reference/auxiliary data files separately:")

for ref_filename in reference_files:
    ref_path = os.path.join(raw_folder, ref_filename)
    try:
        ref_df = pd.read_csv(ref_path, low_memory=False, encoding='utf-8')
        print(f"  ✅ Loaded {ref_filename} with shape {ref_df.shape}")

        ref_df.columns = [col.strip() for col in ref_df.columns]
        date_cols = [col for col in ref_df.columns if 'date' in col.lower()]
        for col in date_cols:
            try:
                ref_df[col] = pd.to_datetime(ref_df[col], errors='coerce')
            except Exception as e:
                print(f"    ⚠️ Could not parse date in column '{col}': {e}")

        output_name = ref_filename.replace(".csv", "_clean.csv")
        output_path = os.path.join(processed_folder, output_name)
        ref_df.to_csv(output_path, index=False)
        print(f"  💾 Saved cleaned data to {output_path}")

    except Exception as e:
        print(f"  ❌ Failed to process {ref_filename}: {e}")

print("\n🎉 Data preprocessing completed!")