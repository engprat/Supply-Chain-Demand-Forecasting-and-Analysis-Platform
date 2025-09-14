#!/usr/bin/env python3
"""
Debug script to check the actual column names in your dataset
"""

import pandas as pd
from pathlib import Path

def debug_dataset():
    """Debug the dataset to find actual column names"""
    
    # Try to find the dataset
    possible_paths = [
        Path("data/processed/dataset.csv"),
        Path("data/dataset.csv"),
        Path("dataset.csv")
    ]
    
    dataset_path = None
    for path in possible_paths:
        if path.exists():
            dataset_path = path
            break
    
    if not dataset_path:
        print("❌ Dataset not found in expected locations:")
        for path in possible_paths:
            print(f"   - {path}")
        return
    
    print(f"✅ Found dataset at: {dataset_path}")
    
    try:
        # Load just the first few rows to check structure
        df = pd.read_csv(dataset_path, nrows=10)
        
        print(f"📊 Dataset shape (first 10 rows): {df.shape}")
        print(f"🏷️  Column names ({len(df.columns)} total):")
        
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. '{col}'")
        
        print("\n📅 Looking for date-related columns:")
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            for col in date_cols:
                print(f"   - {col}")
                sample_vals = df[col].head(3).tolist()
                print(f"     Sample values: {sample_vals}")
        else:
            print("   ❌ No columns with 'date' in the name found")
        
        print("\n📦 Looking for quantity-related columns:")
        qty_cols = [col for col in df.columns if any(word in col.lower() for word in ['qty', 'quantity', 'demand', 'order'])]
        if qty_cols:
            for col in qty_cols:
                sample_vals = df[col].head(3).tolist()
                print(f"   - {col}")
                print(f"     Sample values: {sample_vals}")
        else:
            print("   ❌ No obvious quantity columns found")
        
        print("\n🏢 Looking for SKU/Product columns:")
        sku_cols = [col for col in df.columns if any(word in col.lower() for word in ['sku', 'product', 'item'])]
        if sku_cols:
            for col in sku_cols:
                sample_vals = df[col].head(3).tolist()
                print(f"   - {col}")
                print(f"     Sample values: {sample_vals}")
        
        print("\n🌍 Looking for location/channel columns:")
        loc_cols = [col for col in df.columns if any(word in col.lower() for word in ['location', 'city', 'channel', 'region'])]
        if loc_cols:
            for col in loc_cols:
                sample_vals = df[col].head(3).tolist()
                print(f"   - {col}")
                print(f"     Sample values: {sample_vals}")
        
        print("\n📈 Sample of first 3 rows:")
        print(df.head(3).to_string())
        
        # Try to load the full dataset to get accurate row count
        try:
            full_df = pd.read_csv(dataset_path)
            print(f"\n📊 Full dataset shape: {full_df.shape}")
            
            # Check for any time-related columns
            time_cols = [col for col in full_df.columns if any(word in col.lower() for word in ['time', 'timestamp', 'created', 'updated'])]
            if time_cols:
                print(f"\n⏰ Time-related columns found:")
                for col in time_cols:
                    sample_vals = full_df[col].head(3).tolist()
                    print(f"   - {col}: {sample_vals}")
                    
        except Exception as e:
            print(f"\n⚠️  Could not load full dataset: {e}")
            
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        
        # Try to read just the header
        try:
            with open(dataset_path, 'r') as f:
                header = f.readline().strip()
                columns = [col.strip().strip('"') for col in header.split(',')]
                print(f"📋 Header from file: {header}")
                print(f"🏷️  Parsed columns ({len(columns)}):")
                for i, col in enumerate(columns, 1):
                    print(f"   {i:2d}. '{col}'")
        except Exception as e2:
            print(f"❌ Could not even read header: {e2}")

if __name__ == "__main__":
    debug_dataset()