"""Data Exploration Script for Supply Chain Training Data.

Usage:
    python explore_training_data.py --data_dir path/to/training_data --output summary.json

The script scans every CSV file in the given directory and produces a JSON
summary containing, for each file:
    - number of rows
    - columns with dtype, #unique values (capped), and #missing values

The output is printed to stdout and optionally written to the specified JSON file.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np  # Added for dtype handling

# ---------- Helper Functions -------------------------------------------------


def count_file_rows(file_path: Path) -> int:
    """Count rows in a CSV quickly without loading to DataFrame."""
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        # subtract header line
        return sum(1 for _ in f) - 1


def analyze_csv(file_path: Path, max_unique: int = 100_000) -> Dict[str, Any]:
    """Return summary dict for a single CSV file."""
    try:
        n_rows = count_file_rows(file_path)
    except Exception as e:
        print(f"[WARN] Failed fast row count for {file_path.name}: {e}. Falling back to pandas.")
        n_rows = None

    # Read the file in chunks to gather column metadata and avoid OOM on huge files
    chunk_iter = pd.read_csv(file_path, chunksize=100_000, low_memory=False)
    summary: Dict[str, Any] = {"file": file_path.name, "rows": n_rows, "columns": {}}

    # For unique & null counts
    uniques: Dict[str, set] = {}
    null_counts: Dict[str, int] = {}

    for chunk in chunk_iter:
        if summary["rows"] is None:
            summary["rows"] = summary.get("rows", 0) + len(chunk)
        for col in chunk.columns:
            col_data = chunk[col]
            # dtype
            if col not in summary["columns"]:
                summary["columns"][col] = {
                    "dtype": str(col_data.dtype),
                    "n_unique": 0,
                    "n_null": 0,
                }
                uniques[col] = set()
                null_counts[col] = 0

            # update uniques (with cap)
            if summary["columns"][col]["n_unique"] < max_unique:
                uniques[col].update(col_data.dropna().unique())
                summary["columns"][col]["n_unique"] = min(len(uniques[col]), max_unique)
            # update null count
            null_counts[col] += col_data.isna().sum()

    # finalize null counts
    for col, n_null in null_counts.items():
        summary["columns"][col]["n_null"] = n_null

    return summary


# ---------- Main -------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Explore supply chain training CSV files.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training_data directory")
    parser.add_argument("--output", type=str, default=None, help="Optional path to write JSON summary")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists() or not data_path.is_dir():
        print(f"[ERROR] Provided data_dir '{data_path}' is not a valid directory.")
        sys.exit(1)

    summaries = []
    csv_files = sorted(p for p in data_path.iterdir() if p.suffix.lower() == ".csv")
    if not csv_files:
        print(f"[ERROR] No CSV files found in {data_path}.")
        sys.exit(1)

    for csv_path in csv_files:
        print(f"Analyzing {csv_path.name} ...", flush=True)
        summary = analyze_csv(csv_path)
        summaries.append(summary)

    # output result
    print("\n===== DATA SUMMARY =====")

    def _json_default(obj):
        """Convert NumPy types and other non-serializable objects for JSON."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        # Fallback: stringify anything else (e.g., Timestamp)
        return str(obj)

    print(json.dumps(summaries, indent=2, default=_json_default))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, default=_json_default)
        print(f"\nSummary written to {args.output}")


if __name__ == "__main__":
    main()
