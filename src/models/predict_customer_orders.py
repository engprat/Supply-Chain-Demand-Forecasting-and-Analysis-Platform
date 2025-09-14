# src/models/predict_customer_orders.py
import os, sys, json, logging
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import joblib

# repo root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.constants import DATA_PATH
from src.models.train_customer_orders import (
    add_time_features,
    add_lag_features,
    add_inventory_features,
    maybe_join_promotions_asof,
)

LOGS_DIR = Path("logs")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "customer_order_model.pkl"
FEATS_PATH = LOGS_DIR / "customer_features_used.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("predict_customer_orders")


def _load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not FEATS_PATH.exists():
        raise FileNotFoundError(f"Features list not found at {FEATS_PATH}")
    model = joblib.load(MODEL_PATH)
    data = json.loads(Path(FEATS_PATH).read_text())
    features = data.get("features_used", [])
    if not isinstance(features, list) or len(features) == 0:
        raise ValueError("features_used is empty or malformed in logs/customer_features_used.json")
    return model, features


def _load_history() -> pd.DataFrame:
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"]).sort_values("order_date").reset_index(drop=True)

    for col in ["customer_id", "product_id", "city"]:
        if col not in df.columns:
            df[col] = "Unknown"

    for c in ["order_qty", "price", "discount_percent"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _pick_forecast_index(history: pd.DataFrame, start_date: str, end_date: str,
                         min_recent_days: int = 28,
                         customers: Optional[List[str]] = None,
                         products: Optional[List[str]] = None) -> pd.DataFrame:
    """Build the future 'skeleton' rows to predict for."""
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)
    if end < start:
        raise ValueError("end_date must be >= start_date")

    last_hist_date = history["order_date"].max()
    cutoff = last_hist_date - pd.Timedelta(days=min_recent_days)
    recent = history.loc[history["order_date"] >= cutoff, ["customer_id", "product_id", "city"]].drop_duplicates()

    if customers:
        recent = recent[recent["customer_id"].isin(customers)]
    if products:
        recent = recent[recent["product_id"].isin(products)]

    if recent.empty:
        # fallback: use the most recent N combos overall
        top_recent = (history
                      .sort_values("order_date")
                      .groupby(["customer_id", "product_id", "city"], as_index=False)
                      .tail(1)
                      .sort_values("order_date", ascending=False)
                      .head(1000))[["customer_id", "product_id", "city"]]
        if top_recent.empty:
            raise ValueError("No (customer_id, product_id, city) combos available to forecast.")
        recent = top_recent

    future_dates = pd.date_range(start, end, freq="D")
    future = recent.assign(key=1).merge(
        pd.DataFrame({"order_date": future_dates, "key": 1}), on="key"
    ).drop(columns=["key"])
    # placeholders that feature builders expect
    future["order_qty"] = np.nan
    for c in ["price", "discount_percent"]:
        if c not in future.columns:
            future[c] = np.nan

    # mark as future to avoid filtering issues later
    future["is_future"] = 1
    return future


def _engineer_features(history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """
    Append future to history, build features with only past info, then slice future back out by flag.
    """
    history = history.copy()
    history["is_future"] = 0

    both = pd.concat([history, future], ignore_index=True, sort=False)
    both["order_date"] = pd.to_datetime(both["order_date"], errors="coerce")

    # promotions (will yield zeros if your promo file doesn't cover requested dates)
    both = maybe_join_promotions_asof(both)

    # time / inventory-like / lags (safe: use prior info only)
    both = add_time_features(both)
    both = add_inventory_features(both)
    both = add_lag_features(both)

    fut = both.loc[both["is_future"] == 1].copy().reset_index(drop=True)
    fut.drop(columns=["is_future"], inplace=True, errors="ignore")
    return fut


def _attach_static_encodings(history: pd.DataFrame, fut: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Bring over static columns (e.g., city_encoded, category_encoded, one-hots) from the most recent
    history row for each (customer_id, product_id, city). If the column already exists in `fut`,
    only fill its missing values from history. If it's missing entirely, create it.
    """
    key_cols = ["customer_id", "product_id", "city"]

    def _is_static(col: str) -> bool:
        return (
            col in {"city_encoded", "category_encoded"}
            or col.startswith("delivery_status_")
            or col.startswith("channel_")
            or col.startswith("customer_type_")
            or col.startswith("country_")
        )

    # Only pull columns that are both in features AND in history, and are "static"
    static_cols = [c for c in features if _is_static(c) and c in history.columns]
    if not static_cols:
        return fut

    # Most recent history per (customer, product, city)
    latest = (
        history.sort_values("order_date")
               .groupby(key_cols, as_index=False)
               .tail(1)[key_cols + static_cols]
    )

    # Merge with suffix to avoid overlap, then fill from *_hist into current cols
    merged = fut.merge(latest, on=key_cols, how="left", suffixes=("", "_hist"))

    filled_count = 0
    for c in static_cols:
        hist_col = f"{c}_hist"
        if hist_col in merged.columns:
            # If the column already exists in fut, fill NaNs from history
            if c in merged.columns:
                before_na = merged[c].isna().sum()
                merged[c] = pd.to_numeric(merged[c], errors="coerce")
                merged[c] = merged[c].fillna(pd.to_numeric(merged[hist_col], errors="coerce"))
                after_na = merged[c].isna().sum()
                filled_count += max(0, before_na - after_na)
            else:
                # Column missing in fut → create from history
                merged[c] = pd.to_numeric(merged[hist_col], errors="coerce")

            # Drop the helper column
            merged.drop(columns=[hist_col], inplace=True)

        # Final safety: any residual NaNs → 0.0
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    # Optional: log how many cells we filled (keeps function side-effect free)
    logging.getLogger("predict_customer_orders").info(
        f"Static encodings filled from history: {filled_count} cells across {len(static_cols)} columns."
    )

    return merged



def predict_range(start_date: str, end_date: str,
                  customers: Optional[List[str]] = None,
                  products: Optional[List[str]] = None) -> pd.DataFrame:
    model, features = _load_artifacts()
    hist = _load_history()
    fut_idx = _pick_forecast_index(hist, start_date, end_date, customers=customers, products=products)
    log.info(f"Forecast index size: {len(fut_idx)} rows for {start_date}→{end_date}")

    fut = _engineer_features(hist, fut_idx)
    log.info(f"Future after feature engineering: rows={len(fut)}, cols={len(fut.columns)}")

    # attach static encodings where available
    fut = _attach_static_encodings(hist, fut, features)

    # align to training feature set
    # add any missing feature columns as zeros
    missing_cols = [c for c in features if c not in fut.columns]
    if missing_cols:
        for c in missing_cols:
            fut[c] = 0.0
        log.info(f"Added {len(missing_cols)} missing feature columns as zeros (e.g., {missing_cols[:8]})")

    # final X matrix
    X = fut[features].copy()
    # ensure numeric dtypes first, then fill
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)


    # shape checks
    if X.shape[0] == 0:
        raise ValueError(
            "No future rows were produced. Try widening the date range or reducing min_recent_days in _pick_forecast_index()."
        )
    if X.shape[1] == 0:
        raise ValueError(
            "Feature matrix has 0 columns. Check logs/customer_features_used.json for a valid 'features_used' list."
        )

    yhat = model.predict(X)
    yhat = np.clip(yhat, 0, None)

    out = fut[["order_date", "customer_id", "product_id", "city"]].copy()
    out["pred_order_qty"] = yhat
    return out.sort_values(["order_date", "customer_id", "product_id", "city"]).reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict customer-level orders using trained model.")
    parser.add_argument("--start", required=False, default=None, help="YYYY-MM-DD (defaults: day after last history date)")
    parser.add_argument("--end",   required=False, default=None, help="YYYY-MM-DD (defaults: start + 13 days)")
    parser.add_argument("--customers", nargs="*", default=None, help="Optional list of customer_ids")
    parser.add_argument("--products",  nargs="*", default=None, help="Optional list of product_ids")
    args = parser.parse_args()

    hist = _load_history()
    last = hist["order_date"].max()
    start = args.start or (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end   = args.end   or (last + pd.Timedelta(days=14)).strftime("%Y-%m-%d")

    preds = predict_range(start, end, customers=args.customers, products=args.products)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / "customer_order_predictions.csv"
    preds.to_csv(out_path, index=False)
    log.info(f"Saved predictions -> {out_path} (rows={len(preds)})")
