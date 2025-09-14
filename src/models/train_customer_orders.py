# src/models/train_customer_orders.py
import os
import sys
import json
import logging
from math import sqrt
from pathlib import Path
from datetime import timedelta
from typing import List, Tuple, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
# add right at the top before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add repo root for imports (same pattern as train_model.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.constants import DATA_PATH  # uses your processed dataset path

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_customer_orders")

# ---------------- Paths ----------------
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")
PROCESSED_DIR = Path(DATA_PATH).resolve().parent

CUSTOMER_MODEL_PATH = MODELS_DIR / "customer_order_model.pkl"
METRICS_JSON_PATH = LOGS_DIR / "customer_order_metrics.json"
FEATURE_IMPORTANCE_CSV = LOGS_DIR / "customer_feature_importance.csv"
PRED_SCATTER_PNG = LOGS_DIR / "customer_pred_scatter.png"
RESIDUALS_PNG = LOGS_DIR / "customer_residuals.png"
FEATURES_USED_JSON = LOGS_DIR / "customer_features_used.json"

# Optional auxiliary (joined if present)
INV_SNAPSHOTS_PATH = PROCESSED_DIR / "kc_inventory_snapshots_complete_clean.csv"
PROMO_CAL_PATH = PROCESSED_DIR / "kc_promotional_calendar_complete_clean.csv"  # usually already reflected in is_promotional

# --- add right below imports (helper for key crosswalk) ---
CROSSWALK_PATH = PROCESSED_DIR / "sku_crosswalk.csv"

def _norm_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _canon_pid(s: pd.Series) -> pd.Series:
    s = _norm_str(s).str.replace(r'[^a-z0-9]', '', regex=True)
    return s.str.lstrip('0')

def _apply_crosswalk(df_left: pd.DataFrame, df_right: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optional mapping from orders product ids -> inventory product ids and/or city names.
    Expected columns (any subset): orders_product_id, inv_product_id, orders_city, inv_city
    """
    if not CROSSWALK_PATH.exists():
        logger.info("Crosswalk not found; skipping SKU/city mapping.")
        return df_left, df_right

    try:
        cw = pd.read_csv(CROSSWALK_PATH, low_memory=False)
        cols = {c.lower(): c for c in cw.columns}
        op = cols.get("orders_product_id")
        ip = cols.get("inv_product_id")
        oc = cols.get("orders_city")
        ic = cols.get("inv_city")

        if op and ip:
            m = cw[[op, ip]].dropna()
            m[op] = _canon_pid(m[op])
            m[ip] = _canon_pid(m[ip])
            # map left product_id -> right product_id domain
            df_left = df_left.copy()
            df_left["product_id"] = _canon_pid(df_left["product_id"])
            df_right = df_right.copy()
            df_right["product_id"] = _canon_pid(df_right["product_id"])
            # build dict: left -> right
            d = dict(zip(m[op], m[ip]))
            df_left["product_id"] = df_left["product_id"].map(lambda x: d.get(x, x))

        if oc and ic and ("city" in df_left.columns) and ("city" in df_right.columns):
            m2 = cw[[oc, ic]].dropna()
            m2[oc] = _norm_str(m2[oc])
            m2[ic] = _norm_str(m2[ic])
            d2 = dict(zip(m2[oc], m2[ic]))
            df_left["city"] = _norm_str(df_left["city"]).map(lambda x: d2.get(x, x))
            df_right["city"] = _norm_str(df_right["city"])  # ensure normalized

        logger.info("Applied SKU/city crosswalk mapping.")
        return df_left, df_right
    except Exception as e:
        logger.warning(f"Crosswalk load/apply failed; continuing without it. Error: {e}")
        return df_left, df_right


# ---------------- Data Loading ----------------
def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """
    Load the preprocessed dataset and ensure time types & basic cleanliness.
    We assume your preprocessing produced the columns listed in synthetic_columns.
    """
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    # Order date
    if "order_date" not in df.columns:
        raise ValueError("Expected 'order_date' in dataset")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"]).sort_values("order_date").reset_index(drop=True)

    # Ensure key identifiers exist (filled by your preprocessing)
    for col in ["customer_id", "product_id", "city"]:
        if col not in df.columns:
            logger.warning(f"Missing '{col}' in dataset; adding fallback.")
            df[col] = "Unknown"

    # Numeric target
    if "order_qty" not in df.columns:
        raise ValueError("Expected 'order_qty' column (target variable) in dataset")

    # A few handy casts/fills
    numeric_maybe = [
        "base_demand_qty", "promotional_demand_qty", "discount_percent",
        "price", "temperature", "humidity", "inflation"
    ]
    for c in numeric_maybe:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "is_promotional" in df.columns:
        df["is_promotional"] = df["is_promotional"].astype(int)

    logger.info(f"Loaded shape: {df.shape}")
    logger.info(f"Date range: {df['order_date'].min()} -> {df['order_date'].max()}")
    return df

def _canon_pid(s: pd.Series) -> pd.Series:
    # normalize product ids: lowercase, strip spaces, remove non-alnum, drop leading zeros
    s = s.astype(str).str.strip().str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    return s.str.lstrip('0')

def _pick_snapshot_date_column(inv: pd.DataFrame, df_orders: pd.DataFrame) -> Optional[str]:
    """Pick the inventory date column that overlaps the orders window the most."""
    df_min = pd.to_datetime(df_orders["order_date"], errors="coerce").min()
    df_max = pd.to_datetime(df_orders["order_date"], errors="coerce").max()

    candidates = []
    for c in inv.columns:
        lc = c.lower()
        if any(k in lc for k in ["date", "time", "timestamp", "updated", "asof", "as_of", "snapshot", "effective"]):
            ser = pd.to_datetime(inv[c], errors="coerce")
            if ser.notna().sum() == 0:
                continue
            inv_min, inv_max = ser.min(), ser.max()
            # fraction of non-null dates that fall within the orders window
            within = ((ser >= df_min) & (ser <= df_max)).mean()
            # prefer snapshot/asof/effective columns with a small bonus
            bonus = 0.1 if any(k in lc for k in ["asof", "as_of", "snapshot", "effective"]) else 0.0
            score = within + bonus
            candidates.append((c, inv_min, inv_max, within, score))

    if not candidates:
        logger.info("Inventory: no date-like columns found.")
        return None

    for c, mn, mx, within, score in candidates:
        logger.info(f"Inventory date candidate {c}: {mn} → {mx} | within-orders={within:.1%} | score={score:.3f}")

    # choose highest score; require some overlap to avoid leakage
    best = max(candidates, key=lambda t: t[4])
    col, mn, mx, within, score = best
    if within == 0.0:
        logger.warning("No inventory date column overlaps the orders time range; "
                       "skipping inventory join to avoid future-data leakage.")
        return None
    logger.info(f"Using inventory date column '{col}' ({mn} → {mx}).")
    return col

# --- replace your maybe_join_inventory with this version ---
def maybe_join_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    As-of join inventory snapshots (latest snapshot <= order_date) with robust key canonicalization.
    - Automatically tries multiple product-id candidates on the inventory side (sku/product_code/etc.)
    - Uses key cascade: (product, city, customer) → (product, city) → (product)
    - Chooses the best match-rate result
    - Skips join entirely if no inventory timestamp overlaps the orders window (prevents leakage)
    """
    if not INV_SNAPSHOTS_PATH.exists():
        logger.info("Inventory snapshots not found; skipping join.")
        return df

    try:
        inv = pd.read_csv(INV_SNAPSHOTS_PATH, low_memory=False)

        # ---- detect/prepare snapshot timestamp column ----
        date_cols = [c for c in inv.columns if "date" in c.lower() or "updated" in c.lower()]
        if not date_cols:
            logger.info("No date/timestamp column in inventory snapshots; skipping join.")
            return df
        snap_col = date_cols[0]
        inv[snap_col] = pd.to_datetime(inv[snap_col], errors="coerce")
        inv = inv.dropna(subset=[snap_col])

        # ---- pick the inventory date column that overlaps the orders window (if any) ----
        df_start, df_end = pd.to_datetime(df["order_date"]).min(), pd.to_datetime(df["order_date"]).max()
        inv_start, inv_end = inv[snap_col].min(), inv[snap_col].max()
        logger.info(f"INV debug: DF dates {df_start} → {df_end} | INV dates {inv_start} → {inv_end}")

        # If inventory entirely after orders (or entirely before), we skip to avoid leakage
        overlap = (inv_end >= df_start) and (inv_start <= df_end)
        if not overlap:
            logger.warning("No inventory date column overlaps the orders time range; skipping inventory join to avoid future-data leakage.")
            return df

        # ---- normalize/alias join keys on inventory ----
        key_map = {
            "sku_id": "sku_id", "SKU_ID": "sku_id", "sku": "sku_id", "SKU": "sku_id",
            "product_code": "product_code", "Product Code": "product_code", "SKU Code": "product_code",
            "location": "city", "city_code": "city", "City": "city", "site": "city", "dc": "city", "warehouse": "city",
            "customer": "customer_id", "customer_code": "customer_id", "Customer": "customer_id"
        }
        # copy aliases without clobbering existing columns
        if "city" not in inv.columns:
            for src in ["location", "city_code", "City", "site", "dc", "warehouse"]:
                if src in inv.columns:
                    inv["city"] = inv[src]
                    break
        if "customer_id" not in inv.columns:
            for src in ["customer", "customer_code", "Customer"]:
                if src in inv.columns:
                    inv["customer_id"] = inv[src]
                    break

        # ---- candidate product-id columns on inventory side ----
        pid_candidates = []
        for c in ["product_id", "sku_id", "SKU_ID", "sku", "SKU", "product_code", "Product Code", "SKU Code"]:
            if c in inv.columns:
                pid_candidates.append(c)
        if not pid_candidates:
            logger.warning("No recognizable product-id columns in inventory; skipping join.")
            return df

        # ---- canonicalizers ----
        def norm(s: pd.Series) -> pd.Series:
            return s.astype(str).str.strip().str.lower()

        def canon_pid(s: pd.Series) -> pd.Series:
            # strip non-alnum and leading zeros
            s = norm(s).str.replace(r'[^a-z0-9]', '', regex=True)
            return s.str.lstrip('0')

        # ---- left prep ----
        out = df.copy()
        out["order_date"] = pd.to_datetime(out["order_date"], errors="coerce")
        out = out.dropna(subset=["order_date"]).copy()
        out["row_id"] = np.arange(len(out))

        if "product_id" not in out.columns:
            out["product_id"] = "unknown"
        out["product_id"] = canon_pid(out["product_id"])

        if "city" not in out.columns:
            out["city"] = "unknown"
        out["city"] = norm(out["city"])

        if "customer_id" in out.columns:
            out["customer_id"] = norm(out["customer_id"])

        # ---- helper: run one asof cascade with a specific inventory product column ----
        TOL_DAYS = 90

        def _match_rate(_merged: pd.DataFrame) -> float:
            cols = ["customer_inventory_qty","current_inventory_units","warehouse_inventory_qty","in_transit_qty","backorder_qty"]
            present = [c for c in cols if c in _merged.columns]
            if not present:
                return 0.0
            return float(_merged[present].notna().any(axis=1).mean())

        def _asof(left_df, right_df, keys: List[str]) -> pd.DataFrame:
            try:
                L = left_df.sort_values(keys + ["order_date"], kind="mergesort").reset_index(drop=True)
                R = right_df.sort_values(keys + [snap_col],   kind="mergesort").reset_index(drop=True)
                return pd.merge_asof(
                    L, R,
                    left_on="order_date", right_on=snap_col,
                    by=keys, direction="backward",
                    tolerance=pd.Timedelta(days=TOL_DAYS)
                )
            except Exception as e:
                logger.warning(f"merge_asof failed for {keys} ({e}); using groupwise fallback.")
                parts = []
                for key_vals, lg in left_df.groupby(keys, sort=False, dropna=False):
                    if not isinstance(key_vals, tuple):
                        key_vals = (key_vals,)
                    mask = pd.Series(True, index=right_df.index)
                    for col, val in zip(keys, key_vals):
                        mask &= (right_df[col] == ("" if pd.isna(val) else val))
                    rg = right_df.loc[mask]
                    if rg.empty:
                        parts.append(lg); continue
                    lg = lg.sort_values("order_date", kind="mergesort")
                    rg = rg.sort_values(snap_col,   kind="mergesort")
                    parts.append(pd.merge_asof(
                        lg, rg,
                        left_on="order_date", right_on=snap_col,
                        direction="backward", tolerance=pd.Timedelta(days=TOL_DAYS)
                    ))
                return pd.concat(parts, ignore_index=True)

        best_overall = None
        best_rate = -1.0
        best_keys = None
        best_pidcol = None

        for pidcol in pid_candidates:
            inv_work = inv.copy()
            inv_work["pid_join"] = canon_pid(inv_work[pidcol])

            # right-side normalization
            if "city" in inv_work.columns:
                inv_work["city"] = norm(inv_work["city"])
            if "customer_id" in inv_work.columns:
                inv_work["customer_id"] = norm(inv_work["customer_id"])

            # set left pid to pid_join name
            left = out.rename(columns={"product_id": "pid_join"}).copy()

            # key cascades
            candidates = []
            if "customer_id" in inv_work.columns and "customer_id" in left.columns:
                candidates.append(["pid_join", "city", "customer_id"])
            if "city" in inv_work.columns:
                candidates.append(["pid_join", "city"])
            candidates.append(["pid_join"])

            for keys in candidates:
                m = _asof(left, inv_work, keys)
                if snap_col in m.columns:
                    m = m.drop(columns=[snap_col], errors="ignore")
                rate = _match_rate(m)
                logger.info(f"Inventory match rate with pid={pidcol} & keys={keys}: {rate:.1%} (tol={TOL_DAYS}d)")
                if rate > best_rate:
                    best_overall, best_rate, best_keys, best_pidcol = m, rate, keys, pidcol

        if best_overall is None:
            logger.info("Inventory join produced no result; returning original df.")
            return df

        out = (best_overall
               .rename(columns={"pid_join": "product_id"})
               .sort_values("row_id")
               .drop(columns=["row_id"], errors="ignore")
               .reset_index(drop=True))

        if best_rate <= 0.0:
            logger.warning("Inventory as-of match rate is 0%; verify product IDs (codes vs names) and city labels.")
        else:
            logger.info(f"Using inventory join with pid={best_pidcol} and keys={best_keys} (match rate {best_rate:.1%}).")

        return out

    except Exception as e:
        logger.warning(f"Inventory as-of join failed; continuing without it. Error: {e}")
        return df



# --- helper: optional static fallback (NOT leakage-safe) ---
def _join_inventory_static(df_left: pd.DataFrame, inv_right: pd.DataFrame) -> pd.DataFrame:
    """
    Join per-(product_id, city) static medians as *_static features when temporal overlap is unavailable.
    WARNING: This uses whatever snapshots exist (possibly future vs training); not leakage-safe for model training.
    """
    cols = ["product_id","city","customer_inventory_qty","current_inventory_units",
            "warehouse_inventory_qty","in_transit_qty","backorder_qty"]
    have = [c for c in cols if c in inv_right.columns]
    if not set(["product_id","city"]).issubset(set(have)):
        logger.info("Static fallback impossible (missing product_id/city).")
        return df_left

    tmp = inv_right.copy()
    tmp["product_id"] = _canon_pid(tmp["product_id"])
    tmp["city"] = _norm_str(tmp["city"])
    grp = tmp.groupby(["product_id","city"], dropna=False).median(numeric_only=True)
    grp = grp.add_suffix("_static").reset_index()

    out = df_left.copy()
    if "product_id" not in out.columns: out["product_id"] = "unknown"
    if "city" not in out.columns: out["city"] = "unknown"
    out["product_id"] = _canon_pid(out["product_id"])
    out["city"] = _norm_str(out["city"])

    out = out.merge(grp, on=["product_id","city"], how="left")
    logger.warning("STATIC inventory features joined ( *_static ). This is NOT leakage-safe. "
                   "Use only if you understand the implications.")
    return out


def maybe_join_promotions_asof(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag promo activity on order_date (+discount if available).
    Uses fuzzy column detection for product/channel/start/end/discount and normalizes keys.
    """
    if not PROMO_CAL_PATH.exists():
        logger.info("Promo calendar not found; skipping promo join.")
        df["is_promo_day"] = 0
        df["disc_pct"] = 0.0
        return df

    try:
        import re
        promo = pd.read_csv(PROMO_CAL_PATH, low_memory=False)

        # ---- detect product column ----
        prod_col = None
        for c in promo.columns:
            if re.search(r'(product|sku).*(id|code|number)?', c, flags=re.I):
                prod_col = c
                break
        if prod_col is None:
            logger.warning("Promo file has no recognizable product column; skipping promo join.")
            df["is_promo_day"] = 0
            df["disc_pct"] = 0.0
            return df
        if prod_col != "product_id":
            promo["product_id"] = promo[prod_col]

        # ---- detect channel column (optional) ----
        ch_col = None
        for c in promo.columns:
            if re.search(r'channel', c, flags=re.I):
                ch_col = c
                break
        if ch_col and ch_col != "channel":
            promo["channel"] = promo[ch_col]

        # ---- detect start/end date columns ----
        start_col = None
        end_col = None
        for c in promo.columns:
            if re.search(r'(start).*date', c, flags=re.I) or re.search(r'promo.*start', c, flags=re.I):
                start_col = c; break
        for c in promo.columns:
            if re.search(r'(end).*date', c, flags=re.I) or re.search(r'promo.*end', c, flags=re.I):
                end_col = c; break
        if not start_col or not end_col:
            logger.warning("Promo file missing start/end dates; skipping promo join.")
            df["is_promo_day"] = 0
            df["disc_pct"] = 0.0
            return df

        promo[start_col] = pd.to_datetime(promo[start_col], errors="coerce")
        promo[end_col]   = pd.to_datetime(promo[end_col], errors="coerce")
        promo = promo.dropna(subset=[start_col, end_col])

        # ---- discount column (optional) ----
        disc_col = None
        for c in promo.columns:
            if re.search(r'(discount|disc|%off|markdown)', c, flags=re.I):
                disc_col = c; break

        # ---- normalize keys on both sides ----
        def _norm(s: pd.Series) -> pd.Series:
            return s.astype(str).str.strip().str.lower()

        if "product_id" not in df.columns:
            df["product_id"] = "unknown"
        df["product_id"] = _norm(df["product_id"])
        promo["product_id"] = _norm(promo["product_id"])

        use_channel = ("channel" in promo.columns) and ("channel" in df.columns)
        if use_channel:
            df["channel"] = _norm(df["channel"])
            promo["channel"] = _norm(promo["channel"])

        # ---- build row ids & join keys ----
        out = df.reset_index(drop=False).rename(columns={"index": "row_id"}).copy()
        out["order_date"] = pd.to_datetime(out["order_date"], errors="coerce")

        join_keys = ["product_id", "channel"] if use_channel else ["product_id"]
        merged = out.merge(promo, on=join_keys, how="left", suffixes=("", "_p"))

        active = (merged["order_date"] >= merged[start_col]) & (merged["order_date"] <= merged[end_col])

        cols = ["row_id"]
        if disc_col:
            cols.append(disc_col)
        hits = merged.loc[active, cols].copy()

        if disc_col:
            hits["disc_pct"] = pd.to_numeric(hits[disc_col], errors="coerce").fillna(0.0)
        else:
            hits["disc_pct"] = 1.0  # indicator if we don't have explicit discount

        promo_by_row = hits.groupby("row_id", as_index=False)["disc_pct"].max() if len(hits) else pd.DataFrame({"row_id": [], "disc_pct": []})

        out = out.merge(promo_by_row, on="row_id", how="left")
        out["disc_pct"] = out["disc_pct"].fillna(0.0)
        out["is_promo_day"] = (out["disc_pct"] > 0).astype(int)

        out = out.sort_values("row_id").drop(columns=["row_id"], errors="ignore").reset_index(drop=True)
        return out

    except Exception as e:
        logger.warning(f"Promo join failed; setting zero flags. Error: {e}")
        df["is_promo_day"] = 0
        df["disc_pct"] = 0.0
        return df



# ---------------- Feature Engineering ----------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("order_date").reset_index(drop=True)
    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_day"] = df["order_date"].dt.day
    df["order_day_of_week"] = df["order_date"].dt.dayofweek
    df["quarter"] = df["order_date"].dt.quarter
    df["is_month_start"] = df["order_date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["order_date"].dt.is_month_end.astype(int)
    df["is_weekend"] = (df["order_day_of_week"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lags/rolls scoped by (customer_id, product_id). Uses only past info to avoid leakage.
    """
    df = df.sort_values(["customer_id", "product_id", "order_date"]).reset_index(drop=True)

    # Quantity-based signals (target is order_qty)
    df["qty_lag_1"] = df.groupby(["customer_id", "product_id"])["order_qty"].shift(1)
    df["qty_lag_7"] = df.groupby(["customer_id", "product_id"])["order_qty"].shift(7)
    df["qty_rolling_mean_7"] = (
        df.groupby(["customer_id", "product_id"])["order_qty"]
          .rolling(window=7, min_periods=1)
          .mean()
          .reset_index(level=[0, 1], drop=True)
    )
    df["qty_rolling_std_7"] = (
        df.groupby(["customer_id", "product_id"])["order_qty"]
          .rolling(window=7, min_periods=1)
          .std()
          .reset_index(level=[0, 1], drop=True)
    )

    # Revenue-derived lags (if revenue available)
    if "revenue" in df.columns:
        df["revenue_lag_1"] = df.groupby(["customer_id", "product_id"])["revenue"].shift(1)
        df["revenue_lag_7"] = df.groupby(["customer_id", "product_id"])["revenue"].shift(7)
        df["revenue_rolling_mean_7"] = (
            df.groupby(["customer_id", "product_id"])["revenue"]
              .rolling(window=7, min_periods=1)
              .mean()
              .reset_index(level=[0, 1], drop=True)
        )
        df["revenue_rolling_std_7"] = (
            df.groupby(["customer_id", "product_id"])["revenue"]
              .rolling(window=7, min_periods=1)
              .std()
              .reset_index(level=[0, 1], drop=True)
        )

    # SAFE lags for potential same-day fields (prevents leakage)
    if "discount_percent" in df.columns:
        df["discount_percent_lag_1"] = df.groupby(["customer_id", "product_id"])["discount_percent"].shift(1)
    if "price" in df.columns:
        df["price_lag_1"] = df.groupby(["customer_id", "product_id"])["price"].shift(1)

    # Fill NaNs created by shifts/rolls
    lag_fill_cols = [
        "qty_lag_1","qty_lag_7","qty_rolling_mean_7","qty_rolling_std_7",
        "revenue_lag_1","revenue_lag_7","revenue_rolling_mean_7","revenue_rolling_std_7",
        "discount_percent_lag_1","price_lag_1"
    ]
    for c in lag_fill_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0.0)

    return df


def add_inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived 'inventory-like' features using only past information from orders/prices.
    Still computes qty_roll14 / days_of_supply, and adds proxies:
      - days_since_pos_order
      - zero_streak
      - zero_share_28
      - qty_trend_14
      - qty_cv_28
      - price_ratio_28
      - prev_spike_7
    Index-safe: preserves original order.
    """
    df = df.copy()

    # Preserve original order
    orig_col = "__orig_idx__"
    df[orig_col] = np.arange(len(df))

    # Ensure required columns exist
    for k in ["customer_id", "product_id"]:
        if k not in df.columns:
            df[k] = "Unknown"

    # Ensure usable dtypes
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["order_qty"]  = pd.to_numeric(df.get("order_qty", 0.0), errors="coerce").fillna(0.0)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Sort for groupwise time ops
    df = df.sort_values(["customer_id", "product_id", "order_date"]).reset_index(drop=True)
    gkeys = ["customer_id", "product_id"]
    grp = df.groupby(gkeys, sort=False)

    # --- Core past-only rolling demand ---
    df["qty_roll14"] = grp["order_qty"].transform(
        lambda s: s.shift(1).rolling(14, min_periods=3).mean()
    ).fillna(0.0)

    # On-hand proxy if not present (sum of any available inventory cols, else 0)
    if "on_hand_qty" not in df.columns:
        oh_candidates = ["customer_inventory_qty", "current_inventory_units", "warehouse_inventory_qty"]
        oh_cols = [c for c in oh_candidates if c in df.columns]
        df["on_hand_qty"] = df[oh_cols].sum(axis=1) if oh_cols else 0.0

    # Days of supply (bounded)
    df["days_of_supply"] = (df["on_hand_qty"] / (df["qty_roll14"] + 1e-6)).clip(0, 365)

    # Backorder & stockout flag (robust defaults)
    if "backorder_qty" not in df.columns:
        df["backorder_qty"] = 0.0
    df["stockout_flag"] = (df["on_hand_qty"] <= 0).astype(int)

    # --- Inventory-like proxies (past only) ---

    # 1) Days since last positive order (previous day or earlier)
    prev_pos_mask = grp["order_qty"].shift(1).gt(0)
    prev_pos_date = df["order_date"].where(prev_pos_mask)
    last_prev_pos = prev_pos_date.groupby([df[k] for k in gkeys]).ffill()
    df["days_since_pos_order"] = (df["order_date"] - last_prev_pos).dt.days
    df["days_since_pos_order"] = df["days_since_pos_order"].fillna(999).clip(0, 999).astype(int)

    # 2) Consecutive zero streak ending yesterday (fill before cast to avoid NaN→int error)
    def _zero_streak_prev(s: pd.Series) -> pd.Series:
        b = s.shift(1).eq(0).fillna(False).to_numpy()
        out = np.zeros(len(b), dtype=int)
        run = 0
        for i, z in enumerate(b):
            run = run + 1 if z else 0
            out[i] = run
        return pd.Series(out, index=s.index)

    zstreak = grp["order_qty"].transform(_zero_streak_prev)
    df["zero_streak"] = zstreak.fillna(0).astype(int)

    # 3) Share of zero days in the last 28 (past only)
    df["zero_share_28"] = grp["order_qty"].transform(
        lambda s: s.shift(1).eq(0).rolling(28, min_periods=7).mean()
    ).fillna(0.0)

    # 4) Trend proxy over 14 days: 7d mean (past) minus prior 7d mean
    m7 = grp["order_qty"].transform(lambda s: s.shift(1).rolling(7, min_periods=4).mean())
    df["qty_trend_14"] = (m7 - m7.shift(7)).fillna(0.0)

    # 5) Volatility proxy: coefficient of variation over last 28 days (past only)
    mean28 = grp["order_qty"].transform(lambda s: s.shift(1).rolling(28, min_periods=7).mean())
    std28  = grp["order_qty"].transform(lambda s: s.shift(1).rolling(28, min_periods=7).std())
    df["qty_cv_28"] = (std28 / (mean28 + 1e-6)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 10)

    # 6) Price ratio vs 28d median (past only) — if price exists
    if "price" in df.columns:
        med28 = grp["price"].transform(lambda s: s.shift(1).rolling(28, min_periods=7).median())
        df["price_ratio_28"] = (df["price"].shift(1) / (med28 + 1e-6)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    else:
        df["price_ratio_28"] = 1.0  # neutral

    # 7) Was yesterday a spike vs its own prior 7d mean? (fill before cast)
    prev = grp["order_qty"].transform(lambda s: s.shift(1))
    prev_base = grp["order_qty"].transform(lambda s: s.shift(2).rolling(7, min_periods=4).mean())
    spike_bool = (prev > (1.5 * (prev_base + 1e-6))).fillna(False)
    df["prev_spike_7"] = spike_bool.astype(int)

    # Restore original order
    df = df.sort_values(orig_col).drop(columns=[orig_col], errors="ignore").reset_index(drop=True)
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    # Calendar / encodings
    base_numeric = [
        "order_year","order_month","order_day","order_day_of_week","quarter",
        "is_month_start","is_month_end","is_weekend","city_encoded","category_encoded"
    ]

    # Promo & pricing (safe)
    promo_features = [
        "is_promo_day","disc_pct",          # from promo calendar (our join already guards)
        "discount_percent_lag_1",           # created in add_lag_features
        "price_lag_1"                       # created in add_lag_features
    ]

    # External factors
    external_features = ["temperature","humidity","inflation"]

    # Inventory / behavior features (raw if present + proxies we just added)
    inventory_like = [
        "on_hand_qty","qty_roll14","days_of_supply","stockout_flag",
        "customer_inventory_qty","current_inventory_units",
        "warehouse_inventory_qty","in_transit_qty","backorder_qty",
        # new proxies:
        "days_since_pos_order","zero_streak","zero_share_28",
        "qty_trend_14","qty_cv_28","price_ratio_28","prev_spike_7",
    ]

    # Delivery/channel binaries
    binary_features = ["is_late","channel_Retail","customer_type_Enterprise","country_USA"]
    delivery_onehot = [c for c in df.columns if c.startswith("delivery_status_")]

    # Lags & rolls for demand/revenue
    lag_features = [
        "qty_lag_1","qty_lag_7","qty_rolling_mean_7","qty_rolling_std_7",
        "revenue_lag_1","revenue_lag_7","revenue_rolling_mean_7","revenue_rolling_std_7"
    ]

    candidate_features = (
        base_numeric + promo_features + external_features +
        inventory_like + binary_features + delivery_onehot + lag_features
    )

    available = [c for c in candidate_features if c in df.columns]
    missing = [c for c in candidate_features if c not in df.columns]
    if missing:
        logger.info(f"Skipping {len(missing)} non-existent feature(s): {missing[:10]}{' ...' if len(missing)>10 else ''}")

    X = df[available].copy()
    y = df["order_qty"].astype(float).clip(lower=0)

    # Final NaN guard
    X = X.fillna(X.median(numeric_only=True))

    return X, y, available



# ---------------- Training & Evaluation ----------------
def chronological_split(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    test_ratio: float = 0.2,
    gap_days: int = 7
):
    """
    Chronological split using unique dates with guardrails:
      - uses a gap to reduce look-ahead
      - falls back if the split produces empty sets
      - logs the chosen split date
    """
    assert "order_date" in df.columns

    # Unique, sorted dates
    dates = pd.to_datetime(df["order_date"]).dropna().sort_values().unique()

    # If too few dates, fall back to simple index split
    if dates.size < 10:
        logger.warning("Very few unique dates; using index-based split without gap.")
        n = len(df)
        ix = int(n * (1 - test_ratio))
        return X.iloc[:ix], X.iloc[ix:], y.iloc[:ix], y.iloc[ix:]

    # Choose split date by ratio, clamp to valid range
    split_ix = int(dates.size * (1 - test_ratio))
    split_ix = max(1, min(split_ix, dates.size - 1))
    split_date = pd.Timestamp(dates[split_ix])
    train_end = split_date - pd.Timedelta(days=gap_days)

    # Primary: gap between train end and test start
    train_mask = df["order_date"] < train_end
    test_mask  = df["order_date"] >= split_date

    # If either side is empty, retry with no gap
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        logger.warning(f"Empty train/test after date+gap split (gap_days={gap_days}). Retrying with gap_days=0.")
        train_mask = df["order_date"] < split_date
        test_mask  = df["order_date"] >= split_date

    # If still empty, final fallback: index split
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        logger.warning("Still empty after removing gap; falling back to index-based split.")
        n = len(df)
        ix = int(n * (1 - test_ratio))
        return X.iloc[:ix], X.iloc[ix:], y.iloc[:ix], y.iloc[ix:]

    logger.info(
        f"Split date: {split_date.date()} (gap_days={gap_days}), "
        f"train={int(train_mask.sum())}, test={int(test_mask.sum())}"
    )

    return X.loc[train_mask], X.loc[test_mask], y.loc[train_mask], y.loc[test_mask]



def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="poisson",             # counts
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=0.4,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="poisson",
        callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(200)]
    )
    return model

def _smape(y_true, y_pred) -> float:
    """Symmetric MAPE (robust when counts include zeros)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0.0, 1.0, denom)  # avoid /0
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

def evaluate(model: lgb.LGBMRegressor, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Predict
    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)

    # Poisson objective should be >=0; clip just in case for metric stability
    y_pred_tr = np.clip(y_pred_tr, 0, None)
    y_pred_te = np.clip(y_pred_te, 0, None)

    # Core metrics
    train_mae  = mean_absolute_error(y_train, y_pred_tr)
    test_mae   = mean_absolute_error(y_test,  y_pred_te)
    train_rmse = sqrt(mean_squared_error(y_train, y_pred_tr))
    test_rmse  = sqrt(mean_squared_error(y_test,  y_pred_te))
    train_r2   = r2_score(y_train, y_pred_tr)
    test_r2    = r2_score(y_test,  y_pred_te)

    # SMAPE (scale-free, good for counts)
    train_smape = _smape(y_train, y_pred_tr)
    test_smape  = _smape(y_test,  y_pred_te)

    logger.info("=== Customer Orders Model Performance (target: order_qty) ===")
    logger.info(
        f"Train: MAE={train_mae:.3f} RMSE={train_rmse:.3f} R²={train_r2:.3f} SMAPE={train_smape:.3f}"
    )
    logger.info(
        f"Test : MAE={test_mae:.3f} RMSE={test_rmse:.3f} R²={test_r2:.3f} SMAPE={test_smape:.3f}"
    )

    return {
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "train_smape": train_smape,
        "test_smape": test_smape,
    }


def save_artifacts(model: lgb.LGBMRegressor, features: List[str], metrics: Dict[str, Any], X_test, y_test):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Model
    joblib.dump(model, CUSTOMER_MODEL_PATH)
    logger.info(f"Saved model -> {CUSTOMER_MODEL_PATH}")

    # Metrics (rounded)
    metrics_out = {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info(f"Saved metrics -> {METRICS_JSON_PATH}")

    # Features used
    with open(FEATURES_USED_JSON, "w") as f:
        json.dump({"features_used": features}, f, indent=2)
    logger.info(f"Saved features used -> {FEATURES_USED_JSON}")

    # Feature importance
    try:
        fi = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=False
        )
        fi.to_csv(FEATURE_IMPORTANCE_CSV, index=False)
        logger.info(f"Saved feature importance -> {FEATURE_IMPORTANCE_CSV}")
    except Exception as e:
        logger.warning(f"Could not save feature importance: {e}")

    # ---- Extra: feature importance by GAIN (average loss reduction) ----
    try:
        # Get the underlying Booster no matter which wrapper version is installed
        booster = getattr(model, "booster_", None) or getattr(model, "booster", None)
        if booster is None:
            raise AttributeError("LightGBM booster not found on model (is it fitted?).")

        # Pull names directly from the booster to avoid any column order mismatch
        try:
            feat_names = booster.feature_name()      # preferred (method)
        except TypeError:
            # Older versions expose it as a property/attribute
            feat_names = getattr(booster, "feature_name", None) or getattr(booster, "feature_name_", None)

        if not feat_names:
            # Fallback to the features list we passed into save_artifacts
            feat_names = list(features)

        # Compute importance by gain; keep it simple for maximum compatibility
        gain = booster.feature_importance(importance_type="gain")

        # Align lengths defensively
        if len(gain) != len(feat_names):
            logger.warning(f"Gain importance length ({len(gain)}) != feature name length ({len(feat_names)}). "
                        "Falling back to features list passed into save_artifacts.")
            feat_names = list(features)
            # If still mismatched, truncate/pad so we can save something sane
            if len(gain) != len(feat_names):
                m = min(len(gain), len(feat_names))
                gain = gain[:m]
                feat_names = feat_names[:m]

        fi_gain = (
            pd.DataFrame({"feature": feat_names, "gain": gain})
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )
        out_path = LOGS_DIR / "customer_feature_importance_gain.csv"
        fi_gain.to_csv(out_path, index=False)
        logger.info(f"Saved feature importance (gain) -> {out_path}")

        # Optional: quick top-10 log for convenience
        top10 = fi_gain.head(10).to_string(index=False)
        logger.info("Top-10 gain features:\n" + top10)

    except Exception as e:
        logger.warning(f"Could not save gain importances: {e}")


    # Plots
    try:
        y_pred = model.predict(X_test)
        # Scatter
        plt.figure(figsize=(9, 6))
        plt.scatter(y_test, y_pred, alpha=0.45)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        plt.plot(lims, lims, "r--", linewidth=1.5)
        plt.xlabel("Actual Orders (order_qty)")
        plt.ylabel("Predicted Orders")
        plt.title("Customer-level Orders: Actual vs Predicted")
        plt.tight_layout()
        plt.savefig(PRED_SCATTER_PNG, dpi=220, bbox_inches="tight")
        plt.close()

        # Residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(9, 6))
        plt.scatter(y_pred, residuals, alpha=0.45)
        plt.axhline(0, color="r", linestyle="--", linewidth=1)
        plt.xlabel("Predicted Orders")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted (Customer Orders)")
        plt.tight_layout()
        plt.savefig(RESIDUALS_PNG, dpi=220, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plots -> {PRED_SCATTER_PNG}, {RESIDUALS_PNG}")
    except Exception as e:
        logger.warning(f"Plotting failed (skipping): {e}")

def debug_inventory_overlap(df: pd.DataFrame):
    if not INV_SNAPSHOTS_PATH.exists():
        logger.info("INV debug: snapshots file not found.")
        return
    inv = pd.read_csv(INV_SNAPSHOTS_PATH, low_memory=False)

    # pick product column on inv
    prod_candidates = ["product_id","sku_id","SKU_ID","sku","SKU","product_code","Product Code","SKU Code"]
    prod_inv = next((c for c in prod_candidates if c in inv.columns), None)
    if prod_inv is None:
        logger.info("INV debug: no recognizable product column on inventory file.")
        return

    # normalize helpers
    def norm(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.lower()
    def canon_pid(s: pd.Series) -> pd.Series:
        # remove non-alnum, strip leading zeros
        s = norm(s).str.replace(r'[^a-z0-9]', '', regex=True)
        return s.str.lstrip('0')

    # build normalized keys
    left_pid = canon_pid(df.get("product_id", "unknown"))
    right_pid = canon_pid(inv[prod_inv])

    left_city = norm(df.get("city", "unknown"))
    right_city = norm(inv[inv.columns[inv.columns.str.lower().isin(["city","city_code","location","site","dc","warehouse"])][0]]) \
        if any(c in inv.columns.str.lower() for c in ["city","city_code","location","site","dc","warehouse"]) else pd.Series([], dtype=str)

    # coverage stats
    pid_overlap = len(set(left_pid.unique()) & set(right_pid.unique()))
    logger.info(f"INV debug: product_id overlap count={pid_overlap} "
                f"(left_nuniq={left_pid.nunique()}, right_nuniq={right_pid.nunique()})")

    if len(right_city) > 0:
        city_overlap = len(set(left_city.unique()) & set(right_city.unique()))
        logger.info(f"INV debug: city overlap count={city_overlap} "
                    f"(left_nuniq={left_city.nunique()}, right_nuniq={right_city.nunique()})")
    else:
        logger.info("INV debug: inventory file has no recognizable city/location column; join may need product-only keys.")

    # date coverage
    df_dates = pd.to_datetime(df["order_date"], errors="coerce")
    inv_date_cols = [c for c in inv.columns if "date" in c.lower() or "updated" in c.lower()]
    if inv_date_cols:
        inv_dates = pd.to_datetime(inv[inv_date_cols[0]], errors="coerce").dropna()
        logger.info(f"INV debug: DF dates {df_dates.min()} → {df_dates.max()} | "
                    f"INV dates {inv_dates.min()} → {inv_dates.max()}")


# ---------------- Main ----------------
def main():
    if not Path(DATA_PATH).exists():
        logger.error(f"❌ Dataset not found at {DATA_PATH}")
        sys.exit(1)

    # 1) Load core data
    df = load_and_prepare_data(DATA_PATH)

    debug_inventory_overlap(df)

    # 2) Inventory snapshots (as-of)
    df = maybe_join_inventory(df)

    # 3) Inventory-derived features (DoS, stockout, etc.)
    df = add_inventory_features(df)

    # 3.5) Promotions active on the day (safe)
    df = maybe_join_promotions_asof(df)

    # 4) Time & lag features (use only past info)
    df = add_time_features(df)
    df = add_lag_features(df)

    # 5) Select features / target (leakage-safe)
    X, y, features = prepare_features(df)

    LEAKY = {"order_qty","revenue","discount_percent","price","base_demand_qty","promotional_demand_qty"}
    bad = LEAKY.intersection(set(features))
    if bad:
        raise RuntimeError(f"Leakage guard tripped: these same-day cols are in features: {sorted(bad)}")

    # 6) Chronological split with gap
    X_train, X_test, y_train, y_test = chronological_split(df, X, y, test_ratio=0.2, gap_days=7)

    # 7) Train (Poisson + early stopping)
    model = train_model(X_train, y_train, X_test, y_test)

    # 8) Evaluate & save
    metrics = evaluate(model, X_train, y_train, X_test, y_test)

    # 8.1) Per-customer SMAPE leaderboard (test set only)
    # 8.1) Per-customer SMAPE leaderboard (test set only, guarded)
    if len(X_test) > 0:
        if "customer_id" in df.columns:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)

            # Safe slice: only use columns that exist
            cols = ["customer_id"]
            cols = [c for c in cols if c in df.columns]
            df_test = df.loc[X_test.index, cols].copy()

            # True / predicted
            df_test["y_true"] = y_test.values if hasattr(y_test, "values") else y_test
            # Clip negatives for count targets (Poisson objective)
            y_pred_test = np.clip(model.predict(X_test), a_min=0, a_max=None)
            df_test["y_pred"] = y_pred_test

            # SMAPE per customer (optionally keep groups with enough rows)
            grp = df_test.groupby("customer_id", dropna=False)
            cust_smape = grp.apply(lambda s: _smape(s["y_true"].to_numpy(), s["y_pred"].to_numpy()), include_groups=False)
            cust_cnt = grp.size().rename("n_rows")
            leaderboard = pd.concat([cust_smape.rename("smape"), cust_cnt], axis=1).sort_values("smape")

            # Save leaderboards
            leaderboard.to_csv(LOGS_DIR / "customer_smape_all.csv")
            leaderboard[leaderboard["n_rows"] >= 10].head(20).to_csv(LOGS_DIR / "customer_smape_best20.csv")
            leaderboard[leaderboard["n_rows"] >= 10].tail(20).to_csv(LOGS_DIR / "customer_smape_worst20.csv")

            logger.info("Saved per-customer SMAPE leaderboards in logs/")
        else:
            logger.warning("`customer_id` not found; skipping per-customer SMAPE leaderboard.")
    else:
        logger.info("Test set empty; skipping per-customer SMAPE leaderboard.")


    save_artifacts(model, features, metrics, X_test, y_test)

    logger.info("✅ Customer-level order model training completed successfully.")


if __name__ == "__main__":
    main()
