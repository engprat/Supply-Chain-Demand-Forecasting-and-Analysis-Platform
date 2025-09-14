# src/data/build_sku_crosswalk.py
import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.constants import DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("build_sku_crosswalk")

PROCESSED_DIR = Path(DATA_PATH).resolve().parent
INV_PATH = PROCESSED_DIR / "kc_inventory_snapshots_complete_clean.csv"
OUT_PATH = PROCESSED_DIR / "sku_crosswalk.csv"

def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _canon_pid(s: pd.Series) -> pd.Series:
    s = _norm(s).str.replace(r"[^a-z0-9]", "", regex=True)
    return s.str.lstrip("0")

def top_fuzzy_match(src_vals, tgt_vals, min_ratio=0.92):
    """Simple difflib top1 matcher returning dict src->(tgt, score)."""
    out = {}
    tgt = list(tgt_vals)
    for s in src_vals:
        best = None
        best_score = 0.0
        for t in tgt:
            r = SequenceMatcher(None, s, t).ratio()
            if r > best_score:
                best_score, best = r, t
        if best and best_score >= min_ratio:
            out[s] = (best, best_score)
    return out

def main():
    orders = pd.read_csv(DATA_PATH, low_memory=False)
    if "product_id" not in orders.columns:
        raise ValueError("Orders dataset missing product_id")
    orders_pid_raw = orders["product_id"].dropna().astype(str).unique()
    orders_pid_c = pd.Series(orders_pid_raw, name="orders_product_id_raw")
    orders_pid_c = orders_pid_c.to_frame()
    orders_pid_c["canon_orders_pid"] = _canon_pid(orders_pid_c["orders_product_id_raw"])

    if not INV_PATH.exists():
        raise FileNotFoundError(f"Inventory snapshots not found: {INV_PATH}. Run build_inventory_history first.")
    inv = pd.read_csv(INV_PATH, low_memory=False)
    if "product_id" not in inv.columns:
        raise ValueError("Inventory snapshots missing product_id")
    inv_pid_raw = inv["product_id"].dropna().astype(str).unique()
    inv_pid_c = pd.Series(inv_pid_raw, name="inv_product_id_raw").to_frame()
    inv_pid_c["canon_inv_pid"] = _canon_pid(inv_pid_c["inv_product_id_raw"])

    # 1) exact canonical matches
    exact = (
        orders_pid_c.merge(inv_pid_c, left_on="canon_orders_pid", right_on="canon_inv_pid", how="inner")
        .assign(method="exact_canonical", score=1.0)
    )

    # 2) fuzzy (only for those not matched yet)
    remaining = orders_pid_c[~orders_pid_c["canon_orders_pid"].isin(exact["canon_orders_pid"])]
    tgt = set(inv_pid_c["canon_inv_pid"])
    matches = top_fuzzy_match(list(remaining["canon_orders_pid"]), tgt, min_ratio=0.92)
    if matches:
        fuzzy = pd.DataFrame(
            [(k, v[0], v[1]) for k, v in matches.items()],
            columns=["canon_orders_pid", "canon_inv_pid", "score"]
        )
        fuzzy = (
            remaining.merge(fuzzy, on="canon_orders_pid", how="inner")
            .merge(inv_pid_c[["canon_inv_pid", "inv_product_id_raw"]], on="canon_inv_pid", how="left")
            .assign(method="fuzzy_92")
        )
        crosswalk = pd.concat([exact, fuzzy], ignore_index=True, sort=False)
    else:
        crosswalk = exact

    # Keep only the best mapping per canon_orders_pid
    crosswalk = (
        crosswalk.sort_values(["canon_orders_pid", "score"], ascending=[True, False])
        .drop_duplicates(subset=["canon_orders_pid"], keep="first")
    )

    # Final shape
    out = crosswalk[[
        "orders_product_id_raw", "inv_product_id_raw",
        "canon_orders_pid", "canon_inv_pid",
        "method", "score"
    ]].copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    log.info(f"Saved SKU crosswalk → {OUT_PATH} | rows={len(out)}")
    log.info("Manual review tip: add/override rows in this file if needed (keep canon_* columns consistent).")

if __name__ == "__main__":
    main()
