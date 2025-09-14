"""
Scenario Engine Module - Baseline-first scenario application
- If a baseline (labels + values) is provided from the API, we apply scenario multipliers
  directly to those quantities (fast, shape-preserving).
- If no baseline is provided, we fall back to the existing enhanced ML path.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import hashlib
import time
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ================================
# Scenario transform guardrails
# ================================
MIN_MULT, MAX_MULT = 0.1, 10.0
MIN_PRICE = 0.01


def _clamp(v, lo, hi, default):
    try:
        v = float(v)
    except Exception:
        return default
    return min(max(v, lo), hi)


def _select_training_slice(df: pd.DataFrame, target_col: str, min_rows: int = 200) -> pd.DataFrame:
    """Prefer a slice with non-zero target values, else recent rows with variance."""
    if df.empty or target_col not in df:
        return df

    series = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
    nonzero = df[series > 0]
    if len(nonzero) >= min_rows:
        return nonzero

    recent = df.tail(100000) if len(df) > 100000 else df
    if pd.to_numeric(recent[target_col], errors="coerce").fillna(0.0).std(skipna=True) > 0 and len(recent) >= min_rows:
        return recent

    return df


def _validate_training_ready(df: pd.DataFrame, target_col: str, min_unique: int = 5):
    if df.empty:
        raise ValueError("Filtered dataset is empty.")
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found.")
    tgt = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
    if float(tgt.std(skipna=True) or 0.0) == 0.0:
        raise ValueError(f"Target '{target_col}' has zero variance.")
    if int(tgt.nunique(dropna=True)) < min_unique:
        raise ValueError(f"Target '{target_col}' has <{min_unique} unique values.")


# =====================================
# Cache for trained scenario models
# =====================================
class ScenarioCache:
    """Cache for trained scenario models with TTL and size limits."""
    def __init__(self, max_size: int = 10, ttl_hours: int = 24):
        self.cache = {}      # scenario_hash -> trained_model
        self.timestamps = {} # scenario_hash -> timestamp
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600

    def get_scenario_hash(self, scenario: Dict[str, float]) -> str:
        """Create deterministic hash from scenario parameters."""
        sorted_items = sorted(scenario.items())
        scenario_str = str(sorted_items)
        return hashlib.md5(scenario_str.encode()).hexdigest()[:12]

    def get_cached_model(self, scenario_hash: str):
        """Return cached model if exists and not expired."""
        if scenario_hash not in self.cache:
            return None
        if scenario_hash in self.timestamps:
            age = time.time() - self.timestamps[scenario_hash]
            if age > self.ttl_seconds:
                logger.info(f"Scenario model {scenario_hash} expired (age: {age/3600:.1f}h)")
                self._remove_from_cache(scenario_hash)
                return None
        logger.info(f"Using cached scenario model: {scenario_hash}")
        return self.cache[scenario_hash]

    def cache_model(self, scenario_hash: str, model):
        """Store trained scenario model with timestamp."""
        if len(self.cache) >= self.max_size and self.timestamps:
            oldest_hash = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            self._remove_from_cache(oldest_hash)
            logger.info(f"Evicted oldest scenario model: {oldest_hash}")

        self.cache[scenario_hash] = model
        self.timestamps[scenario_hash] = time.time()
        logger.info(f"Cached new scenario model: {scenario_hash}")

    def _remove_from_cache(self, scenario_hash: str):
        """Remove model from cache."""
        self.cache.pop(scenario_hash, None)
        self.timestamps.pop(scenario_hash, None)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "cached_models": len(self.cache),
            "max_size": self.max_size,
            "ttl_hours": self.ttl_seconds / 3600,
            "model_hashes": list(self.cache.keys())
        }


# =====================================
# Baseline context (optional)
# =====================================
@dataclass
class BaselineContext:
    labels: List[str]            # dates shown on the chart
    values_qty: List[float]      # baseline quantities shown on the chart
    X_future: pd.DataFrame       # exact feature rows used to create the baseline (if any)
    unit: str                    # 'qty' or 'revenue' for the main model output
    avg_price: float             # avg price used if unit == 'revenue'


# =====================================
# Scenario Engine
# =====================================
class ScenarioEngine:
    """Scenario forecasting engine."""

    def __init__(self, data_manager, cache_size: int = 10, cache_ttl_hours: int = 24):
        self.data_manager = data_manager
        self.cache = ScenarioCache(cache_size, cache_ttl_hours)
        self.training_timeout = 30  # seconds
        self.baseline_ctx: Optional[BaselineContext] = None

        # LightGBM params (only used if we fall back to ML training)
        self.quick_train_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'early_stopping_rounds': 10,
            'num_boost_round': 100,
            'verbose': -1,
            'force_col_wise': True,
            'seed': 42,
            'num_threads': -1,
        }

    # ----------------------------------
    # Optional baseline registration
    # ----------------------------------
    def register_baseline(self,
                          labels: List[str],
                          values_qty: List[float],
                          X_future: pd.DataFrame,
                          unit: str = "qty",
                          avg_price: float = 100.0) -> None:
        try:
            unit = (unit or "qty").strip().lower()
        except Exception:
            unit = "qty"
        self.baseline_ctx = BaselineContext(
            labels=list(labels),
            values_qty=list(values_qty),
            X_future=X_future.copy() if isinstance(X_future, pd.DataFrame) else pd.DataFrame(),
            unit=unit,
            avg_price=float(avg_price or 100.0),
        )

    # ----------------------------------
    # Public entry
    # ----------------------------------
    async def generate_forecast(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        If `baseline_override` is provided ({labels, values}), apply scenario multipliers
        directly to those values (fast & shape-preserving). Otherwise, fall back to
        the enhanced ML scenario path.
        """
        try:
            start_time = time.time()
            forecast_days = params.get("forecast_days", 30)
            scenario = params.get("scenario", {}) or {}

            # 1) If we got a baseline from the API (daily-forecast), use it directly
            baseline_override = params.get("baseline_override")
            if baseline_override and isinstance(baseline_override, dict):
                labels = list(baseline_override.get("labels", []))[:forecast_days]
                baseline_values = list(baseline_override.get("values", []))[:forecast_days]
                if labels and baseline_values and len(labels) == len(baseline_values):
                    scenario_values, cu, cl, intensity = self._apply_multipliers_to_baseline(
                        baseline_values, scenario
                    )
                    deltas = [s - b for s, b in zip(scenario_values, baseline_values)]
                    training_time = time.time() - start_time

                    return {
                        "labels": labels,
                        "baseline_values": baseline_values,
                        "scenario_values": scenario_values,
                        "values": scenario_values,
                        "deltas": deltas,
                        "confidence_upper": cu,
                        "confidence_lower": cl,
                        "metadata": {
                            "scenario_hash": self.cache.get_scenario_hash(scenario),
                            "training_time_seconds": round(training_time, 2),
                            "model_cached": True,  # not training in this path
                            "scenario_multipliers": scenario,
                            "forecasting_method": "baseline_multiplier",
                            "baseline_method": "daily_chart",
                            "cache_stats": self.cache.get_cache_stats(),
                            "feature_count": 0,
                            "avg_delta": round(sum(deltas) / len(deltas), 2) if deltas else 0,
                            "max_delta": max(deltas) if deltas else 0,
                            "min_delta": min(deltas) if deltas else 0,
                            "scenario_intensity": intensity,
                        },
                    }

            # 2) Fallback to enhanced ML scenario path (kept for completeness)
            logger.info("No baseline provided; using enhanced ML scenario path.")
            baseline_forecast = await self._generate_baseline_forecast(forecast_days)

            scenario_hash = self.cache.get_scenario_hash(scenario)
            scenario_model = self.cache.get_cached_model(scenario_hash)
            model_was_cached = scenario_model is not None

            if scenario_model is None:
                scenario_model = await self._train_enhanced_scenario_model(scenario)
                if scenario_model is not None:
                    self.cache.cache_model(scenario_hash, scenario_model)
                else:
                    logger.warning("Scenario training failed; using multiplier fallback on baseline.")
                    return self._create_fallback_response(baseline_forecast, scenario, start_time)

            scenario_forecast = await self._generate_enhanced_scenario_forecast(
                scenario_model, forecast_days, scenario
            )

            baseline_values = baseline_forecast["values"]
            deltas = [s - b for s, b in zip(scenario_forecast["values"], baseline_values)]
            training_time = time.time() - start_time

            return {
                "labels": scenario_forecast['labels'],
                "baseline_values": baseline_values,
                "scenario_values": scenario_forecast['values'],
                "values": scenario_forecast['values'],
                "deltas": deltas,
                "confidence_upper": scenario_forecast.get('confidence_upper', []),
                "confidence_lower": scenario_forecast.get('confidence_lower', []),
                "metadata": {
                    "scenario_hash": scenario_hash,
                    "training_time_seconds": round(training_time, 2),
                    "model_cached": model_was_cached,
                    "scenario_multipliers": scenario,
                    "forecasting_method": "enhanced_scenario_retrain",
                    "baseline_method": "stat_or_ml",
                    "cache_stats": self.cache.get_cache_stats(),
                    "feature_count": scenario_forecast.get('feature_count', 0),
                    "avg_delta": round(sum(deltas) / len(deltas), 2) if deltas else 0,
                    "max_delta": max(deltas) if deltas else 0,
                    "min_delta": min(deltas) if deltas else 0,
                }
            }

        except Exception as e:
            logger.exception("Scenario engine failed: %s", e)
            return self._create_error_fallback(str(e))

    # ----------------------------------
    # Baseline multiplier application
    # ----------------------------------
    def _apply_multipliers_to_baseline(
        self, baseline_values: List[float], scenario: Dict[str, float]
    ) -> Tuple[List[int], List[int], List[int], float]:
        """
        Multiply the baseline quantities by clamped scenario multipliers and produce
        dynamic confidence bands. Returns (values, cu, cl, intensity).
        """
        def safe_mult(x, default=1.0):
            try:
                return float(x)
            except Exception:
                return default

        price  = _clamp(safe_mult(scenario.get("price", 1.0)), MIN_MULT, MAX_MULT, 1.0)
        disc_v = scenario.get("discount", 1.0)
        # allow both direct multiplier (0..10) or pct (e.g. 0.2 => 0.8x)
        try:
            disc_v = float(disc_v)
            if not (0 < disc_v <= MAX_MULT):
                disc = 1.0 - float(scenario.get("discount", 0.0))
            else:
                disc = disc_v
        except Exception:
            disc = 1.0
        disc = _clamp(disc, MIN_MULT, MAX_MULT, 1.0)

        infl = _clamp(safe_mult(scenario.get("inflation", 1.0)), MIN_MULT, MAX_MULT, 1.0)
        temp = _clamp(safe_mult(scenario.get("temperature", 1.0)), MIN_MULT, MAX_MULT, 1.0)

        combined = price * disc * infl * temp
        intensity = float(np.sqrt((price-1)**2 + (disc-1)**2 + (infl-1)**2 + (temp-1)**2))

        values = [int(max(0, round(v * combined))) for v in baseline_values]

        # confidence bands scale with intensity
        conf = 0.12 + 0.06 * min(1.0, intensity)  # 12%..18%
        cu = [int(round(v * (1 + conf))) for v in values]
        cl = [int(round(v * (1 - conf))) for v in values]
        return values, cu, cl, intensity

    # ----------------------------------
    # Minimal baseline generators (fallback)
    # ----------------------------------
    async def _generate_baseline_forecast(self, forecast_days: int) -> Dict[str, Any]:
        if self.data_manager.model is not None and not self.data_manager.demand_history.empty:
            return await self._generate_statistical_baseline(forecast_days)  # simple + stable
        return await self._generate_statistical_baseline(forecast_days)

    async def _generate_statistical_baseline(self, forecast_days: int) -> Dict[str, Any]:
        df = self.data_manager.demand_history.copy()
        if df.empty or "order_date" not in df or "order_qty" not in df:
            today = datetime.now()
            labels = [(today + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(forecast_days)]
            values = [100 for _ in range(forecast_days)]
            return {"labels": labels, "values": values, "confidence_upper": [120]*forecast_days, "confidence_lower": [80]*forecast_days}

        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df = df.dropna(subset=["order_date"])
        df["dow"] = df["order_date"].dt.dayofweek
        daily = df.groupby(['order_date', 'dow'])['order_qty'].sum().reset_index()
        dow_avg = daily.groupby('dow')["order_qty"].mean().to_dict()

        last = df["order_date"].max()
        future = [last + pd.Timedelta(days=i+1) for i in range(forecast_days)]
        labels = [d.strftime("%Y-%m-%d") for d in future]
        values = [int(max(0, round(dow_avg.get(d.dayofweek, 100)))) for d in future]

        return {
            "labels": labels,
            "values": values,
            "confidence_upper": [int(v * 1.2) for v in values],
            "confidence_lower": [int(v * 0.8) for v in values],
        }

    # ----------------------------------
    # Enhanced ML path (kept as backup)
    # ----------------------------------
    def _build_daily_qty(self) -> pd.DataFrame:
        df = self.data_manager.demand_history.copy()
        if df.empty or 'order_date' not in df or 'order_qty' not in df:
            return pd.DataFrame(columns=['order_date', 'order_qty'])
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df = df.dropna(subset=['order_date'])
        daily = (df.groupby(df['order_date'].dt.normalize())['order_qty']
                   .sum()
                   .reset_index()
                   .rename(columns={'order_date': 'order_date', 'order_qty': 'order_qty'}))
        return daily.sort_values('order_date').reset_index(drop=True)

    def _apply_daily_scenario(self, daily: pd.DataFrame, scenario: Dict[str, float]) -> pd.DataFrame:
        if daily.empty:
            return daily

        def _safe_mult(x, lo=MIN_MULT, hi=MAX_MULT, default=1.0):
            try:
                x = float(x)
            except Exception:
                return default
            return min(max(x, lo), hi)

        price_mult = _safe_mult(scenario.get('price', 1.0))

        raw_disc = scenario.get('discount', 1.0)
        if raw_disc is None:
            discount_mult = 1.0
        else:
            try:
                rd = float(raw_disc)
                discount_mult = rd if 0 < rd <= MAX_MULT else _safe_mult(1.0 - rd)
            except Exception:
                discount_mult = 1.0

        inflation_mult = _safe_mult(scenario.get('inflation', 1.0))

        out = daily.copy()
        out['order_qty'] = pd.to_numeric(out['order_qty'], errors='coerce').fillna(0.0)
        out['order_qty_scn'] = (out['order_qty'] * price_mult * discount_mult * inflation_mult).clip(lower=0)

        out['scenario_price_mult'] = price_mult
        out['scenario_discount_mult'] = discount_mult
        out['scenario_inflation_mult'] = inflation_mult
        out['scenario_intensity'] = float(np.sqrt((price_mult - 1) ** 2 +
                                                  (discount_mult - 1) ** 2 +
                                                  (inflation_mult - 1) ** 2))
        return out

    def _prepare_enhanced_scenario_dataset(self, scenario: Dict[str, float]) -> pd.DataFrame:
        daily = self._build_daily_qty()
        if daily.empty:
            return daily
        daily = self._apply_daily_scenario(daily, scenario)
        return daily

    def _prepare_enhanced_training_data(
        self, df: pd.DataFrame, force_target_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[float]]:
        if df.empty:
            return pd.DataFrame(), []

        df = df.sort_values('order_date').reset_index(drop=True)

        target_col = force_target_col or ('order_qty_scn' if 'order_qty_scn' in df.columns else 'order_qty')
        target = df[target_col].fillna(0.0).astype(float).tolist()

        leak_cols: List[str] = []
        if target_col == "order_qty_scn" and "order_qty" in df.columns:
            leak_cols.append("order_qty")
        if leak_cols:
            df = df.drop(columns=leak_cols)

        feature_df = pd.DataFrame()

        df['order_date'] = pd.to_datetime(df['order_date'])
        feature_df['year'] = df['order_date'].dt.year
        feature_df['month'] = df['order_date'].dt.month
        feature_df['day'] = df['order_date'].dt.day
        feature_df['dayofweek'] = df['order_date'].dt.dayofweek
        feature_df['quarter'] = df['order_date'].dt.quarter
        feature_df['is_weekend'] = (df['order_date'].dt.dayofweek >= 5).astype(int)

        feature_df['month_sin'] = np.sin(2 * np.pi * feature_df['month'] / 12)
        feature_df['month_cos'] = np.cos(2 * np.pi * feature_df['month'] / 12)
        feature_df['dayofweek_sin'] = np.sin(2 * np.pi * feature_df['dayofweek'] / 7)
        feature_df['dayofweek_cos'] = np.cos(2 * np.pi * feature_df['dayofweek'] / 7)
        feature_df['day_sin'] = np.sin(2 * np.pi * feature_df['day'] / 31)
        feature_df['day_cos'] = np.cos(2 * np.pi * feature_df['day'] / 31)

        for lag in [1, 3, 7, 14]:
            if len(df) > lag:
                feature_df[f'lag_{lag}'] = df[target_col].shift(lag).fillna(df[target_col].mean())
            else:
                feature_df[f'lag_{lag}'] = df[target_col].mean()

        for window in [3, 7, 14]:
            if len(df) > window:
                rolling = df[target_col].rolling(window=window, min_periods=1)
                feature_df[f'rolling_mean_{window}'] = rolling.mean().fillna(df[target_col].mean())
                feature_df[f'rolling_std_{window}'] = rolling.std().fillna(0.0)
                diffs = df[target_col].diff().fillna(0.0)
                feature_df[f'rolling_trend_{window}'] = diffs.rolling(window=window, min_periods=1).mean().fillna(0.0)
            else:
                feature_df[f'rolling_mean_{window}'] = df[target_col].mean()
                feature_df[f'rolling_std_{window}'] = 0.0
                feature_df[f'rolling_trend_{window}'] = 0.0

        for col in [c for c in df.columns if c.startswith('scenario_')]:
            feature_df[col] = df[col]

        return feature_df.fillna(0), target

    async def _train_enhanced_scenario_model(self, scenario: Dict[str, float], timeout_seconds: int = 30):
        start_time = time.time()
        df = self._prepare_enhanced_scenario_dataset(scenario)
        if df.empty:
            raise ValueError("No data available to train scenario model")

        MAX_ROWS = 100_000
        candidate_cols = ["order_qty_scn", "order_qty"]
        df = _pick_recent_nonzero_window(df, candidate_cols, max_rows=MAX_ROWS)
        if len(df) > MAX_ROWS:
            df = df.tail(MAX_ROWS)

        for _col in ("order_qty_scn", "order_qty"):
            if _col in df.columns:
                df[_col] = pd.to_numeric(df[_col], errors="coerce").fillna(0.0)

        target_primary = "order_qty_scn" if ("order_qty_scn" in df.columns and df["order_qty_scn"].std() > 0) else "order_qty"
        if target_primary not in df or float(df[target_primary].std() or 0.0) == 0.0:
            raise ValueError("No non-degenerate target available for scenario training.")

        df = _select_training_slice(df, target_primary, min_rows=200)
        df = df.sort_values("order_date").reset_index(drop=True)

        feature_df, target = self._prepare_enhanced_training_data(df, force_target_col=target_primary)
        if len(feature_df) < 20:
            raise ValueError("Insufficient samples for scenario training (need >=20)")

        _validate_training_ready(pd.DataFrame({target_primary: target}), target_primary)
        y_arr = np.asarray(target, dtype=float)

        split_date = df['order_date'].quantile(0.8)
        train_mask = (df['order_date'] <= split_date).to_numpy()
        val_mask = ~train_mask

        X_train = feature_df.iloc[train_mask]
        X_val = feature_df.iloc[val_mask]
        y_train = y_arr[train_mask]
        y_val = y_arr[val_mask]

        categorical_features = [col for col in X_train.columns if X_train[col].dtype == 'object']
        for col in categorical_features:
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')

        lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features or None)
        lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train, categorical_feature=categorical_features or None)

        model = lgb.train(
            self.quick_train_params,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)]
        )
        try:
            model.set_attr(target_unit="qty")
        except Exception:
            pass

        elapsed = time.time() - start_time
        logger.info("Enhanced scenario model trained in %.2fs with %d trees", elapsed, model.num_trees())
        return model

    async def _generate_enhanced_scenario_forecast(self, model, forecast_days: int, scenario: Dict[str, float]) -> Dict[str, Any]:
        # Build basic future features (quick)
        daily = self._build_daily_qty()
        if daily.empty:
            today = datetime.now()
            labels = [(today + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(forecast_days)]
            values = [100 for _ in range(forecast_days)]
            return {"labels": labels, "values": values, "confidence_upper": [120]*forecast_days, "confidence_lower": [80]*forecast_days, "feature_count": 0}

        last_date = daily["order_date"].max()
        dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]
        labels = [d.strftime("%Y-%m-%d") for d in dates]

        feature_rows = []
        series = daily["order_qty"]
        mean_v = float(series.mean()) if len(series) else 0.0
        for d in dates:
            r = {
                'year': d.year, 'month': d.month, 'day': d.day,
                'dayofweek': d.dayofweek, 'quarter': d.quarter, 'is_weekend': int(d.dayofweek >= 5),
                'month_sin': np.sin(2*np.pi*d.month/12.0), 'month_cos': np.cos(2*np.pi*d.month/12.0),
                'dayofweek_sin': np.sin(2*np.pi*d.dayofweek/7.0), 'dayofweek_cos': np.cos(2*np.pi*d.dayofweek/7.0),
                'day_sin': np.sin(2*np.pi*d.day/31.0), 'day_cos': np.cos(2*np.pi*d.day/31.0),
                'scenario_price_mult': float(scenario.get('price', 1.0)),
                'scenario_discount_mult': float(scenario.get('discount', 1.0)) if 0 < float(scenario.get('discount', 1.0)) <= MAX_MULT else float(1.0 - float(scenario.get('discount', 0.0))),
                'scenario_inflation_mult': float(scenario.get('inflation', 1.0)),
            }
            for lag in (1, 3, 7, 14):
                r[f'lag_{lag}'] = float(series.iloc[-lag]) if len(series) >= lag else mean_v
            for win in (3, 7, 14):
                if len(series) >= win:
                    recent = series.tail(win)
                    r[f'rolling_mean_{win}'] = float(recent.mean())
                    r[f'rolling_std_{win}'] = float(recent.std() or 0.0)
                    r[f'rolling_trend_{win}'] = float(recent.diff().fillna(0.0).mean())
                else:
                    r[f'rolling_mean_{win}'] = mean_v
                    r[f'rolling_std_{win}'] = 0.0
                    r[f'rolling_trend_{win}'] = 0.0
            feature_rows.append(r)

        X_future = pd.DataFrame(feature_rows)
        model_features = model.feature_name()
        missing = set(model_features) - set(X_future.columns)
        for m in missing:
            X_future[m] = 0
        X_future = X_future[model_features]
        preds = model.predict(X_future)

        # Convert (assumed qty) & apply simple uncertainty bands
        values = [int(max(0, round(float(p)))) for p in preds]
        cu = [int(round(v * 1.15)) for v in values]
        cl = [int(round(v * 0.85)) for v in values]
        return {"labels": labels, "values": values, "confidence_upper": cu, "confidence_lower": cl, "feature_count": len(model_features)}

    # ----------------------------------
    # Fallbacks
    # ----------------------------------
    def _create_fallback_response(self, baseline_forecast: Dict[str, Any], scenario: Dict[str, float], start_time: float) -> Dict[str, Any]:
        multiplier = np.mean(list(scenario.values())) if scenario else 1.0
        scenario_values = [int(round(v * multiplier)) for v in baseline_forecast['values']]
        deltas = [s - b for s, b in zip(scenario_values, baseline_forecast['values'])]

        return {
            "labels": baseline_forecast['labels'],
            "baseline_values": baseline_forecast['values'],
            "scenario_values": scenario_values,
            "values": scenario_values,
            "deltas": deltas,
            "confidence_upper": [int(v * 1.2) for v in scenario_values],
            "confidence_lower": [int(v * 0.8) for v in scenario_values],
            "metadata": {
                "scenario_hash": "fallback",
                "training_time_seconds": round(time.time() - start_time, 2),
                "model_cached": False,
                "scenario_multipliers": scenario,
                "forecasting_method": "enhanced_multiplier_fallback",
                "baseline_method": "stat_or_ml",
                "warning": "Training failed; used multiplier fallback",
                "feature_count": 0,
                "avg_delta": round(sum(deltas) / len(deltas), 2) if deltas else 0,
                "max_delta": max(deltas) if deltas else 0,
                "min_delta": min(deltas) if deltas else 0
            }
        }

    def _create_error_fallback(self, error_msg: str) -> Dict[str, Any]:
        today = datetime.now()
        labels = [(today + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(30)]
        values = [100 for _ in labels]
        return {
            "labels": labels,
            "values": values,
            "baseline_values": values,
            "scenario_values": values,
            "deltas": [0]*len(values),
            "confidence_upper": [120]*len(values),
            "confidence_lower": [80]*len(values),
            "metadata": {
                "scenario_hash": "error",
                "training_time_seconds": 0,
                "model_cached": False,
                "scenario_multipliers": {},
                "forecasting_method": "error_fallback",
                "baseline_method": "synthetic",
                "error": error_msg,
                "feature_count": 0,
                "avg_delta": 0,
                "max_delta": 0,
                "min_delta": 0,
                "cache_stats": self.cache.get_cache_stats(),
            }
        }


# ------------------------
# Recent nonzero window picker
# ------------------------
def _pick_recent_nonzero_window(
    df: pd.DataFrame,
    target_cols,
    max_rows: int = 100_000,
    min_nonzero: int = 50,
    blocks = (100_000, 60_000, 40_000, 20_000, 10_000, 5_000, 2_000)
) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.sort_values("order_date").reset_index(drop=True)

    has_signal_any = pd.Series(False, index=df.index)
    for col in target_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            has_signal_any = has_signal_any | (s > 0)

    if has_signal_any.any():
        last_sig_pos = int(np.where(has_signal_any.values)[0].max())
        end = last_sig_pos + 1
        cand = df.iloc[max(0, end - max_rows):end].copy()

        for blk in blocks:
            size = min(blk, max_rows, end)
            start_blk = max(0, end - size)
            sub = df.iloc[start_blk:end].copy()

            ok = False
            for col in target_cols:
                if col not in sub.columns:
                    continue
                series = pd.to_numeric(sub[col], errors="coerce").fillna(0.0)
                if (series > 0).sum() >= min_nonzero and float(series.std(skipna=True) or 0.0) > 0.0:
                    ok = True
                    break
            if ok:
                logger.info("Picked anchored window ending at last signal idx=%d: rows=%d", last_sig_pos, len(sub))
                return sub

        logger.warning("Anchored slice had weak signal; returning it to avoid all-zero tail: rows=%d", len(cand))
        return cand

    size = min(max_rows, len(df))
    logger.warning("No non-zero signal found; using most recent tail(%d) of %d rows", size, len(df))
    return df.tail(size).copy()
"""
Scenario Engine Module - Baseline-first scenario application
- If a baseline (labels + values) is provided from the API, we apply scenario multipliers
  directly to those quantities (fast, shape-preserving).
- If no baseline is provided, we fall back to the existing enhanced ML path.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import hashlib
import time
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ================================
# Scenario transform guardrails
# ================================
MIN_MULT, MAX_MULT = 0.1, 10.0
MIN_PRICE = 0.01


def _clamp(v, lo, hi, default):
    try:
        v = float(v)
    except Exception:
        return default
    return min(max(v, lo), hi)


def _select_training_slice(df: pd.DataFrame, target_col: str, min_rows: int = 200) -> pd.DataFrame:
    """Prefer a slice with non-zero target values, else recent rows with variance."""
    if df.empty or target_col not in df:
        return df

    series = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
    nonzero = df[series > 0]
    if len(nonzero) >= min_rows:
        return nonzero

    recent = df.tail(100000) if len(df) > 100000 else df
    if pd.to_numeric(recent[target_col], errors="coerce").fillna(0.0).std(skipna=True) > 0 and len(recent) >= min_rows:
        return recent

    return df


def _validate_training_ready(df: pd.DataFrame, target_col: str, min_unique: int = 5):
    if df.empty:
        raise ValueError("Filtered dataset is empty.")
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found.")
    tgt = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
    if float(tgt.std(skipna=True) or 0.0) == 0.0:
        raise ValueError(f"Target '{target_col}' has zero variance.")
    if int(tgt.nunique(dropna=True)) < min_unique:
        raise ValueError(f"Target '{target_col}' has <{min_unique} unique values.")


# =====================================
# Cache for trained scenario models
# =====================================
class ScenarioCache:
    """Cache for trained scenario models with TTL and size limits."""
    def __init__(self, max_size: int = 10, ttl_hours: int = 24):
        self.cache = {}      # scenario_hash -> trained_model
        self.timestamps = {} # scenario_hash -> timestamp
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600

    def get_scenario_hash(self, scenario: Dict[str, float]) -> str:
        """Create deterministic hash from scenario parameters."""
        sorted_items = sorted(scenario.items())
        scenario_str = str(sorted_items)
        return hashlib.md5(scenario_str.encode()).hexdigest()[:12]

    def get_cached_model(self, scenario_hash: str):
        """Return cached model if exists and not expired."""
        if scenario_hash not in self.cache:
            return None
        if scenario_hash in self.timestamps:
            age = time.time() - self.timestamps[scenario_hash]
            if age > self.ttl_seconds:
                logger.info(f"Scenario model {scenario_hash} expired (age: {age/3600:.1f}h)")
                self._remove_from_cache(scenario_hash)
                return None
        logger.info(f"Using cached scenario model: {scenario_hash}")
        return self.cache[scenario_hash]

    def cache_model(self, scenario_hash: str, model):
        """Store trained scenario model with timestamp."""
        if len(self.cache) >= self.max_size and self.timestamps:
            oldest_hash = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            self._remove_from_cache(oldest_hash)
            logger.info(f"Evicted oldest scenario model: {oldest_hash}")

        self.cache[scenario_hash] = model
        self.timestamps[scenario_hash] = time.time()
        logger.info(f"Cached new scenario model: {scenario_hash}")

    def _remove_from_cache(self, scenario_hash: str):
        """Remove model from cache."""
        self.cache.pop(scenario_hash, None)
        self.timestamps.pop(scenario_hash, None)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "cached_models": len(self.cache),
            "max_size": self.max_size,
            "ttl_hours": self.ttl_seconds / 3600,
            "model_hashes": list(self.cache.keys())
        }


# =====================================
# Baseline context (optional)
# =====================================
@dataclass
class BaselineContext:
    labels: List[str]            # dates shown on the chart
    values_qty: List[float]      # baseline quantities shown on the chart
    X_future: pd.DataFrame       # exact feature rows used to create the baseline (if any)
    unit: str                    # 'qty' or 'revenue' for the main model output
    avg_price: float             # avg price used if unit == 'revenue'


# =====================================
# Scenario Engine
# =====================================
class ScenarioEngine:
    """Scenario forecasting engine."""

    def __init__(self, data_manager, cache_size: int = 10, cache_ttl_hours: int = 24):
        self.data_manager = data_manager
        self.cache = ScenarioCache(cache_size, cache_ttl_hours)
        self.training_timeout = 30  # seconds
        self.baseline_ctx: Optional[BaselineContext] = None

        # LightGBM params (only used if we fall back to ML training)
        self.quick_train_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'early_stopping_rounds': 10,
            'num_boost_round': 100,
            'verbose': -1,
            'force_col_wise': True,
            'seed': 42,
            'num_threads': -1,
        }

    # ----------------------------------
    # Optional baseline registration
    # ----------------------------------
    def register_baseline(self,
                          labels: List[str],
                          values_qty: List[float],
                          X_future: pd.DataFrame,
                          unit: str = "qty",
                          avg_price: float = 100.0) -> None:
        try:
            unit = (unit or "qty").strip().lower()
        except Exception:
            unit = "qty"
        self.baseline_ctx = BaselineContext(
            labels=list(labels),
            values_qty=list(values_qty),
            X_future=X_future.copy() if isinstance(X_future, pd.DataFrame) else pd.DataFrame(),
            unit=unit,
            avg_price=float(avg_price or 100.0),
        )

    # ----------------------------------
    # Public entry
    # ----------------------------------
    async def generate_forecast(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        If `baseline_override` is provided ({labels, values}), apply scenario multipliers
        directly to those values (fast & shape-preserving). Otherwise, fall back to
        the enhanced ML scenario path.
        """
        try:
            start_time = time.time()
            forecast_days = params.get("forecast_days", 30)
            scenario = params.get("scenario", {}) or {}

            # 1) If we got a baseline from the API (daily-forecast), use it directly
            baseline_override = params.get("baseline_override")
            if baseline_override and isinstance(baseline_override, dict):
                labels = list(baseline_override.get("labels", []))[:forecast_days]
                baseline_values = list(baseline_override.get("values", []))[:forecast_days]
                if labels and baseline_values and len(labels) == len(baseline_values):
                    scenario_values, cu, cl, intensity = self._apply_multipliers_to_baseline(
                        baseline_values, scenario
                    )
                    deltas = [s - b for s, b in zip(scenario_values, baseline_values)]
                    training_time = time.time() - start_time

                    return {
                        "labels": labels,
                        "baseline_values": baseline_values,
                        "scenario_values": scenario_values,
                        "values": scenario_values,
                        "deltas": deltas,
                        "confidence_upper": cu,
                        "confidence_lower": cl,
                        "metadata": {
                            "scenario_hash": self.cache.get_scenario_hash(scenario),
                            "training_time_seconds": round(training_time, 2),
                            "model_cached": True,  # not training in this path
                            "scenario_multipliers": scenario,
                            "forecasting_method": "baseline_multiplier",
                            "baseline_method": "daily_chart",
                            "cache_stats": self.cache.get_cache_stats(),
                            "feature_count": 0,
                            "avg_delta": round(sum(deltas) / len(deltas), 2) if deltas else 0,
                            "max_delta": max(deltas) if deltas else 0,
                            "min_delta": min(deltas) if deltas else 0,
                            "scenario_intensity": intensity,
                        },
                    }

            # 2) Fallback to enhanced ML scenario path (kept for completeness)
            logger.info("No baseline provided; using enhanced ML scenario path.")
            baseline_forecast = await self._generate_baseline_forecast(forecast_days)

            scenario_hash = self.cache.get_scenario_hash(scenario)
            scenario_model = self.cache.get_cached_model(scenario_hash)
            model_was_cached = scenario_model is not None

            if scenario_model is None:
                scenario_model = await self._train_enhanced_scenario_model(scenario)
                if scenario_model is not None:
                    self.cache.cache_model(scenario_hash, scenario_model)
                else:
                    logger.warning("Scenario training failed; using multiplier fallback on baseline.")
                    return self._create_fallback_response(baseline_forecast, scenario, start_time)

            scenario_forecast = await self._generate_enhanced_scenario_forecast(
                scenario_model, forecast_days, scenario
            )

            baseline_values = baseline_forecast["values"]
            deltas = [s - b for s, b in zip(scenario_forecast["values"], baseline_values)]
            training_time = time.time() - start_time

            return {
                "labels": scenario_forecast['labels'],
                "baseline_values": baseline_values,
                "scenario_values": scenario_forecast['values'],
                "values": scenario_forecast['values'],
                "deltas": deltas,
                "confidence_upper": scenario_forecast.get('confidence_upper', []),
                "confidence_lower": scenario_forecast.get('confidence_lower', []),
                "metadata": {
                    "scenario_hash": scenario_hash,
                    "training_time_seconds": round(training_time, 2),
                    "model_cached": model_was_cached,
                    "scenario_multipliers": scenario,
                    "forecasting_method": "enhanced_scenario_retrain",
                    "baseline_method": "stat_or_ml",
                    "cache_stats": self.cache.get_cache_stats(),
                    "feature_count": scenario_forecast.get('feature_count', 0),
                    "avg_delta": round(sum(deltas) / len(deltas), 2) if deltas else 0,
                    "max_delta": max(deltas) if deltas else 0,
                    "min_delta": min(deltas) if deltas else 0,
                }
            }

        except Exception as e:
            logger.exception("Scenario engine failed: %s", e)
            return self._create_error_fallback(str(e))

    # ----------------------------------
    # Baseline multiplier application
    # ----------------------------------
    def _apply_multipliers_to_baseline(
        self, baseline_values: List[float], scenario: Dict[str, float]
    ) -> Tuple[List[int], List[int], List[int], float]:
        """
        Multiply the baseline quantities by clamped scenario multipliers and produce
        dynamic confidence bands. Returns (values, cu, cl, intensity).
        """
        def safe_mult(x, default=1.0):
            try:
                return float(x)
            except Exception:
                return default

        price  = _clamp(safe_mult(scenario.get("price", 1.0)), MIN_MULT, MAX_MULT, 1.0)
        disc_v = scenario.get("discount", 1.0)
        # allow both direct multiplier (0..10) or pct (e.g. 0.2 => 0.8x)
        try:
            disc_v = float(disc_v)
            if not (0 < disc_v <= MAX_MULT):
                disc = 1.0 - float(scenario.get("discount", 0.0))
            else:
                disc = disc_v
        except Exception:
            disc = 1.0
        disc = _clamp(disc, MIN_MULT, MAX_MULT, 1.0)

        infl = _clamp(safe_mult(scenario.get("inflation", 1.0)), MIN_MULT, MAX_MULT, 1.0)
        temp = _clamp(safe_mult(scenario.get("temperature", 1.0)), MIN_MULT, MAX_MULT, 1.0)

        combined = price * disc * infl * temp
        intensity = float(np.sqrt((price-1)**2 + (disc-1)**2 + (infl-1)**2 + (temp-1)**2))

        values = [int(max(0, round(v * combined))) for v in baseline_values]

        # confidence bands scale with intensity
        conf = 0.12 + 0.06 * min(1.0, intensity)  # 12%..18%
        cu = [int(round(v * (1 + conf))) for v in values]
        cl = [int(round(v * (1 - conf))) for v in values]
        return values, cu, cl, intensity

    # ----------------------------------
    # Minimal baseline generators (fallback)
    # ----------------------------------
    async def _generate_baseline_forecast(self, forecast_days: int) -> Dict[str, Any]:
        if self.data_manager.model is not None and not self.data_manager.demand_history.empty:
            return await self._generate_statistical_baseline(forecast_days)  # simple + stable
        return await self._generate_statistical_baseline(forecast_days)

    async def _generate_statistical_baseline(self, forecast_days: int) -> Dict[str, Any]:
        df = self.data_manager.demand_history.copy()
        if df.empty or "order_date" not in df or "order_qty" not in df:
            today = datetime.now()
            labels = [(today + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(forecast_days)]
            values = [100 for _ in range(forecast_days)]
            return {"labels": labels, "values": values, "confidence_upper": [120]*forecast_days, "confidence_lower": [80]*forecast_days}

        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df = df.dropna(subset=["order_date"])
        df["dow"] = df["order_date"].dt.dayofweek
        daily = df.groupby(['order_date', 'dow'])['order_qty'].sum().reset_index()
        dow_avg = daily.groupby('dow')["order_qty"].mean().to_dict()

        last = df["order_date"].max()
        future = [last + pd.Timedelta(days=i+1) for i in range(forecast_days)]
        labels = [d.strftime("%Y-%m-%d") for d in future]
        values = [int(max(0, round(dow_avg.get(d.dayofweek, 100)))) for d in future]

        return {
            "labels": labels,
            "values": values,
            "confidence_upper": [int(v * 1.2) for v in values],
            "confidence_lower": [int(v * 0.8) for v in values],
        }

    # ----------------------------------
    # Enhanced ML path (kept as backup)
    # ----------------------------------
    def _build_daily_qty(self) -> pd.DataFrame:
        df = self.data_manager.demand_history.copy()
        if df.empty or 'order_date' not in df or 'order_qty' not in df:
            return pd.DataFrame(columns=['order_date', 'order_qty'])
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df = df.dropna(subset=['order_date'])
        daily = (df.groupby(df['order_date'].dt.normalize())['order_qty']
                   .sum()
                   .reset_index()
                   .rename(columns={'order_date': 'order_date', 'order_qty': 'order_qty'}))
        return daily.sort_values('order_date').reset_index(drop=True)

    def _apply_daily_scenario(self, daily: pd.DataFrame, scenario: Dict[str, float]) -> pd.DataFrame:
        if daily.empty:
            return daily

        def _safe_mult(x, lo=MIN_MULT, hi=MAX_MULT, default=1.0):
            try:
                x = float(x)
            except Exception:
                return default
            return min(max(x, lo), hi)

        price_mult = _safe_mult(scenario.get('price', 1.0))

        raw_disc = scenario.get('discount', 1.0)
        if raw_disc is None:
            discount_mult = 1.0
        else:
            try:
                rd = float(raw_disc)
                discount_mult = rd if 0 < rd <= MAX_MULT else _safe_mult(1.0 - rd)
            except Exception:
                discount_mult = 1.0

        inflation_mult = _safe_mult(scenario.get('inflation', 1.0))

        out = daily.copy()
        out['order_qty'] = pd.to_numeric(out['order_qty'], errors='coerce').fillna(0.0)
        out['order_qty_scn'] = (out['order_qty'] * price_mult * discount_mult * inflation_mult).clip(lower=0)

        out['scenario_price_mult'] = price_mult
        out['scenario_discount_mult'] = discount_mult
        out['scenario_inflation_mult'] = inflation_mult
        out['scenario_intensity'] = float(np.sqrt((price_mult - 1) ** 2 +
                                                  (discount_mult - 1) ** 2 +
                                                  (inflation_mult - 1) ** 2))
        return out

    def _prepare_enhanced_scenario_dataset(self, scenario: Dict[str, float]) -> pd.DataFrame:
        daily = self._build_daily_qty()
        if daily.empty:
            return daily
        daily = self._apply_daily_scenario(daily, scenario)
        return daily

    def _prepare_enhanced_training_data(
        self, df: pd.DataFrame, force_target_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[float]]:
        if df.empty:
            return pd.DataFrame(), []

        df = df.sort_values('order_date').reset_index(drop=True)

        target_col = force_target_col or ('order_qty_scn' if 'order_qty_scn' in df.columns else 'order_qty')
        target = df[target_col].fillna(0.0).astype(float).tolist()

        leak_cols: List[str] = []
        if target_col == "order_qty_scn" and "order_qty" in df.columns:
            leak_cols.append("order_qty")
        if leak_cols:
            df = df.drop(columns=leak_cols)

        feature_df = pd.DataFrame()

        df['order_date'] = pd.to_datetime(df['order_date'])
        feature_df['year'] = df['order_date'].dt.year
        feature_df['month'] = df['order_date'].dt.month
        feature_df['day'] = df['order_date'].dt.day
        feature_df['dayofweek'] = df['order_date'].dt.dayofweek
        feature_df['quarter'] = df['order_date'].dt.quarter
        feature_df['is_weekend'] = (df['order_date'].dt.dayofweek >= 5).astype(int)

        feature_df['month_sin'] = np.sin(2 * np.pi * feature_df['month'] / 12)
        feature_df['month_cos'] = np.cos(2 * np.pi * feature_df['month'] / 12)
        feature_df['dayofweek_sin'] = np.sin(2 * np.pi * feature_df['dayofweek'] / 7)
        feature_df['dayofweek_cos'] = np.cos(2 * np.pi * feature_df['dayofweek'] / 7)
        feature_df['day_sin'] = np.sin(2 * np.pi * feature_df['day'] / 31)
        feature_df['day_cos'] = np.cos(2 * np.pi * feature_df['day'] / 31)

        for lag in [1, 3, 7, 14]:
            if len(df) > lag:
                feature_df[f'lag_{lag}'] = df[target_col].shift(lag).fillna(df[target_col].mean())
            else:
                feature_df[f'lag_{lag}'] = df[target_col].mean()

        for window in [3, 7, 14]:
            if len(df) > window:
                rolling = df[target_col].rolling(window=window, min_periods=1)
                feature_df[f'rolling_mean_{window}'] = rolling.mean().fillna(df[target_col].mean())
                feature_df[f'rolling_std_{window}'] = rolling.std().fillna(0.0)
                diffs = df[target_col].diff().fillna(0.0)
                feature_df[f'rolling_trend_{window}'] = diffs.rolling(window=window, min_periods=1).mean().fillna(0.0)
            else:
                feature_df[f'rolling_mean_{window}'] = df[target_col].mean()
                feature_df[f'rolling_std_{window}'] = 0.0
                feature_df[f'rolling_trend_{window}'] = 0.0

        for col in [c for c in df.columns if c.startswith('scenario_')]:
            feature_df[col] = df[col]

        return feature_df.fillna(0), target

    async def _train_enhanced_scenario_model(self, scenario: Dict[str, float], timeout_seconds: int = 30):
        start_time = time.time()
        df = self._prepare_enhanced_scenario_dataset(scenario)
        if df.empty:
            raise ValueError("No data available to train scenario model")

        MAX_ROWS = 100_000
        candidate_cols = ["order_qty_scn", "order_qty"]
        df = _pick_recent_nonzero_window(df, candidate_cols, max_rows=MAX_ROWS)
        if len(df) > MAX_ROWS:
            df = df.tail(MAX_ROWS)

        for _col in ("order_qty_scn", "order_qty"):
            if _col in df.columns:
                df[_col] = pd.to_numeric(df[_col], errors="coerce").fillna(0.0)

        target_primary = "order_qty_scn" if ("order_qty_scn" in df.columns and df["order_qty_scn"].std() > 0) else "order_qty"
        if target_primary not in df or float(df[target_primary].std() or 0.0) == 0.0:
            raise ValueError("No non-degenerate target available for scenario training.")

        df = _select_training_slice(df, target_primary, min_rows=200)
        df = df.sort_values("order_date").reset_index(drop=True)

        feature_df, target = self._prepare_enhanced_training_data(df, force_target_col=target_primary)
        if len(feature_df) < 20:
            raise ValueError("Insufficient samples for scenario training (need >=20)")

        _validate_training_ready(pd.DataFrame({target_primary: target}), target_primary)
        y_arr = np.asarray(target, dtype=float)

        split_date = df['order_date'].quantile(0.8)
        train_mask = (df['order_date'] <= split_date).to_numpy()
        val_mask = ~train_mask

        X_train = feature_df.iloc[train_mask]
        X_val = feature_df.iloc[val_mask]
        y_train = y_arr[train_mask]
        y_val = y_arr[val_mask]

        categorical_features = [col for col in X_train.columns if X_train[col].dtype == 'object']
        for col in categorical_features:
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')

        lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features or None)
        lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train, categorical_feature=categorical_features or None)

        model = lgb.train(
            self.quick_train_params,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)]
        )
        try:
            model.set_attr(target_unit="qty")
        except Exception:
            pass

        elapsed = time.time() - start_time
        logger.info("Enhanced scenario model trained in %.2fs with %d trees", elapsed, model.num_trees())
        return model

    async def _generate_enhanced_scenario_forecast(self, model, forecast_days: int, scenario: Dict[str, float]) -> Dict[str, Any]:
        # Build basic future features (quick)
        daily = self._build_daily_qty()
        if daily.empty:
            today = datetime.now()
            labels = [(today + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(forecast_days)]
            values = [100 for _ in range(forecast_days)]
            return {"labels": labels, "values": values, "confidence_upper": [120]*forecast_days, "confidence_lower": [80]*forecast_days, "feature_count": 0}

        last_date = daily["order_date"].max()
        dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]
        labels = [d.strftime("%Y-%m-%d") for d in dates]

        feature_rows = []
        series = daily["order_qty"]
        mean_v = float(series.mean()) if len(series) else 0.0
        for d in dates:
            r = {
                'year': d.year, 'month': d.month, 'day': d.day,
                'dayofweek': d.dayofweek, 'quarter': d.quarter, 'is_weekend': int(d.dayofweek >= 5),
                'month_sin': np.sin(2*np.pi*d.month/12.0), 'month_cos': np.cos(2*np.pi*d.month/12.0),
                'dayofweek_sin': np.sin(2*np.pi*d.dayofweek/7.0), 'dayofweek_cos': np.cos(2*np.pi*d.dayofweek/7.0),
                'day_sin': np.sin(2*np.pi*d.day/31.0), 'day_cos': np.cos(2*np.pi*d.day/31.0),
                'scenario_price_mult': float(scenario.get('price', 1.0)),
                'scenario_discount_mult': float(scenario.get('discount', 1.0)) if 0 < float(scenario.get('discount', 1.0)) <= MAX_MULT else float(1.0 - float(scenario.get('discount', 0.0))),
                'scenario_inflation_mult': float(scenario.get('inflation', 1.0)),
            }
            for lag in (1, 3, 7, 14):
                r[f'lag_{lag}'] = float(series.iloc[-lag]) if len(series) >= lag else mean_v
            for win in (3, 7, 14):
                if len(series) >= win:
                    recent = series.tail(win)
                    r[f'rolling_mean_{win}'] = float(recent.mean())
                    r[f'rolling_std_{win}'] = float(recent.std() or 0.0)
                    r[f'rolling_trend_{win}'] = float(recent.diff().fillna(0.0).mean())
                else:
                    r[f'rolling_mean_{win}'] = mean_v
                    r[f'rolling_std_{win}'] = 0.0
                    r[f'rolling_trend_{win}'] = 0.0
            feature_rows.append(r)

        X_future = pd.DataFrame(feature_rows)
        model_features = model.feature_name()
        missing = set(model_features) - set(X_future.columns)
        for m in missing:
            X_future[m] = 0
        X_future = X_future[model_features]
        preds = model.predict(X_future)

        # Convert (assumed qty) & apply simple uncertainty bands
        values = [int(max(0, round(float(p)))) for p in preds]
        cu = [int(round(v * 1.15)) for v in values]
        cl = [int(round(v * 0.85)) for v in values]
        return {"labels": labels, "values": values, "confidence_upper": cu, "confidence_lower": cl, "feature_count": len(model_features)}

    # ----------------------------------
    # Fallbacks
    # ----------------------------------
    def _create_fallback_response(self, baseline_forecast: Dict[str, Any], scenario: Dict[str, float], start_time: float) -> Dict[str, Any]:
        multiplier = np.mean(list(scenario.values())) if scenario else 1.0
        scenario_values = [int(round(v * multiplier)) for v in baseline_forecast['values']]
        deltas = [s - b for s, b in zip(scenario_values, baseline_forecast['values'])]

        return {
            "labels": baseline_forecast['labels'],
            "baseline_values": baseline_forecast['values'],
            "scenario_values": scenario_values,
            "values": scenario_values,
            "deltas": deltas,
            "confidence_upper": [int(v * 1.2) for v in scenario_values],
            "confidence_lower": [int(v * 0.8) for v in scenario_values],
            "metadata": {
                "scenario_hash": "fallback",
                "training_time_seconds": round(time.time() - start_time, 2),
                "model_cached": False,
                "scenario_multipliers": scenario,
                "forecasting_method": "enhanced_multiplier_fallback",
                "baseline_method": "stat_or_ml",
                "warning": "Training failed; used multiplier fallback",
                "feature_count": 0,
                "avg_delta": round(sum(deltas) / len(deltas), 2) if deltas else 0,
                "max_delta": max(deltas) if deltas else 0,
                "min_delta": min(deltas) if deltas else 0
            }
        }

    def _create_error_fallback(self, error_msg: str) -> Dict[str, Any]:
        today = datetime.now()
        labels = [(today + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(30)]
        values = [100 for _ in labels]
        return {
            "labels": labels,
            "values": values,
            "baseline_values": values,
            "scenario_values": values,
            "deltas": [0]*len(values),
            "confidence_upper": [120]*len(values),
            "confidence_lower": [80]*len(values),
            "metadata": {
                "scenario_hash": "error",
                "training_time_seconds": 0,
                "model_cached": False,
                "scenario_multipliers": {},
                "forecasting_method": "error_fallback",
                "baseline_method": "synthetic",
                "error": error_msg,
                "feature_count": 0,
                "avg_delta": 0,
                "max_delta": 0,
                "min_delta": 0,
                "cache_stats": self.cache.get_cache_stats(),
            }
        }


# ------------------------
# Recent nonzero window picker
# ------------------------
def _pick_recent_nonzero_window(
    df: pd.DataFrame,
    target_cols,
    max_rows: int = 100_000,
    min_nonzero: int = 50,
    blocks = (100_000, 60_000, 40_000, 20_000, 10_000, 5_000, 2_000)
) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.sort_values("order_date").reset_index(drop=True)

    has_signal_any = pd.Series(False, index=df.index)
    for col in target_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            has_signal_any = has_signal_any | (s > 0)

    if has_signal_any.any():
        last_sig_pos = int(np.where(has_signal_any.values)[0].max())
        end = last_sig_pos + 1
        cand = df.iloc[max(0, end - max_rows):end].copy()

        for blk in blocks:
            size = min(blk, max_rows, end)
            start_blk = max(0, end - size)
            sub = df.iloc[start_blk:end].copy()

            ok = False
            for col in target_cols:
                if col not in sub.columns:
                    continue
                series = pd.to_numeric(sub[col], errors="coerce").fillna(0.0)
                if (series > 0).sum() >= min_nonzero and float(series.std(skipna=True) or 0.0) > 0.0:
                    ok = True
                    break
            if ok:
                logger.info("Picked anchored window ending at last signal idx=%d: rows=%d", last_sig_pos, len(sub))
                return sub

        logger.warning("Anchored slice had weak signal; returning it to avoid all-zero tail: rows=%d", len(cand))
        return cand

    size = min(max_rows, len(df))
    logger.warning("No non-zero signal found; using most recent tail(%d) of %d rows", size, len(df))
    return df.tail(size).copy()
