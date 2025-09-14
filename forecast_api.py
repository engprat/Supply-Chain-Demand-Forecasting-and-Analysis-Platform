#new version
from fastapi import FastAPI, HTTPException, Query, Request, Depends, Response, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, date, timedelta, timezone 
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import math
import os
import uuid
import tracemalloc
import time
import hashlib
import json
import io

# Import scenario engine
from src.models.scenario_engine import ScenarioEngine

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reuse the CLI predictor to keep one source of truth
try:
    from src.models.predict_customer_orders import predict_range as _customer_orders_predict_range
except Exception as _imp_err:
    _customer_orders_predict_range = None
    logger.warning(f"customer_orders predict_range not importable: {_imp_err}")


# ---------- FastAPI App ----------
app = FastAPI(
    title="Forecast Management API",
    description="API for managing SKU-Channel-Location-Day level forecasts with promotional/non-promotional distinction",
    version="1.0.0"
)

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY_ENV = "FORECAST_API_KEY"

def require_api_key(request: Request):
    """If FORECAST_API_KEY is set, require x-api-key header to match."""
    expected = os.getenv(API_KEY_ENV)
    if not expected:
        return  # auth disabled
    provided = request.headers.get("x-api-key")
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# key -> {"ts": epoch_seconds, "data": <list[dict]>}
_PRED_CACHE: Dict[str, Dict[str, Any]] = {}
_PRED_TTL_SEC = 600  # 10 minutes

def _cache_get(key: str):
    rec = _PRED_CACHE.get(key)
    if not rec:
        return None
    if time.time() - rec["ts"] > _PRED_TTL_SEC:
        _PRED_CACHE.pop(key, None)
        return None
    return rec["data"]

def _cache_put(key: str, data: List[Dict[str, Any]]):
    _PRED_CACHE[key] = {"ts": time.time(), "data": data}


from datetime import datetime, date  # you already import these above

def safe_json_convert(obj):
    """Recursively convert object to be JSON serializable"""
    if isinstance(obj, dict):
        return {k: safe_json_convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_convert(item) for item in obj]
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return str(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj) or obj is None:
        return None
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    elif isinstance(obj, (datetime, date)):   # add this branch
        return obj.isoformat()
    else:
        return str(obj)

def enhanced_safe_float_conversion(value):
    """Safely convert values to float, handling NaN, inf, and other edge cases"""
    try:
        if pd.isna(value):
            return 0.0
        if isinstance(value, (int, float)):
            if pd.isinf(value):
                return 0.0
            return float(value)
        if isinstance(value, str):
            try:
                converted = float(value)
                if pd.isinf(converted) or pd.isna(converted):
                    return 0.0
                return converted
            except (ValueError, TypeError):
                return 0.0
        return 0.0
    except Exception:
        return 0.0


# ---------- Enums / Schemas ----------
class ChannelType(str, Enum):
    ONLINE = "online"
    RETAIL = "retail"
    WHOLESALE = "wholesale"
    MARKETPLACE = "marketplace"

class PromoType(str, Enum):
    PERCENTAGE_DISCOUNT = "percentage_discount"
    FIXED_AMOUNT = "fixed_amount"
    BOGO = "buy_one_get_one"
    SEASONAL = "seasonal"
    CLEARANCE = "clearance"

class AdHocAdjustmentRequest(BaseModel):
    sku_id: str
    channel: Optional[str] = None
    location: Optional[str] = None
    target_date: str
    current_forecast: float
    adjustment_type: str  # 'absolute', 'percentage', 'multiplier'
    adjustment_value: float
    reason_category: str
    notes: Optional[str] = None
    granularity_level: str
    new_forecast_value: float
    timestamp: str

class ScenarioParams(BaseModel):
    forecast_days: int = Field(default=30, ge=1, le=730)
    start_date: Optional[str] = None
    sku_id: Optional[str] = None  # Add this
    channel: Optional[str] = None  # Add this  
    location: Optional[str] = None  # Add this
    scenario: dict  # Scenario multipliers from frontend
    granularity_level: str = "sku_channel_location"  # You might also need this

class ForecastBase(BaseModel):
    sku_id: str
    location: str
    channel: ChannelType
    forecast_date: date
    base_demand_qty: float = Field(ge=0)
    promotional_demand_qty: float = Field(ge=0, default=0)
    is_promotional: bool = False
    confidence_level: float = Field(ge=0, le=1, default=0.8)

class ForecastCreate(ForecastBase):
    promo_id: Optional[str] = None
    promo_type: Optional[PromoType] = None
    discount_percent: Optional[float] = Field(None, ge=0, le=100)
    promo_start_date: Optional[date] = None
    promo_end_date: Optional[date] = None

class ForecastUpdate(BaseModel):
    base_demand_qty: Optional[float] = Field(None, ge=0)
    promotional_demand_qty: Optional[float] = Field(None, ge=0)
    is_promotional: Optional[bool] = None
    confidence_level: Optional[float] = Field(None, ge=0, le=1)
    promo_id: Optional[str] = None
    promo_type: Optional[PromoType] = None
    discount_percent: Optional[float] = Field(None, ge=0, le=100)
    promo_start_date: Optional[date] = None
    promo_end_date: Optional[date] = None

class ForecastResponse(ForecastBase):
    forecast_id: str
    total_demand_qty: float
    created_at: datetime
    updated_at: datetime
    promo_id: Optional[str] = None
    promo_type: Optional[str] = None
    discount_percent: Optional[float] = None
    promo_start_date: Optional[date] = None
    promo_end_date: Optional[date] = None

class ForecastFilter(BaseModel):
    sku_ids: Optional[List[str]] = None
    locations: Optional[List[str]] = None
    channels: Optional[List[ChannelType]] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    is_promotional: Optional[bool] = None
    promo_types: Optional[List[PromoType]] = None

# Frontend models
class DailyForecastParams(BaseModel):
    sku_id: Optional[str] = None
    channel: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    forecast_days: int = 30
    granularity_level: str = "sku_channel_location"

class CustomerForecastParams(BaseModel):
    customer_id: str
    sku_id: Optional[str] = None
    location: Optional[str] = None
    channel: Optional[str] = None
    forecast_days: int = Field(default=30, ge=1, le=365)
    start_date: Optional[date] = None

class PromotionalForecastParams(BaseModel):
    sku_id: Optional[str] = None
    channel: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    forecast_days: int = 30
    include_base_demand: bool = True
    include_promotional_demand: bool = True
    promo_type_filter: Optional[str] = None
    region: str = "USA"
    forecast_granularity: str = "daily"  
    promo_modeling_approach: str = "separated" 

class CustomerOrdersPredictRequest(BaseModel):
    start: date
    end: date
    customers: Optional[List[str]] = None
    products: Optional[List[str]] = None
    cities: Optional[List[str]] = None          # ← NEW
    # min_qty: Optional[float] = None
    # min_qty: Optional[float] = Field(None, ge=0) # ← NEW
    min_qty: Optional[float] = Field(None, ge=0, description="Minimum predicted qty filter (>= 0)")
    # api_key: Optional[str] = None  # keep if you’re using auth


# ---------- Data Manager ----------
class DataManager:
    def __init__(self):
        self.data_path = Path("data")
        self.processed_path = self.data_path / "processed"
        self.model_path = Path("models") / "supply_chain_model.pkl"
        self.forecasts_data = {}  # In-memory storage for demo
        self.adhoc_adjustments = []  # Add this for ad-hoc adjustments
        
        # Initialize all DataFrames
        self.weather_data = pd.DataFrame()
        self.sku_master = pd.DataFrame()
        self.promotional_calendar = pd.DataFrame()
        self.inventory_snapshots = pd.DataFrame()
        self.forecast_scenarios = pd.DataFrame()
        self.external_factors = pd.DataFrame()
        self.demand_history = pd.DataFrame()  # This will use your real processed dataset
        self.customer_master = pd.DataFrame()
        self.alerts_exceptions = pd.DataFrame()
        self.model = None
        
        # Load all data
        self.load_reference_data()
        self.load_processed_data()
        self.load_model()
    
    def load_reference_data(self):
        """Load reference data - enhanced to use both processed and raw data"""
        try:
            def maybe_read_processed(name):
                p = self.processed_path / name
                return pd.read_csv(p) if p.exists() else pd.DataFrame()
            
            def maybe_read_raw(name):
                p = self.data_path / name
                return pd.read_csv(p) if p.exists() else pd.DataFrame()
            
            # Load processed data first, fallback to raw
            self.weather_data = maybe_read_processed("kc_weather_data_complete_clean.csv")
            if self.weather_data.empty:
                self.weather_data = maybe_read_raw("kc_weather_data_complete.csv")
            
            self.sku_master = maybe_read_processed("kc_sku_master_data_complete_clean.csv")
            if self.sku_master.empty:
                self.sku_master = maybe_read_raw("kc_sku_master_data_complete.csv")
            
            self.promotional_calendar = maybe_read_processed("kc_promotional_calendar_complete_clean.csv")
            if self.promotional_calendar.empty:
                self.promotional_calendar = maybe_read_raw("kc_promotional_calendar_complete.csv")
            
            self.inventory_snapshots = maybe_read_processed("kc_inventory_snapshots_complete_clean.csv")
            if self.inventory_snapshots.empty:
                self.inventory_snapshots = maybe_read_raw("kc_inventory_snapshots_complete.csv")
            
            self.forecast_scenarios = maybe_read_processed("kc_forecast_scenarios_complete_clean.csv")
            if self.forecast_scenarios.empty:
                self.forecast_scenarios = maybe_read_raw("kc_forecast_scenarios_complete.csv")
            
            self.external_factors = maybe_read_processed("kc_external_factors_complete_clean.csv")
            if self.external_factors.empty:
                self.external_factors = maybe_read_raw("kc_external_factors_complete.csv")
            
            self.customer_master = maybe_read_processed("kc_customer_master_data_complete_clean.csv")
            if self.customer_master.empty:
                self.customer_master = maybe_read_raw("kc_customer_master_data_complete.csv")
            
            self.alerts_exceptions = maybe_read_processed("kc_alerts_exceptions_complete_clean.csv")
            if self.alerts_exceptions.empty:
                self.alerts_exceptions = maybe_read_raw("kc_alerts_exceptions_complete.csv")
            
            logger.info("Successfully loaded reference data files")
            
            # Convert date columns safely
            for df_name, df in [
                ("promotional_calendar", self.promotional_calendar),
                ("forecast_scenarios", self.forecast_scenarios),
                ("weather_data", self.weather_data),
                ("external_factors", self.external_factors),
                ("inventory_snapshots", self.inventory_snapshots),
                ("alerts_exceptions", self.alerts_exceptions)
            ]:
                if isinstance(df, pd.DataFrame) and not df.empty:
                    date_columns = [col for col in df.columns if 'date' in col.lower()]
                    for col in date_columns:
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except Exception as e:
                            logger.warning(f"Could not convert {col} in {df_name}: {e}")
                        
        except Exception as e:
            logger.error(f"Error loading reference data: {str(e)}")
            # Initialize empty DataFrames if files don't exist
            for attr in ['weather_data', 'sku_master', 'promotional_calendar', 
                        'inventory_snapshots', 'forecast_scenarios', 'external_factors',
                        'customer_master', 'alerts_exceptions']:
                setattr(self, attr, pd.DataFrame())
    
    def load_processed_data(self):
        """Load your main processed dataset"""
        try:
            # Load your real processed dataset
            dataset_path = self.processed_path / "dataset.csv"
            
            if dataset_path.exists():
                self.demand_history = pd.read_csv(
                    dataset_path,
                    low_memory=False,
                    dtype={"order_id": "string", "customer_id": "string", "product_id": "string"}
                )
                
                if not self.demand_history.empty:
                    # Convert order_date to datetime
                    if 'order_date' in self.demand_history.columns:
                        self.demand_history['order_date'] = pd.to_datetime(
                            self.demand_history['order_date'], errors='coerce'
                        )
                        # Drop rows with invalid dates
                        before_count = len(self.demand_history)
                        self.demand_history = self.demand_history.dropna(subset=['order_date'])
                        after_count = len(self.demand_history)
                        if before_count != after_count:
                            logger.warning(f"Dropped {before_count - after_count} rows with invalid dates")
                    
                    # Sort by date for consistency
                    if 'order_date' in self.demand_history.columns:
                        self.demand_history = self.demand_history.sort_values('order_date').reset_index(drop=True)
                    
                    logger.info(f"✅ Loaded processed dataset: {self.demand_history.shape}")
                    logger.info(f"📊 Date range: {self.demand_history['order_date'].min()} to {self.demand_history['order_date'].max()}")
                    logger.info(f"🏷️  Columns: {list(self.demand_history.columns)}")
                else:
                    logger.warning("Processed dataset is empty")
            else:
                logger.error(f"❌ Processed dataset not found at {dataset_path}")
                # Fallback to raw demand history
                raw_path = self.data_path / "kc_demand_history_complete.csv"
                if raw_path.exists():
                    self.demand_history = pd.read_csv(raw_path)
                    logger.info(f"Using raw demand history as fallback: {self.demand_history.shape}")
                else:
                    logger.error("No demand history data available")
                    self.demand_history = pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"Failed to load processed dataset: {e}")
            self.demand_history = pd.DataFrame()
    
    def load_model(self):
        """Load trained ML model if available"""
        try:
            if self.model_path.exists():
                import joblib
                self.model = joblib.load(self.model_path)
                logger.info(f"✅ Loaded ML model from {self.model_path}")
            else:
                logger.warning("⚠️  No trained model found - using statistical methods")
                self.model = None
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            self.model = None
    
    def get_sku_key(self) -> str:
        """Get the SKU column name from the dataset"""
        if 'sku_id' in self.demand_history.columns:
            return 'sku_id'
        elif 'product_id' in self.demand_history.columns:
            return 'product_id'
        else:
            return 'sku_id'  # default
    
    def get_location_key(self) -> str:
        """Get the location column name from the dataset"""
        if 'location' in self.demand_history.columns:
            return 'location'
        elif 'city' in self.demand_history.columns:
            return 'city'
        else:
            return 'location'  # default

    
    def get_sku_info(self, sku_id: str) -> Dict:
        """Get SKU information from master data - enhanced for real data"""
        if self.sku_master.empty:
            return {}
        
        sku_col = None
        for col in ['sku_id', 'product_id', 'SKU_ID', 'PRODUCT_ID']:
            if col in self.sku_master.columns:
                sku_col = col
                break
        
        if sku_col:
            sku_info = self.sku_master[self.sku_master[sku_col].astype(str) == str(sku_id)]
            if not sku_info.empty:
                return sku_info.iloc[0].to_dict()
        return {}

    def get_promotional_info(self, promo_id: str = None, forecast_date: date = None) -> Dict:
        """Get promotional information - enhanced for real data"""
        if self.promotional_calendar.empty:
            return {}
        
        promo_data = self.promotional_calendar.copy()
        
        if promo_id and 'promo_id' in promo_data.columns:
            promo_data = promo_data[promo_data['promo_id'].astype(str) == str(promo_id)]
        
        if forecast_date:
            start_col = None
            end_col = None
            
            for start_name in ['promo_start_date', 'start_date', 'Start_Date']:
                if start_name in promo_data.columns:
                    start_col = start_name
                    break
            
            for end_name in ['promo_end_date', 'end_date', 'End_Date']:
                if end_name in promo_data.columns:
                    end_col = end_name
                    break
            
            if start_col and end_col:
                try:
                    promo_data[start_col] = pd.to_datetime(promo_data[start_col], errors='coerce')
                    promo_data[end_col] = pd.to_datetime(promo_data[end_col], errors='coerce')
                    
                    promo_data = promo_data[
                        (promo_data[start_col].dt.date <= forecast_date) &
                        (promo_data[end_col].dt.date >= forecast_date)
                    ]
                except Exception as e:
                    logger.warning(f"Error filtering promotions by date: {e}")
        
        if not promo_data.empty:
            return promo_data.iloc[0].to_dict()
        return {}

    def get_demand_history(self, sku_id: str, location: str, channel: str, days: int = 30) -> List[Dict]:
        """Get historical demand data using real dataset"""
        if self.demand_history.empty:
            return []
        
        try:
            df = self.demand_history.copy()
            
            sku_col = self.get_sku_key()
            loc_col = self.get_location_key()
            
            if sku_id and sku_col in df.columns:
                df = df[df[sku_col].astype(str) == str(sku_id)]
            
            if location and loc_col in df.columns:
                df = df[df[loc_col].astype(str) == str(location)]
            
            if channel and 'channel' in df.columns:
                df = df[df['channel'].astype(str) == str(channel)]
            
            if 'order_date' in df.columns:
                df = df.sort_values('order_date').tail(days)
            else:
                df = df.tail(days)
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error getting demand history: {e}")
            return []

    def get_current_forecast(self, sku_id: str, channel: str, location: str, target_date: date) -> Optional[float]:
        """Get intelligent current forecast using real historical patterns"""
        try:
            if self.demand_history.empty:
                return None
            
            df = self.demand_history.copy()
            sku_col = self.get_sku_key()
            loc_col = self.get_location_key()
            
            original_size = len(df)
            if sku_id and sku_col in df.columns:
                df = df[df[sku_col].astype(str) == str(sku_id)]
            if channel and 'channel' in df.columns:
                df = df[df['channel'].astype(str) == str(channel)]
            if location and loc_col in df.columns:
                df = df[df[loc_col].astype(str) == str(location)]
            
            logger.info(f"Forecast calculation: filtered from {original_size} to {len(df)} records")
            
            if df.empty or 'order_qty' not in df.columns:
                return None
            
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
            df = df.dropna(subset=['order_date']).sort_values('order_date')
            
            target_dow = pd.Timestamp(target_date).dayofweek
            target_month = pd.Timestamp(target_date).month
            
            # 1. Day-of-week patterns (most specific)
            dow_data = df[df['order_date'].dt.dayofweek == target_dow]
            if not dow_data.empty and len(dow_data) >= 3:
                
                recent_dow = dow_data.tail(6)['order_qty'].mean()
                older_dow = dow_data.head(max(1, len(dow_data)-6))['order_qty'].mean() if len(dow_data) > 6 else recent_dow
                
                
                if older_dow > 0:
                    trend_factor = recent_dow / older_dow
                    forecast = recent_dow * min(max(trend_factor, 0.5), 2.0) 
                else:
                    forecast = recent_dow
                
                logger.info(f"Day-of-week forecast for {target_date} ({target_dow}): {forecast:.1f}")
                return float(forecast)
            
            # 2. Monthly seasonality
            month_data = df[df['order_date'].dt.month == target_month]
            if not month_data.empty and len(month_data) >= 5:
                monthly_avg = month_data['order_qty'].mean()
                recent_overall = df.tail(30)['order_qty'].mean()
                
                if recent_overall > 0:
                    seasonal_factor = monthly_avg / df['order_qty'].mean()
                    forecast = recent_overall * seasonal_factor
                    logger.info(f"Monthly seasonality forecast: {forecast:.1f}")
                    return float(forecast)
            
            # 3. Recent trend analysis
            if len(df) >= 14:
                recent_period = df.tail(7)['order_qty'].mean()
                older_period = df.tail(14).head(7)['order_qty'].mean()
                
                if older_period > 0:
                    trend = recent_period / older_period
                    base_forecast = recent_period
                    trended_forecast = base_forecast * min(max(trend, 0.7), 1.5)  
                    logger.info(f"Trend-based forecast: {trended_forecast:.1f}")
                    return float(trended_forecast)
            
            # 4. Simple recent average (fallback)
            recent_avg = df.tail(min(30, len(df)))['order_qty'].mean()
            logger.info(f"Simple average forecast: {recent_avg:.1f}")
            return float(recent_avg)
            
        except Exception as e:
            logger.error(f"Error calculating current forecast: {e}")
            return None
    
    def validate_sku_location_channel(self, sku_id: str, channel: str, location: str) -> Dict:
        """Enhanced validation using real processed dataset"""
        validation_result = {
            'valid': True,
            'sku_exists': False,
            'location_exists': False,
            'channel_exists': False,
            'combination_exists': False,
            'warnings': [],
            'suggestions': [],
            'historical_data_points': 0
        }
        
        try:
            if self.demand_history.empty:
                validation_result['warnings'].append("No historical data available for validation")
                return validation_result
            
            df = self.demand_history.copy()
            sku_col = self.get_sku_key()
            loc_col = self.get_location_key()
            
            if sku_col in df.columns:
                sku_exists = str(sku_id) in df[sku_col].astype(str).values
                validation_result['sku_exists'] = sku_exists
                if not sku_exists:
                    validation_result['warnings'].append(f"SKU {sku_id} not found in historical data")
                    all_skus = df[sku_col].astype(str).unique()
                    similar = [s for s in all_skus if sku_id.upper() in s.upper() or s.upper() in sku_id.upper()][:3]
                    if similar:
                        validation_result['suggestions'].append(f"Similar SKUs found: {', '.join(similar)}")
            
            if loc_col in df.columns:
                location_exists = str(location) in df[loc_col].astype(str).values
                validation_result['location_exists'] = location_exists
                if not location_exists and location:
                    validation_result['warnings'].append(f"Location {location} not found in historical data")
                    all_locations = df[loc_col].astype(str).unique()
                    similar = [l for l in all_locations if location.upper() in l.upper() or l.upper() in location.upper()][:3]
                    if similar:
                        validation_result['suggestions'].append(f"Similar locations found: {', '.join(similar)}")
            
            if 'channel' in df.columns:
                channel_exists = str(channel) in df['channel'].astype(str).values
                validation_result['channel_exists'] = channel_exists
                if not channel_exists and channel:
                    validation_result['warnings'].append(f"Channel {channel} not found in historical data")
                    all_channels = df['channel'].astype(str).unique()
                    validation_result['suggestions'].append(f"Available channels: {', '.join(all_channels)}")
            
            filtered_df = df.copy()
            if sku_id and sku_col in df.columns:
                filtered_df = filtered_df[filtered_df[sku_col].astype(str) == str(sku_id)]
            if channel and 'channel' in df.columns:
                filtered_df = filtered_df[filtered_df['channel'].astype(str) == str(channel)]
            if location and loc_col in df.columns:
                filtered_df = filtered_df[filtered_df[loc_col].astype(str) == str(location)]
            
            validation_result['combination_exists'] = not filtered_df.empty
            validation_result['historical_data_points'] = len(filtered_df)
            
            if filtered_df.empty:
                validation_result['warnings'].append(
                    f"No historical data for exact combination: SKU={sku_id}, Channel={channel}, Location={location}"
                )
            else:
                if 'order_date' in filtered_df.columns:
                    date_range = filtered_df['order_date'].agg(['min', 'max'])
                    validation_result['suggestions'].append(
                        f"Historical data available from {date_range['min'].strftime('%Y-%m-%d')} to {date_range['max'].strftime('%Y-%m-%d')}"
                    )
                
                if 'order_qty' in filtered_df.columns:
                    avg_qty = filtered_df['order_qty'].mean()
                    validation_result['suggestions'].append(f"Historical average quantity: {avg_qty:.1f}")
            
            if not self.sku_master.empty:
                sku_master_cols = [col for col in ['sku_id', 'product_id', 'SKU_ID', 'PRODUCT_ID'] if col in self.sku_master.columns]
                if sku_master_cols:
                    master_sku_col = sku_master_cols[0]
                    sku_in_master = str(sku_id) in self.sku_master[master_sku_col].astype(str).values
                    if not sku_in_master:
                        validation_result['warnings'].append(f"SKU {sku_id} not found in master catalog")
                    else:
                        sku_info = self.sku_master[self.sku_master[master_sku_col].astype(str) == str(sku_id)]
                        if not sku_info.empty and 'category' in sku_info.columns:
                            validation_result['suggestions'].append(f"SKU Category: {sku_info.iloc[0]['category']}")
        
        except Exception as e:
            logger.error(f"Enhanced validation error: {e}")
            validation_result['warnings'].append(f"Validation error: {str(e)}")
        
        return validation_result

    def get_adjustment_context(self, sku_id: str, channel: str, location: str, target_date: date) -> Dict:
        """Get rich context for adjustment decision making with safe serialization"""
        context = {
            'historical_patterns': {},
            'recent_trends': {},
            'seasonality': {},
            'promotional_context': {},
            'recommendations': []
        }
        
        try:
            if self.demand_history.empty:
                return context
            
            df = self.demand_history.copy()
            sku_col = self.get_sku_key()
            loc_col = self.get_location_key()
            
            if sku_id and sku_col in df.columns:
                df = df[df[sku_col].astype(str) == str(sku_id)]
            if channel and 'channel' in df.columns:
                df = df[df['channel'].astype(str) == str(channel)]
            if location and loc_col in df.columns:
                df = df[df[loc_col].astype(str) == str(location)]
            
            if df.empty:
                return context
            
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
            df = df.dropna(subset=['order_date']).sort_values('order_date')
            
            if 'order_qty' in df.columns:
                qty_series = df['order_qty'].dropna()
                if not qty_series.empty:
                    context['historical_patterns'] = {
                        'total_records': int(len(df)),
                        'avg_quantity': enhanced_safe_float_conversion(qty_series.mean()),
                        'min_quantity': enhanced_safe_float_conversion(qty_series.min()),
                        'max_quantity': enhanced_safe_float_conversion(qty_series.max()),
                        'std_quantity': enhanced_safe_float_conversion(qty_series.std()),
                        'date_range': {
                            'start': df['order_date'].min().strftime('%Y-%m-%d'),
                            'end': df['order_date'].max().strftime('%Y-%m-%d')
                        }
                    }
            
            if len(df) >= 10:
                try:
                    recent_30 = df.tail(min(30, len(df)//2))
                    older_30 = df.head(min(30, len(df)//2))
                    
                    if not recent_30.empty and not older_30.empty:
                        recent_avg = enhanced_safe_float_conversion(recent_30['order_qty'].mean())
                        older_avg = enhanced_safe_float_conversion(older_30['order_qty'].mean())
                        
                        trend_magnitude = 0.0
                        if older_avg > 0:
                            trend_magnitude = enhanced_safe_float_conversion(
                                abs(recent_avg - older_avg) / older_avg * 100
                            )
                        
                        context['recent_trends'] = {
                            'recent_avg': recent_avg,
                            'older_avg': older_avg,
                            'trend_direction': 'increasing' if recent_avg > older_avg else 'decreasing',
                            'trend_magnitude': trend_magnitude
                        }
                except Exception as e:
                    logger.warning(f"Error calculating recent trends: {e}")
            
            try:
                target_dow = pd.Timestamp(target_date).dayofweek
                dow_patterns = df.groupby(df['order_date'].dt.dayofweek)['order_qty'].agg(['mean', 'count'])
                
                if not dow_patterns.empty:
                    dow_dict = {}
                    for dow_idx in dow_patterns.index:
                        dow_dict[int(dow_idx)] = {
                            'mean': enhanced_safe_float_conversion(dow_patterns.loc[dow_idx, 'mean']),
                            'count': int(dow_patterns.loc[dow_idx, 'count'])
                        }
                    
                    context['seasonality']['day_of_week'] = dow_dict
                    
                    if target_dow in dow_patterns.index:
                        context['seasonality']['target_day_pattern'] = {
                            'avg_quantity': enhanced_safe_float_conversion(dow_patterns.loc[target_dow, 'mean']),
                            'data_points': int(dow_patterns.loc[target_dow, 'count'])
                        }
            except Exception as e:
                logger.warning(f"Error calculating day-of-week patterns: {e}")
            
            try:
                target_month = pd.Timestamp(target_date).month
                month_patterns = df.groupby(df['order_date'].dt.month)['order_qty'].agg(['mean', 'count'])
                
                if not month_patterns.empty and target_month in month_patterns.index:
                    context['seasonality']['target_month_pattern'] = {
                        'avg_quantity': enhanced_safe_float_conversion(month_patterns.loc[target_month, 'mean']),
                        'data_points': int(month_patterns.loc[target_month, 'count'])
                    }
            except Exception as e:
                logger.warning(f"Error calculating monthly patterns: {e}")
            
            try:
                if 'is_promotional' in df.columns:
                    promo_data = df[df['is_promotional'] == True]
                    base_data = df[df['is_promotional'] == False]
                    
                    if not promo_data.empty and not base_data.empty:
                        promo_avg = enhanced_safe_float_conversion(promo_data['order_qty'].mean())
                        base_avg = enhanced_safe_float_conversion(base_data['order_qty'].mean())
                        
                        promotional_lift = 0.0
                        if base_avg > 0:
                            promotional_lift = enhanced_safe_float_conversion(
                                (promo_avg / base_avg - 1) * 100
                            )
                        
                        context['promotional_context'] = {
                            'promotional_avg': promo_avg,
                            'base_avg': base_avg,
                            'promotional_lift': promotional_lift,
                            'promotional_frequency': enhanced_safe_float_conversion(
                                len(promo_data) / len(df) * 100
                            )
                        }
            except Exception as e:
                logger.warning(f"Error calculating promotional context: {e}")
            
            recommendations = []
            try:
                trend_magnitude = context.get('recent_trends', {}).get('trend_magnitude', 0)
                if trend_magnitude > 20:
                    trend_dir = context['recent_trends']['trend_direction']
                    recommendations.append(f"Strong {trend_dir} trend detected ({trend_magnitude:.1f}% change)")
                
                target_day_pattern = context.get('seasonality', {}).get('target_day_pattern', {})
                if target_day_pattern.get('data_points', 0) >= 3:
                    dow_avg = target_day_pattern['avg_quantity']
                    overall_avg = context.get('historical_patterns', {}).get('avg_quantity', 0)
                    if overall_avg > 0 and abs(dow_avg - overall_avg) / overall_avg > 0.15:
                        recommendations.append(f"Day-of-week pattern: {dow_avg:.1f} vs overall avg {overall_avg:.1f}")
                
                promotional_lift = context.get('promotional_context', {}).get('promotional_lift', 0)
                if promotional_lift > 20:
                    recommendations.append(f"Historical promotional lift: {promotional_lift:.1f}%")
                
                context['recommendations'] = recommendations
            except Exception as e:
                logger.warning(f"Error generating recommendations: {e}")
            
        except Exception as e:
            logger.error(f"Error generating adjustment context: {e}")
            context['error'] = str(e)
        
        return safe_json_convert(context)
    
    def store_adhoc_adjustment(self, adjustment_data: Dict) -> str:
        """Store ad-hoc forecast adjustment - keeping your working logic"""
        adjustment_id = str(uuid.uuid4())
        
        if not hasattr(self, 'adhoc_adjustments'):
            self.adhoc_adjustments = []
        
        adjustment_record = {
            **adjustment_data,
            'adjustment_id': adjustment_id,
            'created_at': datetime.now().isoformat()
        }
        
        self.adhoc_adjustments.append(adjustment_record)
        
        try:
            adjustments_df = pd.DataFrame(self.adhoc_adjustments)
            adjustments_df.to_csv(self.data_path / "adhoc_adjustments.csv", index=False)
            logger.info(f"Ad-hoc adjustment {adjustment_id} saved to CSV")
        except Exception as e:
            logger.warning(f"Could not save adjustment to CSV: {e}")
        
        return adjustment_id
    
    def generate_sample_data_summary(self):
        """Generate data summary using REAL data instead of synthetic"""
        try:
            if self.demand_history.empty:
                return {
                    "basic_stats": {"total_records": 0, "date_range": {"start": None, "end": None}},
                    "fr1_granularity_stats": {"unique_skus": 0, "unique_channels": 0, "unique_locations": 0,
                                              "unique_sku_channel_location_combinations": 0},
                    "fr2_promotional_stats": {"total_promotional_records": 0, "promotional_percentage": 0.0}
                }
            
            df = self.demand_history
            sku_col = self.get_sku_key()
            loc_col = self.get_location_key()
            
            total_records = len(df)
            
            start_date = None
            end_date = None
            if 'order_date' in df.columns:
                date_series = pd.to_datetime(df['order_date'], errors='coerce')
                if not date_series.isna().all():
                    start_date = date_series.min().strftime('%Y-%m-%d')
                    end_date = date_series.max().strftime('%Y-%m-%d')
            
            unique_skus = df[sku_col].nunique() if sku_col in df.columns else 0
            unique_locations = df[loc_col].nunique() if loc_col in df.columns else 0
            
            unique_channels = 0
            if 'channel' in df.columns:
                unique_channels = df['channel'].nunique()
            else:
                channel_cols = [col for col in df.columns if col.lower().startswith('channel_')]
                unique_channels = len(channel_cols)
            
            unique_combinations = 0
            if sku_col in df.columns and loc_col in df.columns:
                if 'channel' in df.columns:
                    unique_combinations = df[[sku_col, loc_col, 'channel']].drop_duplicates().shape[0]
                else:
                    unique_combinations = df[[sku_col, loc_col]].drop_duplicates().shape[0]
            
            promotional_records = 0
            promotional_percentage = 0.0
            if 'is_promotional' in df.columns:
                promotional_records = int(df['is_promotional'].sum())
                promotional_percentage = (promotional_records / total_records) * 100 if total_records > 0 else 0.0
            
            return {
                "basic_stats": {
                    "total_records": total_records,
                    "date_range": {"start": start_date, "end": end_date}
                },
                "fr1_granularity_stats": {
                    "unique_skus": unique_skus,
                    "unique_channels": unique_channels,
                    "unique_locations": unique_locations,
                    "unique_sku_channel_location_combinations": unique_combinations
                },
                "fr2_promotional_stats": {
                    "total_promotional_records": promotional_records,
                    "promotional_percentage": float(promotional_percentage)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating data summary: {e}")
            return {
                "error": f"Failed to generate summary: {str(e)}",
                "basic_stats": {"total_records": 0, "date_range": {"start": None, "end": None}},
                "fr1_granularity_stats": {"unique_skus": 0, "unique_channels": 0, "unique_locations": 0,
                                          "unique_sku_channel_location_combinations": 0},
                "fr2_promotional_stats": {"total_promotional_records": 0, "promotional_percentage": 0.0}
            }

class AggregationLevel(str, Enum):
    NATIONAL_DAY = "national_day"
    BRAND_DAY = "brand_day" 
    BRAND_LOCATION_DAY = "brand_location_day"
    BRAND_CHANNEL_DAY = "brand_channel_day"
    SKU_DAY = "sku_day"
    SKU_LOCATION_DAY = "sku_location_day"
    SKU_CHANNEL_DAY = "sku_channel_day"

class DrillDirection(str, Enum):
    DOWN = "down"    # More granular
    UP = "up"        # More aggregated

class AggregationRequest(BaseModel):
    level: AggregationLevel
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    forecast_days: int = Field(default=30, ge=1, le=730)
    
    sku_id: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None  
    location: Optional[str] = None
    channel: Optional[str] = None
    customer: Optional[str] = None
    
    # Options
    include_promotional: bool = True
    include_base_demand: bool = True
    currency: str = "USD"
    
    # Advanced
    parent_filters: Optional[Dict[str, Any]] = None

class DrillDownRequest(BaseModel):
    """Request model for drill-down operations"""
    from_level: AggregationLevel
    to_level: AggregationLevel
    direction: DrillDirection
    
    current_filters: Dict
    target_dimension: str 
    target_value: Optional[str] = None  
    
    # Time parameters
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    forecast_days: int = 30

class AggregationResponse(BaseModel):
    """Response model for aggregated forecasts"""
    level: AggregationLevel
    aggregation_key: str
    forecast_data: Dict
    summary_metrics: Dict
    drill_down_options: List[Dict]
    roll_up_options: List[Dict]
    data_quality: Dict
    drill_context: Optional[Dict] = None

class ForecastAggregationEngine:
    """Fixed drill-down implementation"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.hierarchy_map = self._build_hierarchy_map()
        self.aggregation_functions = self._setup_aggregation_functions()
    
    def _build_hierarchy_map(self) -> Dict:
        return {
            AggregationLevel.SKU_LOCATION_DAY: {
                "dimensions": ["sku_id", "location", "transaction_date"],
                "parent_levels": [AggregationLevel.SKU_DAY],
                "child_levels": []  # Most granular
            },
            AggregationLevel.SKU_CHANNEL_DAY: {
                "dimensions": ["sku_id", "channel", "transaction_date"],
                "parent_levels": [AggregationLevel.SKU_DAY],
                "child_levels": []
            },
            AggregationLevel.SKU_DAY: {
                "dimensions": ["sku_id", "transaction_date"],
                "parent_levels": [AggregationLevel.BRAND_DAY],  
                "child_levels": [AggregationLevel.SKU_LOCATION_DAY, AggregationLevel.SKU_CHANNEL_DAY]
            },
            AggregationLevel.BRAND_DAY: {
                "dimensions": ["brand", "transaction_date"],
                "parent_levels": [AggregationLevel.NATIONAL_DAY],
                "child_levels": [AggregationLevel.SKU_DAY, AggregationLevel.BRAND_LOCATION_DAY, AggregationLevel.BRAND_CHANNEL_DAY]
            },
            AggregationLevel.BRAND_LOCATION_DAY: {
                "dimensions": ["brand", "location", "transaction_date"],
                "parent_levels": [AggregationLevel.BRAND_DAY],
                "child_levels": [AggregationLevel.SKU_LOCATION_DAY]
            },
            AggregationLevel.BRAND_CHANNEL_DAY: {
                "dimensions": ["brand", "channel", "transaction_date"],
                "parent_levels": [AggregationLevel.BRAND_DAY],
                "child_levels": [AggregationLevel.SKU_CHANNEL_DAY]
            },
            AggregationLevel.NATIONAL_DAY: {
                "dimensions": ["transaction_date"],
                "parent_levels": [],  # Top level
                "child_levels": [AggregationLevel.BRAND_DAY] 
            }
        }
    
    def _setup_aggregation_functions(self) -> Dict:
        """Define how different metrics should be aggregated"""
        return {
            "additive": [
                "total_demand_qty", "base_demand_qty", "promotional_demand_qty",
                "current_inventory_units", "warehouse_inventory_qty",
                "customer_inventory_qty", "in_transit_qty"
            ],
            "weighted_avg": [
                "discount_percent", "expected_lift_factor", "seasonality_index",
                "demand_volatility", "competitor_activity_score"
            ],
            "simple_avg": [
                "historical_avg_demand_7d", "historical_avg_demand_30d",
                "prediction_interval_lower", "prediction_interval_upper",
                "forecast_confidence", "economic_indicator", "consumer_confidence"
            ],
            "max_latest": [
                "is_promotional", "manual_adjustment_flag", "is_frozen",
                "is_approved_version", "lifecycle_stage", "abc_classification"
            ]
        }
    
    def aggregate_forecast(self, request: AggregationRequest) -> AggregationResponse:
        """Main aggregation function"""
        try:
            logger.info(f"Aggregating forecast at level: {request.level}")
            
            df = self._get_filtered_dataset(request)
            
            if df.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for aggregation level {request.level}"
                )
            
            aggregated_data = self._perform_aggregation(df, request.level)
            
            forecast_data = self._generate_aggregated_forecast(
                aggregated_data, request
            )
            
            summary_metrics = self._calculate_summary_metrics(
                aggregated_data, request.level
            )
            
            drill_options = self._get_drill_down_options(request.level, request)
            rollup_options = self._get_roll_up_options(request.level, request)
            
            data_quality = self._assess_data_quality(aggregated_data, request.level)
            
            agg_key = self._generate_aggregation_key(request)
            
            return AggregationResponse(
                level=request.level,
                aggregation_key=agg_key,
                forecast_data=forecast_data,
                summary_metrics=summary_metrics,
                drill_down_options=drill_options,
                roll_up_options=rollup_options,
                data_quality=data_quality
            )
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Forecast aggregation failed: {str(e)}"
            )
    
    def drill_down_forecast(self, request: DrillDownRequest) -> List[AggregationResponse]:
        try:
            logger.info(f"Drilling down from {request.from_level} to {request.to_level}")
            
            if not self._validate_drill_path(request.from_level, request.to_level, request.direction):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid drill path from {request.from_level} to {request.to_level}"
                )
            
            drill_dimension = self._get_drill_dimension(request.from_level, request.to_level)
            
            if not drill_dimension:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot determine drill dimension from {request.from_level} to {request.to_level}"
                )
            
            target_values = self._get_drill_target_values(request, drill_dimension)
            
            if not target_values:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for drill dimension: {drill_dimension}"
                )
            
            logger.info(f"Drilling into {drill_dimension}, found {len(target_values)} values: {target_values[:5]}...")
            
            drill_results = []
            for value in target_values:
                try:
                   
                    agg_request = self._build_drill_aggregation_request(request, value, drill_dimension)
             
                    agg_response = self.aggregate_forecast(agg_request)
                    
                    agg_response.drill_context = {
                        "drill_dimension": drill_dimension,
                        "drill_value": value,
                        "parent_level": request.from_level.value,
                        "current_level": request.to_level.value
                    }
                    
                    drill_results.append(agg_response)
                    
                except Exception as e:
                    logger.warning(f"Failed to aggregate for {drill_dimension}={value}: {e}")
                    continue
            
            if not drill_results:
                raise HTTPException(
                    status_code=404,
                    detail=f"No successful drill results generated for {drill_dimension}"
                )
            
            logger.info(f"Drill-down successful: {len(drill_results)} results")
            return drill_results
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Drill-down failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Drill-down operation failed: {str(e)}"
            )
    
    def _get_drill_dimension(self, from_level: AggregationLevel, to_level: AggregationLevel) -> str:
        """FIXED: Determine which dimension we're drilling into"""
        try:
            from_dims = set(self.hierarchy_map[from_level]["dimensions"])
            to_dims = set(self.hierarchy_map[to_level]["dimensions"])
            
            new_dimensions = to_dims - from_dims
            
            if len(new_dimensions) == 1:
                return list(new_dimensions)[0]
            elif len(new_dimensions) > 1:
                priority_order = ["sku_id", "location", "channel", "brand", "customer"]
                for dim in priority_order:
                    if dim in new_dimensions:
                        return dim
                return list(new_dimensions)[0]  
            else:
                logger.warning(f"No new dimensions found: from={from_dims}, to={to_dims}")
                return "sku_id" 
                
        except Exception as e:
            logger.error(f"Error determining drill dimension: {e}")
            return "sku_id"  

    def _get_drill_target_values(self, request: DrillDownRequest, drill_dimension: str) -> List[str]:
        try:
            df = self.data_manager.demand_history.copy()
        
            if df.empty:
                logger.warning("No data available for drill target values")
                return self._get_fallback_dimension_values(drill_dimension)
        
            df = self._apply_drill_filters(df, request.current_filters)
        
            if df.empty:
                logger.warning("No data after applying current filters, using broader dataset")
                df = self.data_manager.demand_history.copy()
            
                filtered_filters = {k: v for k, v in request.current_filters.items() 
                              if k != 'sku_id' and v}
                df = self._apply_drill_filters(df, filtered_filters)
        
            column_mapping = {
                "sku_id": self.data_manager.get_sku_key(),
                "location": self.data_manager.get_location_key(),
                "channel": "channel",
                "brand": "brand",
                "customer": "customer",
                "category": "category",
                "transaction_date": "order_date"
            }
        
            actual_column = column_mapping.get(drill_dimension, drill_dimension)
            logger.info(f"Drill dimension '{drill_dimension}' mapped to column '{actual_column}'")
        
            if actual_column not in df.columns:
                logger.error(f"Column '{actual_column}' not found in data. Available: {list(df.columns)}")
                return self._get_fallback_dimension_values(drill_dimension)
        
            unique_values = df[actual_column].dropna()
            unique_values = unique_values[unique_values.astype(str).str.strip() != ""]
            unique_values = unique_values.astype(str).unique()
        
            unique_values_list = sorted(unique_values)[:50]
        
            logger.info(f"Found {len(unique_values_list)} unique values for {drill_dimension}")
        
            if not unique_values_list:
                return self._get_fallback_dimension_values(drill_dimension)
        
            return unique_values_list
        
        except Exception as e:
            logger.error(f"Error getting drill target values: {e}")
            return self._get_fallback_dimension_values(drill_dimension)

    
    def _apply_drill_filters(self, df: pd.DataFrame, current_filters: Dict) -> pd.DataFrame:
        try:
            filtered_df = df.copy()
            
            for filter_key, filter_value in current_filters.items():
                if not filter_value:  
                    continue
                
                column_mapping = {
                    "sku_id": self.data_manager.get_sku_key(),
                    "location": self.data_manager.get_location_key(),
                    "channel": "channel",
                    "brand": "brand",
                    "customer": "customer",
                    "category": "category"
                }
                
                actual_column = column_mapping.get(filter_key, filter_key)
                
                if actual_column in filtered_df.columns:
                    before_count = len(filtered_df)
                    filtered_df = filtered_df[filtered_df[actual_column].astype(str) == str(filter_value)]
                    after_count = len(filtered_df)
                    logger.info(f"Applied filter {filter_key}={filter_value}: {before_count} -> {after_count} records")
                else:
                    logger.warning(f"Filter column '{actual_column}' not found for filter '{filter_key}'")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error applying drill filters: {e}")
            return df
    

    def _get_filtered_dataset(self, request: AggregationRequest) -> pd.DataFrame:
        df = self.data_manager.demand_history.copy()
    
        if df.empty:
            logger.warning("Empty demand history dataset")
            return df
    
        logger.info(f"Starting with {len(df)} records")
    
        if 'order_date' in df.columns and 'transaction_date' not in df.columns:
            df['transaction_date'] = df['order_date']
            logger.info("Mapped order_date to transaction_date")
    
        if 'order_qty' in df.columns and 'total_demand_qty' not in df.columns:
            df['total_demand_qty'] = df['order_qty']
            logger.info("Mapped order_qty to total_demand_qty")
    
        original_count = len(df)

        if request.sku_id:
            sku_col = self.data_manager.get_sku_key()
            if sku_col in df.columns:
                before = len(df)
                df = df[df[sku_col].astype(str) == str(request.sku_id)]
                logger.info(f"SKU filter: {before} -> {len(df)} records")
    
        if request.brand:
            if 'brand' in df.columns:
                before = len(df)
                df = df[df['brand'].astype(str) == str(request.brand)]
                logger.info(f"Brand filter: {before} -> {len(df)} records")
            else:
                logger.info("Brand column not found, creating synthetic brand mapping")
                sku_col = self.data_manager.get_sku_key()
                if sku_col in df.columns:
                    df['brand'] = df[sku_col].astype(str).str[:3].map({
                        'HUG': 'Huggies',
                        'KLX': 'Kleenex', 
                        'SCT': 'Scott',
                        'KTX': 'Kotex',
                        'DEP': 'Depend',
                        'PSE': 'Poise'
                    }).fillna('Other')
                
                    before = len(df)
                    df = df[df['brand'].astype(str) == str(request.brand)]
                    logger.info(f"Synthetic brand filter: {before} -> {len(df)} records")
                else:
                    df['brand'] = 'Unknown'
                    if request.brand != 'Unknown':
                        df = df.iloc[0:0]  
                        logger.info("No SKU column found, filtered to empty for brand search")

        if request.brand and 'brand' not in df.columns:
            logger.info("Brand column not found, creating enhanced synthetic brand mapping")
            sku_col = self.data_manager.get_sku_key()
    
            def smart_brand_mapping(sku):
                sku_str = str(sku).upper()
                if any(x in sku_str for x in ['HUG', 'HUGG', 'H']):
                    return 'Huggies'
                elif any(x in sku_str for x in ['KLX', 'KLE', 'KLEEN']):
                    return 'Kleenex'  
                elif any(x in sku_str for x in ['SCT', 'SCO', 'SCOTT']):
                    return 'Scott'
                elif any(x in sku_str for x in ['KTX', 'KOT', 'KOTEX']):
                    return 'Kotex'
                elif any(x in sku_str for x in ['DEP', 'DEPEND']):
                    return 'Depend'
                return 'Other'
    
            df['brand'] = df[sku_col].apply(smart_brand_mapping)
    
        if request.category and 'category' in df.columns:
            before = len(df)
            df = df[df['category'].astype(str) == str(request.category)]
            logger.info(f"Category filter: {before} -> {len(df)} records")
    
        if request.location:
            loc_col = self.data_manager.get_location_key()
            if loc_col in df.columns:
                before = len(df)
                df = df[df[loc_col].astype(str) == str(request.location)]
                logger.info(f"Location filter: {before} -> {len(df)} records")
    
        if request.channel and 'channel' in df.columns:
            before = len(df)
            df = df[df['channel'].astype(str) == str(request.channel)]
            logger.info(f"Channel filter: {before} -> {len(df)} records")
    
        if request.customer and 'customer' in df.columns:
            before = len(df)
            df = df[df['customer'].astype(str) == str(request.customer)]
            logger.info(f"Customer filter: {before} -> {len(df)} records")
    
        if 'transaction_date' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        
            if request.start_date:
                start_dt = pd.to_datetime(request.start_date)
                before = len(df)
                df = df[df['transaction_date'] >= start_dt]
                logger.info(f"Start date filter: {before} -> {len(df)} records")
        
            if request.end_date:
                end_dt = pd.to_datetime(request.end_date)
                before = len(df)
                df = df[df['transaction_date'] <= end_dt]
                logger.info(f"End date filter: {before} -> {len(df)} records")
    
        if not request.include_promotional and 'is_promotional' in df.columns:
            before = len(df)
            df = df[df['is_promotional'] == False]
            logger.info(f"Promotional filter: {before} -> {len(df)} records")
    
        result_df = df.dropna(subset=['transaction_date']) if 'transaction_date' in df.columns else df
    
        final_count = len(df)
        logger.info(f"Final filtered dataset: {final_count} records (filtered from {original_count})")
        
        if df.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found after applying filters. Original: {original_count} records, Final: 0 records"
            )
    
        return result_df
    
    def _get_fallback_dimension_values(self, drill_dimension: str) -> List[str]:
        fallback_values = {
            "sku_id": ["HUG1001", "KLX2001", "SCT3001", "KTX4001", "DEP5001"],
            "location": ["USA-Northeast", "USA-Southeast", "USA-Midwest", "USA-West", "China-Tier1","China-Tier2","Korea-Seoul","Korea-Other","Brazil-Urban","Mexico-Urban "],
            "channel": ["Online", "Retail", "Wholesale", "Marketplace"],
            "brand": ["Huggies", "Kleenex", "Scott", "Kotex", "Depend"],
            "customer": ["Walmart", "Target", "CVS", "Amazon", "Kroger"],
            "category": ["Personal Care", "Tissue", "Baby Care", "Adult Care"]
        }
        return fallback_values.get(drill_dimension, ["Sample1", "Sample2", "Sample3"])
    
    def _perform_aggregation(self, df: pd.DataFrame, level: AggregationLevel) -> pd.DataFrame:
        hierarchy_info = self.hierarchy_map.get(level)
        if not hierarchy_info:
            raise ValueError(f"Unknown aggregation level: {level}")

        dimension_mapping = {
            "sku_id": self.data_manager.get_sku_key(),
            "location": self.data_manager.get_location_key(),
            "transaction_date": "transaction_date", 
            "channel": "channel",
            "customer": "customer",
            "brand": "brand",
            "category": "category"
        }

        group_dimensions = []
        for dim in hierarchy_info["dimensions"]:
            actual_col = dimension_mapping.get(dim, dim)
            if actual_col and actual_col in df.columns:
                group_dimensions.append(actual_col)

        if not group_dimensions:
            if 'transaction_date' in df.columns:
                group_dimensions = ['transaction_date']
            else:
                raise ValueError(f"No valid dimensions found for level {level}")
        
        if 'transaction_date' in group_dimensions and 'transaction_date' in df.columns:
            df = df.copy()
            df['transaction_date'] = pd.to_datetime(df['transaction_date']).dt.date
            logger.info("Converting transaction_date to daily grouping")

        agg_dict = {}

        if 'total_demand_qty' in df.columns:
            agg_dict['total_demand_qty'] = 'sum'
        if 'order_qty' in df.columns:
            agg_dict['order_qty'] = 'sum'
        if 'base_demand_qty' in df.columns:
            agg_dict['base_demand_qty'] = 'sum'
        if 'promotional_demand_qty' in df.columns:
            agg_dict['promotional_demand_qty'] = 'sum'

        count_column = None
        for col in ['total_demand_qty', 'order_qty', 'product_id', 'order_id']:
            if col in df.columns:
                count_column = col
                break
    
        if count_column:
            agg_dict['record_count'] = 'count'

        for col in ['revenue', 'price', 'discount_percent']:
            if col in df.columns:
                agg_dict[col] = 'sum'

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (col not in agg_dict and 
                col not in group_dimensions and 
                col != count_column): 
                agg_dict[col] = 'sum'

        logger.info(f"Fixed aggregation functions: {agg_dict}")
        logger.info(f"Grouping by: {group_dimensions}")

        try:
            if count_column and 'record_count' in agg_dict:
                count_agg = {count_column: 'count'}
                other_agg = {k: v for k, v in agg_dict.items() if k != 'record_count'}
            
                if other_agg:
                    aggregated = df.groupby(group_dimensions).agg(other_agg).reset_index()
                else:
                    aggregated = df.groupby(group_dimensions).size().reset_index(name='temp_size')
            
                record_counts = df.groupby(group_dimensions).size().reset_index(name='record_count')
            
                if other_agg:
                    aggregated = aggregated.merge(record_counts, on=group_dimensions, how='left')
                else:
                    aggregated = record_counts
                    if 'temp_size' in aggregated.columns:
                        aggregated = aggregated.drop('temp_size', axis=1)
            else:
                aggregated = df.groupby(group_dimensions).agg(agg_dict).reset_index()
        
            if isinstance(aggregated.columns, pd.MultiIndex):
                aggregated.columns = ['_'.join(col).strip() if isinstance(col, tuple) and col[1] else 
                            (col[0] if isinstance(col, tuple) else col)
                            for col in aggregated.columns.values]
        
            logger.info(f"Aggregation successful: {len(aggregated)} aggregated records")
            logger.info(f"Final aggregated columns: {list(aggregated.columns)}")
            return aggregated
    
        except Exception as e:
            logger.error(f"Fixed aggregation failed with error: {e}")
            logger.error(f"Attempted agg_dict: {agg_dict}")
            logger.error(f"Group dimensions: {group_dimensions}")
            logger.error(f"Available columns: {list(df.columns)}")
    
            try:
                logger.info("Attempting fallback aggregation...")
            
                if 'total_demand_qty' in df.columns:
                    simple_agg = df.groupby(group_dimensions)['total_demand_qty'].sum().reset_index()
                    record_counts = df.groupby(group_dimensions).size().reset_index(name='record_count')
                    simple_agg = simple_agg.merge(record_counts, on=group_dimensions, how='left')
                    logger.info("Fallback successful with total_demand_qty")
                    return simple_agg
                
                elif 'order_qty' in df.columns:
                    simple_agg = df.groupby(group_dimensions)['order_qty'].sum().reset_index()
                    simple_agg.rename(columns={'order_qty': 'total_demand_qty'}, inplace=True)
                    record_counts = df.groupby(group_dimensions).size().reset_index(name='record_count')
                    simple_agg = simple_agg.merge(record_counts, on=group_dimensions, how='left')
                    logger.info("Fallback successful with order_qty")
                    return simple_agg
                
                else:
                    logger.info("Using ultimate fallback - count records only")
                    result = df.groupby(group_dimensions[0] if group_dimensions else 'transaction_date').size().reset_index(name='total_demand_qty')
                    result['record_count'] = result['total_demand_qty']  # Use count as demand
                    logger.info("Ultimate fallback successful")
                    return result
                
            except Exception as fallback_error:
                logger.error(f"Even fallback aggregation failed: {fallback_error}")
                return pd.DataFrame({
                    group_dimensions[0] if group_dimensions else 'transaction_date': [pd.Timestamp.now().date()],
                    'total_demand_qty': [1000],
                    'record_count': [1]
                })

    
    def _generate_aggregated_forecast(self, aggregated_data: pd.DataFrame, 
                                request: AggregationRequest) -> Dict:
        if aggregated_data.empty:
            return {"error": "No aggregated data available for forecasting"}
    
        try:
            logger.info(f"Generating forecast from {len(aggregated_data)} aggregated records")
            logger.info(f"Aggregated data columns: {list(aggregated_data.columns)}")
    
            if 'transaction_date' in aggregated_data.columns:
                aggregated_data = aggregated_data.sort_values('transaction_date')
                logger.info(f"Date range in aggregated data: {aggregated_data['transaction_date'].min()} to {aggregated_data['transaction_date'].max()}")

            forecast_values = []
            dates = []
            confidence_upper = []
            confidence_lower = []
        
            qty_column = None
            for col in ['total_demand_qty', 'order_qty', 'demand_qty', 'quantity']:
                if col in aggregated_data.columns:
                    qty_column = col
                    break
    
            if qty_column is None:
                logger.error(f"No quantity column found in aggregated data: {list(aggregated_data.columns)}")
                return {"error": "No quantity column found for forecasting"}
    
            logger.info(f"Using quantity column: {qty_column}")
    
            qty_series = aggregated_data[qty_column].dropna()
            if qty_series.empty:
                logger.error("Quantity series is empty after removing nulls")
                return {"error": "No valid quantity data for forecasting"}
    
            recent_avg = float(qty_series.tail(min(30, len(qty_series))).mean())
            recent_std = float(qty_series.tail(min(30, len(qty_series))).std())
            overall_avg = float(qty_series.mean())
    
            logger.info(f"Forecast base stats - Recent avg: {recent_avg:.1f}, Overall avg: {overall_avg:.1f}, Std: {recent_std:.1f}")
    
            if recent_avg <= 0 or pd.isna(recent_avg):
                recent_avg = max(overall_avg, 100) 
                logger.warning(f"Recent avg was invalid, using fallback: {recent_avg}")
    
            if recent_std <= 0 or pd.isna(recent_std):
                recent_std = recent_avg * 0.2  
    
            trend_slope = 0
            if len(qty_series) >= 14:
                try:
                    recent_period = qty_series.tail(7).mean()
                    older_period = qty_series.tail(14).head(7).mean()
            
                    if older_period > 0 and not pd.isna(recent_period) and not pd.isna(older_period):
                        trend_slope = (recent_period - older_period) / 7
                        trend_slope = max(-recent_avg * 0.1, min(trend_slope, recent_avg * 0.1))
                        logger.info(f"Calculated trend slope: {trend_slope:.2f}")
                except Exception as e:
                    logger.warning(f"Trend calculation failed: {e}")
                    trend_slope = 0
    
            last_date = pd.Timestamp.now()
            if 'transaction_date' in aggregated_data.columns and not aggregated_data.empty:
                try:
                    last_date = aggregated_data['transaction_date'].max()
                    if pd.isna(last_date):
                        last_date = pd.Timestamp.now()
                except Exception as e:
                    logger.warning(f"Error getting last date: {e}")
                    last_date = pd.Timestamp.now()
    
            logger.info(f"Starting forecast from date: {last_date}")
    
            for i in range(request.forecast_days):
                forecast_date = last_date + pd.Timedelta(days=i+1)
                dates.append(forecast_date.strftime('%Y-%m-%d'))
        
                base_forecast = recent_avg
        
                base_forecast += trend_slope * (i + 1)
        
                try:
                    dow = pd.Timestamp(forecast_date).dayofweek
                    dow_multipliers = [0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.15]  # Mon-Sun
                    base_forecast *= dow_multipliers[dow]
                except Exception as e:
                    logger.warning(f"Day-of-week adjustment failed: {e}")
        
                try:
                    variation = np.random.normal(0, recent_std * 0.1)  # Small variation
                    base_forecast += variation
                except Exception as e:
                    logger.warning(f"Variation calculation failed: {e}")
        
                base_forecast = max(1, base_forecast)  # Minimum of 1
        
                forecast_value = round(base_forecast)
                forecast_values.append(forecast_value)
        
                upper_bound = round(base_forecast + 1.96 * recent_std)
                lower_bound = round(max(0, base_forecast - 1.96 * recent_std))
        
                confidence_upper.append(upper_bound)
                confidence_lower.append(lower_bound)
    
            logger.info(f"Generated forecast range: {min(forecast_values)} - {max(forecast_values)}")
    
            forecast_result = {
                "dates": dates,
                "forecast_values": forecast_values,
                "confidence_upper": confidence_upper,
                "confidence_lower": confidence_lower,
                "aggregation_level": request.level.value,
                "base_period_avg": round(recent_avg),
                "forecast_method": f"aggregated_{request.level.value}",
                "trend_slope": round(trend_slope, 3),
                "data_points_used": len(qty_series),
                "quantity_column_used": qty_column,
                "forecast_stats": {
                    "min_forecast": min(forecast_values) if forecast_values else 0,
                    "max_forecast": max(forecast_values) if forecast_values else 0,
                    "avg_forecast": round(sum(forecast_values) / len(forecast_values)) if forecast_values else 0,
                    "total_forecast": sum(forecast_values) if forecast_values else 0
                }
            }
    
            return forecast_result
    
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            logger.exception("Full forecast generation error:")
            return {
                "error": f"Forecast generation failed: {str(e)}",
                "aggregated_data_shape": list(aggregated_data.shape) if not aggregated_data.empty else [0, 0],
                "aggregated_data_columns": list(aggregated_data.columns) if not aggregated_data.empty else []
            }
            
    
    def _calculate_summary_metrics(self, aggregated_data: pd.DataFrame, 
                                 level: AggregationLevel) -> Dict:
        """Calculate summary metrics for aggregated data"""
        metrics = {
            "total_records": len(aggregated_data),
            "aggregation_level": level.value,
            "date_range": {},
            "demand_metrics": {},
            "promotional_metrics": {},
            "geographical_coverage": {}
        }
        
        try:
            # Date range metrics
            if 'transaction_date' in aggregated_data.columns:
                metrics["date_range"] = {
                    "start_date": aggregated_data['transaction_date'].min().strftime('%Y-%m-%d'),
                    "end_date": aggregated_data['transaction_date'].max().strftime('%Y-%m-%d'),
                    "total_days": (aggregated_data['transaction_date'].max() - 
                                 aggregated_data['transaction_date'].min()).days
                }
            
            # Demand metrics
            if 'total_demand_qty' in aggregated_data.columns:
                metrics["demand_metrics"] = {
                    "total_demand": int(aggregated_data['total_demand_qty'].sum()),
                    "average_daily_demand": round(aggregated_data['total_demand_qty'].mean()),
                    "demand_volatility": round(aggregated_data['total_demand_qty'].std(), 2),
                    "min_demand": int(aggregated_data['total_demand_qty'].min()),
                    "max_demand": int(aggregated_data['total_demand_qty'].max())
                }
            
            # Promotional metrics
            if 'promotional_demand_qty' in aggregated_data.columns:
                total_promo = aggregated_data['promotional_demand_qty'].sum()
                total_base = aggregated_data.get('base_demand_qty', pd.Series([0])).sum()
                
                metrics["promotional_metrics"] = {
                    "total_promotional_demand": int(total_promo),
                    "promotional_percentage": round(
                        (total_promo / (total_promo + total_base)) * 100, 2
                    ) if (total_promo + total_base) > 0 else 0,
                    "average_promotional_lift": round(
                        aggregated_data.get('expected_lift_factor', pd.Series([0])).mean(), 2
                    )
                }
            
            # Geographical coverage
            if 'location' in aggregated_data.columns:
                metrics["geographical_coverage"] = {
                    "unique_locations": aggregated_data['location'].nunique(),
                    "top_locations": aggregated_data['location'].value_counts().head(5).to_dict()
                }
            
        except Exception as e:
            logger.warning(f"Error calculating summary metrics: {e}")
            metrics["calculation_error"] = str(e)
        
        return metrics
    
    def _get_drill_down_options(self, current_level: AggregationLevel, 
                          request: AggregationRequest) -> List[Dict]:
        options = []
    
        drill_paths = {
            AggregationLevel.NATIONAL_DAY: [
                AggregationLevel.BRAND_DAY
            ],
            AggregationLevel.BRAND_DAY: [
                AggregationLevel.BRAND_LOCATION_DAY,
                AggregationLevel.BRAND_CHANNEL_DAY,
                AggregationLevel.SKU_DAY
            ],
            AggregationLevel.BRAND_LOCATION_DAY: [
                AggregationLevel.SKU_LOCATION_DAY
            ],
            AggregationLevel.BRAND_CHANNEL_DAY: [
                AggregationLevel.SKU_CHANNEL_DAY
            ],
            AggregationLevel.SKU_DAY: [
                AggregationLevel.SKU_LOCATION_DAY,
                AggregationLevel.SKU_CHANNEL_DAY
        ]
        }
    
        available_levels = drill_paths.get(current_level, [])
    
        for level in available_levels:
            dimension_preview = self._get_dimension_preview(level, request)
        
            options.append({
                "target_level": level.value,
                "display_name": level.value.replace('_', ' ').title(),
                "available_dimensions": dimension_preview,
                "estimated_records": self._estimate_drill_records(level, request)
            })
    
        return options
    
    def _get_roll_up_options(self, current_level: AggregationLevel, 
                           request: AggregationRequest) -> List[Dict]:
        """Get available roll-up options from current level"""
        hierarchy_info = self.hierarchy_map.get(current_level, {})
        parent_levels = hierarchy_info.get("parent_levels", [])
        
        options = []
        for level in parent_levels:
            options.append({
                "target_level": level.value,
                "display_name": level.value.replace('_', ' ').title(),
                "aggregation_benefit": "Broader view with reduced granularity"
            })
        
        return options
    
    def _assess_data_quality(self, aggregated_data: pd.DataFrame, 
                           level: AggregationLevel) -> Dict:
        """Assess data quality for the aggregated dataset"""
        quality = {
            "overall_score": 0,
            "completeness": {},
            "consistency": {},
            "temporal_coverage": {},
            "recommendations": []
        }
        
        try:
            if aggregated_data.empty:
                quality["overall_score"] = 0
                quality["recommendations"].append("No data available for assessment")
                return quality
            
            # Completeness assessment
            total_cols = len(aggregated_data.columns)
            complete_cols = len(aggregated_data.columns) - aggregated_data.isnull().any().sum()
            completeness_score = (complete_cols / total_cols) * 100 if total_cols > 0 else 0
            
            quality["completeness"] = {
                "score": round(completeness_score, 2),
                "missing_columns": aggregated_data.columns[aggregated_data.isnull().any()].tolist(),
                "complete_records": len(aggregated_data.dropna())
            }
            
            # Temporal coverage
            if 'transaction_date' in aggregated_data.columns:
                date_range = (aggregated_data['transaction_date'].max() - 
                            aggregated_data['transaction_date'].min()).days
                expected_records = date_range  
                actual_records = len(aggregated_data)
                coverage_score = min((actual_records / expected_records) * 100, 100) if expected_records > 0 else 100
                
                quality["temporal_coverage"] = {
                    "score": round(coverage_score, 2),
                    "date_gaps": expected_records - actual_records,
                    "continuity": "Good" if coverage_score > 80 else "Fair" if coverage_score > 60 else "Poor"
                }
            
            # Overall score calculation
            scores = [
                quality["completeness"]["score"],
                quality["temporal_coverage"].get("score", 100)
            ]
            quality["overall_score"] = round(sum(scores) / len(scores), 2)
            
            # Recommendations
            if completeness_score < 90:
                quality["recommendations"].append("Some data fields have missing values")
            if quality["temporal_coverage"].get("score", 100) < 80:
                quality["recommendations"].append("Temporal data coverage could be improved")
            if len(aggregated_data) < 30:
                quality["recommendations"].append("Limited data points may affect forecast accuracy")
            
        except Exception as e:
            logger.warning(f"Data quality assessment failed: {e}")
            quality["assessment_error"] = str(e)
        
        return quality
    
    def _generate_aggregation_key(self, request: AggregationRequest) -> str:
        """Generate unique key for aggregation request"""
        key_parts = [
            request.level.value,
            request.sku_id or "all",
            request.brand or "all",
            request.category or "all", 
            request.location or "all",
            request.channel or "all",
            request.customer or "all"
        ]
        return "_".join(key_parts)
    
    def _validate_drill_path(self, from_level: AggregationLevel, 
                           to_level: AggregationLevel, direction: DrillDirection) -> bool:
        try:
            from_info = self.hierarchy_map.get(from_level)
            to_info = self.hierarchy_map.get(to_level)
            
            if not from_info or not to_info:
                logger.error(f"Unknown levels: from={from_level}, to={to_level}")
                return False
            
            if direction == DrillDirection.DOWN:
                valid_children = from_info.get("child_levels", [])
                is_valid = to_level in valid_children
                logger.info(f"Drill down validation: {from_level} -> {to_level}, valid children: {valid_children}, is_valid: {is_valid}")
                return is_valid
                
            elif direction == DrillDirection.UP:
                valid_parents = from_info.get("parent_levels", [])
                is_valid = to_level in valid_parents
                logger.info(f"Drill up validation: {from_level} -> {to_level}, valid parents: {valid_parents}, is_valid: {is_valid}")
                return is_valid
            
            return False
            
        except Exception as e:
            logger.error(f"Error validating drill path: {e}")
            return False

    def _build_drill_filters(self, request: DrillDownRequest) -> Dict:
        """Build filters for drill-down operation"""
        try:
            return request.current_filters.copy()
        except Exception as e:
            logger.error(f"Error building drill filters: {e}")
            return {}

    def _get_target_dimension_values(self, request: DrillDownRequest, 
                                   filters: Dict) -> List[str]:
        """Get unique values for the target drill dimension"""
        try:
            df = self.data_manager.demand_history.copy()
            
            if df.empty:
                logger.warning("No data available for drill dimension values")
                return []
            
            for key, value in filters.items():
                if key in df.columns and value:
                    df = df[df[key].astype(str) == str(value)]
            
            dimension_mapping = {
                "sku_id": self.data_manager.get_sku_key(),
                "location": self.data_manager.get_location_key(),
                "channel": "channel",
                "customer": "customer",
                "brand": "brand",
                "category": "category"
            }
            
            actual_column = dimension_mapping.get(request.target_dimension, request.target_dimension)
            
            if actual_column in df.columns:
                unique_values = df[actual_column].dropna().astype(str).unique()
                return unique_values[:20].tolist()
            
            logger.warning(f"Target dimension '{request.target_dimension}' (mapped to '{actual_column}') not found in data")
            return []
            
        except Exception as e:
            logger.error(f"Error getting target dimension values: {e}")
            return []

    def _build_drill_aggregation_request(self, drill_request: DrillDownRequest, 
                                       target_value: str, drill_dimension: str) -> AggregationRequest:
        try:
           
            filters = drill_request.current_filters.copy()
            
            filters[drill_dimension] = target_value
            
            logger.info(f"Building aggregation request with filters: {filters}")
            
            agg_request = AggregationRequest(
                level=drill_request.to_level,
                forecast_days=drill_request.forecast_days
            )
            
            if drill_request.start_date:
                agg_request.start_date = drill_request.start_date
            if drill_request.end_date:
                agg_request.end_date = drill_request.end_date
            
            if 'sku_id' in filters and filters['sku_id']:
                agg_request.sku_id = filters['sku_id']
            if 'brand' in filters and filters['brand']:
                agg_request.brand = filters['brand']
            if 'category' in filters and filters['category']:
                agg_request.category = filters['category']
            if 'location' in filters and filters['location']:
                agg_request.location = filters['location']
            if 'channel' in filters and filters['channel']:
                agg_request.channel = filters['channel']
            if 'customer' in filters and filters['customer']:
                agg_request.customer = filters['customer']
            
            logger.info(f"Created aggregation request: {agg_request.model_dump()}")
            return agg_request
            
        except Exception as e:
            logger.error(f"Error building drill aggregation request: {e}")
          
            return AggregationRequest(
                level=drill_request.to_level,
                forecast_days=drill_request.forecast_days
            )

    def _get_dimension_preview(self, level: AggregationLevel, 
                         request: AggregationRequest) -> Dict:
        try:
            df = self.data_manager.demand_history
        
            if df.empty:
                return {
                    "sample_values": [],
                    "total_unique_values": 0
                }
        
            dimension_map = {
            AggregationLevel.SKU_DAY: self.data_manager.get_sku_key(),
            AggregationLevel.BRAND_DAY: "brand", 
            AggregationLevel.SKU_LOCATION_DAY: self.data_manager.get_location_key(),
            AggregationLevel.SKU_CHANNEL_DAY: "channel",
            AggregationLevel.BRAND_LOCATION_DAY: "brand",  
            AggregationLevel.BRAND_CHANNEL_DAY: "brand", 
            AggregationLevel.NATIONAL_DAY: "total"  
        }
        
            target_column = dimension_map.get(level)
        
            if target_column and target_column != "total" and target_column in df.columns:
                unique_values = df[target_column].dropna().astype(str).unique()
                sample_values = unique_values[:3].tolist()
                total_unique = len(unique_values)
            
                return {
                    "sample_values": sample_values,
                    "total_unique_values": total_unique
                }
        
            return {
                "sample_values": ["National"],
                "total_unique_values": 1
            }
        
        except Exception as e:
            logger.error(f"Error getting dimension preview: {e}")
            return {
                "sample_values": [],
                "total_unique_values": 0
            }

    def _estimate_drill_records(self, level: AggregationLevel, 
                          request: AggregationRequest) -> int:
        try:
            df = self.data_manager.demand_history
        
            if df.empty:
                return 0
        
            filtered_df = df.copy()
        
            if request.sku_id:
                sku_col = self.data_manager.get_sku_key()
                if sku_col in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[sku_col].astype(str) == str(request.sku_id)]
        
            if request.location:
                loc_col = self.data_manager.get_location_key()
                if loc_col in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[loc_col].astype(str) == str(request.location)]
        
            if request.channel and 'channel' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['channel'].astype(str) == str(request.channel)]
        
            level_multipliers = {
            AggregationLevel.NATIONAL_DAY: 1,
            AggregationLevel.BRAND_DAY: 10, 
            AggregationLevel.SKU_DAY: 100,
            AggregationLevel.BRAND_LOCATION_DAY: 50, 
            AggregationLevel.BRAND_CHANNEL_DAY: 30, 
            AggregationLevel.SKU_LOCATION_DAY: 500,
            AggregationLevel.SKU_CHANNEL_DAY: 300
        }
        
            base_estimate = len(filtered_df) // 30 
            multiplier = level_multipliers.get(level, 100)
        
            return min(base_estimate * multiplier // 100, len(filtered_df))
        
        except Exception as e:
            logger.error(f"Error estimating drill records: {e}")
            return 100 
    
    def get_filter_options(self, filter_type: str, existing_filters: Dict = None) -> Dict:
        """Get available options for filter dropdowns"""
        try:
            df = self.data_manager.demand_history.copy()
            
            if df.empty:
                logger.warning("No data available for filter options")
                return {"options": [], "total": 0}
            
            if existing_filters:
                for key, value in existing_filters.items():
                    if key != filter_type and value and key in df.columns:
                        df = df[df[key].astype(str) == str(value)]
            
            if filter_type not in df.columns:
                logger.warning(f"Column '{filter_type}' not found in dataset")
                return {"options": [], "total": 0}
            
            value_counts = df[filter_type].dropna().value_counts().head(100) 
            
            options = []
            for value, count in value_counts.items():
                options.append({
                    "value": str(value),
                    "label": str(value),
                    "count": int(count)
                })
            
            return {
                "options": options,
                "total": len(value_counts),
                "column_name": filter_type,
                "filter_type": filter_type
            }
            
        except Exception as e:
            logger.error(f"Error getting filter options for {filter_type}: {e}")
            return {"options": [], "total": 0, "error": str(e)}
    
    def get_all_filter_options(self) -> Dict:
        """Get all available filter options for the UI - automatically detect from dataset"""
        try:
            df = self.data_manager.demand_history.copy()
            
            if df.empty:
                return {"error": "No data available"}
            
            filter_options = {}
            
            exclude_patterns = [
                'qty', 'quantity', 'amount', 'price', 'cost', 'value', 'total', 
                'sum', 'count', 'avg', 'mean', 'std', 'min', 'max', 'percent',
                'rate', 'ratio', 'index', 'score', 'weight', 'volume', 'units'
            ]
            
            date_patterns = ['date', 'time', 'timestamp', 'day', 'month', 'year']
            
            potential_filter_columns = []
            
            for col in df.columns:
                col_lower = col.lower()
                
                is_metric = any(pattern in col_lower for pattern in exclude_patterns)
                is_date = any(pattern in col_lower for pattern in date_patterns)
                
                if not is_metric and not is_date:
                    unique_count = df[col].nunique()
                    total_count = len(df)
                    
                    if 1 < unique_count <= min(1000, total_count * 0.5):
                        potential_filter_columns.append({
                            'column': col,
                            'unique_count': unique_count,
                            'data_type': str(df[col].dtype)
                        })
            
            potential_filter_columns.sort(key=lambda x: x['unique_count'])
            
            for col_info in potential_filter_columns:
                col_name = col_info['column']
                filter_options[col_name] = self.get_filter_options(col_name)
                filter_options[col_name]['is_detected'] = True
                filter_options[col_name]['unique_count'] = col_info['unique_count']
                filter_options[col_name]['data_type'] = col_info['data_type']
            
            date_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in date_patterns):
                    try:
                        pd.to_datetime(df[col].dropna().head(100), errors='raise')
                        date_columns.append(col)
                    except:
                        pass
            
            if date_columns:
                filter_options["date_columns"] = {}
                for date_col in date_columns:
                    try:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        filter_options["date_columns"][date_col] = {
                            "min_date": df[date_col].min().strftime('%Y-%m-%d') if not df[date_col].min() is pd.NaT else None,
                            "max_date": df[date_col].max().strftime('%Y-%m-%d') if not df[date_col].max() is pd.NaT else None,
                            "total_days": (df[date_col].max() - df[date_col].min()).days if not df[date_col].isnull().all() else 0,
                            "null_count": df[date_col].isnull().sum()
                        }
                    except Exception as e:
                        logger.warning(f"Error processing date column {date_col}: {e}")
            
            filter_options["dataset_summary"] = {
                "total_records": len(df),
                "total_columns": len(df.columns),
                "detected_filter_columns": len(potential_filter_columns),
                "detected_date_columns": len(date_columns),
                "all_columns": df.columns.tolist(),
                "column_types": {col: str(df[col].dtype) for col in df.columns}
            }
            
            return filter_options
            
        except Exception as e:
            logger.error(f"Error getting all filter options: {e}")
            return {"error": str(e)}
    
    def get_dataset_structure(self) -> Dict:
        """Analyze and return the complete structure of the dataset"""
        try:
            df = self.data_manager.demand_history
            
            if df.empty:
                return {"valid": False, "error": "No data available"}
            
            structure_info = {
                "valid": True,
                "basic_info": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "memory_usage": df.memory_usage(deep=True).sum()
                },
                "columns": {},
                "recommended_filters": [],
                "recommended_metrics": [],
                "data_quality": {}
            }
            
            for col in df.columns:
                col_info = {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "unique_count": df[col].nunique(),
                    "null_count": df[col].isnull().sum(),
                    "null_percentage": round((df[col].isnull().sum() / len(df)) * 100, 2),
                    "sample_values": df[col].dropna().astype(str).head(5).tolist()
                }
                
                unique_ratio = col_info["unique_count"] / len(df)
                col_lower = col.lower()
                
                if any(pattern in col_lower for pattern in ['date', 'time']):
                    col_info["category"] = "date"
                    col_info["recommended_use"] = "time_filter"
                elif any(pattern in col_lower for pattern in ['qty', 'amount', 'price', 'total', 'count']):
                    col_info["category"] = "metric"
                    col_info["recommended_use"] = "aggregation_target"
                    structure_info["recommended_metrics"].append(col)
                elif unique_ratio < 0.1:  
                    col_info["category"] = "categorical"
                    col_info["recommended_use"] = "filter"
                    structure_info["recommended_filters"].append({
                        "column": col,
                        "unique_count": col_info["unique_count"],
                        "priority": "high" if unique_ratio < 0.05 else "medium"
                    })
                elif unique_ratio < 0.5:  
                    col_info["category"] = "categorical"
                    col_info["recommended_use"] = "filter"
                    structure_info["recommended_filters"].append({
                        "column": col,
                        "unique_count": col_info["unique_count"],
                        "priority": "medium"
                    })
                else:
                    col_info["category"] = "identifier"
                    col_info["recommended_use"] = "identifier"
                
                structure_info["columns"][col] = col_info
            
            total_cells = len(df) * len(df.columns)
            null_cells = df.isnull().sum().sum()
            
            structure_info["data_quality"] = {
                "completeness_percentage": round(((total_cells - null_cells) / total_cells) * 100, 2),
                "duplicate_rows": df.duplicated().sum(),
                "columns_with_nulls": df.isnull().any().sum(),
                "recommendations": []
            }
            
            if structure_info["data_quality"]["completeness_percentage"] < 90:
                structure_info["data_quality"]["recommendations"].append("Dataset has significant missing values")
            
            if structure_info["data_quality"]["duplicate_rows"] > 0:
                structure_info["data_quality"]["recommendations"].append(f"Dataset contains {structure_info['data_quality']['duplicate_rows']} duplicate rows")
            
            if len(structure_info["recommended_filters"]) == 0:
                structure_info["data_quality"]["recommendations"].append("No suitable filter columns detected")
            
            return structure_info
            
        except Exception as e:
            logger.error(f"Error analyzing dataset structure: {e}")
            return {"valid": False, "error": str(e)}

def generate_realistic_forecast_from_data(data_manager: DataManager, sku_id: str = None, 
                                        channel: str = None, location: str = None, 
                                        days: int = 30) -> Dict:
    """Generate realistic forecast using actual historical patterns"""
    try:
        if data_manager.demand_history.empty:
            logger.warning("No real data available, using synthetic fallback")
            return generate_forecast_data(days)
        
        df = data_manager.demand_history.copy()
        sku_col = data_manager.get_sku_key()
        loc_col = data_manager.get_location_key()
        
        if sku_id and sku_col in df.columns:
            df = df[df[sku_col].astype(str) == str(sku_id)]
        
        if channel and 'channel' in df.columns:
            df = df[df['channel'].astype(str) == str(channel)]
        
        if location and loc_col in df.columns:
            df = df[df[loc_col].astype(str) == str(location)]
        
        if df.empty:
            logger.warning("No data after filtering, using synthetic fallback")
            return generate_forecast_data(days)
        
        if 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
            df = df.dropna(subset=['order_date'])
        
        if df.empty or 'order_qty' not in df.columns:
            return generate_forecast_data(days)
        
        labels = []
        values = []
        confidence_upper = []
        confidence_lower = []
        
        dow_patterns = df.groupby(df['order_date'].dt.dayofweek)['order_qty'].mean()
        recent_avg = df['order_qty'].tail(30).mean()
        recent_std = df['order_qty'].tail(30).std()
        
        if len(df) > 7:
            recent_trend = (df['order_qty'].tail(7).mean() - df['order_qty'].tail(14).mean()) / 7
        else:
            recent_trend = 0
        
        start_date = datetime.now()
        
        for i in range(days):
            date = start_date + timedelta(days=i)
            labels.append(date.strftime('%b %d'))
            
            dow = date.weekday()
            base_value = float(dow_patterns.get(dow, recent_avg))
            
            trend_value = base_value + (recent_trend * i)
            
            variation = np.random.normal(0, recent_std * 0.1)
            final_value = max(0, trend_value + variation)
            
            values.append(round(final_value))
            confidence_upper.append(round(final_value * 1.2))
            confidence_lower.append(round(final_value * 0.8))
        
        feature_count = len([col for col in df.columns if col not in ['order_date']])
        features_used = ["real_historical_demand", "day_of_week_patterns", "trend_analysis"]
        
        if data_manager.model is not None:
            features_used.append("ml_model_available")
        
        return {
            "labels": labels,
            "values": values,
            "confidence_upper": confidence_upper,
            "confidence_lower": confidence_lower,
            "feature_count": feature_count,
            "features_used": features_used,
            "data_source": "real_processed_dataset"
        }
        
    except Exception as e:
        logger.error(f"Error generating realistic forecast: {e}")
        return generate_forecast_data(days)

@app.post("/api/customer-forecast")
async def generate_customer_forecast(params: CustomerForecastParams):
    try:
        # 1. Validate model exists
        model_path = Path("models/customer_order_model.pkl")
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Customer order model not trained yet")

        model = joblib.load(model_path)

        # 2. Get history subset
        df = _build_customer_history(params)

        if df.empty:
            raise HTTPException(status_code=404, detail="No historical data for requested filters")

        # 3. Build future frame (forecast horizon)
        future = _build_future_frame(df, params.forecast_days)

        # 4. Merge external + promotional data
        future = _merge_external_factors(future, params)

        # 5. Predict
        preds = model.predict(future)

        # 6. Return results
        return {
            "customer_id": params.customer_id,
            "sku_id": params.sku_id,
            "dates": future["order_date"].astype(str).tolist(),
            "predicted_orders": [int(max(0, p)) for p in preds],
            "features_used": list(future.columns),
            "data_source": "customer_forecast_pipeline"
        }

    except Exception as e:
        logger.error(f"Customer forecast failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/customer-orders/predict")
async def customer_orders_predict(
    body: CustomerOrdersPredictRequest,
    page: int = Query(1, ge=1),
    page_size: int = Query(500, ge=1, le=5000),
    min_pct: Optional[float] = Query(None, ge=0, le=100),  # e.g. 90 means p90
    sort_by: Optional[str] = Query(None, description="e.g. 'pred_order_qty' or 'date'"),
    sort_dir: Literal["asc", "desc"] = Query("desc"),
    format: Literal["json", "csv"] = Query("json"),
    download_all: bool = Query(False, description="If format=csv, return all filtered rows instead of just the page"),
):
    """
    Serve customer-order predictions using the trained LightGBM model and
    the same feature engineering used by the CLI (predict_customer_orders.py).
    """
    t0 = time.perf_counter() 
    try:
        if _customer_orders_predict_range is None:
            raise HTTPException(
                status_code=500,
                detail=(
                    "predict_range not available (import failed). "
                    "Ensure src.models.predict_customer_orders is importable."
                ),
            )

        model_path = Path("models/customer_order_model.pkl")
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "Customer order model not found. Train it via: "
                    "python -m src.models.train_customer_orders"
                ),
            )

        # 1) Parse dates FIRST
        start_ts = pd.to_datetime(body.start)
        end_ts   = pd.to_datetime(body.end)

        # --- POST-FILTER GUARD (case-insensitive) ---
        def _norm(x: object) -> str:
            return str(x).strip().lower() if x is not None else ""

        def _norm_list(xs) -> list[str]:
            return [_norm(v) for v in xs] if xs else []


        # Validate dates early
        if pd.isna(start_ts) or pd.isna(end_ts):
            raise HTTPException(status_code=422, detail="Invalid start/end date.")
        if end_ts < start_ts:
            raise HTTPException(status_code=422, detail="'end' must be >= 'start'.")
        max_days = 180
        if (end_ts - start_ts).days + 1 > max_days:
            raise HTTPException(status_code=422, detail=f"Date range too large; max {max_days} days.")

        customers = body.customers or []
        products  = body.products or []

        # 2) Build a JSON-serializable cache key
        key_src = {
           "start": start_ts.date().isoformat(),
            "end": end_ts.date().isoformat(),
            "customers": sorted(_norm_list(body.customers)) if body.customers else None,
            "products": sorted(_norm_list(body.products)) if body.products else None,
            "cities": sorted(_norm_list(getattr(body, "cities", None))) if getattr(body, "cities", None) else None,
            "min_qty": float(body.min_qty) if body.min_qty is not None else None,
            "min_pct": float(min_pct) if min_pct is not None else None,   # <-- add
            "sort_by": sort_by,                                           # <-- add (optional)
            "sort_dir": sort_dir,                                         # <-- add (optional)
            "format": format,                                             # <-- add (optional)
            "download_all": download_all,                                 # <-- add (optional)
            "page": page,
            "page_size": page_size,
        }
        cache_key = hashlib.sha1(json.dumps(key_src, sort_keys=True).encode()).hexdigest()

        # 3) Run the same pipeline the CLI uses
        df = _customer_orders_predict_range(
            start_ts, end_ts,
            customers=customers,
            products=products,
        )

        if df is None or (hasattr(df, "empty") and df.empty):
            return {
                "status": "ok",
                "message": "No predictions for the requested window / filters.",
                "start": key_src["start"],
                "end": key_src["end"],
                "count": 0,
                "page": page,
                "page_size": page_size,
                "returned": 0,
                "predictions": [],
            }

        # 4) Normalize column names
        out_df = df.copy()
        for cand in ["date", "order_date", "forecast_date"]:
            if cand in out_df.columns:
                out_df.rename(columns={cand: "date"}, inplace=True)
                break

        # 5) Make all datetime-like columns JSON-safe
        for col in out_df.columns:
            if pd.api.types.is_datetime64_any_dtype(out_df[col]):
                out_df[col] = out_df[col].dt.strftime("%Y-%m-%d")
            elif out_df[col].dtype == "object":
                out_df[col] = out_df[col].apply(
                    lambda x: x.isoformat() if isinstance(x, (datetime, date)) else x
                )

        # --- POST-FILTER GUARD (apply even if the CLI helper didn't) ---
        # normalize types
        # Ensure string dtype on id/location-like cols
        # Normalize dtypes for safe string ops
        for c in ("customer_id", "product_id", "city", "location"):
            if c in out_df.columns:
                out_df[c] = out_df[c].astype("string").fillna("")

        # Apply filters (case-insensitive, single-pass)
        if body.customers and "customer_id" in out_df.columns:
            cust_set = set(_norm_list(body.customers))
            out_df = out_df[out_df["customer_id"].map(_norm).isin(cust_set)]

        if body.products and "product_id" in out_df.columns:
            prod_set = set(_norm_list(body.products))
            out_df = out_df[out_df["product_id"].map(_norm).isin(prod_set)]

        # Cities: dataset uses "city" (but support "location" if present)
        if getattr(body, "cities", None):
            city_col = "city" if "city" in out_df.columns else ("location" if "location" in out_df.columns else None)
            if city_col:
                city_set = set(_norm_list(body.cities))
                out_df = out_df[out_df[city_col].map(_norm).isin(city_set)]

            # ---- quantity column & filters (BEFORE pagination) ----
        qty_col = next((c for c in ["pred_order_qty", "prediction", "yhat", "forecast_qty"]
                        if c in out_df.columns), None)

        stats = None
        if qty_col:
            # work with numeric series
            s_all = pd.to_numeric(out_df[qty_col], errors="coerce")

            # (1) percentile filter
            if min_pct is not None and s_all.notna().any():
                pct_threshold = float(s_all.quantile(min_pct / 100.0))
                out_df = out_df[s_all >= pct_threshold]
                s_all = pd.to_numeric(out_df[qty_col], errors="coerce")  # refresh

            # (2) numeric min_qty
            if body.min_qty is not None:
                out_df = out_df[pd.to_numeric(out_df[qty_col], errors="coerce") >= float(body.min_qty)]
                s_all = pd.to_numeric(out_df[qty_col], errors="coerce")  # refresh

            # (3) stats on the post-filter set
            s = s_all.dropna()
            if not s.empty:
                q = s.quantile([0.5, 0.9, 0.95]).to_dict()
                stats = {
                    "count": int(s.size),
                    "min": float(s.min()),
                    "median": float(q.get(0.5, float("nan"))),
                    "p90": float(q.get(0.9, float("nan"))),
                    "p95": float(q.get(0.95, float("nan"))),
                    "max": float(s.max()),
                    "mean": float(s.mean()),
                    "std": float(s.std(ddof=1) if s.size > 1 else 0.0),
                }

            # (4) round for presentation
            out_df[qty_col] = pd.to_numeric(out_df[qty_col], errors="coerce").fillna(0).round().astype(int)
        # -------------------------------------------------------

        # ---------- Sorting (deterministic) ----------
        sort_col = sort_by or (qty_col if qty_col else ("date" if "date" in out_df.columns else None))
        if sort_col and sort_col in out_df.columns:
            ascending = (sort_dir == "asc")
            # if sorting by qty_col ensure it's numeric (already int if qty_col block ran)
            if qty_col and sort_col == qty_col:
                out_df[qty_col] = pd.to_numeric(out_df[qty_col], errors="coerce").fillna(0)

            by_cols = [sort_col] + (["date"] if sort_col != "date" and "date" in out_df.columns else [])
            out_df = out_df.sort_values(by=by_cols, ascending=ascending)
        else:
            fallback_cols = [c for c in ["date","customer_id","product_id","city","location"] if c in out_df.columns]
            if fallback_cols:
                out_df = out_df.sort_values(fallback_cols, ascending=True)


        # ---------------------------------------------
        # helpful debug
        logger.info("Post-filter rows=%s (customers=%s, products=%s)",
            len(out_df), body.customers, body.products)

        # 6) Pagination
        total = len(out_df)
        start_idx = (page - 1) * page_size
        end_idx   = start_idx + page_size
        page_df   = out_df.iloc[start_idx:end_idx]

        # --- CSV export (before JSON) ---
        if format and format.lower() == "csv":
            export_df = out_df if download_all else page_df

            # (optional) make sure quantity column is int for a clean CSV
            if qty_col and qty_col in export_df.columns:
                export_df[qty_col] = pd.to_numeric(export_df[qty_col], errors="coerce").fillna(0).round().astype(int)

            buf = io.StringIO()
            export_df.to_csv(buf, index=False)
            buf.seek(0)

            fname = f"customer_orders_{key_src['start']}_{key_src['end']}.csv"
            headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
            return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=headers)
        # --------------------------------

        records = [safe_json_convert(rec) for rec in page_df.to_dict(orient="records")]
        generated_ms = int((time.perf_counter() - t0) * 1000)

        return {
            "status": "ok",
            "start": key_src["start"],
            "end": key_src["end"],
            "count": total,
            "page": page,
            "page_size": page_size,
            "returned": len(records),
            "predictions": records,
            "stats": stats,
            "cache_key": cache_key,
            "generated_ms": generated_ms,          # <-- add this
            "filters_applied": {                   # optional but helpful
                "customers": body.customers,
                "products": body.products,
                "cities": getattr(body, "cities", None),
                "min_qty": body.min_qty,
                "min_pct": min_pct,
                "sort_by": sort_by,
                "sort_dir": sort_dir,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Customer orders prediction failed")
        raise HTTPException(status_code=500, detail=f"Customer orders prediction failed: {e}")

@app.get("/api/customer-orders/options")
async def customer_orders_options(
    q: Optional[str] = None,
    column: Optional[str] = None,          # "customer_id" | "product_id" | "city"
    limit: int = Query(50, ge=1, le=500),
):
    """
    Return distinct values for customers/products/cities to populate dropdowns.
    Optional q (case-insensitive contains) and per-column fetch.
    """
    try:
        df = data_manager.demand_history.copy()
        if df.empty:
            return {"customers": [], "products": [], "cities": [], "total_rows": 0}

        def _vals(col: str) -> list[str]:
            if col not in df.columns:
                return []
            s = df[col].dropna().astype(str)
            if q:
                s = s[s.str.contains(str(q), case=False, na=False)]
            # most frequent first
            return s.value_counts().head(limit).index.tolist()

        if column:
            return {column: _vals(column), "total_rows": int(len(df))}

        return {
            "customers": _vals("customer_id"),
            "products":  _vals("product_id"),
            "cities":    _vals("city"),
            "total_rows": int(len(df)),
        }
    except Exception as e:
        logger.exception("options endpoint failed")
        raise HTTPException(status_code=500, detail=f"Options lookup failed: {e}")

data_manager = DataManager()
data_manager.load_reference_data()
data_manager.load_processed_data()
data_manager.load_model()
aggregation_engine = ForecastAggregationEngine(data_manager)

# --- SCENARIO ENGINE (module-scope singleton) ---
scenario_engine = ScenarioEngine(data_manager)
SCENARIO_MAX_FORECAST_DAYS = 365

if data_manager.demand_history.empty:
    logger.error("❌ CRITICAL: No demand history data loaded! Check data/processed/dataset.csv")
else:
    logger.info(f"✅ SUCCESS: Loaded {len(data_manager.demand_history)} demand history records")
    logger.info(f"📊 Data range: {data_manager.demand_history['order_date'].min()} to {data_manager.demand_history['order_date'].max()}")
    
# ---------- Helper Functions ----------
def generate_forecast_id(sku_id: str, location: str, channel: str, forecast_date: date) -> str:
    return f"{sku_id}_{location}_{channel}_{forecast_date.strftime('%Y%m%d')}"

def calculate_total_demand(base_demand: float, promotional_demand: float = 0) -> float:
    return base_demand + promotional_demand

def synthetic_series(days: int = 30, scenario_multipliers: Dict = None):
    """(Only used as hard fallback if no data)"""
    labels, values, cu, cl = [], [], [], []
    start_date = datetime.now()
    for i in range(days):
        d = start_date + timedelta(days=i)
        labels.append(d.strftime('%b %d'))
        seasonal_trend = math.sin(i / 15) * 1200
        weekly_pattern = math.sin(i / 7) * 600
        growth_trend = i * 12
        random_variation = np.random.normal(0, 500)
        base_value = 2500
        mult = (sum(scenario_multipliers.values()) / len(scenario_multipliers)) if scenario_multipliers else 1.0
        pred = max(1000, (base_value + seasonal_trend + weekly_pattern + growth_trend + random_variation) * mult)
        values.append(round(pred))
        cu.append(round(pred * 1.15))
        cl.append(round(pred * 0.85))
    return {
        "labels": labels,
        "values": values,
        "confidence_upper": cu,
        "confidence_lower": cl,
        "feature_count": 0,
        "features_used": ["historical_fallback_synthetic"]
    }

def safe_float_conversion(value):
    """Safely convert values to JSON-serializable floats"""
    if value is None or pd.isna(value) or np.isnan(value) or np.isinf(value):
        return 0.0
    return float(value)

# ---------- Frontend endpoints powered by processed data ----------

@app.get("/api/data-summary")
async def get_data_summary():
    """Get data summary for dashboard using REAL data"""
    try:
        summary = data_manager.generate_sample_data_summary()
        
        if summary.get("basic_stats", {}).get("total_records", 0) == 0:
            logger.warning("⚠️  Dashboard showing empty data - check dataset loading")
        else:
            logger.info(f"✅ Dashboard serving real data: {summary['basic_stats']['total_records']} records")
        
        return summary
        
    except Exception as e:
        logger.error(f"Dashboard data summary failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load dashboard data: {str(e)}")


@app.get("/api/forecast")
async def get_simple_forecast(days: int = Query(30, ge=1, le=730)):
    """Get forecast using real data patterns"""
    return generate_realistic_forecast_from_data(data_manager, days=days)

@app.post("/api/daily-forecast")
async def generate_daily_forecast(params: DailyForecastParams):
    """Generate daily forecast using REAL data with proper variation"""
    try:
        logger.info(f"Daily forecast request: {params}")
        
        if data_manager.demand_history.empty:
            logger.warning("No real data available - falling back to synthetic")
            days = params.forecast_days
            start_date = datetime.now()
            
            dates = []
            base_forecast = []
            promotional_forecast = []
            total_forecast = []
            
            for i in range(days):
                date = start_date + timedelta(days=i)
                dates.append(date.strftime('%Y-%m-%d'))
                
                
                base_demand = max(500, 1500 + math.sin(i / 10) * 300 + np.random.normal(0, 100))
                
                promo_demand = 0
                if np.random.random() < 0.3:  # 30% chance of promotion
                    promo_demand = base_demand * np.random.uniform(0.2, 0.8)
                
                base_forecast.append(round(base_demand))
                promotional_forecast.append(round(promo_demand))
                total_forecast.append(round(base_demand + promo_demand))
            
            forecast_data = {
                "sku_id": params.sku_id or "SAMPLE_SKU",
                "dates": dates,
                "base_forecast": base_forecast,
                "promotional_forecast": promotional_forecast,
                "total_forecast": total_forecast,
                "has_historical_data": False, 
                "historical_base_avg": np.mean(base_forecast),
                "historical_promo_avg": np.mean(promotional_forecast),
                "trend_slope": np.random.uniform(-0.5, 0.5),
                "forecasting_method": "synthetic_fallback",
                "features_used": ["synthetic_patterns"],
                "data_source": "synthetic_fallback"
            }
            
            return {
                "forecasts": [forecast_data],
                "granularity_level": params.granularity_level,
                "parameters_used": params.model_dump()
            }
        
        df = data_manager.demand_history.copy()
        sku_col = data_manager.get_sku_key()
        loc_col = data_manager.get_location_key()
        
        logger.info(f"Using columns - SKU: {sku_col}, Location: {loc_col}")
        
        # Apply filters
        original_size = len(df)
        if params.sku_id and sku_col in df.columns:
            df = df[df[sku_col].astype(str) == str(params.sku_id)]
        if params.channel and 'channel' in df.columns:
            df = df[df['channel'].astype(str) == str(params.channel)]
        if params.location and loc_col in df.columns:
            df = df[df[loc_col].astype(str) == str(params.location)]
        
        logger.info(f"Filtered from {original_size} to {len(df)} records")
        
        # Validate required columns
        if 'order_date' not in df.columns or 'order_qty' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Required columns missing. Available: {list(df.columns)}"
            )
        
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df = df.dropna(subset=['order_date', 'order_qty']).sort_values('order_date')
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No valid data after cleaning")
        
        logger.info(f"Clean data: {len(df)} records from {df['order_date'].min()} to {df['order_date'].max()}")
        
        daily_data = df.groupby(df['order_date'].dt.date).agg({
            'order_qty': 'sum',
            'is_promotional': 'max' if 'is_promotional' in df.columns else lambda x: False
        }).reset_index()
        
        if 'is_promotional' in daily_data.columns:
            base_days = daily_data[~daily_data['is_promotional']]
            promo_days = daily_data[daily_data['is_promotional']]
            
            historical_base_avg = float(base_days['order_qty'].mean()) if not base_days.empty else float(daily_data['order_qty'].mean())
            historical_promo_avg = float(promo_days['order_qty'].mean()) if not promo_days.empty else 0.0
            
            if historical_base_avg > 0 and historical_promo_avg > 0:
                promo_lift = max(0.1, min(3.0, (historical_promo_avg / historical_base_avg) - 1.0))
            else:
                promo_lift = 0.3
            
            promo_frequency = len(promo_days) / len(daily_data) if len(daily_data) > 0 else 0.1
        else:
            historical_base_avg = float(daily_data['order_qty'].mean())
            historical_promo_avg = 0.0
            promo_lift = 0.3
            promo_frequency = 0.1
        
        df_dow = df.copy()
        df_dow['day_of_week'] = df_dow['order_date'].dt.dayofweek
        dow_patterns = df_dow.groupby('day_of_week')['order_qty'].agg(['mean', 'std']).fillna(0)
        
        if len(daily_data) >= 14:
            recent_avg = daily_data['order_qty'].tail(7).mean()
            older_avg = daily_data['order_qty'].tail(14).head(7).mean()
            trend_slope = (recent_avg - older_avg) / 7 if older_avg > 0 else 0.0
        else:
            trend_slope = 0.0
        
        logger.info(f"Real patterns - Base avg: {historical_base_avg:.1f}, Promo lift: {promo_lift:.2f}, Trend: {trend_slope:.2f}")
        
        days = max(1, params.forecast_days)
        last_date = df['order_date'].max()
        start_date = last_date + pd.Timedelta(days=1)
        
        dates = []
        base_forecast = []
        promotional_forecast = []
        
        for i in range(days):
            future_date = start_date + pd.Timedelta(days=i)
            dates.append(future_date.strftime('%Y-%m-%d'))
            dow = future_date.dayofweek
            
            if dow in dow_patterns.index:
                dow_mean = dow_patterns.loc[dow, 'mean']
                dow_std = dow_patterns.loc[dow, 'std']
                
                base_demand = max(10, dow_mean + np.random.normal(0, max(10, dow_std * 0.3)))
            else:
                base_demand = max(10, historical_base_avg + np.random.normal(0, historical_base_avg * 0.1))
            
            base_demand += trend_slope * i
            
            base_demand = max(10, base_demand)
            
            promo_demand = 0.0
            if np.random.random() < promo_frequency:
                promo_demand = base_demand * promo_lift
            
            base_forecast.append(round(base_demand))
            promotional_forecast.append(round(promo_demand))
        
        total_forecast = [b + p for b, p in zip(base_forecast, promotional_forecast)]
        
        logger.info(f"Generated forecast range: {min(base_forecast)} - {max(base_forecast)}")
        
        forecast_data = {
            "sku_id": params.sku_id or "aggregated",
            "dates": dates,
            "base_forecast": base_forecast,
            "promotional_forecast": promotional_forecast,
            "total_forecast": total_forecast,
            "has_historical_data": True,
            "historical_base_avg": historical_base_avg,
            "historical_promo_avg": historical_promo_avg,
            "trend_slope": trend_slope,
            "forecasting_method": "real_data_enhanced",
            "features_used": ["real_data", "day_of_week_variation", "trend_analysis", "promotional_patterns"],
            "data_quality": {
                "total_historical_records": len(df),
                "promotional_records": len(df[df['is_promotional'] == True]) if 'is_promotional' in df.columns else 0,
                "base_records": len(df[df['is_promotional'] == False]) if 'is_promotional' in df.columns else len(df),
                "date_range": {
                    "start": df['order_date'].min().strftime('%Y-%m-%d'),
                    "end": df['order_date'].max().strftime('%Y-%m-%d')
                },
                "promotional_lift_calculated": promo_lift,
                "promotional_frequency": promo_frequency,
                "dow_patterns_available": len(dow_patterns)
            },
            "data_source": "real_processed_dataset"
        }
        
        return {
            "forecasts": [forecast_data],
            "granularity_level": params.granularity_level,
            "parameters_used": params.model_dump(),
            "data_source": "real_processed_dataset"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Daily forecast failed: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")


@app.post("/api/promotional-forecast")
async def generate_promotional_forecast(params: PromotionalForecastParams):
    """Generate promotional forecast using REAL dataset and promotional calendar"""
    try:
        logger.info(f"Promotional forecast request: {params}")
        
        if data_manager.demand_history.empty:
            raise HTTPException(
                status_code=404, 
                detail="No historical data available for promotional analysis"
            )
        
        df = data_manager.demand_history.copy()
        promo_calendar = data_manager.promotional_calendar.copy()
        
        sku_col = data_manager.get_sku_key()
        loc_col = data_manager.get_location_key()
        
        logger.info(f"Dataset shape: {df.shape}, Promo calendar shape: {promo_calendar.shape}")
        
        if params.sku_id and sku_col in df.columns:
            df = df[df[sku_col].astype(str) == str(params.sku_id)]
        
        if params.channel and 'channel' in df.columns:
            df = df[df['channel'].astype(str) == str(params.channel)]
        
        if params.location and loc_col in df.columns:
            df = df[df[loc_col].astype(str) == str(params.location)]
        
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df = df.dropna(subset=['order_date']).sort_values('order_date')
        
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data after filtering for SKU={params.sku_id}, Channel={params.channel}, Location={params.location}"
            )
        
        if 'is_promotional' in df.columns:
            promo_data = df[df['is_promotional'] == True]
            base_data = df[df['is_promotional'] == False]
            
            if not base_data.empty:
                base_daily = base_data.groupby(base_data['order_date'].dt.date)['order_qty'].sum()
                historical_base_avg = float(base_daily.mean())
            else:
                historical_base_avg = float(df['order_qty'].mean())
            
            if not promo_data.empty:
                promo_daily = promo_data.groupby(promo_data['order_date'].dt.date)['order_qty'].sum()
                historical_promo_avg = float(promo_daily.mean())
                promo_lift = (historical_promo_avg / historical_base_avg) - 1 if historical_base_avg > 0 else 0.5
                promo_lift = max(0.0, min(promo_lift, 3.0))  # Cap between 0-300%
            else:
                historical_promo_avg = 0.0
                promo_lift = 0.5  # Default 50% lift
        else:
            historical_base_avg = float(df['order_qty'].mean())
            historical_promo_avg = 0.0
            promo_lift = 0.5
        
        days = params.forecast_days
        last_date = df['order_date'].max()
        start_date = last_date + pd.Timedelta(days=1)
        
        dow_patterns = df.groupby(df['order_date'].dt.dayofweek)['order_qty'].mean()
        
        dates = []
        base_demand_forecast = []
        promotional_demand_forecast = []
        
        for i in range(days):
            future_date = start_date + pd.Timedelta(days=i)
            dates.append(future_date.strftime('%Y-%m-%d'))
            
           
            dow = future_date.dayofweek
            base_demand = float(dow_patterns.get(dow, historical_base_avg))
            
            promo_demand = 0.0
            if not promo_calendar.empty:
                current_date = future_date.date()
                start_col = None
                end_col = None
                for start_name in ['promo_start_date', 'start_date', 'Start_Date']:
                    if start_name in promo_calendar.columns:
                        start_col = start_name
                        break
                
                for end_name in ['promo_end_date', 'end_date', 'End_Date']:
                    if end_name in promo_calendar.columns:
                        end_col = end_name
                        break
                
                if start_col and end_col:
                    try:
                        promo_calendar[start_col] = pd.to_datetime(promo_calendar[start_col], errors='coerce')
                        promo_calendar[end_col] = pd.to_datetime(promo_calendar[end_col], errors='coerce')
                        
                        active_promos = promo_calendar[
                            (promo_calendar[start_col].dt.date <= current_date) &
                            (promo_calendar[end_col].dt.date >= current_date)
                        ]
                        
                        if not active_promos.empty:
                            promo_demand = base_demand * promo_lift
                    except Exception as e:
                        logger.warning(f"Error processing promotional calendar: {e}")
            
            base_demand_forecast.append(round(max(0, base_demand)))
            promotional_demand_forecast.append(round(max(0, promo_demand)))
        
        total_base = sum(base_demand_forecast)
        total_promo = sum(promotional_demand_forecast)
        
        historical_promo_days = len(df[df['is_promotional'] == True].groupby(df['order_date'].dt.date)) if 'is_promotional' in df.columns else 0
        total_historical_days = len(df.groupby(df['order_date'].dt.date))

        regional_config = {
        "region": params.region,
        "forecast_granularity": params.forecast_granularity,
        "promo_modeling_approach": params.promo_modeling_approach
        }

        if params.forecast_granularity == "monthly":
            monthly_dates = []
            monthly_base = []
            monthly_promo = []
        
            current_month = None
            month_base_sum = 0
            month_promo_sum = 0
        
            for i, date_str in enumerate(dates):
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                month_key = date_obj.strftime('%Y-%m')
            
                if current_month != month_key:
                    if current_month is not None:
                        monthly_dates.append(current_month)
                        monthly_base.append(month_base_sum)
                        monthly_promo.append(month_promo_sum)
                
                    current_month = month_key
                    month_base_sum = base_demand_forecast[i]
                    month_promo_sum = promotional_demand_forecast[i]
                else:
                    month_base_sum += base_demand_forecast[i]
                    month_promo_sum += promotional_demand_forecast[i]
        
            if current_month:
                monthly_dates.append(current_month)
                monthly_base.append(month_base_sum)
                monthly_promo.append(month_promo_sum)
        
            dates = monthly_dates
            base_demand_forecast = monthly_base
            promotional_demand_forecast = monthly_promo
    
        elif params.forecast_granularity == "weekly":
            weekly_dates = []
            weekly_base = []
            weekly_promo = []
        
            for i in range(0, len(dates), 7):
                week_end = min(i + 7, len(dates))
                week_start_date = dates[i]
                week_base_sum = sum(base_demand_forecast[i:week_end])
                week_promo_sum = sum(promotional_demand_forecast[i:week_end])
            
                weekly_dates.append(f"Week {week_start_date}")
                weekly_base.append(week_base_sum)
                weekly_promo.append(week_promo_sum)
        
            dates = weekly_dates
            base_demand_forecast = weekly_base
            promotional_demand_forecast = weekly_promo
    
        if params.promo_modeling_approach == "embedded":
            combined_forecast = [base + promo for base, promo in zip(base_demand_forecast, promotional_demand_forecast)]
            base_demand_forecast = combined_forecast
            promotional_demand_forecast = [0] * len(combined_forecast) 
    
        monthly_summary = None
        if params.region == "China":
            total_monthly_demand = sum(base_demand_forecast) + sum(promotional_demand_forecast)
            monthly_summary = {
                "total_monthly_demand": total_monthly_demand,
                "monthly_growth_rate": 2.5, 
                "promotion_intensity": "High" if sum(promotional_demand_forecast) > sum(base_demand_forecast) * 0.3 else "Medium"
            }
        
        
        return {
            "dates": dates,
            "base_demand_forecast": base_demand_forecast,
            "promotional_demand_forecast": promotional_demand_forecast,
            "regional_config": regional_config,
            "monthly_summary": monthly_summary,
            "historical_summary": {
                "total_base_demand": total_base,
                "total_promotional_demand": total_promo,
                "promotional_vs_base_ratio": total_promo / total_base if total_base > 0 else 0,
                "historical_base_avg": historical_base_avg,
                "historical_promo_avg": historical_promo_avg
            },
            "promotional_impact_analysis": {
                "total_promotional_days": len([x for x in promotional_demand_forecast if x > 0]),
                "avg_promotional_lift": promo_lift,
                "promotional_frequency": historical_promo_days / total_historical_days if total_historical_days > 0 else 0,
                "data_source": "real_promotional_calendar_and_history"
            },
            "data_quality": {
                "historical_records_used": len(df),
                "promotional_records_found": len(df[df['is_promotional'] == True]) if 'is_promotional' in df.columns else 0,
                "base_period_records": len(df[df['is_promotional'] == False]) if 'is_promotional' in df.columns else len(df),
                "promotional_calendar_entries": len(promo_calendar),
                "date_range_analyzed": {
                    "start": df['order_date'].min().strftime('%Y-%m-%d'),
                    "end": df['order_date'].max().strftime('%Y-%m-%d')
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Promotional forecast generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate promotional forecast: {str(e)}")


@app.post("/api/adhoc-adjustment")
async def apply_adhoc_adjustment(request: AdHocAdjustmentRequest):
    try:
        logger.info(f"Ad-hoc adjustment request: {request}")
        
        target_date = datetime.strptime(request.target_date, '%Y-%m-%d').date()

        validation = data_manager.validate_sku_location_channel(
            request.sku_id, request.channel or '', request.location or ''
        )
        
        logger.info(f"Enhanced validation result: {validation}")
        
        adjustment_context = data_manager.get_adjustment_context(
            request.sku_id, request.channel or '', request.location or '', target_date
        )
        
        sku_info = data_manager.get_sku_info(request.sku_id)
        promo_info = data_manager.get_promotional_info(forecast_date=target_date)

        current = enhanced_safe_float_conversion(request.current_forecast)
        adjustment = enhanced_safe_float_conversion(request.adjustment_value)
        new_value = enhanced_safe_float_conversion(request.new_forecast_value)

        if request.adjustment_type == 'absolute':
            expected = adjustment
        elif request.adjustment_type == 'percentage':
            expected = current * (1 + adjustment / 100) if current != 0 else adjustment
        elif request.adjustment_type == 'multiplier':
            expected = current * adjustment if current != 0 else adjustment
        else:
            raise HTTPException(status_code=400, detail="Unknown adjustment_type")

        if abs(expected - new_value) > 1.0:
            logger.warning(f"Calculation mismatch: expected={expected}, provided={new_value}")
            expected = new_value

        system_forecast = data_manager.get_current_forecast(
            request.sku_id, request.channel or '', request.location or '', target_date
        )
        system_forecast = enhanced_safe_float_conversion(system_forecast) if system_forecast is not None else None

        adjustment_data = {
            "sku_id": request.sku_id,
            "channel": request.channel,
            "location": request.location,
            "target_date": request.target_date,
            "previous_value": current,
            "new_value": new_value,
            "adjustment_type": request.adjustment_type,
            "adjustment_value": adjustment,
            "reason_category": request.reason_category,
            "notes": request.notes,
            "applied_by": "system_user",
            "applied_at": request.timestamp,
            "granularity_level": request.granularity_level,
            "system_forecast": system_forecast,
            "validation_warnings": validation.get('warnings', []),
            "adjustment_context": safe_json_convert(adjustment_context),
            "data_quality_score": len(adjustment_context.get('historical_patterns', {}))
        }

        adj_id = data_manager.store_adhoc_adjustment(adjustment_data)

        change_pct = None
        if current != 0:
            change_pct = enhanced_safe_float_conversion((new_value - current) / current * 100)

        resp = {
            "status": "success",
            "message": "Forecast adjustment applied successfully",
            "adjustment_id": adj_id,
            "applied_at": request.timestamp,
            "adjustment_summary": {
                "previous_value": current,
                "new_value": new_value,
                "change": enhanced_safe_float_conversion(new_value - current),
                "change_percentage": change_pct,
                "adjustment_type": request.adjustment_type,
                "reason": request.reason_category
            },
            "data_context": {
                "sku_info": safe_json_convert(sku_info) if sku_info else {},
                "validation": {
                    "warnings": validation.get('warnings', []),
                    "suggestions": validation.get('suggestions', []),
                    "historical_data_points": validation.get('historical_data_points', 0),
                    "combination_exists": validation.get('combination_exists', False)
                },
                "historical_patterns": safe_json_convert(adjustment_context.get('historical_patterns', {})),
                "recommendations": adjustment_context.get('recommendations', [])
            },
            "promotional_context": {
                "active_promotion": bool(promo_info), 
                "promo_details": safe_json_convert(promo_info) if promo_info else {},
                "promotional_patterns": safe_json_convert(adjustment_context.get('promotional_context', {}))
            },
        }
        
        if system_forecast is not None:
            resp["system_forecast_comparison"] = {
                "intelligent_system_forecast": system_forecast,
                "user_input_vs_system": enhanced_safe_float_conversion(current - system_forecast),
                "adjustment_vs_system": enhanced_safe_float_conversion(new_value - system_forecast),
                "system_confidence": "high" if adjustment_context.get('historical_patterns', {}).get('total_records', 0) > 50 else "medium"
            }
        
        safe_resp = safe_json_convert(resp)
        
        logger.info(f"Enhanced ad-hoc adjustment successful: {adj_id}")
        return safe_resp

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Enhanced ad-hoc adjustment failed")
        raise HTTPException(status_code=500, detail=f"Failed to apply forecast adjustment: {str(e)}")


@app.get("/api/adhoc-adjustments/history")
async def get_adjustment_history(
    sku_id: str = None,
    channel: str = None,
    location: str = None,
    days_back: int = 30,
    limit: int = 50
):
    """Get historical adjustments with analytics"""
    try:
        if not hasattr(data_manager, 'adhoc_adjustments') or not data_manager.adhoc_adjustments:
            return {
                "adjustments": [],
                "summary": {"total_adjustments": 0},
                "patterns": {}
            }
        
        adjustments = data_manager.adhoc_adjustments.copy()
        
        if sku_id:
            adjustments = [adj for adj in adjustments if adj.get('sku_id') == sku_id]
        if channel:
            adjustments = [adj for adj in adjustments if adj.get('channel') == channel]
        if location:
            adjustments = [adj for adj in adjustments if adj.get('location') == location]
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        adjustments = [
            adj for adj in adjustments 
            if datetime.fromisoformat(adj.get('created_at', '').replace('Z', '+00:00')) >= cutoff_date
        ]
        
        adjustments.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        limited_adjustments = adjustments[:limit]
        
        if adjustments:
            changes = [
                adj.get('new_value', 0) - adj.get('previous_value', 0) 
                for adj in adjustments 
                if adj.get('new_value') is not None and adj.get('previous_value') is not None
            ]
            
            summary = {
                "total_adjustments": len(adjustments),
                "avg_change": sum(changes) / len(changes) if changes else 0,
                "positive_adjustments": len([c for c in changes if c > 0]),
                "negative_adjustments": len([c for c in changes if c < 0]),
                "most_common_reason": max(
                    [adj.get('reason_category', 'unknown') for adj in adjustments],
                    key=[adj.get('reason_category', 'unknown') for adj in adjustments].count
                ) if adjustments else None
            }
        else:
            summary = {"total_adjustments": 0}
        
        return {
            "adjustments": limited_adjustments,
            "summary": summary,
            "filters_applied": {
                "sku_id": sku_id,
                "channel": channel, 
                "location": location,
                "days_back": days_back
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting adjustment history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get adjustment history: {str(e)}")


@app.get("/api/debug/data-status")
async def get_data_status():
    """Debug endpoint with comprehensive error handling"""
    try:
        response_data = {
            "status": "checking",
            "demand_history": {"loaded": False, "shape": [0, 0], "columns": []},
            "available_data": {"locations": [], "skus": [], "location_key": None, "sku_key": None},
            "sku_master": {"loaded": False, "shape": [0, 0], "columns": []},
            "promotional_calendar": {"loaded": False, "shape": [0, 0], "columns": []},
            "model_info": {"loaded": False, "path": ""},
            "system_status": {"using_real_data": False}
        }
        
        try:
            if hasattr(data_manager, 'demand_history'):
                df = data_manager.demand_history
                response_data["demand_history"]["loaded"] = not df.empty
                if not df.empty:
                    response_data["demand_history"]["shape"] = list(df.shape)
                    response_data["demand_history"]["columns"] = list(df.columns)
                    response_data["system_status"]["using_real_data"] = True
                    
                    if 'order_date' in df.columns:
                        try:
                            date_series = pd.to_datetime(df['order_date'], errors='coerce')
                            if not date_series.isna().all():
                                response_data["demand_history"]["date_range"] = {
                                    "min": str(date_series.min()),
                                    "max": str(date_series.max())
                                }
                        except Exception as e:
                            logger.warning(f"Date range error: {e}")
                    
                    # Safe SKU/location info
                    try:
                        sku_key = data_manager.get_sku_key()
                        loc_key = data_manager.get_location_key()
                        response_data["available_data"]["sku_key"] = sku_key
                        response_data["available_data"]["location_key"] = loc_key
                        
                        if sku_key in df.columns:
                            response_data["available_data"]["skus"] = df[sku_key].dropna().unique()[:10].tolist()
                        if loc_key in df.columns:
                            response_data["available_data"]["locations"] = df[loc_key].dropna().unique()[:10].tolist()
                    except Exception as e:
                        logger.warning(f"SKU/location error: {e}")
        except Exception as e:
            logger.error(f"Demand history check failed: {e}")
            response_data["demand_history"]["error"] = str(e)
        
        try:
            if hasattr(data_manager, 'sku_master') and not data_manager.sku_master.empty:
                response_data["sku_master"]["loaded"] = True
                response_data["sku_master"]["shape"] = list(data_manager.sku_master.shape)
                response_data["sku_master"]["columns"] = list(data_manager.sku_master.columns)
        except Exception as e:
            logger.warning(f"SKU master check failed: {e}")
        
        try:
            if hasattr(data_manager, 'promotional_calendar') and not data_manager.promotional_calendar.empty:
                response_data["promotional_calendar"]["loaded"] = True
                response_data["promotional_calendar"]["shape"] = list(data_manager.promotional_calendar.shape)
                response_data["promotional_calendar"]["columns"] = list(data_manager.promotional_calendar.columns)
        except Exception as e:
            logger.warning(f"Promotional calendar check failed: {e}")
        
        try:
            response_data["model_info"]["loaded"] = hasattr(data_manager, 'model') and data_manager.model is not None
            if hasattr(data_manager, 'model_path'):
                response_data["model_info"]["path"] = str(data_manager.model_path)
        except Exception as e:
            logger.warning(f"Model check failed: {e}")
        
        response_data["status"] = "completed"
        return response_data
        
    except Exception as e:
        logger.error(f"Debug endpoint failed completely: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__,
            "system_status": {"using_real_data": False}
        }

@app.post("/api/get-baseline-forecast")
async def get_baseline_forecast(params: ScenarioParams):
    """
    Build and return the same baseline that powers the Daily Forecast chart
    for the provided filters/horizon. Used by Scenario to mirror the shape exactly.
    """
    daily_req = DailyForecastParams(
        sku_id=params.sku_id,
        channel=params.channel,
        location=params.location,
        start_date=params.start_date,
        forecast_days=params.forecast_days,
        granularity_level=params.granularity_level,
    )
    daily = await generate_daily_forecast(daily_req)
    if not daily or "forecasts" not in daily or not daily["forecasts"]:
        raise HTTPException(status_code=500, detail="Could not build baseline from daily-forecast")
    f0 = daily["forecasts"][0]
    return {
        "labels": f0["dates"],
        "values": f0["total_forecast"],
        "confidence_upper": [int(v * 1.1) for v in f0["total_forecast"]],
        "confidence_lower": [int(v * 0.9) for v in f0["total_forecast"]],
    }


@app.post("/api/scenario-forecast")
async def generate_scenario_forecast(params: ScenarioParams):
    """
       Generate ML-powered scenario forecast with quick retrain capability.
       Scenario forecast that reuses the exact baseline used in the Daily Forecast chart
       (same filters, same dates), then applies scenario multipliers on top.
    """
    try:
        if params.forecast_days > SCENARIO_MAX_FORECAST_DAYS:
            raise HTTPException(status_code=400, detail=f"forecast_days exceeds max {SCENARIO_MAX_FORECAST_DAYS}")

        # 1) Build the baseline via the SAME path as the chart
        baseline = await get_baseline_forecast(params)

        # 2) Ask the scenario engine to simulate using that baseline
        tracemalloc.start()
        result = await scenario_engine.generate_forecast({
            "forecast_days": params.forecast_days,
            "scenario": params.scenario,
            "filters": {
                "sku_id": params.sku_id, "channel": params.channel, "location": params.location
            },
            "baseline_override": baseline,   # <-- key change
        })
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        meta = result.get("metadata", {})
        meta["peak_memory_mb"] = round(peak_mem / (1024 * 1024), 2)
        result["metadata"] = meta

        logger.info(
            "ScenarioForecast: hash=%s cached=%s train_time=%ss avg_delta=%s feature_count=%s",
            meta.get("scenario_hash"),
            meta.get("model_cached"),
            meta.get("training_time_seconds"),
            meta.get("avg_delta"),
            meta.get("feature_count"),
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Scenario forecast failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Scenario forecast failed: {str(e)}")

@app.post("/api/aggregate-forecast")
async def aggregate_forecast_endpoint(request: AggregationRequest):
    """Main aggregation endpoint - roll up forecasts to any level"""
    try:
        logger.info(f"Aggregation request received: {request.model_dump()}")
        response = aggregation_engine.aggregate_forecast(request)
        logger.info(f"Aggregation successful: {response.level}")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/drill-down")
async def drill_down_forecast_endpoint(request: DrillDownRequest):
    """FIXED: Drill down with comprehensive error handling"""
    try:
        logger.info(f"Drill-down request: {request.model_dump()}")
        
        if request.from_level == request.to_level:
            raise HTTPException(
                status_code=400, 
                detail="From and to levels cannot be the same"
            )
        
        if request.direction == DrillDirection.DOWN and not request.target_dimension:
            drill_dim = aggregation_engine._get_drill_dimension(request.from_level, request.to_level)
            request.target_dimension = drill_dim
            logger.info(f"Auto-determined drill dimension: {drill_dim}")
        
        if data_manager.demand_history.empty:
            raise HTTPException(
                status_code=404,
                detail="No historical data available for drill-down operation"
            )
        
        results = aggregation_engine.drill_down_forecast(request)
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No drill-down results generated. Check if data exists for the specified filters."
            )
        
        logger.info(f"Drill-down successful: {len(results)} results")
        
        return {
            "drill_results": results,
            "target_level": request.to_level.value,
            "drill_dimension": request.target_dimension,
            "metadata": {
                "total_results": len(results),
                "from_level": request.from_level.value,
                "to_level": request.to_level.value,
                "direction": request.direction.value,
                "filters_applied": request.current_filters
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Drill-down endpoint failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Drill-down operation failed: {str(e)}"
        )
    

@app.get("/api/aggregation-levels")
async def get_aggregation_levels():
    """Get available aggregation levels with hierarchy info - simplified"""
    return {
        "levels": [
            {
                "id": "sku_location_customer_day",
                "name": "SKU-Location-Customer-Day",
                "granularity": "most_detailed",
                "can_drill_up_to": ["sku_location_day"]
            },
            {
                "id": "sku_location_day", 
                "name": "SKU-Location-Day",
                "granularity": "detailed",
                "can_drill_up_to": ["sku_day"],
                "can_drill_down_to": ["sku_location_customer_day"]
            },
            {
                "id": "sku_channel_day",
                "name": "SKU-Channel-Day",
                "granularity": "detailed",
                "can_drill_up_to": ["sku_day"]
            },
            {
                "id": "sku_day",
                "name": "SKU-Day", 
                "granularity": "medium",
                "can_drill_up_to": ["national_day"],
                "can_drill_down_to": ["sku_location_day", "sku_channel_day"]
            },
            {
                "id": "national_day",
                "name": "National-Day",
                "granularity": "highest",
                "can_drill_down_to": ["sku_day"]
            }
        ],
        "hierarchy": "national → sku → sku+location/channel → sku+location+customer",
        "dimensions": ["sku", "location", "channel", "customer", "time"]
    }


@app.get("/api/quick-aggregate/{level}")
async def quick_aggregate(
    level: str,
    forecast_days: int = Query(30, ge=1, le=730),
    sku_id: str = None,
    location: str = None,
    channel: str = None
):
    """Quick aggregation for common use cases - simplified"""
    try:
        level_map = {
            "sku": AggregationLevel.SKU_DAY,
            "national": AggregationLevel.NATIONAL_DAY,
            "sku_location": AggregationLevel.SKU_LOCATION_DAY,
            "sku_channel": AggregationLevel.SKU_CHANNEL_DAY
        }
        
        if level not in level_map:
            raise HTTPException(status_code=400, detail=f"Invalid level: {level}. Available: {list(level_map.keys())}")
        
        request = AggregationRequest(
            level=level_map[level],
            forecast_days=forecast_days,
            sku_id=sku_id,
            location=location,
            channel=channel
        )
        
        response = aggregation_engine.aggregate_forecast(request)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dimension-values/{dimension}")
async def get_dimension_values(dimension: str, limit: int = Query(200, le=500)):
    """Get available values for a dimension with KC-specific handling"""
    try:
        df = data_manager.demand_history
        
        kc_values = {
            "brand": ["Huggies", "Kleenex", "Scott", "Kotex", "Depend", "Poise"],
            "location": [
                "USA-Northeast", "USA-Southeast", "USA-Midwest", "USA-West",
                "China-Tier1", "China-Tier2", "Korea-Seoul", "Korea-Other",
                "Brazil-Urban", "Mexico-Urban"
            ],
            "channel": ["E-commerce", "Mass Retail", "Drug Store", "Club Store", "Dollar Store"],
            "customer": ["Walmart", "Target", "CVS", "Walgreens", "Amazon", "Kroger", "Costco"],
            "sku": ["HUG1001", "KLX2001", "SCT3001", "KTX4001", "DEP5001", "PSE6001"]
        }
        
        if df.empty:
            values = kc_values.get(dimension, [])
            logger.info(f"Using KC fallback values for {dimension}: {len(values)} values")
            return {
                "dimension": dimension,
                "values": values,
                "total": len(values),
                "source": "kc_fallback_values"
            }
        
        column_map = {
            "sku": data_manager.get_sku_key(),
            "location": data_manager.get_location_key(),
            "channel": "channel",
            "customer": "customer", 
            "brand": "brand",
            "category": "category"
        }
        
        if dimension not in column_map:
            values = kc_values.get(dimension, [])
            return {
                "dimension": dimension,
                "values": values,
                "total": len(values),
                "source": "kc_fallback_unknown"
            }
        
        col_name = column_map[dimension]
        
        if col_name not in df.columns:
            logger.warning(f"Column {col_name} not found, using KC fallback for {dimension}")
            values = kc_values.get(dimension, [])
            return {
                "dimension": dimension,
                "values": values,
                "total": len(values),
                "source": "kc_fallback_missing_column",
                "column_searched": col_name,
                "available_columns": list(df.columns)
            }
        
        unique_values = df[col_name].dropna()
        unique_values = unique_values[unique_values != ""]
        values = unique_values.unique()[:limit].tolist()
        
        if len(values) < 3 and dimension in kc_values:
            kc_supplements = [v for v in kc_values[dimension] if v not in values]
            values.extend(kc_supplements[:10])  # Add up to 10 KC values
            source = "real_data_plus_kc_supplement"
        else:
            source = "real_data"
        
        logger.info(f"Dimension {dimension}: {len(values)} values from {source}")
        
        return {
            "dimension": dimension,
            "values": values,
            "total": len(values),
            "column_used": col_name,
            "source": source
        }
        
    except Exception as e:
        logger.error(f"Error getting dimension values for {dimension}: {e}")
        
        values = kc_values.get(dimension, [])
        return {
            "dimension": dimension,
            "values": values,
            "total": len(values),
            "source": "kc_fallback_error",
            "error": str(e)
        }

@app.get("/api/drill-options/{current_level}")
async def get_drill_options(current_level: str):
    """ADAPTIVE: Get drill options based on actual available data"""
    try:
        hierarchy_map = aggregation_engine.hierarchy_map
        
        try:
            current_enum = AggregationLevel(current_level)
        except ValueError:
            available_levels = [level.value for level in hierarchy_map.keys()]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid level: {current_level}. Available: {available_levels}"
            )
        
        if current_enum not in hierarchy_map:
            available_levels = [level.value for level in hierarchy_map.keys()]
            raise HTTPException(
                status_code=400, 
                detail=f"Level not in hierarchy: {current_level}. Available: {available_levels}"
            )
        
        level_info = hierarchy_map[current_enum]
        
        result = {
            "drill_down": [level.value for level in level_info.get("child_levels", [])],
            "drill_up": [level.value for level in level_info.get("parent_levels", [])],
            "description": f"{current_level.replace('_', ' ').title()} level aggregation",
            "available_dimensions": []
        }
        
        child_levels = level_info.get("child_levels", [])
        current_dimensions = set(level_info.get("dimensions", []))
        
        available_dimensions = set()
        for child_level in child_levels:
            if child_level in hierarchy_map:
                child_dimensions = set(hierarchy_map[child_level].get("dimensions", []))
                new_dims = child_dimensions - current_dimensions
                available_dimensions.update(new_dims)
        
        dimension_mapping = {
            data_manager.get_sku_key(): "sku",
            data_manager.get_location_key(): "location",
            "channel": "channel",
            "brand": "brand",
            "customer": "customer"
        }
        
        user_friendly_dims = []
        for dim in available_dimensions:
            if dim != "transaction_date":  
                user_friendly = dimension_mapping.get(dim, dim)
                user_friendly_dims.append(user_friendly)
        
        result["available_dimensions"] = user_friendly_dims
        
        # Add dimension value counts from real data
        if hasattr(data_manager, 'demand_history') and not data_manager.demand_history.empty:
            df = data_manager.demand_history
            dimension_counts = {}
            
            for dim in user_friendly_dims:
                reverse_mapping = {v: k for k, v in dimension_mapping.items()}
                actual_column = reverse_mapping.get(dim, dim)
                
                if actual_column in df.columns:
                    dimension_counts[dim] = int(df[actual_column].nunique())
                else:
                    dimension_counts[dim] = 0
            
            result["dimension_counts"] = dimension_counts
        
        logger.info(f"✅ Adaptive drill options for {current_level}: {result}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Drill options failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/aggregation-health")
async def check_aggregation_health():
    """Quick health check for aggregation system"""
    try:
        test_request = AggregationRequest(
            level=AggregationLevel.SKU_DAY,  
            forecast_days=7
        )
        
        test_response = aggregation_engine.aggregate_forecast(test_request)
        
        return {
            "status": "healthy",
            "aggregation_engine_loaded": True,
            "test_aggregation_successful": bool(test_response),
            "available_levels": len(AggregationLevel),
            "data_available": not data_manager.demand_history.empty,
            "total_records": len(data_manager.demand_history)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "aggregation_engine_loaded": hasattr(aggregation_engine, 'aggregate_forecast')
        }


@app.get("/api/debug/check-data")
async def check_specific_data(
    sku_id: str = "HUG5760",
    channel: str = "Online", 
    location: str = "USA-West"
):
    """Debug endpoint to check if specific data exists"""
    try:
        if data_manager.demand_history.empty:
            return {"error": "No demand history loaded"}
        
        df = data_manager.demand_history.copy()
        sku_col = data_manager.get_sku_key()
        loc_col = data_manager.get_location_key()
        
        available_skus = df[sku_col].unique()[:10].tolist() if sku_col in df.columns else []
        available_channels = df['channel'].unique().tolist() if 'channel' in df.columns else []
        available_locations = df[loc_col].unique()[:10].tolist() if loc_col in df.columns else []
        
        sku_match = str(sku_id) in df[sku_col].astype(str).values if sku_col in df.columns else False
        channel_match = str(channel) in df['channel'].astype(str).values if 'channel' in df.columns else False
        location_match = str(location) in df[loc_col].astype(str).values if loc_col in df.columns else False
        
        filtered_df = df.copy()
        if sku_col in df.columns:
            filtered_df = filtered_df[filtered_df[sku_col].astype(str) == str(sku_id)]
        if 'channel' in df.columns:
            filtered_df = filtered_df[filtered_df['channel'].astype(str) == str(channel)]
        if loc_col in df.columns:
            filtered_df = filtered_df[filtered_df[loc_col].astype(str) == str(location)]
        
        return {
            "total_records": len(df),
            "columns": list(df.columns),
            "sku_key": sku_col,
            "location_key": loc_col,
            "available_samples": {
                "skus": available_skus,
                "channels": available_channels,
                "locations": available_locations
            },
            "exact_matches": {
                "sku_exists": sku_match,
                "channel_exists": channel_match,
                "location_exists": location_match
            },
            "filtered_results": {
                "records_found": len(filtered_df),
                "date_range": {
                    "min": filtered_df['order_date'].min().strftime('%Y-%m-%d') if not filtered_df.empty and 'order_date' in filtered_df.columns else None,
                    "max": filtered_df['order_date'].max().strftime('%Y-%m-%d') if not filtered_df.empty and 'order_date' in filtered_df.columns else None
                } if not filtered_df.empty else None
            }
        }
        
    except Exception as e:
        return {"error": str(e)}


# ---------- Health ----------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "total_forecasts": len(data_manager.forecasts_data),
        "api_version": "1.0.0",
        "endpoints_available": [
            "/api/data-summary",
            "/api/forecast",
            "/api/daily-forecast",
            "/api/promotional-forecast",
            "/api/scenario-forecast",
            "/health"
        ],
        "data_files_loaded": {
            "weather_data": not data_manager.weather_data.empty,
            "sku_master": not data_manager.sku_master.empty,
            "promotional_calendar": not data_manager.promotional_calendar.empty,
            "inventory_snapshots": not data_manager.inventory_snapshots.empty,
            "forecast_scenarios": not data_manager.forecast_scenarios.empty,
            "external_factors": not data_manager.external_factors.empty,
            "demand_history": not data_manager.demand_history.empty,
            "customer_master": not data_manager.customer_master.empty,
            "alerts_exceptions": not data_manager.alerts_exceptions.empty
        }
    }

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)