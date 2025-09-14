# forecast_helpers.py
"""
Helper functions, classes, and utilities for the Forecast Management API.
Contains all data management, calculations, and business logic.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
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

# Import scenario engine
from src.models.scenario_engine import ScenarioEngine

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Date utils ----------
def _coerce_transaction_date(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or 'transaction_date' not in df.columns:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df['transaction_date']):
        df = df.copy()
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce', utc=False)
    return df.dropna(subset=['transaction_date'])

def _prepare_model_features(df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """Prepare features for model inference matching train_model.py requirements"""
    if df.empty:
        return pd.DataFrame()
    
    # Create a row for the target date with time-based features
    feature_row = pd.DataFrame({
        'transaction_date': [target_date],
        'order_year': [target_date.year],
        'order_month': [target_date.month], 
        'order_day': [target_date.day],
        'order_day_of_week': [target_date.dayofweek],
        'quarter': [target_date.quarter],
        'is_month_start': [int(target_date.is_month_start)],
        'is_month_end': [int(target_date.is_month_end)],
        'is_weekend': [int(target_date.dayofweek >= 5)]
    })
    
    # Add lag and rolling features from historical data
    if len(df) >= 7:
        # Use recent values for lag features
        recent_revenue = df['revenue'].tail(7).mean() if 'revenue' in df.columns else df['order_qty'].tail(7).mean() * 100
        recent_qty = df['order_qty'].tail(7).mean()
        
        feature_row['revenue_lag_1'] = recent_revenue
        feature_row['revenue_lag_7'] = recent_revenue  
        feature_row['revenue_rolling_mean_7'] = recent_revenue
        feature_row['revenue_rolling_std_7'] = df['revenue'].tail(7).std() if 'revenue' in df.columns else recent_revenue * 0.2
        feature_row['qty_lag_1'] = recent_qty
        feature_row['qty_rolling_mean_7'] = recent_qty
        feature_row['order_qty'] = recent_qty  # Use as baseline
    else:
        # Fallback for sparse data
        avg_qty = df['order_qty'].mean() if len(df) > 0 else 1000
        avg_revenue = df['revenue'].mean() if 'revenue' in df.columns and len(df) > 0 else avg_qty * 100
        
        for col in ['revenue_lag_1', 'revenue_lag_7', 'revenue_rolling_mean_7', 'revenue_rolling_std_7']:
            feature_row[col] = avg_revenue
        for col in ['qty_lag_1', 'qty_rolling_mean_7', 'order_qty']:
            feature_row[col] = avg_qty
    
    # Add encoded categorical features with defaults
    feature_row['city_encoded'] = 0  # Default encoding
    feature_row['category_encoded'] = 0  # Default encoding
    
    # Add binary features with defaults
    feature_row['is_late'] = 0
    feature_row['channel_Retail'] = 1  # Default to retail
    feature_row['customer_type_Enterprise'] = 0
    feature_row['country_USA'] = 1  # Default to USA
    
    # Add delivery status features (one-hot encoded) - default to 'Delivered'
    delivery_statuses = ['Cancelled', 'Delivered', 'In_Transit', 'Pending', 'Returned']
    for status in delivery_statuses:
        feature_row[f'delivery_status_{status}'] = 1 if status == 'Delivered' else 0
    
    return feature_row

# ---------- Model utils ----------
def _align_features_to_model(feature_df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Ensure the feature frame matches the model's expected columns:
    - add any missing columns filled with 0
    - drop extras
    - reorder to match the model
    Works for LightGBM (Booster) and scikit-learn (feature_names_in_).
    """
    if model is None or feature_df is None or feature_df.empty:
        return feature_df

    feat_names = None
    # LightGBM Booster
    if hasattr(model, "feature_name"):
        try:
            fn = model.feature_name() if callable(model.feature_name) else model.feature_name
            if fn:
                feat_names = list(fn)
        except Exception:
            pass
    # scikit-learn models
    if feat_names is None and hasattr(model, "feature_names_in_"):
        try:
            feat_names = list(model.feature_names_in_)
        except Exception:
            pass

    if not feat_names:
        # last resort: trust incoming columns
        return feature_df

    for col in feat_names:
        if col not in feature_df.columns:
            feature_df[col] = 0
    # Drop extras and reorder
    feature_df = feature_df[feat_names]
    return feature_df


def safe_json_convert(obj):
    """Recursively convert object to be JSON serializable"""
    if isinstance(obj, dict):
        return {k: safe_json_convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_convert(item) for item in obj]
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
    forecast_days: int = Field(default=30, ge=1, le=365)
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

class ForecastAccuracyTracker:
    """Calculate and track forecast accuracy metrics"""
    
    @staticmethod
    def calculate_accuracy_metrics(actual: float, forecast: float) -> Dict:
        """Calculate individual accuracy metrics"""
        try:
            actual = float(actual)
            forecast = float(forecast)
            
            if actual == 0 and forecast == 0:
                return {
                    'absolute_error': 0.0,
                    'absolute_percentage_error': 0.0,
                    'bias': 0.0,
                    'squared_error': 0.0
                }
            
            error = forecast - actual
            abs_error = abs(error)
            
            # Handle division by zero for percentage error
            if actual != 0:
                abs_pct_error = (abs_error / abs(actual)) * 100
            else:
                abs_pct_error = float('inf') if forecast != 0 else 0.0
            
            return {
                'absolute_error': enhanced_safe_float_conversion(abs_error),
                'absolute_percentage_error': enhanced_safe_float_conversion(abs_pct_error),
                'bias': enhanced_safe_float_conversion(error),
                'squared_error': enhanced_safe_float_conversion(error ** 2)
            }
        except Exception as e:
            logger.warning(f"Error calculating accuracy metrics: {e}")
            return {
                'absolute_error': 0.0,
                'absolute_percentage_error': 0.0,
                'bias': 0.0,
                'squared_error': 0.0
            }
    
    @staticmethod
    def calculate_aggregate_metrics(actuals: List[float], forecasts: List[float]) -> Dict:
        """Calculate aggregate accuracy metrics for a series"""
        try:
            if not actuals or not forecasts or len(actuals) != len(forecasts):
                return {}
            
            actuals = [enhanced_safe_float_conversion(a) for a in actuals]
            forecasts = [enhanced_safe_float_conversion(f) for f in forecasts]
            
            errors = [f - a for a, f in zip(actuals, forecasts)]
            abs_errors = [abs(e) for e in errors]
            
            # MAPE calculation with handling for zero actuals
            ape_values = []
            for a, f in zip(actuals, forecasts):
                if a != 0:
                    ape_values.append(abs(f - a) / abs(a) * 100)
                elif f == 0:
                    ape_values.append(0.0)
                # Skip cases where actual=0 but forecast!=0 for MAPE
            
            mape = sum(ape_values) / len(ape_values) if ape_values else float('inf')
            
            return {
                'mae': enhanced_safe_float_conversion(sum(abs_errors) / len(abs_errors)),
                'mape': enhanced_safe_float_conversion(mape),
                'bias': enhanced_safe_float_conversion(sum(errors) / len(errors)),
                'bias_percentage': enhanced_safe_float_conversion(sum(errors) / sum(actuals) * 100) if sum(actuals) > 0 else 0.0,
                'rmse': enhanced_safe_float_conversion((sum(e**2 for e in errors) / len(errors)) ** 0.5),
                'data_points': len(actuals)
            }
        except Exception as e:
            logger.warning(f"Error calculating aggregate metrics: {e}")
            return {}
    
    @staticmethod
    def get_accuracy_grade(mape: float) -> str:
        """Convert MAPE to letter grade"""
        try:
            if pd.isna(mape) or mape == float('inf'):
                return 'N/A'
            
            mape = float(mape)
            if mape <= 10:
                return 'Excellent'
            elif mape <= 20:
                return 'Good'
            elif mape <= 30:
                return 'Fair'
            elif mape <= 50:
                return 'Poor'
            else:
                return 'Very Poor'
        except:
            return 'N/A'
        
    @staticmethod
    def get_bias_recommendations(summary_metrics: Dict) -> List[str]:
        """Generate recommendations based on bias analysis"""
        recommendations = []
        
        try:
            bias_pct = summary_metrics.get('bias_percentage', 0)
            
            if abs(bias_pct) <= 5:
                recommendations.append("Bias is within acceptable range (±5%)")
            elif bias_pct > 15:
                recommendations.append("Significant over-forecasting detected. Consider reducing forecast levels.")
            elif bias_pct < -15:
                recommendations.append("Significant under-forecasting detected. Consider increasing forecast levels.")
            elif bias_pct > 5:
                recommendations.append("Moderate over-forecasting. Monitor and adjust if trend continues.")
            elif bias_pct < -5:
                recommendations.append("Moderate under-forecasting. Monitor and adjust if trend continues.")
            
            mape = metrics.get('mape', 0)
            if mape > 30:
                recommendations.append("High forecast error detected. Consider model recalibration or alternative methods.")
            
        except Exception as e:
            logger.warning(f"Error generating bias recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to data issues.")
        
        return recommendations if recommendations else ["No specific recommendations available."]
    
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
            dataset_path = self.processed_path / "dataset.csv"
            if dataset_path.exists():
                self.demand_history = pd.read_csv(dataset_path, low_memory=False)
                if not self.demand_history.empty:
                    # Map order_date to transaction_date
                    if 'order_date' in self.demand_history.columns:
                        self.demand_history['transaction_date'] = pd.to_datetime(self.demand_history['order_date'], errors='coerce')
                        self.demand_history = self.demand_history.dropna(subset=['transaction_date'])
                        self.demand_history = self.demand_history.sort_values('transaction_date').reset_index(drop=True)
                        logger.info(f"✅ Loaded processed dataset: {self.demand_history.shape}")
                        logger.info(f"📊 Date range: {self.demand_history['transaction_date'].min()} to {self.demand_history['transaction_date'].max()}")
                    else:
                        logger.error("No order_date column found")
                else:
                    logger.warning("Dataset is empty")
            else:
                logger.error(f"Dataset not found at {dataset_path}")
                self.demand_history = pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
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

    # ENHANCED FORECAST ACCURACY METHODS - START
    def get_forecast_accuracy_data_updated(self, 
                             sku_id: str = None, 
                             days_back: int = 30,
                             granularity: str = 'daily') -> Dict:
        try:
            if self.demand_history.empty:
                return {'error': 'No demand history data available for analysis'}
        
            # Use YOUR exact columns - not the old transaction_date/order_qty
            demand_data = self.demand_history.copy()
        
            logger.info(f"FR9: Starting accuracy analysis with {len(demand_data)} total records")
        
            # Apply SKU filter using YOUR sku_id column
            if sku_id and 'sku_id' in demand_data.columns:
                demand_data = demand_data[demand_data['sku_id'].astype(str) == str(sku_id)]
                logger.info(f"FR9: SKU filter applied: {len(demand_data)} records for SKU {sku_id}")
        
            if demand_data.empty:
                return {'error': f'No historical data found for SKU {sku_id}'}
        
            # Use YOUR transaction_date column
            if 'transaction_date' in demand_data.columns:
                demand_data['transaction_date'] = pd.to_datetime(demand_data['transaction_date'], errors='coerce')
                demand_data = demand_data.dropna(subset=['transaction_date'])
            
                if len(demand_data) > 0:
                    latest_date = demand_data['transaction_date'].max()
                    cutoff_date = latest_date - pd.Timedelta(days=days_back)
                
                    logger.info(f"FR9: Date filtering from {cutoff_date} to {latest_date}")
                
                    recent_data = demand_data[demand_data['transaction_date'] >= cutoff_date]
                    if len(recent_data) >= 5:
                        demand_data = recent_data
                        logger.info(f"FR9: Using recent data: {len(demand_data)} records")
        
            if demand_data.empty:
                return {'error': 'No valid data found after date filtering'}
        
            # Group by time grain using YOUR columns
            if granularity == 'daily':
                demand_data['date_key'] = demand_data['transaction_date'].dt.date
                grouped_actuals = demand_data.groupby('date_key').agg({
                    'total_demand_qty': 'sum',  # YOUR total demand column
                    'base_demand_qty': 'sum',   # YOUR base demand column
                    'promotional_demand_qty': 'sum',  # YOUR promo demand column
                    'sku_id': 'first' if sku_id else lambda x: 'aggregated'
                }).reset_index()
                grouped_actuals.rename(columns={'date_key': 'analysis_date'}, inplace=True)
        
            version_results = {}
            if 'forecast_version_name' in demand_data.columns:
                versions = demand_data['forecast_version_name'].unique()
                logger.info(f"FR9: Found forecast versions: {list(versions)}")
            
                for version in versions:
                    version_data = demand_data[demand_data['forecast_version_name'] == version]
                
                    if len(version_data) < 5:
                        continue
                
                    # Calculate accuracy using YOUR data structure
                    version_accuracy = []
                    daily_version = version_data.groupby(version_data['transaction_date'].dt.date)['total_demand_qty'].sum()
                
                    for idx, (date, actual_qty) in enumerate(daily_version.items()):
                        if idx < 3:  # Need lookback for forecast
                            continue
                    
                        # Use moving average as forecast baseline
                        forecast_qty = daily_version.iloc[max(0, idx-7):idx].mean()
                    
                        if not pd.isna(actual_qty) and not pd.isna(forecast_qty) and forecast_qty > 0:
                            metrics = ForecastAccuracyTracker.calculate_accuracy_metrics(actual_qty, forecast_qty)
                        
                            accuracy_record = {
                                'analysis_date': str(date),
                                'sku_id': str(sku_id) if sku_id else 'aggregated',
                                'actual': enhanced_safe_float_conversion(actual_qty),
                                'forecast': enhanced_safe_float_conversion(forecast_qty),
                                'forecast_version': version,
                                'granularity': granularity,
                                **metrics
                            }
                            version_accuracy.append(accuracy_record)
                
                    if version_accuracy:
                        actuals = [item['actual'] for item in version_accuracy]
                        forecasts = [item['forecast'] for item in version_accuracy]
                    
                        version_results[version] = {
                            'summary_metrics': ForecastAccuracyTracker.calculate_aggregate_metrics(actuals, forecasts),
                            'data_points': len(version_accuracy),
                            'coverage': len(version_accuracy) / len(grouped_actuals) * 100 if len(grouped_actuals) > 0 else 0
                        }
        
            # Find best performing version
            best_version = None
            best_mape = float('inf')
        
            for version, results in version_results.items():
                version_mape = results['summary_metrics'].get('mape', float('inf'))
                if version_mape < best_mape:
                    best_mape = version_mape
                    best_version = version
        
            # Enhanced SKU-Channel-Time analysis using YOUR columns
            sku_time_analysis = self._analyze_accuracy_by_sku_time_updated(demand_data, sku_id)
        
            return {
                'summary': {
                    'total_data_points': sum(v['data_points'] for v in version_results.values()),
                    'unique_time_periods': len(grouped_actuals),
                    'analysis_period': {
                        'start_date': demand_data['transaction_date'].min().strftime('%Y-%m-%d'),
                        'end_date': demand_data['transaction_date'].max().strftime('%Y-%m-%d'),
                        'days_analyzed': days_back
                    },
                    'granularity': granularity,
                    'best_performing_version': {
                        'version': best_version,
                        'mape': best_mape,
                        'accuracy_grade': ForecastAccuracyTracker.get_accuracy_grade(best_mape)
                    }
                },
                'version_results': version_results,
                'sku_time_analysis': sku_time_analysis,
                'detail_data': [],  # Can be populated if needed
                'filters_applied': {
                    'sku_id': sku_id,
                    'days_back': days_back,
                    'granularity': granularity
                },
                'methodology': 'fr9_sku_channel_time_grain_analysis',
                'data_source': 'real_forecast_versions'
            }
        
        except Exception as e:
            logger.error(f"FR9: Enhanced forecast accuracy analysis failed: {e}")
            return {'error': f'Analysis failed: {str(e)}'}
    

    def _generate_enhanced_forecast_versions(self, data: pd.DataFrame) -> Dict:
        """Generate enhanced forecast versions for SKU-Time analysis"""
        forecast_versions = {}
        
        # Version 1: Day-of-week with trend
        def dow_trend_forecast(df, row, date):
            try:
                if isinstance(date, str):
                    target_date = pd.to_datetime(date)
                else:
                    target_date = pd.Timestamp(date)
                
                dow = target_date.dayofweek
                dow_data = df[df['transaction_date'].dt.dayofweek == dow]
                
                if not dow_data.empty and len(dow_data) >= 3:
                    # Calculate trend within day-of-week
                    if len(dow_data) >= 6:
                        recent_dow = dow_data.tail(3)['order_qty'].mean()
                        older_dow = dow_data.head(3)['order_qty'].mean()
                        trend_factor = recent_dow / older_dow if older_dow > 0 else 1.0
                        base_value = recent_dow
                    else:
                        base_value = dow_data['order_qty'].mean()
                        trend_factor = 1.0
                    
                    return float(base_value * min(max(trend_factor, 0.7), 1.5))
                else:
                    return float(df['order_qty'].mean())
            except:
                return float(df['order_qty'].mean()) if not df.empty else 0
        
        # Version 2: Exponential smoothing
        def exp_smoothing_forecast(df, row, date):
            try:
                alpha = 0.3
                sorted_data = df.sort_values('transaction_date')['order_qty'].values
                if len(sorted_data) > 0:
                    smooth = sorted_data[0]
                    for val in sorted_data[1:]:
                        smooth = alpha * val + (1 - alpha) * smooth
                    return float(smooth)
                return 0
            except:
                return float(df['order_qty'].mean()) if not df.empty else 0
        
        # Version 3: Moving average with seasonal adjustment
        def seasonal_ma_forecast(df, row, date):
            try:
                if isinstance(date, str):
                    target_date = pd.to_datetime(date)
                else:
                    target_date = pd.Timestamp(date)
                
                # Get recent moving average
                recent_avg = df['order_qty'].tail(min(7, len(df))).mean()
                
                # Seasonal adjustment based on month
                month = target_date.month
                month_data = df[df['transaction_date'].dt.month == month]
                if not month_data.empty:
                    seasonal_factor = month_data['order_qty'].mean() / df['order_qty'].mean()
                    seasonal_factor = min(max(seasonal_factor, 0.5), 2.0)  # Cap factor
                    return float(recent_avg * seasonal_factor)
                else:
                    return float(recent_avg)
            except:
                return float(df['order_qty'].mean()) if not df.empty else 0
        
        # Version 4: Linear trend projection
        def linear_trend_forecast(df, row, date):
            try:
                if len(df) >= 10:
                    # Calculate trend over recent periods
                    recent_data = df.tail(min(14, len(df)))
                    recent_data = recent_data.sort_values('transaction_date').reset_index(drop=True)
                    
                    x_vals = range(len(recent_data))
                    y_vals = recent_data['order_qty'].values
                    
                    # Simple linear regression
                    n = len(x_vals)
                    if n > 1:
                        x_mean = sum(x_vals) / n
                        y_mean = sum(y_vals) / n
                        
                        slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals)) / \
                               sum((x - x_mean) ** 2 for x in x_vals)
                        intercept = y_mean - slope * x_mean
                        
                        # Project forward
                        projected = intercept + slope * n
                        return float(max(0, projected))
                
                return float(df['order_qty'].mean())
            except:
                return float(df['order_qty'].mean()) if not df.empty else 0
        
        # Version 5: Weighted recent average
        def weighted_recent_forecast(df, row, date):
            try:
                recent_data = df.tail(min(10, len(df)))['order_qty'].values
                if len(recent_data) > 0:
                    # Give more weight to recent values
                    weights = [i + 1 for i in range(len(recent_data))]
                    weighted_avg = sum(val * weight for val, weight in zip(recent_data, weights)) / sum(weights)
                    return float(weighted_avg)
                return 0
            except:
                return float(df['order_qty'].mean()) if not df.empty else 0
        
        forecast_versions = {
            'day_of_week_trend': dow_trend_forecast,
            'exponential_smoothing': exp_smoothing_forecast,
            'seasonal_moving_average': seasonal_ma_forecast,
            'linear_trend_projection': linear_trend_forecast,
            'weighted_recent_average': weighted_recent_forecast
        }
        
        return forecast_versions

    def _analyze_accuracy_by_sku_time_updated(self, data: pd.DataFrame, sku_id: str = None) -> Dict:
        analysis = {
            'by_sku': {},
            'by_channel': {},
            'by_time_pattern': {},
            'by_promotional_vs_base': {},
            'insights': []
        }
    
        try:
            if data.empty:
                return analysis
        
            # Analysis by SKU (if multiple SKUs in data)
            if not sku_id and 'sku_id' in data.columns:
                sku_groups = data.groupby('sku_id').agg({
                    'total_demand_qty': ['mean', 'std', 'count'],
                    'base_demand_qty': 'mean',
                    'promotional_demand_qty': 'mean'
                }).round(2)
            
                for sku in sku_groups.index:
                    analysis['by_sku'][sku] = {
                        'avg_total_demand': enhanced_safe_float_conversion(sku_groups.loc[sku, ('total_demand_qty', 'mean')]),
                        'demand_volatility': enhanced_safe_float_conversion(sku_groups.loc[sku, ('total_demand_qty', 'std')]),
                        'avg_base_demand': enhanced_safe_float_conversion(sku_groups.loc[sku, ('base_demand_qty', 'mean')]),
                        'avg_promo_demand': enhanced_safe_float_conversion(sku_groups.loc[sku, ('promotional_demand_qty', 'mean')]),
                        'data_points': int(sku_groups.loc[sku, ('total_demand_qty', 'count')])
                    }
        
            # Analysis by Channel using YOUR channel column
            if 'channel' in data.columns:
                channel_groups = data.groupby('channel').agg({
                    'total_demand_qty': ['mean', 'std', 'count'],
                    'is_promotional': 'mean'
                }).round(2)
            
                for channel in channel_groups.index:
                    analysis['by_channel'][channel] = {
                        'avg_demand': enhanced_safe_float_conversion(channel_groups.loc[channel, ('total_demand_qty', 'mean')]),
                        'demand_volatility': enhanced_safe_float_conversion(channel_groups.loc[channel, ('total_demand_qty', 'std')]),
                        'promotional_frequency': enhanced_safe_float_conversion(channel_groups.loc[channel, ('is_promotional', 'mean')]) * 100,
                        'data_points': int(channel_groups.loc[channel, ('total_demand_qty', 'count')])
                    }
        
            # Analysis by time patterns using YOUR transaction_date
            if 'transaction_date' in data.columns:
                data['day_of_week'] = pd.to_datetime(data['transaction_date']).dt.day_name()
                dow_groups = data.groupby('day_of_week').agg({
                    'total_demand_qty': ['mean', 'count']
                }).round(2)
            
                for dow in dow_groups.index:
                    analysis['by_time_pattern'][dow] = {
                        'avg_demand': enhanced_safe_float_conversion(dow_groups.loc[dow, ('total_demand_qty', 'mean')]),
                        'data_points': int(dow_groups.loc[dow, ('total_demand_qty', 'count')])
                    }
        
            # Analysis by promotional vs base using YOUR promotional flags
            if 'is_promotional' in data.columns:
                promo_analysis = data.groupby('is_promotional').agg({
                    'total_demand_qty': ['mean', 'count'],
                    'base_demand_qty': 'mean',
                    'promotional_demand_qty': 'mean'
                }).round(2)
            
                for is_promo in promo_analysis.index:
                    period_type = 'promotional' if is_promo else 'base'
                    analysis['by_promotional_vs_base'][period_type] = {
                        'avg_total_demand': enhanced_safe_float_conversion(promo_analysis.loc[is_promo, ('total_demand_qty', 'mean')]),
                        'avg_base_demand': enhanced_safe_float_conversion(promo_analysis.loc[is_promo, ('base_demand_qty', 'mean')]),
                        'avg_promo_demand': enhanced_safe_float_conversion(promo_analysis.loc[is_promo, ('promotional_demand_qty', 'mean')]),
                        'data_points': int(promo_analysis.loc[is_promo, ('total_demand_qty', 'count')])
                    }
        
            # Generate insights
            insights = []
        
            if analysis['by_channel']:
                best_channel = max(analysis['by_channel'].items(), key=lambda x: x[1]['avg_demand'])
                insights.append(f"Highest demand channel: {best_channel[0]} (Avg: {best_channel[1]['avg_demand']:.1f})")
        
            if analysis['by_time_pattern']:
                best_day = max(analysis['by_time_pattern'].items(), key=lambda x: x[1]['avg_demand'])
                insights.append(f"Highest demand day: {best_day[0]} (Avg: {best_day[1]['avg_demand']:.1f})")
        
            if analysis['by_promotional_vs_base']:
                promo_data = analysis['by_promotional_vs_base'].get('promotional', {})
                base_data = analysis['by_promotional_vs_base'].get('base', {})
                if promo_data and base_data:
                    lift = (promo_data['avg_total_demand'] / base_data['avg_total_demand'] - 1) * 100 if base_data['avg_total_demand'] > 0 else 0
                    insights.append(f"Promotional lift: {lift:.1f}%")
        
            analysis['insights'] = insights
        
        except Exception as e:
            logger.error(f"FR9: Error in SKU-Time analysis: {e}")
            analysis['error'] = str(e)
    
        return analysis

    # ENHANCED FORECAST ACCURACY METHODS - END

    def get_forecast_bias_analysis_updated(self, 
                             sku_id: str = None,
                             channel: str = None, 
                             location: str = None,
                             days_back: int = 30) -> Dict:
        try:
            # Get accuracy data using updated method
            accuracy_data = self.get_forecast_accuracy_data_updated(sku_id, days_back)
        
            if 'error' in accuracy_data:
                return accuracy_data
        
            version_results = accuracy_data.get('version_results', {})
        
            # Enhanced bias analysis by YOUR forecast_version_name
            version_bias_analysis = {}
        
            for version_name, version_data in version_results.items():
                summary_metrics = version_data.get('summary_metrics', {})
            
                # Classify bias using YOUR data
                bias_pct = summary_metrics.get('bias_percentage', 0)
                bias_direction = 'neutral'
                if bias_pct and abs(bias_pct) > 5:
                    bias_direction = 'over_forecasting' if bias_pct > 0 else 'under_forecasting'
            
                bias_severity = 'low'
                if abs(bias_pct) > 15:
                    bias_severity = 'high'
                elif abs(bias_pct) > 8:
                    bias_severity = 'medium'
            
                version_bias_analysis[version_name] = {
                    'bias_direction': bias_direction,
                    'bias_severity': bias_severity,
                    'bias_percentage': enhanced_safe_float_conversion(bias_pct),
                    'bias_absolute': enhanced_safe_float_conversion(summary_metrics.get('bias', 0)),
                    'recommendations': self._generate_bias_recommendations_for_version(version_name, summary_metrics)
                }
        
            return {
                'bias_summary': {
                    'overall_bias_direction': self._determine_overall_bias(version_bias_analysis),
                    'version_count': len(version_bias_analysis),
                    'consistent_bias_versions': len([v for v in version_bias_analysis.values() if v['bias_severity'] != 'low']),
                    'total_data_points': accuracy_data.get('summary', {}).get('total_data_points', 0)
                },
                'version_bias_analysis': version_bias_analysis,
                'recommendations': self._generate_enhanced_bias_recommendations(version_bias_analysis),
                'filters_applied': {
                    'sku_id': sku_id,
                    'channel': channel,
                    'location': location,
                    'days_back': days_back
                },
                'methodology': 'fr9_version_bias_tracking',
                'data_source': 'real_forecast_versions'
            }
        
        except Exception as e:
            logger.error(f"FR9: Enhanced bias analysis failed: {e}")
            return {'error': f'Bias analysis failed: {str(e)}'}

    def _determine_overall_bias(self, version_bias_analysis: Dict) -> str:
        """Determine overall bias direction across versions"""
        if not version_bias_analysis:
            return 'neutral'
        
        over_count = len([v for v in version_bias_analysis.values() if v['bias_direction'] == 'over_forecasting'])
        under_count = len([v for v in version_bias_analysis.values() if v['bias_direction'] == 'under_forecasting'])
        
        if over_count > under_count:
            return 'predominantly_over_forecasting'
        elif under_count > over_count:
            return 'predominantly_under_forecasting'
        else:
            return 'mixed_bias_patterns'

    def _analyze_bias_consistency(self, df: pd.DataFrame) -> Dict:
        """Analyze bias consistency over time"""
        consistency = {}
        
        try:
            if 'analysis_date' in df.columns and len(df) > 7:
                df['analysis_date'] = pd.to_datetime(df['analysis_date'], errors='coerce')
                df = df.sort_values('analysis_date')
                
                # Rolling bias analysis
                df['rolling_bias'] = df['bias'].rolling(window=7, min_periods=3).mean()
                
                # Bias trend
                bias_trend = df['bias'].corr(pd.Series(range(len(df))))
                
                consistency = {
                    'bias_trend_correlation': enhanced_safe_float_conversion(bias_trend),
                    'trend_direction': 'increasing' if bias_trend > 0.1 else 'decreasing' if bias_trend < -0.1 else 'stable',
                    'bias_volatility': enhanced_safe_float_conversion(df['bias'].std()),
                    'consistent_bias_periods': len(df[df['rolling_bias'].abs() > 10]) if 'rolling_bias' in df.columns else 0
                }
        except Exception as e:
            logger.warning(f"Error in bias consistency analysis: {e}")
            consistency['error'] = str(e)
        
        return consistency

    def _analyze_bias_by_grain(self, df: pd.DataFrame) -> Dict:
        """Analyze bias patterns by grain"""
        grain_bias = {}
        
        try:
            # Bias by SKU
            if 'sku_id' in df.columns:
                sku_bias = df.groupby('sku_id')['bias'].agg(['mean', 'std', 'count']).round(2)
                grain_bias['by_sku'] = {}
                for sku in sku_bias.index:
                    grain_bias['by_sku'][sku] = {
                        'avg_bias': enhanced_safe_float_conversion(sku_bias.loc[sku, 'mean']),
                        'bias_volatility': enhanced_safe_float_conversion(sku_bias.loc[sku, 'std']),
                        'data_points': int(sku_bias.loc[sku, 'count'])
                    }
                    
        except Exception as e:
            logger.warning(f"Error in grain bias analysis: {e}")
            grain_bias['error'] = str(e)
        
        return grain_bias

    def _generate_enhanced_bias_recommendations(self, version_bias_analysis: Dict) -> List[str]:
        """Generate enhanced bias recommendations"""
        recommendations = []
        
        if not version_bias_analysis:
            return ["No version analysis available for bias recommendations"]
        
        # Find versions with high bias
        high_bias_versions = [name for name, data in version_bias_analysis.items() 
                             if data['bias_severity'] == 'high']
        
        if high_bias_versions:
            recommendations.append(
                f"High bias detected in models: {', '.join(high_bias_versions)}. Consider model recalibration."
            )
        
        # Find consistently over/under forecasting versions
        over_versions = [name for name, data in version_bias_analysis.items() 
                        if data['bias_direction'] == 'over_forecasting']
        under_versions = [name for name, data in version_bias_analysis.items() 
                         if data['bias_direction'] == 'under_forecasting']
        
        if len(over_versions) > len(under_versions):
            recommendations.append(
                "Systematic over-forecasting detected across multiple models. Consider reducing baseline forecasts."
            )
        elif len(under_versions) > len(over_versions):
            recommendations.append(
                "Systematic under-forecasting detected across multiple models. Consider increasing baseline forecasts."
            )
        
        # Model-specific recommendations
        if version_bias_analysis:
            best_version = min(version_bias_analysis.items(), 
                              key=lambda x: abs(x[1]['bias_percentage']))
            
            if abs(best_version[1]['bias_percentage']) < 5:
                recommendations.append(
                    f"Model '{best_version[0]}' shows minimal bias ({best_version[1]['bias_percentage']:.1f}%) - consider as primary model."
                )
        
        if not recommendations:
            recommendations.append("Bias levels are within acceptable ranges across all models.")
        
        return recommendations

    def get_sku_info(self, sku_id: str) -> Dict:
        """Get SKU information from master data - enhanced for real data"""
        if self.sku_master.empty:
            return {}
        
        # Try different column names that might exist in your data
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
            # Try different column name combinations that might exist
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
            
            # Apply filters based on available columns
            sku_col = self.get_sku_key()
            loc_col = self.get_location_key()
            
            if sku_id and sku_col in df.columns:
                df = df[df[sku_col].astype(str) == str(sku_id)]
            
            if location and loc_col in df.columns:
                df = df[df[loc_col].astype(str) == str(location)]
            
            if channel and 'channel' in df.columns:
                df = df[df['channel'].astype(str) == str(channel)]
            
            # Get recent records
            if 'transaction_date' in df.columns:
                df = df.sort_values('transaction_date').tail(days)
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
            
            # Apply filters
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
            
            # Clean and sort data
            df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
            df = df.dropna(subset=['transaction_date']).sort_values('transaction_date')
            
            # Multiple forecasting approaches based on data availability
            target_dow = pd.Timestamp(target_date).dayofweek
            target_month = pd.Timestamp(target_date).month
            
            # 1. Day-of-week patterns (most specific)
            dow_data = df[df['transaction_date'].dt.dayofweek == target_dow]
            if not dow_data.empty and len(dow_data) >= 3:
                # Use recent day-of-week pattern with trend
                recent_dow = dow_data.tail(6)['order_qty'].mean()
                older_dow = dow_data.head(max(1, len(dow_data)-6))['order_qty'].mean() if len(dow_data) > 6 else recent_dow
                
                # Calculate trend
                if older_dow > 0:
                    trend_factor = recent_dow / older_dow
                    forecast = recent_dow * min(max(trend_factor, 0.5), 2.0)  # Cap trend between 50%-200%
                else:
                    forecast = recent_dow
                
                logger.info(f"Day-of-week forecast for {target_date} ({target_dow}): {forecast:.1f}")
                return float(forecast)
            
            # 2. Monthly seasonality
            month_data = df[df['transaction_date'].dt.month == target_month]
            if not month_data.empty and len(month_data) >= 5:
                monthly_avg = month_data['order_qty'].mean()
                recent_overall = df.tail(30)['order_qty'].mean()
                
                # Adjust monthly pattern with recent trends
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
                    trended_forecast = base_forecast * min(max(trend, 0.7), 1.5)  # Cap trend
                    logger.info(f"Trend-based forecast: {trended_forecast:.1f}")
                    return float(trended_forecast)
            
            # 4. Simple recent average (fallback)
            recent_avg = df.tail(min(30, len(df)))['order_qty'].mean()
            logger.info(f"Simple average forecast: {recent_avg:.1f}")
            return float(recent_avg)
            
        except Exception as e:
            logger.error(f"Error calculating current forecast: {e}")
            return None
    
    def _generate_bias_recommendations_for_version(self, version_name: str, summary_metrics: Dict) -> List[str]:
        recommendations = []
        try:
            bias_pct = summary_metrics.get('bias_percentage', 0)
            if abs(bias_pct) <= 5:
                recommendations.append(f"{version_name}: Bias within acceptable range")
            elif bias_pct > 15:
                recommendations.append(f"{version_name}: High over-forecasting")
            elif bias_pct < -15:
                recommendations.append(f"{version_name}: High under-forecasting")
        except:
            pass
        return recommendations if recommendations else [f"{version_name}: No recommendations"]



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
            
            # Check SKU existence in real data
            if sku_col in df.columns:
                sku_exists = str(sku_id) in df[sku_col].astype(str).values
                validation_result['sku_exists'] = sku_exists
                if not sku_exists:
                    validation_result['warnings'].append(f"SKU {sku_id} not found in historical data")
                    # Suggest similar SKUs
                    all_skus = df[sku_col].astype(str).unique()
                    similar = [s for s in all_skus if sku_id.upper() in s.upper() or s.upper() in sku_id.upper()][:3]
                    if similar:
                        validation_result['suggestions'].append(f"Similar SKUs found: {', '.join(similar)}")
            
            # Check location existence
            if loc_col in df.columns:
                location_exists = str(location) in df[loc_col].astype(str).values
                validation_result['location_exists'] = location_exists
                if not location_exists and location:
                    validation_result['warnings'].append(f"Location {location} not found in historical data")
                    # Suggest similar locations
                    all_locations = df[loc_col].astype(str).unique()
                    similar = [l for l in all_locations if location.upper() in l.upper() or l.upper() in location.upper()][:3]
                    if similar:
                        validation_result['suggestions'].append(f"Similar locations found: {', '.join(similar)}")
            
            # Check channel existence
            if 'channel' in df.columns:
                channel_exists = str(channel) in df['channel'].astype(str).values
                validation_result['channel_exists'] = channel_exists
                if not channel_exists and channel:
                    validation_result['warnings'].append(f"Channel {channel} not found in historical data")
                    all_channels = df['channel'].astype(str).unique()
                    validation_result['suggestions'].append(f"Available channels: {', '.join(all_channels)}")
            
            # Check if exact combination exists
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
                # Add data quality insights
                if 'transaction_date' in filtered_df.columns:
                    date_range = filtered_df['transaction_date'].agg(['min', 'max'])
                    validation_result['suggestions'].append(
                        f"Historical data available from {date_range['min'].strftime('%Y-%m-%d')} to {date_range['max'].strftime('%Y-%m-%d')}"
                    )
                
                if 'order_qty' in filtered_df.columns:
                    avg_qty = filtered_df['order_qty'].mean()
                    validation_result['suggestions'].append(f"Historical average quantity: {avg_qty:.1f}")
            
            # Check SKU in master data if available
            if not self.sku_master.empty:
                sku_master_cols = [col for col in ['sku_id', 'product_id', 'SKU_ID', 'PRODUCT_ID'] if col in self.sku_master.columns]
                if sku_master_cols:
                    master_sku_col = sku_master_cols[0]
                    sku_in_master = str(sku_id) in self.sku_master[master_sku_col].astype(str).values
                    if not sku_in_master:
                        validation_result['warnings'].append(f"SKU {sku_id} not found in master catalog")
                    else:
                        # Get SKU details from master
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
            
            # Filter data
            if sku_id and sku_col in df.columns:
                df = df[df[sku_col].astype(str) == str(sku_id)]
            if channel and 'channel' in df.columns:
                df = df[df['channel'].astype(str) == str(channel)]
            if location and loc_col in df.columns:
                df = df[df[loc_col].astype(str) == str(location)]
            
            if df.empty:
                return context
            
            df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
            df = df.dropna(subset=['transaction_date']).sort_values('transaction_date')
            
            # Historical patterns with safe conversion
            if 'order_qty' in df.columns:
                # Calculate stats safely
                qty_series = df['order_qty'].dropna()
                if not qty_series.empty:
                    context['historical_patterns'] = {
                        'total_records': int(len(df)),
                        'avg_quantity': enhanced_safe_float_conversion(qty_series.mean()),
                        'min_quantity': enhanced_safe_float_conversion(qty_series.min()),
                        'max_quantity': enhanced_safe_float_conversion(qty_series.max()),
                        'std_quantity': enhanced_safe_float_conversion(qty_series.std()),
                        'date_range': {
                            'start': df['transaction_date'].min().strftime('%Y-%m-%d'),
                            'end': df['transaction_date'].max().strftime('%Y-%m-%d')
                        }
                    }
            
            # Recent trends with safe conversion
            if len(df) >= 10:
                try:
                    recent_30 = df.tail(min(30, len(df)//2))
                    older_30 = df.head(min(30, len(df)//2))
                    
                    if not recent_30.empty and not older_30.empty:
                        recent_avg = enhanced_safe_float_conversion(recent_30['order_qty'].mean())
                        older_avg = enhanced_safe_float_conversion(older_30['order_qty'].mean())
                        
                        # Safe trend calculation
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
            
            # Day-of-week patterns with safe conversion
            try:
                target_dow = pd.Timestamp(target_date).dayofweek
                dow_patterns = df.groupby(df['transaction_date'].dt.dayofweek)['order_qty'].agg(['mean', 'count'])
                
                if not dow_patterns.empty:
                    # Convert to safe dictionary
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
            
            # Monthly patterns with safe conversion
            try:
                target_month = pd.Timestamp(target_date).month
                month_patterns = df.groupby(df['transaction_date'].dt.month)['order_qty'].agg(['mean', 'count'])
                
                if not month_patterns.empty and target_month in month_patterns.index:
                    context['seasonality']['target_month_pattern'] = {
                        'avg_quantity': enhanced_safe_float_conversion(month_patterns.loc[target_month, 'mean']),
                        'data_points': int(month_patterns.loc[target_month, 'count'])
                    }
            except Exception as e:
                logger.warning(f"Error calculating monthly patterns: {e}")
            
            # Promotional context with safe conversion
            try:
                if 'is_promotional' in df.columns:
                    promo_data = df[df['is_promotional'] == True]
                    base_data = df[df['is_promotional'] == False]
                    
                    if not promo_data.empty and not base_data.empty:
                        promo_avg = enhanced_safe_float_conversion(promo_data['order_qty'].mean())
                        base_avg = enhanced_safe_float_conversion(base_data['order_qty'].mean())
                        
                        # Safe promotional lift calculation
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
            
            # Generate safe recommendations
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
        
        # Final safety check - convert entire context to safe JSON
        return safe_json_convert(context)
    
    def store_adhoc_adjustment(self, adjustment_data: Dict) -> str:
        """Store ad-hoc forecast adjustment - keeping your working logic"""
        adjustment_id = str(uuid.uuid4())
        
        # Store in memory 
        if not hasattr(self, 'adhoc_adjustments'):
            self.adhoc_adjustments = []
        
        adjustment_record = {
            **adjustment_data,
            'adjustment_id': adjustment_id,
            'created_at': datetime.now().isoformat()
        }
        
        self.adhoc_adjustments.append(adjustment_record)
        
        # Save to CSV for persistence
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
            
            # Basic stats
            total_records = len(df)
            
            # Date range
            start_date = None
            end_date = None
            if 'transaction_date' in df.columns:
                date_series = pd.to_datetime(df['transaction_date'], errors='coerce')
                if not date_series.isna().all():
                    start_date = date_series.min().strftime('%Y-%m-%d')
                    end_date = date_series.max().strftime('%Y-%m-%d')
            
            # Granularity stats using REAL data
            unique_skus = df[sku_col].nunique() if sku_col in df.columns else 0
            unique_locations = df[loc_col].nunique() if loc_col in df.columns else 0
            
            # Count channels from real data
            unique_channels = 0
            if 'channel' in df.columns:
                unique_channels = df['channel'].nunique()
            else:
                # Count channel_* columns if they exist
                channel_cols = [col for col in df.columns if col.lower().startswith('channel_')]
                unique_channels = len(channel_cols)
            
            # Unique combinations
            unique_combinations = 0
            if sku_col in df.columns and loc_col in df.columns:
                if 'channel' in df.columns:
                    unique_combinations = df[[sku_col, loc_col, 'channel']].drop_duplicates().shape[0]
                else:
                    unique_combinations = df[[sku_col, loc_col]].drop_duplicates().shape[0]
            
            # Promotional stats from REAL data
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

# Enhanced forecast generation using real data
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
        
        # Apply filters if provided
        if sku_id and sku_col in df.columns:
            df = df[df[sku_col].astype(str) == str(sku_id)]
        
        if channel and 'channel' in df.columns:
            df = df[df['channel'].astype(str) == str(channel)]
        
        if location and loc_col in df.columns:
            df = df[df[loc_col].astype(str) == str(location)]
        
        if df.empty:
            logger.warning("No data after filtering, using synthetic fallback")
            return generate_forecast_data(days)
        
        # Prepare date column
        if 'transaction_date' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
            df = df.dropna(subset=['transaction_date'])
        
        if df.empty or 'order_qty' not in df.columns:
            return generate_forecast_data(days)
        
        # Generate forecast based on real patterns
        labels = []
        values = []
        confidence_upper = []
        confidence_lower = []
        
        # Calculate day-of-week patterns
        dow_patterns = df.groupby(df['transaction_date'].dt.dayofweek)['order_qty'].mean()
        recent_avg = df['order_qty'].tail(30).mean()
        recent_std = df['order_qty'].tail(30).std()
        
        # Calculate trend
        if len(df) > 7:
            recent_trend = (df['order_qty'].tail(7).mean() - df['order_qty'].tail(14).mean()) / 7
        else:
            recent_trend = 0
        
        start_date = datetime.now()
        
        for i in range(days):
            date = start_date + timedelta(days=i)
            labels.append(date.strftime('%b %d'))
            
            # Use day-of-week pattern
            dow = date.weekday()
            base_value = float(dow_patterns.get(dow, recent_avg))
            
            # Add trend
            trend_value = base_value + (recent_trend * i)
            
            # Add some realistic variation
            variation = np.random.normal(0, recent_std * 0.1)
            final_value = max(0, trend_value + variation)
            
            values.append(round(final_value))
            confidence_upper.append(round(final_value * 1.2))
            confidence_lower.append(round(final_value * 0.8))
        
        feature_count = len([col for col in df.columns if col not in ['transaction_date']])
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

