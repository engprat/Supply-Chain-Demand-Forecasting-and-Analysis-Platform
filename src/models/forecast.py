import numpy as np
import pandas as pd
from lightgbm import Booster
from typing import Dict, Any

def simulate_scenario_forecast(model: Booster, features_df: pd.DataFrame, scenario: Dict[str, Any]) -> np.ndarray:
    """
    Adjusts features according to scenario dict and returns scenario-adjusted predictions.
    Supports: temperature, price, discount, promotion, inflation, and any other feature in scenario.
    """
    scenario_df = features_df.copy()
    for feature, value in scenario.items():
        if feature in scenario_df.columns:
            # Apply multiplier (default) or override if not numeric
            if isinstance(value, (int, float)):
                scenario_df[feature] = scenario_df[feature] * value
            else:
                scenario_df[feature] = value
    preds = model.predict(scenario_df)
    return preds
