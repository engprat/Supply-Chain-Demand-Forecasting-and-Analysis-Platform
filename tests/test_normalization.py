import pytest
import pandas as pd
import numpy as np
from src.utils.constants import DATA_PATH, NUMERIC_COLS, CATEGORICAL_COLS
from src.validation.normalization import encode_categorical, scale_numeric

# === Test: Loading the Data ===
@pytest.fixture(scope="module")
def raw_data():
    """Load the actual dataset from disk."""
    return pd.read_csv(DATA_PATH, encoding="ISO-8859-1")

# === Test: Encoding Categorical Columns ===
def test_encode_categorical_columns(raw_data):
    df = raw_data.copy()
    
    # Check if CATEGORICAL_COLS are in the DataFrame
    missing_cols = [col for col in CATEGORICAL_COLS if col not in df.columns]
    if missing_cols:
        pytest.fail(f"Missing categorical columns: {', '.join(missing_cols)}")
    
    # Drop rows where any categorical column has NaN values
    df = df.dropna(subset=CATEGORICAL_COLS)

    mappings = encode_categorical(df, CATEGORICAL_COLS)
    
    assert isinstance(mappings, dict)
    for col in CATEGORICAL_COLS:
        enc_col = f"{col}_enc"
        assert enc_col in df.columns
        assert pd.api.types.is_integer_dtype(df[enc_col])
        assert col in mappings
        assert isinstance(mappings[col], dict)

# === Test: Scaling Numeric Columns ===
@pytest.mark.parametrize("method", ["MinMax", "Standard"])
def test_scale_numeric_columns(raw_data, method):
    df = raw_data.copy()
    
    # Drop rows where any numeric column has NaN values
    df = df.dropna(subset=NUMERIC_COLS)

    df_scaled = scale_numeric(df, NUMERIC_COLS, method=method)
    
    for col in NUMERIC_COLS:
        scaled_col = f"{col}_{method}"
        # Scaled column exists
        assert scaled_col in df_scaled.columns, f"Scaled column missing: {scaled_col}"
        # Scaled column is float dtype
        assert pd.api.types.is_float_dtype(df_scaled[scaled_col]), f"Scaled column {scaled_col} not float type"
        # Value sanity check
        if method == "MinMax":
            # All values should be in [0,1]
            assert df_scaled[scaled_col].between(0, 1).all(), f"MinMax scaled column {scaled_col} not in [0,1]"
        elif method == "Standard":
            # mean ~ 0, std ~ 1 (not strict due to dropna & real data noise)
            mean_val = df_scaled[scaled_col].mean()
            std_val = df_scaled[scaled_col].std()
            assert abs(mean_val) < 0.2, f"Standard scaled column {scaled_col} mean not ~0"
            assert 0.8 < std_val < 1.2, f"Standard scaled column {scaled_col} std not ~1"

# === Test: Scaling Constant Column (Edge Case) ===
def test_scale_constant_column():
    df = pd.DataFrame({'num': [5, 5, 5, 5]})
    minmax_scaled = scale_numeric(df, ['num'], method='MinMax')
    std_scaled = scale_numeric(df, ['num'], method='Standard')
    assert minmax_scaled['num_MinMax'].eq(0).all(), 'MinMax scaling of constant column should be 0'
    assert std_scaled['num_Standard'].eq(0).all(), 'Standard scaling of constant column should be 0'
