'''src/data/load_data.py'''
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd

logger = logging.getLogger(__name__)


def load_raw_data(
    data_path: Union[str, Path], 
    encoding: str = 'utf-8',
    chunksize: Optional[int] = None
) -> pd.DataFrame:
    """
    Load raw CSV data with robust error handling.
    
    Args:
        data_path: Path to the CSV file
        encoding: File encoding (default: utf-8)
        chunksize: Optional chunk size for large files
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be parsed
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        # Try primary encoding first
        df = pd.read_csv(
            data_path, 
            encoding=encoding,
            chunksize=chunksize,
            low_memory=False  # Prevent mixed types warning
        )
        
        # Handle chunked reading
        if chunksize is not None:
            df = pd.concat(df, ignore_index=True)
            
        logger.info(f"Successfully loaded data: {df.shape} from {data_path}")
        return df
        
    except UnicodeDecodeError:
        # Fallback to different encodings
        fallback_encodings = ['latin-1', 'iso-8859-1', 'cp1252']
        
        for fallback_encoding in fallback_encodings:
            try:
                logger.warning(f"Trying fallback encoding: {fallback_encoding}")
                df = pd.read_csv(
                    data_path, 
                    encoding=fallback_encoding,
                    chunksize=chunksize,
                    low_memory=False
                )
                
                if chunksize is not None:
                    df = pd.concat(df, ignore_index=True)
                    
                logger.info(f"Successfully loaded with {fallback_encoding}: {df.shape}")
                return df
                
            except UnicodeDecodeError:
                continue
                
        raise ValueError(f"Could not decode file with any encoding: {data_path}")
        
    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty or corrupted CSV file: {data_path}")
        
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise ValueError(f"Failed to load data from {data_path}: {str(e)}")


def validate_data_structure(
    df: pd.DataFrame, 
    essential_columns: List[str],
    max_missing_essential: float = 0.5
) -> Dict[str, any]:
    """
    Validate the basic structure of loaded data.
    
    Args:
        df: DataFrame to validate
        essential_columns: List of columns that must be present
        max_missing_essential: Maximum fraction of essential columns that can be missing
        
    Returns:
        Dictionary with validation results
        
    Raises:
        ValueError: If too many essential columns are missing
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns_found': list(df.columns),
        'missing_essential': [],
        'validation_passed': True,
        'warnings': []
    }
    
    # Check for missing essential columns
    missing_essential = [
        col for col in essential_columns 
        if col not in df.columns
    ]
    
    validation_results['missing_essential'] = missing_essential
    
    # Check if too many essential columns are missing
    missing_ratio = len(missing_essential) / len(essential_columns)
    if missing_ratio > max_missing_essential:
        validation_results['validation_passed'] = False
        raise ValueError(
            f"Too many essential columns missing ({missing_ratio:.1%}): {missing_essential}"
        )
    
    # Add warnings for missing columns
    if missing_essential:
        warning_msg = f"Missing essential columns: {missing_essential}"
        validation_results['warnings'].append(warning_msg)
        logger.warning(warning_msg)
    
    # Check for empty DataFrame
    if df.empty:
        validation_results['validation_passed'] = False
        raise ValueError("Loaded DataFrame is empty")
    
    # Check for duplicate column names
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        warning_msg = f"Duplicate column names found: {duplicate_columns}"
        validation_results['warnings'].append(warning_msg)
        logger.warning(warning_msg)
    
    logger.info(f"Data structure validation completed: {validation_results}")
    return validation_results


def load_and_validate_data(
    data_path: Union[str, Path],
    essential_columns: List[str],
    encoding: str = 'utf-8',
    max_missing_essential: float = 0.5
) -> tuple[pd.DataFrame, Dict[str, any]]:
    """
    Load data and perform initial validation in one step.
    
    Args:
        data_path: Path to the CSV file
        essential_columns: List of columns that must be present
        encoding: File encoding
        max_missing_essential: Maximum fraction of essential columns that can be missing
        
    Returns:
        Tuple of (DataFrame, validation_results)
    """
    try:
        # Load the data
        df = load_raw_data(data_path, encoding)
        
        # Validate structure
        validation_results = validate_data_structure(
            df, essential_columns, max_missing_essential
        )
        
        logger.info("Data loading and validation completed successfully")
        return df, validation_results
        
    except Exception as e:
        logger.error(f"Error in load_and_validate_data: {e}")
        raise


def get_data_sample(
    df: pd.DataFrame, 
    n_rows: int = 5, 
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Get a sample of the data for inspection.
    
    Args:
        df: DataFrame to sample
        n_rows: Number of rows to sample
        random_state: Random seed for reproducibility
        
    Returns:
        Sample DataFrame
    """
    if len(df) <= n_rows:
        return df.copy()
    
    return df.sample(n=n_rows, random_state=random_state)


def get_basic_info(df: pd.DataFrame) -> Dict[str, any]:
    """
    Get basic information about the DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with basic info
    """
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'null_counts': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum()
    }