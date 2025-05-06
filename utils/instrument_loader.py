"""
Instrument Loader Utility

This module provides functions to load instruments from CSV files
and prepare them for analysis.
"""

import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_instruments_from_csv(file_path):
    """
    Load instruments from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing instruments
        
    Returns:
        DataFrame containing instrument data or None if loading fails
    """
    if not os.path.exists(file_path):
        logger.error(f"Instrument file not found: {file_path}")
        return None
    
    try:
        # Read the CSV file with expected columns: symbol,name,sector,industry,market_cap
        instruments_df = pd.read_csv(file_path)
        
        # Log the columns found in the file
        logger.info(f"Loaded instrument file with columns: {instruments_df.columns.tolist()}")
        
        # Basic validation - ensure we have the required columns
        required_columns = ['symbol']
        missing_columns = [col for col in required_columns if col not in instruments_df.columns]
        
        if missing_columns:
            logger.error(f"Required columns missing from instrument file: {missing_columns}")
            return None
        
        # Clean up symbols (remove whitespace, convert to uppercase)
        instruments_df['symbol'] = instruments_df['symbol'].str.strip().str.upper()
        
        # Remove any duplicate symbols
        if instruments_df.duplicated('symbol').any():
            logger.warning(f"Found {instruments_df.duplicated('symbol').sum()} duplicate symbols in instrument file")
            instruments_df.drop_duplicates('symbol', inplace=True)
        
        # Convert market cap to numeric if possible (e.g., "2.5T" -> 2500000000000)
        if 'market_cap' in instruments_df.columns:
            try:
                instruments_df['market_cap_numeric'] = instruments_df['market_cap'].apply(convert_market_cap)
            except Exception as e:
                logger.warning(f"Could not convert market cap to numeric: {e}")
        
        logger.info(f"Successfully loaded {len(instruments_df)} instruments")
        return instruments_df
    
    except Exception as e:
        logger.error(f"Error loading instrument file: {e}")
        return None

def convert_market_cap(market_cap_str):
    """Convert market cap string (e.g., '2.5T') to numeric value"""
    if not isinstance(market_cap_str, str):
        return market_cap_str
    
    market_cap_str = market_cap_str.strip().upper()
    
    # Handle different suffixes
    multipliers = {
        'T': 1e12,  # Trillion
        'B': 1e9,   # Billion
        'M': 1e6,   # Million
        'K': 1e3    # Thousand
    }
    
    # Extract the numeric part and suffix
    for suffix, multiplier in multipliers.items():
        if market_cap_str.endswith(suffix):
            try:
                value = float(market_cap_str[:-1]) * multiplier
                return value
            except ValueError:
                return None
    
    # If no suffix, try to convert directly
    try:
        return float(market_cap_str)
    except ValueError:
        return None

def filter_instruments(instruments_df, criteria=None):
    """
    Filter instruments based on specified criteria.
    
    Args:
        instruments_df: DataFrame containing instrument data
        criteria: Dictionary of filter criteria, examples:
                 {'sector': 'Technology'} - exact match on sector
                 {'sector': ['Technology', 'Healthcare']} - match any in list
                 {'market_cap_numeric': {'min': 1e9}} - minimum market cap
                 {'market_cap_numeric': {'max': 1e12}} - maximum market cap
        
    Returns:
        Filtered DataFrame
    """
    if instruments_df is None or instruments_df.empty:
        return None
    
    if not criteria:
        return instruments_df
    
    filtered_df = instruments_df.copy()
    
    for column, value in criteria.items():
        if column not in filtered_df.columns:
            logger.warning(f"Filter column '{column}' not found in instruments data")
            continue
            
        if isinstance(value, dict):
            # Handle range filters (min/max)
            if 'min' in value and value['min'] is not None:
                filtered_df = filtered_df[filtered_df[column] >= value['min']]
            if 'max' in value and value['max'] is not None:
                filtered_df = filtered_df[filtered_df[column] <= value['max']]
        elif isinstance(value, list):
            # Handle list of values (any match)
            filtered_df = filtered_df[filtered_df[column].isin(value)]
        else:
            # Handle exact match
            filtered_df = filtered_df[filtered_df[column] == value]
    
    logger.info(f"Filtered instruments from {len(instruments_df)} to {len(filtered_df)} based on criteria")
    return filtered_df

def get_instrument_list(file_path, criteria=None, limit=None, sort_by=None, ascending=True):
    """
    Load and filter instruments, returning a list of symbols.
    
    Args:
        file_path: Path to the CSV file containing instruments
        criteria: Dictionary of filter criteria (optional)
        limit: Maximum number of instruments to return (optional)
        sort_by: Column to sort by (optional)
        ascending: Sort order (True for ascending, False for descending)
        
    Returns:
        List of instrument symbols
    """
    instruments_df = load_instruments_from_csv(file_path)
    if instruments_df is None:
        return []
    
    if criteria:
        instruments_df = filter_instruments(instruments_df, criteria)
    
    # Sort if requested
    if sort_by and sort_by in instruments_df.columns:
        instruments_df = instruments_df.sort_values(by=sort_by, ascending=ascending)
    
    # Limit if requested
    if limit and limit > 0:
        instruments_df = instruments_df.head(limit)
    
    return instruments_df['symbol'].tolist()
