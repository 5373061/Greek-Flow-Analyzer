"""
Greek Flow Calculator Module

Provides functions for calculating option Greeks and related metrics.
"""

import pandas as pd
import numpy as np
import math


def calculate_greeks(options_df, current_price):
    """
    Calculate Greeks for each option in the dataframe.
    
    Args:
        options_df: DataFrame containing options data
        current_price: Current price of the underlying asset
        
    Returns:
        DataFrame with calculated Greeks
    """
    if 'strike' not in options_df.columns or 'impliedVolatility' not in options_df.columns:
        raise ValueError("Options dataframe must have 'strike' and 'impliedVolatility' columns")
    
    greek_df = options_df.copy()
    
    # Calculate basic Greeks if not already in the dataframe
    if 'delta' not in greek_df.columns:
        greek_df['delta'] = greek_df.apply(
            lambda row: calculate_delta(
                current_price, 
                row['strike'], 
                row['impliedVolatility'],
                row['isCall'] if 'isCall' in greek_df.columns else True
            ), 
            axis=1
        )
    
    if 'gamma' not in greek_df.columns:
        greek_df['gamma'] = greek_df.apply(
            lambda row: calculate_gamma(
                current_price, 
                row['strike'], 
                row['impliedVolatility']
            ), 
            axis=1
        )
    
    if 'theta' not in greek_df.columns:
        greek_df['theta'] = greek_df.apply(
            lambda row: calculate_theta(
                current_price, 
                row['strike'], 
                row['impliedVolatility'],
                row['isCall'] if 'isCall' in greek_df.columns else True
            ), 
            axis=1
        )
    
    if 'vega' not in greek_df.columns:
        greek_df['vega'] = greek_df.apply(
            lambda row: calculate_vega(
                current_price, 
                row['strike'], 
                row['impliedVolatility']
            ), 
            axis=1
        )
    
    # Calculate secondary Greeks
    greek_df['vanna'] = calculate_vanna(greek_df)
    greek_df['charm'] = calculate_charm(greek_df)
    
    return greek_df


def calculate_delta(stock_price, strike_price, implied_volatility, is_call=True):
    """Calculate option delta using Black-Scholes approximation."""
    # Simplified calculation for demonstration
    moneyness = stock_price / strike_price
    if is_call:
        return 0.5 + 0.5 * (moneyness - 1) / (implied_volatility + 0.1)
    else:
        return -0.5 - 0.5 * (moneyness - 1) / (implied_volatility + 0.1)


def calculate_gamma(stock_price, strike_price, implied_volatility):
    """Calculate option gamma using Black-Scholes approximation."""
    # Simplified calculation for demonstration
    moneyness = stock_price / strike_price
    return 0.1 * math.exp(-0.5 * ((moneyness - 1) / implied_volatility) ** 2)


def calculate_theta(stock_price, strike_price, implied_volatility, is_call=True):
    """Calculate option theta using Black-Scholes approximation."""
    # Simplified calculation for demonstration
    moneyness = stock_price / strike_price
    base_theta = -0.01 * stock_price * implied_volatility / math.sqrt(30)
    if is_call:
        adjustment = 0.5 * (moneyness - 1)
    else:
        adjustment = -0.5 * (moneyness - 1)
    return base_theta * (1 + adjustment)


def calculate_vega(stock_price, strike_price, implied_volatility):
    """Calculate option vega using Black-Scholes approximation."""
    # Simplified calculation for demonstration
    moneyness = stock_price / strike_price
    return 0.01 * stock_price * math.exp(-0.5 * ((moneyness - 1) / implied_volatility) ** 2)


def calculate_vanna(greek_df):
    """Calculate vanna (d(delta)/d(vol)) from delta and vega."""
    # Simplified calculation - for real implementation would use proper formulas
    if 'delta' in greek_df.columns and 'vega' in greek_df.columns:
        return greek_df['delta'] * greek_df['vega'] * 0.1
    return pd.Series([0] * len(greek_df))


def calculate_charm(greek_df):
    """Calculate charm (d(delta)/d(time)) from delta and theta."""
    # Simplified calculation - for real implementation would use proper formulas
    if 'delta' in greek_df.columns and 'theta' in greek_df.columns:
        return greek_df['delta'] * greek_df['theta'] * 0.05
    return pd.Series([0] * len(greek_df))


def calculate_put_call_ratio(options_df):
    """
    Calculate the put/call ratio from options data.
    
    Args:
        options_df: DataFrame containing options data with 'isCall' column
        
    Returns:
        Float representing the put/call ratio
    """
    if 'isCall' not in options_df.columns:
        # Try to infer from option symbols if possible
        if 'symbol' in options_df.columns:
            options