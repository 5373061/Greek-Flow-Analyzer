import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy.stats import norm

class OptionsFeatureEngineer:
    """Engineer features for options ML models"""
    
    def __init__(self, lookback_periods: List[int] = [5, 10, 20]):
        self.lookback_periods = lookback_periods
        
    def create_features(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Generate features from options chain data"""
        df = options_data.copy()
        
        # Volatility features
        df['iv_percentile'] = df.groupby('symbol')['implied_vol'].transform(
            lambda x: x.rank(pct=True))
        df['iv_term_structure'] = df.groupby('date')['implied_vol'].transform(
            lambda x: x.diff() / x.shift(1))
            
        # Greek interaction features
        df['gamma_theta_ratio'] = df['gamma_exposure'] / df['theta_exposure'].abs()
        df['vanna_charm_ratio'] = df['vanna_exposure'] / df['charm_exposure'].abs()
        
        # Volume and liquidity features
        df['volume_oi_ratio'] = df['volume'] / df['open_interest']
        df['bid_ask_spread'] = (df['ask'] - df['bid']) / df['underlying_price']
        
        # Strike positioning
        df['moneyness'] = np.log(df['underlying_price'] / df['strike'])
        df['strike_density'] = self._calculate_strike_density(df)
        
        # Time decay features
        df['theta_acceleration'] = self._calculate_theta_acceleration(df)
        df['gamma_decay'] = self._calculate_gamma_decay(df)
        
        # Add rolling features
        for period in self.lookback_periods:
            # Volatility trends
            df[f'iv_ma_{period}'] = df.groupby('symbol')['implied_vol'].transform(
                lambda x: x.rolling(period).mean())
            df[f'iv_std_{period}'] = df.groupby('symbol')['implied_vol'].transform(
                lambda x: x.rolling(period).std())
            
            # Greek trends
            df[f'gamma_trend_{period}'] = df.groupby('symbol')['gamma_exposure'].transform(
                lambda x: x.rolling(period).mean() / x)
            
            # Volume trends
            df[f'volume_trend_{period}'] = df.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(period).mean() / x.rolling(5).mean())
        
        return df
    
    def _calculate_strike_density(self, df: pd.DataFrame) -> pd.Series:
        """Calculate the density of strikes around current price"""
        return df.groupby('date').apply(
            lambda x: len(x[(x['strike'] >= 0.9 * x['underlying_price']) & 
                          (x['strike'] <= 1.1 * x['underlying_price'])])
        )
    
    def _calculate_theta_acceleration(self, df: pd.DataFrame) -> pd.Series:
        """Calculate the rate of change of theta decay"""
        return df.groupby('symbol')['theta_exposure'].transform(
            lambda x: x.diff() / x.shift(1))
    
    def _calculate_gamma_decay(self, df: pd.DataFrame) -> pd.Series:
        """Calculate the expected gamma decay rate"""
        return df['gamma_exposure'] * np.sqrt(df['days_to_expiry'])