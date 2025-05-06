from typing import Tuple, Optional, Dict, List, Any
from datetime import datetime
import logging
import pandas as pd
from .cache_manager import MarketDataCache
from api_fetcher import fetch_underlying_snapshot, fetch_options_chain_snapshot
from .greek_utils import calculate_black_scholes_greeks

class OptionsDataPipeline:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache = MarketDataCache()
    
    def fetch_symbol_data(self, symbol: str) -> tuple:
        """Fetch options data"""
        try:
            # Handle both module and dictionary configs
            if hasattr(self.config, 'get'):
                # It's a dictionary-like object
                api_key = self.config.get('POLYGON_API_KEY')
            else:
                # It's a module
                api_key = getattr(self.config, 'POLYGON_API_KEY', None)
            
            if not api_key:
                self.logger.error("No API key found in configuration")
                return None, None
            
            underlying_data = fetch_underlying_snapshot(symbol, api_key)
            
            # Validate underlying data first
            if not underlying_data:
                self.logger.error(f"Failed to fetch underlying data for {symbol}")
                return None, None
            
            # Validate only the fields we actually use
            required_fields = [
                'ticker',
                'min',      # For last price
                'prevDay'   # For volume
            ]
            missing_fields = [f for f in required_fields if f not in underlying_data]
            if missing_fields:
                self.logger.error(f"Missing required fields: {missing_fields}")
                return None, None
            
            # Map fields and cache the result
            underlying_mapped = {
                'ticker': underlying_data['ticker'],
                'last': underlying_data['min']['c'],
                'volume': underlying_data['prevDay'].get('v', 0)
            }
            
            options_data = fetch_options_chain_snapshot(symbol, api_key)
            
            # Cache the fetched data
            self.cache.set(symbol, {
                'underlying': underlying_mapped,
                'options': options_data
            })
            
            return underlying_mapped, options_data
            
        except Exception as e:
            self.logger.exception(f"Error fetching data for {symbol}: {str(e)}")
            return None, None
        
    def prepare_analysis_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Prepare options data for analysis."""
        try:
            # Fetch options data - using fetch_symbol_data instead of fetch_options_chain
            underlying_data, options_data = self.fetch_symbol_data(symbol)
            
            if options_data is None or len(options_data) == 0:
                logging.error("No data received from API")
                return None
                
            # Process the data
            df = pd.DataFrame(options_data)
            
            # Ensure required columns exist
            if 'impliedVolatility' not in df.columns and 'implied_volatility' not in df.columns:
                logging.warning("Adding synthetic impliedVolatility column")
                # Add a synthetic impliedVolatility column based on a reasonable default
                df['implied_volatility'] = 0.3  # Default 30% IV
            elif 'impliedVolatility' in df.columns and 'implied_volatility' not in df.columns:
                # Rename to standardized column name
                df['implied_volatility'] = df['impliedVolatility']
                
            # Add underlying data
            if underlying_data:
                df['underlying_price'] = underlying_data.get('last', 0)
                df['underlying_ticker'] = underlying_data.get('ticker', symbol)
                df['underlying_volume'] = underlying_data.get('volume', 0)
            
            # Add fetch timestamp
            df['fetch_time'] = datetime.now()
            
            # Extract contract details
            if 'details' in df.columns:
                # Extract nested details if present
                for idx, row in df.iterrows():
                    if isinstance(row['details'], dict):
                        details = row['details']
                        df.loc[idx, 'strike'] = details.get('strike_price')
                        df.loc[idx, 'expiration'] = pd.to_datetime(details.get('expiration_date'))
                        df.loc[idx, 'type'] = details.get('contract_type', '').lower()
                        
            # Standardize column names
            if 'openInterest' not in df.columns and 'open_interest' in df.columns:
                df['openInterest'] = df['open_interest']
            
            # Convert expiration to datetime if it's not already
            if 'expiration' in df.columns and not pd.api.types.is_datetime64_dtype(df['expiration']):
                df['expiration'] = pd.to_datetime(df['expiration'])
            
            logging.info(f"Successfully prepared {len(df)} options records")
            return df
            
        except Exception as e:
            logging.error(f"Error preparing analysis data: {e}")
            return None
        
    def calculate_greeks(self, analysis_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate option Greeks and energy metrics"""
        try:
            if analysis_df.empty:
                self.logger.warning("Empty DataFrame provided for Greek calculations")
                return None
                
            df = analysis_df.copy()
            
            # Pre-calculate time to expiration for all options
            now = datetime.now()
            df['time_to_expiry'] = df['expiration'].apply(lambda x: max((x - now).days / 365, 0.0001))
            
            # Calculate Greeks with vectorized operations
            for idx, row in df.iterrows():
                # Calculate Greeks with better handling of small values
                delta, gamma, theta, vega = calculate_black_scholes_greeks(
                    S=row['underlying_price'],
                    K=row['strike'],
                    T=row['time_to_expiry'],
                    r=0.05,  # Risk-free rate
                    sigma=max(row['implied_volatility'], 0.01),
                    option_type=row['type']
                )
                
                # Store raw Greeks
                df.loc[idx, 'raw_delta'] = delta
                df.loc[idx, 'raw_gamma'] = gamma
                df.loc[idx, 'raw_theta'] = theta
                df.loc[idx, 'raw_vega'] = vega
                
                # Weight Greeks by open interest
                df.loc[idx, 'delta'] = delta * row['openInterest']
                df.loc[idx, 'gamma'] = abs(gamma * row['openInterest'])
                df.loc[idx, 'theta'] = theta * row['openInterest']
                df.loc[idx, 'vega'] = vega * row['openInterest']
            
            # Calculate energy metrics with safeguards
            total_gamma = df['gamma'].sum()
            if total_gamma > 0:
                df['gamma_contribution'] = df['gamma'] / total_gamma
            else:
                df['gamma_contribution'] = df['gamma'].apply(lambda x: 1.0 if x > 0 else 0.0)
                
            df['delta_weight'] = df['delta']
            df['energy_level'] = df['gamma'] * df['implied_volatility']
            
            # Add dollar impact metrics
            df['price_sensitivity'] = df['delta'] * df['underlying_price'] / 100
            
            # Debug logging
            self.logger.debug(f"Greek calculation summary:")
            self.logger.debug(f"Total Gamma: {total_gamma:.6f}")
            self.logger.debug(f"Max Gamma Contribution: {df['gamma_contribution'].max():.2%}")
            self.logger.debug(f"Net Delta Weight: {df['delta_weight'].sum():.2f}")
            
            return df
            
        except Exception as e:
            self.logger.exception("Error calculating Greeks: %s", str(e))
            return None
