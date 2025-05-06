# enhanced_api_fetcher.py
# Enhanced functions for fetching data from Polygon API and preprocessing
# New version with improved error handling, caching, and class-based structure

import requests
import pandas as pd
import numpy as np
import json
import logging
import os
import glob
import time
import functools
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Configure logging
logger = logging.getLogger(__name__)  # Use module-level logger

# Constants
BASE_URL = "https://api.polygon.io"
CACHE_EXPIRY = 300  # Cache expiry in seconds (5 minutes)


class PolygonAPIError(Exception):
    """Base exception for Polygon API errors"""
    pass


class RateLimitExceeded(PolygonAPIError):
    """Exception for when API rate limits are exceeded"""
    pass


class NetworkError(PolygonAPIError):
    """Exception for network-related errors"""
    pass


class DataError(PolygonAPIError):
    """Exception for data-related errors"""
    pass


@dataclass
class OptionsContract:
    """Data class representing an options contract with its attributes"""
    strike_price: float
    expiration_date: str
    contract_type: str
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: Optional[float] = None
    vega: Optional[float] = None
    
    @classmethod
    def from_api_response(cls, contract_data: Dict[str, Any]) -> Optional['OptionsContract']:
        """Create an OptionsContract from API response data, return None if invalid"""
        if not isinstance(contract_data, dict):
            return None
            
        details = contract_data.get('details', {})
        greeks = contract_data.get('greeks', {})
        
        if not details or not greeks:
            return None
            
        try:
            strike = float(details.get('strike_price', 0))
            expiry_str = details.get('expiration_date', '')
            otype = details.get('contract_type', '').lower()
            oi = int(contract_data.get('open_interest', 0))
            iv = contract_data.get('implied_volatility')
            
            # Clean IV value
            if isinstance(iv, (int, float)) and pd.notna(iv):
                iv_clean = float(iv)
            else:
                return None
                
            # Get greeks with safety checks
            delta = float(greeks.get('delta', 0))
            gamma = float(greeks.get('gamma', 0))
            theta = float(greeks.get('theta', 0)) if greeks.get('theta') is not None else None
            vega = float(greeks.get('vega', 0)) if greeks.get('vega') is not None else None
            
            # Validation checks
            if otype not in ['call', 'put']:
                return None
            if iv_clean <= 1e-6 or pd.isna(iv_clean):
                return None
            if not (-1.01 <= delta <= 1.01):
                return None
            if gamma < 0:
                return None
                
            return cls(
                strike_price=strike,
                expiration_date=expiry_str,
                contract_type=otype,
                open_interest=oi,
                implied_volatility=iv_clean,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega
            )
            
        except (ValueError, TypeError, OverflowError):
            return None


class RequestsCache:
    """Simple cache for API responses"""
    def __init__(self, expiry_seconds: int = CACHE_EXPIRY):
        self.cache = {}
        self.expiry = expiry_seconds
        
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache if it exists and hasn't expired"""
        if key in self.cache:
            timestamp, value = self.cache[key]
            if time.time() - timestamp < self.expiry:
                logger.debug(f"Cache hit for key: {key}")
                return value
            else:
                logger.debug(f"Cache expired for key: {key}")
                del self.cache[key]
        return None
        
    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache with the current timestamp"""
        self.cache[key] = (time.time(), value)
        logger.debug(f"Cached value for key: {key}")
        
    def clear(self) -> None:
        """Clear all items from the cache"""
        self.cache.clear()
        logger.debug("Cache cleared")


class PolygonClient:
    """Client for interacting with the Polygon.io API"""
    
    def __init__(self, api_key: str, cache_expiry: int = CACHE_EXPIRY):
        self.api_key = api_key
        self.base_url = BASE_URL
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.cache = RequestsCache(cache_expiry)
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic"""
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
        
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a request to the Polygon API with caching and error handling"""
        url = f"{self.base_url}{endpoint}"
        cache_key = f"{url}_{json.dumps(params or {})}"
        
        # Check cache first
        cached_response = self.cache.get(cache_key)
        if cached_response is not None:
            return cached_response
            
        try:
            response = self.session.get(
                url, 
                headers=self.headers, 
                params=params or {}, 
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            # Check for API-level errors
            if "error" in data:
                error_msg = data.get("error", {}).get("message", "Unknown API error")
                logger.error(f"API returned error: {error_msg}")
                raise PolygonAPIError(error_msg)
                
            # Cache successful responses
            self.cache.set(cache_key, data)
            return data
            
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 429:
                logger.warning("Rate limit exceeded, backing off")
                raise RateLimitExceeded(f"Rate limit exceeded: {http_err}")
            logger.error(f"HTTP Error: {http_err}")
            if response is not None:
                logger.error(f"Response Text: {response.text}")
            raise NetworkError(f"HTTP Error: {http_err}")
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out for {url}")
            raise NetworkError("Request timed out")
            
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request Exception: {req_err}")
            raise NetworkError(f"Request failed: {req_err}")
            
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON response")
            if response is not None:
                logger.error(f"Response Text: {response.text}")
            raise DataError("Invalid JSON response")
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise PolygonAPIError(f"Unexpected error: {e}")
            
    def fetch_options_chain_snapshot(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch the full options chain snapshot for a ticker"""
        logger.info(f"Fetching Options Chain Snapshot for {ticker}...")
        endpoint = f"/v3/snapshot/options/{ticker}"
        
        try:
            data = self._make_request(endpoint)
            
            if data.get("status") == "OK" and "results" in data:
                results = data["results"]
                logger.info(f"Successfully fetched {len(results)} options contracts.")
                return results
            elif data.get("status") == "OK" and "results" in data and len(data["results"]) == 0:
                logger.warning(f"API returned OK but 0 options contracts found for {ticker}.")
                return []
            else:
                logger.warning(f"API options status: {data.get('status')}. Msg: {data.get('message', 'N/A')}")
                return []
                
        except PolygonAPIError as e:
            logger.error(f"Error fetching options chain: {e}")
            return []
            
    def fetch_underlying_snapshot(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch the snapshot for the underlying stock ticker using V2 endpoint"""
        logger.info(f"Fetching Underlying Snapshot for {ticker} (using V2)...")
        endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        
        try:
            data = self._make_request(endpoint)
            
            if data.get("status") == "OK" and "ticker" in data:
                logger.info("Successfully fetched underlying snapshot (v2).")
                return data["ticker"]
            else:
                logger.warning(f"API underlying status (v2): {data.get('status')}. Response:")
                return None
                
        except PolygonAPIError as e:
            logger.error(f"Error fetching underlying snapshot: {e}")
            return None
            
    def get_spot_price(self, ticker: str) -> Optional[float]:
        """Get the spot price for a ticker, handling multiple possible data sources"""
        underlying_data = self.fetch_underlying_snapshot(ticker)
        if not underlying_data:
            return None
            
        return self.extract_spot_price(underlying_data)
        
    def extract_spot_price(self, ticker_data: Dict[str, Any]) -> Optional[float]:
        """Extract spot price from V2 snapshot ticker data using a cleaner approach"""
        if not ticker_data or not isinstance(ticker_data, dict):
            return None
            
        # Define price extractors in order of preference
        extractors = [
            lambda d: d.get('lastTrade', {}).get('p'),
            lambda d: d.get('day', {}).get('c'),
            lambda d: d.get('prevDay', {}).get('c'),
            lambda d: d.get('lastQuote', {}).get('p'),
            lambda d: d.get('lastQuote', {}).get('P')
        ]
        
        # Try each extractor in sequence
        for extract in extractors:
            try:
                price = extract(ticker_data)
                if price is not None:
                    return float(price)
            except (TypeError, ValueError):
                continue
                
        logger.warning("Could not determine spot price from V2 underlying snapshot data.")
        return None
        
    def process_options_data(self, options_data: List[Dict[str, Any]], analysis_date: str) -> pd.DataFrame:
        """Process raw options data into a clean DataFrame"""
        if not options_data:
            logger.warning("Received empty list for options preprocessing.")
            return pd.DataFrame()
            
        logger.info(f"Processing {len(options_data)} options contracts...")
        
        valid_contracts = []
        for contract_data in options_data:
            contract = OptionsContract.from_api_response(contract_data)
            if contract:
                valid_contracts.append({
                    'strike': contract.strike_price,
                    'expiration': pd.to_datetime(contract.expiration_date),
                    'type': contract.contract_type,
                    'openInterest': contract.open_interest,
                    'impliedVolatility': contract.implied_volatility,
                    'delta': contract.delta,
                    'gamma': contract.gamma,
                    'theta': contract.theta,
                    'vega': contract.vega
                })
                
        if not valid_contracts:
            logger.warning("No valid option contract records created.")
            return pd.DataFrame()
            
        options_df = pd.DataFrame(valid_contracts)
        
        # Add days to expiration column
        options_df['DTE'] = (options_df['expiration'] - pd.Timestamp(analysis_date)).dt.days
        
        # Remove any rows with missing essential data
        essential_numeric = ['strike', 'openInterest', 'impliedVolatility', 'delta', 'gamma']
        options_df.dropna(subset=essential_numeric, inplace=True)
        
        logger.info(f"Processing complete. Valid: {len(options_df)}. Skipped: {len(options_data) - len(options_df)}.")
        return options_df


class HistoricalDataManager:
    """Class for managing historical data operations"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
    def find_latest_overview_file(self, ticker: str) -> Optional[str]:
        """Find the most recent options overview file for a ticker"""
        if not os.path.isdir(self.data_dir):
            logger.warning(f"Data dir not found: {self.data_dir}")
            return None
            
        overview_hist_pattern = os.path.join(self.data_dir, f"{ticker.lower()}_options-overview-history-*.csv")
        overview_hist_files = sorted(glob.glob(overview_hist_pattern), reverse=True)
        
        if not overview_hist_files:
            logger.warning(f"No overview file found for {ticker} in {self.data_dir}")
            return None
            
        latest_file = overview_hist_files[0]
        logger.debug(f"Found overview file: {os.path.basename(latest_file)}")
        return latest_file
        
    def load_historical_iv(self, ticker: str) -> pd.DataFrame:
        """Load historical implied volatility data for a ticker"""
        overview_file = self.find_latest_overview_file(ticker)
        return self.load_historical_iv_from_file(overview_file)
        
    def load_historical_iv_from_file(self, options_overview_file: Optional[str]) -> pd.DataFrame:
        """Load and clean historical implied volatility data from a file"""
        historical_iv_data = pd.DataFrame(columns=['Date', 'Imp Vol'])  # Default empty
        
        if not options_overview_file or not os.path.exists(options_overview_file):
            logger.warning(f"Options overview file not found or specified: {options_overview_file}")
            return historical_iv_data
            
        try:
            logger.info(f"Loading historical IV from {os.path.basename(options_overview_file)}")
            hist_opts = pd.read_csv(options_overview_file, skipfooter=1, engine="python", thousands=',')
            
            if hist_opts.empty:
                logger.warning("Overview file is empty.")
                return historical_iv_data
                
            # Map columns with different possible names
            col_map = {
                'Date': ['Date', 'Time', 'date', 'time'],
                'Imp Vol': ['Imp Vol', 'IV', 'Implied Volatility', 'impliedVolatility']
            }
            
            rename_dict = {}
            final_cols = {}
            found_cols = set(hist_opts.columns)
            
            # Find and map column names
            for target, possibles in col_map.items():
                found = False
                
                for poss in possibles:
                    # Direct match
                    if poss in found_cols:
                        if poss != target:
                            rename_dict[poss] = target
                            final_cols[target] = target
                            found = True
                            break
                    
                    # Case-insensitive match
                    poss_lower = poss.lower()
                    current_lower = {c.lower(): c for c in found_cols}
                    if poss_lower in current_lower:
                        original_col = current_lower[poss_lower]
                        if original_col != target:
                            rename_dict[original_col] = target
                            final_cols[target] = target
                            found = True
                            break
                
                if not found and target in ['Date', 'Imp Vol']:
                    logger.error(f"Essential '{target}' column not found in {options_overview_file}")
                    return historical_iv_data
            
            # Rename columns to standardized names
            hist_opts.rename(columns=rename_dict, inplace=True)
            
            # Process date and IV columns
            date_col = final_cols.get('Date', 'Date')
            iv_col = final_cols.get('Imp Vol', 'Imp Vol')
            
            hist_opts[date_col] = pd.to_datetime(hist_opts[date_col], errors='coerce')
            
            # Clean IV values
            if hist_opts[iv_col].dtype == 'object':
                hist_opts[iv_col] = hist_opts[iv_col].astype(str).str.replace('%', '', regex=False).str.strip()
            
            hist_opts[iv_col] = pd.to_numeric(hist_opts[iv_col], errors='coerce')
            
            # Convert percentage to decimal if needed
            if not hist_opts[iv_col].empty and pd.notna(hist_opts[iv_col].max()) and hist_opts[iv_col].max(skipna=True) > 1.5:
                hist_opts[iv_col] = hist_opts[iv_col] / 100.0
            
            # Remove rows with missing values
            hist_opts.dropna(subset=[date_col, iv_col], inplace=True)
            
            # Create final DataFrame
            historical_iv_data = hist_opts[['Date', 'Imp Vol']].copy()
            
            if not historical_iv_data.empty:
                historical_iv_data.sort_values('Date', inplace=True)
                logger.info(f"Loaded {len(historical_iv_data)} rows of historical IV.")
            else:
                logger.warning("No valid historical IV rows after cleaning.")
                
        except Exception as e:
            logger.error(f"Error loading historical IV: {e}", exc_info=True)
            
        return historical_iv_data


class OptionsAnalyzer:
    """Class for analyzing options data"""
    
    def __init__(self, polygon_client: PolygonClient, hist_data_manager: HistoricalDataManager):
        self.polygon = polygon_client
        self.hist_data = hist_data_manager
        
    def fetch_and_process_options(self, ticker: str, analysis_date: str = None) -> Tuple[pd.DataFrame, Optional[float], pd.DataFrame]:
        """Fetch and process all options data for a ticker"""
        if analysis_date is None:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Analyzing options for {ticker} as of {analysis_date}")
        
        # Fetch data
        options_data = self.polygon.fetch_options_chain_snapshot(ticker)
        spot_price = self.polygon.get_spot_price(ticker)
        historical_iv = self.hist_data.load_historical_iv(ticker)
        
        # Process options data
        options_df = self.polygon.process_options_data(options_data, analysis_date)
        
        if options_df.empty:
            logger.warning(f"No valid options data found for {ticker}")
            
        if spot_price is None:
            logger.warning(f"Could not determine spot price for {ticker}")
            
        return options_df, spot_price, historical_iv
        
    def calculate_metrics(self, options_df: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
        """Calculate key metrics from options data"""
        if options_df.empty or spot_price is None:
            return {}
            
        try:
            # Filtering for relevant options
            near_term = options_df[options_df['DTE'].between(10, 60)]
            atm_options = near_term[near_term['strike'].between(spot_price * 0.9, spot_price * 1.1)]
            
            if atm_options.empty:
                logger.warning("No suitable ATM options found for metrics calculation")
                return {}
                
            # Calculate metrics
            metrics = {
                'avg_iv': atm_options['impliedVolatility'].mean(),
                'median_iv': atm_options['impliedVolatility'].median(),
                'weighted_iv': (atm_options['impliedVolatility'] * atm_options['openInterest']).sum() / atm_options['openInterest'].sum(),
                'call_put_ratio': len(atm_options[atm_options['type'] == 'call']) / max(1, len(atm_options[atm_options['type'] == 'put'])),
                'total_open_interest': options_df['openInterest'].sum(),
                'call_oi': options_df[options_df['type'] == 'call']['openInterest'].sum(),
                'put_oi': options_df[options_df['type'] == 'put']['openInterest'].sum()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}", exc_info=True)
            return {}


# Factory function to create the full set of components
def create_options_analyzer(api_key: str, data_dir: str) -> OptionsAnalyzer:
    """Create and configure an OptionsAnalyzer with all dependencies"""
    polygon_client = PolygonClient(api_key)
    hist_data_manager = HistoricalDataManager(data_dir)
    return OptionsAnalyzer(polygon_client, hist_data_manager)


# Convenience function to run a full analysis
def analyze_ticker(ticker: str, api_key: str, data_dir: str) -> Dict[str, Any]:
    """Run a complete analysis for a ticker and return all relevant data"""
    analyzer = create_options_analyzer(api_key, data_dir)
    options_df, spot_price, historical_iv = analyzer.fetch_and_process_options(ticker)
    
    results = {
        'ticker': ticker,
        'spot_price': spot_price,
        'analysis_date': datetime.now().strftime('%Y-%m-%d'),
        'options_count': len(options_df),
        'has_historical_iv': not historical_iv.empty
    }
    
    if not options_df.empty and spot_price is not None:
        metrics = analyzer.calculate_metrics(options_df, spot_price)
        results.update(metrics)
    
    return results


# Backwards compatibility functions
def fetch_options_chain_snapshot(ticker, api_key):
    """Legacy function for compatibility with old code"""
    client = PolygonClient(api_key)
    result = client.fetch_options_chain_snapshot(ticker)
    
    # Add special debug logging for problematic tickers
    if ticker in ["TSLA", "ZM"] and result:
        logger.info(f"DEBUG: {ticker} options data received, count: {len(result)}")
        if len(result) > 0:
            # Print structure of first contract
            logger.info(f"DEBUG: First contract structure: {list(result[0].keys())}")
            # Check for required sections
            has_details = 'details' in result[0]
            has_greeks = 'greeks' in result[0]
            has_day = 'day' in result[0]
            logger.info(f"DEBUG: Has required sections? details: {has_details}, greeks: {has_greeks}, day: {has_day}")
            
            # If details exist, check its structure
            if has_details:
                logger.info(f"DEBUG: details keys: {list(result[0]['details'].keys())}")
            
            # If greeks exist, check its structure
            if has_greeks:
                logger.info(f"DEBUG: greeks keys: {list(result[0]['greeks'].keys())}")
    
    return result
    
def fetch_underlying_snapshot(ticker, api_key):
    """Legacy function for compatibility with old code"""
    client = PolygonClient(api_key)
    return client.fetch_underlying_snapshot(ticker)
    
def get_spot_price_from_snapshot(underlying_ticker_data):
    """Legacy function for compatibility with old code"""
    # Input is now V2 'ticker' block
    client = PolygonClient("dummy")  # API key not needed for this function
    return client.extract_spot_price(underlying_ticker_data)
    
def preprocess_api_options_data(api_results_list, analysis_date):
    """Legacy function for compatibility with old code"""
    client = PolygonClient("dummy")  # API key not needed for this function
    return client.process_options_data(api_results_list, analysis_date)

# Stand-alone function for loading historical IV data - needed by pattern_analyzer.py
def load_historical_iv_from_file(options_overview_file):
    """
    Load and clean historical implied volatility data from a file
    
    Args:
        options_overview_file (str): Path to options overview CSV file
        
    Returns:
        pd.DataFrame: DataFrame with Date and Imp Vol columns
    """
    import pandas as pd
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    historical_iv_data = pd.DataFrame(columns=['Date', 'Imp Vol'])  # Default empty
    
    if not options_overview_file or not os.path.exists(options_overview_file):
        logger.warning(f"Options overview file not found or specified: {options_overview_file}")
        return historical_iv_data
        
    try:
        logger.info(f"Loading historical IV from {os.path.basename(options_overview_file)}")
        hist_opts = pd.read_csv(options_overview_file, skipfooter=1, engine="python", thousands=',')
        
        if hist_opts.empty:
            logger.warning("Overview file is empty.")
            return historical_iv_data
            
        # Map columns with different possible names
        col_map = {
            'Date': ['Date', 'Time', 'date', 'time'],
            'Imp Vol': ['Imp Vol', 'IV', 'Implied Volatility', 'impliedVolatility']
        }
        
        rename_dict = {}
        final_cols = {}
        found_cols = set(hist_opts.columns)
        
        # Find and map column names
        for target, possibles in col_map.items():
            found = False
            
            for poss in possibles:
                # Direct match
                if poss in found_cols:
                    if poss != target:
                        rename_dict[poss] = target
                        final_cols[target] = target
                        found = True
                        break
                
                # Case-insensitive match
                poss_lower = poss.lower()
                current_lower = {c.lower(): c for c in found_cols}
                if poss_lower in current_lower:
                    original_col = current_lower[poss_lower]
                    if original_col != target:
                        rename_dict[original_col] = target
                        final_cols[target] = target
                        found = True
                        break
            
            if not found and target in ['Date', 'Imp Vol']:
                logger.error(f"Essential '{target}' column not found in {options_overview_file}")
                return historical_iv_data
        
        # Rename columns to standardized names
        hist_opts.rename(columns=rename_dict, inplace=True)
        
        # Process date and IV columns
        date_col = final_cols.get('Date', 'Date')
        iv_col = final_cols.get('Imp Vol', 'Imp Vol')
        
        hist_opts[date_col] = pd.to_datetime(hist_opts[date_col], errors='coerce')
        
        # Clean IV values
        if hist_opts[iv_col].dtype == 'object':
            hist_opts[iv_col] = hist_opts[iv_col].astype(str).str.replace('%', '', regex=False).str.strip()
        
        hist_opts[iv_col] = pd.to_numeric(hist_opts[iv_col], errors='coerce')
        
        # Convert percentage to decimal if needed
        if not hist_opts[iv_col].empty and pd.notna(hist_opts[iv_col].max()) and hist_opts[iv_col].max(skipna=True) > 1.5:
            hist_opts[iv_col] = hist_opts[iv_col] / 100.0
        
        # Remove rows with missing values
        hist_opts.dropna(subset=[date_col, iv_col], inplace=True)
        
        # Create final DataFrame
        historical_iv_data = hist_opts[['Date', 'Imp Vol']].copy()
        
        if not historical_iv_data.empty:
            historical_iv_data.sort_values('Date', inplace=True)
            logger.info(f"Loaded {len(historical_iv_data)} rows of historical IV.")
        else:
            logger.warning("No valid historical IV rows after cleaning.")
            
    except Exception as e:
        logger.error(f"Error loading historical IV: {e}", exc_info=True)
        
    return historical_iv_data

# Stand-alone function for finding latest overview file - needed by pattern_analyzer.py
def find_latest_overview_file(symbol, data_dir="data/company"):
    """
    Find the latest company overview file for the given symbol.
    
    Args:
        symbol (str): Ticker symbol
        data_dir (str): Directory containing company data files
        
    Returns:
        str: Path to the latest overview file or None if not found
    """
    import os
    import glob
    
    # Ensure directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Find all overview files for this symbol
    pattern = os.path.join(data_dir, f"{symbol.lower()}_overview_*.json")
    alt_pattern = os.path.join(data_dir, f"{symbol.upper()}_overview_*.json")
    
    files = glob.glob(pattern)
    if not files:
        files = glob.glob(alt_pattern)
        
    if not files:
        logger.warning(f"No overview files found for {symbol} in {data_dir}")
        return None
    
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    
    # Return the newest file
    latest_file = files[0]
    logger.debug(f"Found overview file: {os.path.basename(latest_file)} for {symbol}")
    return latest_file

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example with environment variables
    api_key = os.environ.get("POLYGON_API_KEY")
    data_dir = os.environ.get("DATA_DIR", "./data")
    ticker = "AAPL"
    
    if api_key:
        results = analyze_ticker(ticker, api_key, data_dir)
        print(f"Analysis Results for {ticker}:")
        for k, v in results.items():
            print(f"  {k}: {v}")
    else:
        print("POLYGON_API_KEY environment variable not set")


