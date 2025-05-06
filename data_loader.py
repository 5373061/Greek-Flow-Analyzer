# data_loader.py (API Price, File Options, TZ Fix, Corrected Date Merge Logic, Correct Pandas dtype check)
import pandas as pd
import numpy as np
import os
import re
import logging
import requests # Needed for API calls
import time
from datetime import date, timedelta, datetime

# Import config here to access API key, paths, etc.
try:
    import config
except ImportError:
    logging.error("config.py not found. Please ensure it's in the same directory or accessible.")
    class MockConfig: # Define fallback config
        DATA_DIR = "."
        POLYGON_API_KEY = None
        PRICE_HISTORY_YEARS = 3
    config = MockConfig()
    if not hasattr(config, 'POLYGON_API_KEY') or not config.POLYGON_API_KEY:
         raise ImportError("CRITICAL ERROR: config.py not found or POLYGON_API_KEY missing/empty in config.")

# --- Helper Functions ---

def _detect_date_col(df: pd.DataFrame) -> str | None:
    """Helper to find date/time column (prioritizes 'Date' for options)."""
    if df is None or df.empty: return None
    df_columns_lower = [str(c).lower() for c in df.columns]
    # Check specifically for 'Date' first, common in daily options files
    if 'date' in df_columns_lower:
        date_col_index = df_columns_lower.index('date')
        # Check if it's the *only* date-like column, otherwise might be ambiguous
        other_date_cols = [c for c in df_columns_lower if any(k in c for k in ['datetime', 'time']) and c != 'date']
        if not other_date_cols:
            return df.columns[date_col_index]

    # If 'Date' wasn't found or was ambiguous, check other common names
    potential_cols = ["datetime", "time"] # Check these AFTER checking 'Date'
    for col_name in potential_cols:
        if col_name in df_columns_lower:
             date_col_index = df_columns_lower.index(col_name)
             return df.columns[date_col_index]

    # Fallback if only standard names found ('Date' already checked)
    if 'date' in df_columns_lower:
        date_col_index = df_columns_lower.index('date')
        return df.columns[date_col_index]


    logging.warning(f"Could not auto-detect date/time column in columns: {df.columns.tolist()}")
    return None

def _fetch_polygon_price_data(symbol: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame | None:
    """Fetches 165-minute aggregate bars from Polygon API."""
    multiplier = 165
    timespan = "minute"
    base_url = "https://api.polygon.io/v2/aggs/ticker"
    safe_symbol = requests.utils.quote(symbol)
    url = f"{base_url}/{safe_symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}
    all_results = []
    request_count = 0
    current_url = url
    headers = {"Authorization": f"Bearer {api_key}"}

    logging.info(f"Fetching Polygon data: {symbol} {multiplier}/{timespan} from {start_date} to {end_date}")

    try:
        while current_url:
            # Add API key to headers for subsequent paginated requests if needed
            current_headers = headers.copy()
            if request_count > 0 and 'apiKey' in current_url: # Polygon might include apiKey in next_url
                logging.debug("API key detected in next_url, relying on that.")
            elif request_count > 0: # Ensure key is always present
                logging.debug("Adding auth header to paginated request.")
            else: # First request uses params
                 current_headers = headers # Use original headers for first request

            current_params = params if request_count == 0 else {} # Use original params only for the first request

            logging.debug(f"  Fetching page {request_count + 1}...")
            response = requests.get(current_url, params=current_params, headers=current_headers)
            request_count += 1

            if response.status_code == 429:
                 logging.warning(f"Rate Limit Exceeded (429) fetching page {request_count} for {symbol}. Waiting 60s.")
                 time.sleep(60)
                 logging.info(f"Retrying page {request_count} for {symbol} after rate limit wait...")
                 response = requests.get(current_url, params=current_params, headers=current_headers)
                 if response.status_code != 200:
                      logging.error(f"API Error {response.status_code} on retry fetching {symbol}. Aborting.")
                      return None
            elif response.status_code != 200:
                logging.error(f"API Error {response.status_code} fetching page {request_count} for {symbol}: {response.text[:500]}")
                # If it's a permission issue, no point retrying
                if response.status_code in [401, 403]:
                     logging.critical("API Key Unauthorized/Forbidden. Check Polygon subscription/key.")
                     return None
                # Optionally add more specific error handling or retries here
                return None

            try: data = response.json()
            except requests.exceptions.JSONDecodeError:
                logging.error(f"Failed to decode JSON response page {request_count}, symbol {symbol}.")
                return None

            results = data.get('results', [])
            if results: all_results.extend(results)
            logging.debug(f"  Status: {data.get('status')}, Results on page: {len(results)}, Total: {len(all_results)}")

            current_url = data.get('next_url')
            # IMPORTANT: Append API key to next_url if it's not already there for subsequent requests
            if current_url and 'apiKey=' not in current_url:
                 api_key_param = f"&apiKey={api_key}"
                 current_url += api_key_param
                 logging.debug("  Appended API key to next_url for pagination.")

            if request_count > 150: # Safety break
                 logging.warning(f"Exceeded 150 pages fetching {symbol}, stopping pagination.")
                 break

        if not all_results:
            logging.warning(f"No price data returned from Polygon API for {symbol} in range {start_date} - {end_date}.")
            return pd.DataFrame()

        df_price = pd.DataFrame(all_results)
        rename_dict = {'t': 'DateTime', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}
        df_price = df_price.rename(columns=rename_dict)
        try:
            # Convert epoch ms to datetime, make UTC explicit, then convert to desired TZ
            df_price['DateTime'] = pd.to_datetime(df_price['DateTime'], unit='ms', utc=True).dt.tz_convert('America/New_York')
        except Exception as tz_e:
            logging.warning(f"Could not convert timezone to America/New_York for {symbol}, keeping UTC: {tz_e}")
            df_price['DateTime'] = pd.to_datetime(df_price['DateTime'], unit='ms', utc=True)

        required_cols = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df_price.columns]
        if missing_cols:
            logging.error(f"API response for {symbol} missing required columns after rename: {missing_cols}")
            return None

        df_price = df_price[required_cols]
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df_price[col] = pd.to_numeric(df_price[col], errors='coerce')

        df_price.sort_values("DateTime", inplace=True)
        df_price.dropna(subset=['DateTime', 'Open', 'High', 'Low', 'Close'], inplace=True) # Keep Volume NaNs for now if any
        df_price.drop_duplicates(subset=['DateTime'], keep='last', inplace=True)
        df_price.reset_index(drop=True, inplace=True)

        logging.info(f"Successfully fetched and processed {len(df_price)} price bars for {symbol} from Polygon.")
        return df_price

    except requests.exceptions.RequestException as e:
        logging.error(f"Network error fetching {symbol} from Polygon: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"Unexpected error fetching or processing Polygon data for {symbol}: {e}", exc_info=True)
        return None


def _find_options_file(symbol: str) -> str | None:
    """Finds the path for the options file for a given symbol."""
    symbol_lower = symbol.lower()
    opts_path = None
    opts_pattern = f"{symbol_lower}_options"

    try:
        data_dir = getattr(config, 'DATA_DIR', None)
        if not data_dir or not os.path.isdir(data_dir):
             logging.error(f"DATA_DIR '{data_dir}' is not configured correctly or doesn't exist.")
             return None

        # Prioritize exact match _options-overview-history-full.csv
        exact_full_pattern = f"^{symbol_lower}_options-overview-history-full\\.csv$"
        opts_files_exact_full = [f for f in os.listdir(data_dir) if re.match(exact_full_pattern, f, re.IGNORECASE)]

        # Fallback: Starts with symbol_options*.csv
        start_pattern = f"^{symbol_lower}_options.*\\.csv$"
        opts_files_start = [f for f in os.listdir(data_dir) if re.match(start_pattern, f, re.IGNORECASE)]

        # Fallback: Contains symbol_options*.csv
        contain_pattern = f"{symbol_lower}_options.*\\.csv$"
        opts_files_contain = [f for f in os.listdir(data_dir) if re.search(contain_pattern, f, re.IGNORECASE)]


        chosen_file = None
        if opts_files_exact_full:
             if len(opts_files_exact_full) > 1: logging.warning(f"Multiple options files matching exact full pattern found for {symbol}, using first: {opts_files_exact_full[0]}")
             chosen_file = opts_files_exact_full[0]
        elif opts_files_start:
             if len(opts_files_start) > 1: logging.warning(f"Multiple options files STARTING with pattern found for {symbol} (no exact full match), using first: {opts_files_start[0]}")
             chosen_file = opts_files_start[0]
        elif opts_files_contain:
             if len(opts_files_contain) > 1: logging.warning(f"Multiple options files CONTAINING pattern found for {symbol} (no exact start/full match), using first: {opts_files_contain[0]}")
             chosen_file = opts_files_contain[0]


        if chosen_file:
            opts_path = os.path.join(data_dir, chosen_file)
            logging.info(f"Found options file for {symbol}: {opts_path}")
            return opts_path
        else:
             logging.warning(f"No options file found for {symbol} matching pattern '{opts_pattern}' in {data_dir}")
             return None

    except Exception as e:
         logging.error(f"Error searching for {symbol} options file in {config.DATA_DIR}: {e}", exc_info=True)
         return None


# --- Main Loader Function (Refactored with Correct Date Merge & Pandas dtype check) ---

def load_intraday_data(symbol: str, config) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Loads price data from Polygon API and options data from local CSV. Merges them.
    Corrects the date standardization logic for merging and uses correct pandas dtype check.
    """
    price_df = None
    opts_df = None
    merged_df = None

    # 1. Fetch Price Data from API
    try:
        end_date = date.today()
        history_years = getattr(config, 'PRICE_HISTORY_YEARS', 3)
        start_date = end_date - timedelta(days=history_years * 365)
        api_key = getattr(config, 'POLYGON_API_KEY', None)
        if not api_key: raise ValueError("POLYGON_API_KEY not found in config.")

        price_df = _fetch_polygon_price_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), api_key)

        if price_df is None:
             logging.error(f"Polygon price fetch returned None for {symbol}.")
             return None, None
        elif price_df.empty:
             logging.warning(f"Polygon price fetch returned empty DataFrame for {symbol}.")
             # Still return empty DF for options if needed later
             return pd.DataFrame(), pd.DataFrame() # Return empty DFs

        # *** CORRECTED DATE STANDARDIZATION FOR PRICE DATA & PANDAS DTYPE CHECK ***
        # Use is_datetime64tz_dtype instead of is_datetime64_tz_dtype
        if pd.api.types.is_datetime64tz_dtype(price_df['DateTime']):
            logging.debug("Standardizing price DateTime to naive UTC date for merging.")
            # Convert to UTC -> Normalize (sets time to 00:00:00 UTC) -> Remove TZ
            price_df['Date'] = price_df['DateTime'].dt.tz_convert('UTC').dt.normalize().dt.tz_localize(None)
        else:
            # If already naive (e.g., from file load), just normalize
            logging.debug("Standardizing price DateTime (naive) to naive UTC date for merging.")
            price_df['Date'] = price_df['DateTime'].dt.normalize()
        # Ensure final type just in case
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        logging.debug(f"Price df 'Date' column dtype after standardization: {price_df['Date'].dtype}")
        # *** END CORRECTION ***

    except Exception as e_price:
        logging.error(f"Error in price data fetching stage for {symbol}: {e_price}", exc_info=True)
        # Ensure we return None, None or appropriate error indication
        # If price_df exists but failed during date processing, return None
        return None, None


    # 2. Load Options Data from File
    opts_df = pd.DataFrame()
    opts_path = _find_options_file(symbol)
    if not opts_path:
        logging.warning(f"No options file found for {symbol}, proceeding without options merge.")
    else:
        try:
            logging.debug(f"Loading options data from: {opts_path}")
            # Attempt to read header to find date column robustly
            try:
                header_df = pd.read_csv(opts_path, nrows=1, comment='#', engine='python')
                opts_date_col = _detect_date_col(header_df)
                if not opts_date_col:
                    raise ValueError(f"Date column could not be auto-detected in options header: {opts_path}")
                logging.info(f"Detected options date column: '{opts_date_col}'")
            except Exception as header_e:
                 logging.error(f"Error detecting header/date column in {opts_path}: {header_e}. Attempting load anyway.")
                 # If header detection fails, proceed but might error later if col is wrong
                 opts_date_col = 'Date' # Default guess


            opts_df = pd.read_csv(opts_path, engine="python", comment='#', header=0)
            # Rename detected/guessed date column to 'Date'
            if opts_date_col != 'Date' and opts_date_col in opts_df.columns:
                 opts_df.rename(columns={opts_date_col: "Date"}, inplace=True)
            elif 'Date' not in opts_df.columns and opts_date_col not in opts_df.columns:
                 # If neither 'Date' nor the detected column exists after loading... error
                 raise ValueError(f"Date column ('{opts_date_col}' or 'Date') not found after loading options data: {opts_path}")

            # Standardize options date: To datetime -> Normalize -> Ensure Naive
            opts_df["Date"] = pd.to_datetime(opts_df["Date"], errors='coerce').dt.normalize()
            opts_df.dropna(subset=["Date"], inplace=True) # Drop rows where date parsing failed
            # Ensure timezone-naive using the correct pandas function name
            if pd.api.types.is_datetime64tz_dtype(opts_df["Date"]): # <-- CORRECTED HERE
                logging.debug(f"Removing timezone info from opts_df 'Date' column for merging.")
                opts_df["Date"] = opts_df["Date"].dt.tz_localize(None)
            opts_df["Date"] = pd.to_datetime(opts_df["Date"]) # Ensure final type
            logging.debug(f"Options df 'Date' column dtype after standardization: {opts_df['Date'].dtype}")

            # Process required options columns
            required_opts_cols = ["Imp Vol", "IV Pctl"] # Use exact names expected later
            standard_cols = {} # Store mapping if names differ
            missing_opts_str = []

            for req_col in required_opts_cols:
                found_col = None
                # Check exact match first
                if req_col in opts_df.columns:
                    found_col = req_col
                else:
                    # Check case-insensitive variations
                    for col in opts_df.columns:
                        if col.lower() == req_col.lower():
                            found_col = col
                            break
                if found_col:
                    standard_cols[req_col] = found_col
                    if req_col != found_col:
                         logging.info(f"Found options column '{found_col}' for required '{req_col}'.")
                else:
                    missing_opts_str.append(req_col)

            if missing_opts_str:
                 logging.warning(f"Missing required options columns in {opts_path}: Need: {', '.join(missing_opts_str)}. Some features may be NaN.")
                 for col in missing_opts_str:
                      opts_df[col] = np.nan # Create missing columns with NaNs

            # Standardize and clean the columns that were found
            if "Imp Vol" in standard_cols:
                original_col = standard_cols["Imp Vol"]
                opts_df["Imp Vol"] = pd.to_numeric(opts_df[original_col].astype(str).str.rstrip("%").replace('', np.nan), errors='coerce') / 100
                if original_col != "Imp Vol": opts_df.drop(columns=[original_col], inplace=True) # Drop original if renamed
            elif "Imp Vol" not in opts_df.columns: # Ensure column exists if totally missing
                 opts_df["Imp Vol"] = np.nan


            if "IV Pctl" in standard_cols:
                original_col = standard_cols["IV Pctl"]
                # Create the standardized 'IV_Pctl' column
                opts_df["IV_Pctl"] = pd.to_numeric(opts_df[original_col].astype(str).str.rstrip("%").replace('', np.nan), errors='coerce') / 100
                if original_col != "IV Pctl": opts_df.drop(columns=[original_col], inplace=True) # Drop original if renamed
            elif "IV_Pctl" not in opts_df.columns: # Ensure column exists if totally missing
                 opts_df["IV_Pctl"] = np.nan

            # Optional columns
            optional_cols = {"Total OI": "Total OI", "P/C OI": "PC_Ratio"}
            for opt_col, target_col in optional_cols.items():
                 found_opt_col = None
                 if opt_col in opts_df.columns:
                      found_opt_col = opt_col
                 else:
                     for col in opts_df.columns:
                          if col.lower() == opt_col.lower():
                              found_opt_col = col
                              break
                 if found_opt_col:
                    opts_df[target_col] = pd.to_numeric(opts_df[found_opt_col], errors="coerce")
                    if found_opt_col != target_col: opts_df.drop(columns=[found_opt_col], inplace=True)
                 elif target_col not in opts_df.columns:
                      opts_df[target_col] = np.nan # Ensure column exists even if not found


            opts_df.sort_values("Date", inplace=True)
            opts_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
            opts_df.reset_index(drop=True, inplace=True)
            logging.info(f"Loaded and processed {len(opts_df)} options rows for {symbol}.")

            if not opts_df.empty:
                logging.info(f"DIAG ({symbol}): Options Date Range: {opts_df['Date'].min().date()} to {opts_df['Date'].max().date()}")
                logging.info(f"DIAG ({symbol}): NaNs in opts_df['Imp Vol'] head(10): {opts_df['Imp Vol'].head(10).isnull().sum()}")
                logging.info(f"DIAG ({symbol}): NaNs in opts_df['IV_Pctl'] head(10): {opts_df['IV_Pctl'].head(10).isnull().sum()}")

        except Exception as e_opts:
            logging.error(f"Error loading/processing options data for {symbol} from {opts_path}: {e_opts}", exc_info=True)
            opts_df = pd.DataFrame() # Ensure it's an empty DF on error
            logging.warning("Proceeding without options data due to loading error.")


    # 3. Merge Data
    try:
        # Check if price_df is valid before proceeding with merge logic
        if price_df is None:
             logging.error(f"Price data loading failed earlier for {symbol}, cannot proceed to merge.")
             return None, opts_df # Return None for merged, return opts_df if it loaded

        logging.debug(f"Attempting merge on 'Date' columns. Price date range: {price_df['Date'].min().date() if not price_df.empty else 'N/A'} to {price_df['Date'].max().date() if not price_df.empty else 'N/A'}")
        logging.debug(f"Price dtype: {price_df['Date'].dtype}, Options dtype: {opts_df['Date'].dtype if not opts_df.empty else 'N/A'}")

        if not opts_df.empty:
            # Define columns expected from options df after processing
            opts_cols_to_merge = ["Date", "Imp Vol", "IV_Pctl"]
            optional_cols_processed = ["Total OI", "PC_Ratio"] # Use standardized names
            for col in optional_cols_processed:
                if col in opts_df.columns: opts_cols_to_merge.append(col)

            # Filter to only columns that *actually exist* in the processed opts_df
            opts_cols_to_merge = [col for col in opts_cols_to_merge if col in opts_df.columns]

            if 'Date' not in opts_cols_to_merge:
                 logging.error(f"CRITICAL: 'Date' column missing from options data columns to merge for {symbol}. Cannot merge options.")
                 merged_df = price_df.copy()
                 # Ensure the essential options columns exist, even if NaN
                 if "Imp Vol" not in merged_df.columns: merged_df["Imp Vol"] = np.nan
                 if "IV_Pctl" not in merged_df.columns: merged_df["IV_Pctl"] = np.nan
            else:
                logging.debug(f"Merging price data ({len(price_df)} rows) with options ({len(opts_df)} rows) using columns: {opts_cols_to_merge}")
                merged_df = pd.merge(price_df, opts_df[opts_cols_to_merge], on="Date", how="left")
                # === DIAGNOSTIC AFTER MERGE ===
                logging.info(f"DIAG ({symbol}): Shape AFTER MERGE: {merged_df.shape}")
                # Check NaNs *after* merge, *before* ffill
                nan_imp_vol_after_merge = merged_df['Imp Vol'].isnull().sum() if 'Imp Vol' in merged_df.columns else 'N/A'
                nan_iv_pctl_after_merge = merged_df['IV_Pctl'].isnull().sum() if 'IV_Pctl' in merged_df.columns else 'N/A'
                logging.info(f"DIAG ({symbol}): NaN count Imp Vol AFTER MERGE: {nan_imp_vol_after_merge}")
                logging.info(f"DIAG ({symbol}): NaN count IV_Pctl AFTER MERGE: {nan_iv_pctl_after_merge}")
                # ==============================

                # Forward fill options data to intraday bars
                ffill_cols = [col for col in opts_cols_to_merge if col != 'Date']
                # Check if columns actually exist before trying to fill
                ffill_cols_exist = [col for col in ffill_cols if col in merged_df.columns]
                if ffill_cols_exist:
                     merged_df[ffill_cols_exist] = merged_df[ffill_cols_exist].ffill()
                     logging.info(f"Forward filled options data for columns: {ffill_cols_exist}")
                     # === DIAGNOSTIC AFTER FFILL ===
                     nan_imp_vol_after_ffill = merged_df['Imp Vol'].isnull().sum() if 'Imp Vol' in merged_df.columns else 'N/A'
                     nan_iv_pctl_after_ffill = merged_df['IV_Pctl'].isnull().sum() if 'IV_Pctl' in merged_df.columns else 'N/A'
                     logging.info(f"DIAG ({symbol}): Shape after ffill: {merged_df.shape}. NaN count Imp Vol: {nan_imp_vol_after_ffill}, IV_Pctl: {nan_iv_pctl_after_ffill}")
                     # ===============================
                else:
                     logging.warning(f"No columns found to forward fill for {symbol}.")

                logging.info(f"Price and Options data merged and ffilled for {symbol}. Final shape: {merged_df.shape}")

        else: # If opts_df was empty
            logging.warning(f"No options data loaded for {symbol}. Proceeding with price data only.")
            merged_df = price_df.copy()
            # Ensure required columns exist, even if all NaN
            if "Imp Vol" not in merged_df.columns: merged_df["Imp Vol"] = np.nan
            if "IV_Pctl" not in merged_df.columns: merged_df["IV_Pctl"] = np.nan


        if merged_df is None or merged_df.empty:
            if price_df is not None and not price_df.empty:
                 logging.error(f"Merged DataFrame became empty for {symbol} after processing. Check merge keys, options data quality, or ffill step.")
            else:
                 # Price data failure already logged, this is redundant
                 # logging.error(f"Merged DataFrame is empty for {symbol} because price data failed to load or was empty.")
                 pass # Error logged earlier
            return None, opts_df # Return None for merged, but return original opts_df if loaded

        logging.info(f"Data loaded and merged successfully for {symbol}. Final row count: {len(merged_df)}")
        # Add final check for NaNs in critical options columns after all processing
        final_nan_imp_vol = merged_df['Imp Vol'].isnull().sum() if 'Imp Vol' in merged_df.columns else len(merged_df)
        final_nan_iv_pctl = merged_df['IV_Pctl'].isnull().sum() if 'IV_Pctl' in merged_df.columns else len(merged_df)
        if final_nan_imp_vol > 0 or final_nan_iv_pctl > 0:
            logging.warning(f"NaNs remain in critical options columns for {symbol} after load/merge/ffill. Imp Vol NaNs: {final_nan_imp_vol}, IV_Pctl NaNs: {final_nan_iv_pctl}. This might affect subsequent calculations.")

        return merged_df, opts_df # Return merged_df and the original loaded opts_df

    except Exception as e_merge:
        logging.error(f"Error during data merging/processing stage for {symbol}: {e_merge}", exc_info=True)
        # Attempt to return price_df if merge failed but price loaded, otherwise None
        return price_df if price_df is not None else None, opts_df if opts_df is not None else pd.DataFrame() # Ensure opts_df is DF