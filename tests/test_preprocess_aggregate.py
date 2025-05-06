# test_api_and_preprocess.py
# Cleaned test script focusing on API fetch and preprocessing steps.

import requests
import pandas as pd
import numpy as np
import json
import sys
import os
import logging
from pprint import pprint
from datetime import datetime, timedelta
import glob

# --- Configuration ---
try:
    import config
    print("✅ Successfully imported config.py")
except ImportError:
    print("❌ ERROR: config.py not found.")
    class MockConfig:
        POLYGON_API_KEY = "YOUR_API_KEY_HERE" # Replace if needed
        DATA_DIR = "."
    config = MockConfig()

# --- Import only necessary functions for this test ---
try:
    from api_fetcher import (fetch_options_chain_snapshot, fetch_underlying_snapshot,
                             get_spot_price_from_snapshot, find_latest_overview_file,
                             preprocess_api_options_data) # Import preprocessing too
    print("✅ Successfully imported necessary functions from api_fetcher.")
except ImportError as e:
    print(f"❌ ERROR: Failed to import from api_fetcher.py: {e}")
    print("Ensure 'api_fetcher.py' is in the correct location and contains the functions.")
    sys.exit(1)

# --- Constants ---
TICKER_TO_TEST = "AAPL" # Change as needed
API_KEY = config.POLYGON_API_KEY
if not API_KEY or "YOUR" in API_KEY:
    print(f"❌ ERROR: POLYGON_API_KEY is not set correctly.")
    sys.exit(1)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', force=True)
logger = logging.getLogger("APITest")

# --- Main Test Execution ---
if __name__ == "__main__":
    logger.info(f"--- Starting API Fetch & Preprocess Test for: {TICKER_TO_TEST} ---")

    # 1. Fetch API Data
    api_options_data = fetch_options_chain_snapshot(TICKER_TO_TEST, API_KEY)
    api_underlying_data = fetch_underlying_snapshot(TICKER_TO_TEST, API_KEY)

    if not api_options_data or not api_underlying_data:
        logger.error("Failed to fetch required data from API. Aborting test.")
        sys.exit(1)

    # 2. Get Spot Price
    spot_price = get_spot_price_from_snapshot(api_underlying_data)
    if spot_price is None:
        logger.error("Failed to get spot price. Aborting test.")
        sys.exit(1)
    logger.info(f"Extracted Spot Price: {spot_price}")

    # 3. Preprocess Options Data
    analysis_date_today = datetime.now().date()
    try:
        options_df_processed = preprocess_api_options_data(api_options_data, analysis_date_today)

        if options_df_processed.empty:
            logger.error("No valid options data remained after preprocessing.")
        else:
            logger.info(f"Successfully preprocessed {len(options_df_processed)} options contracts.")
            print("\n--- Sample Preprocessed Options Data ---")
            print(options_df_processed.head().to_string())
            # Check for essential columns after processing
            required_cols = ['strike', 'expiration', 'type', 'openInterest', 'impliedVolatility', 'delta', 'gamma']
            missing = [col for col in required_cols if col not in options_df_processed.columns]
            if missing:
                logger.error(f"❌ Preprocessed data MISSING essential columns: {missing}")
            else:
                logger.info("✅ Preprocessed data contains essential columns.")

    except Exception as e:
        logger.error(f"Error during preprocessing step: {e}", exc_info=True)

    logger.info(f"--- API Fetch & Preprocess Test for {TICKER_TO_TEST} Complete ---")