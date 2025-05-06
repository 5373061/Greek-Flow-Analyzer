"""
Batch Analysis Runner

This script demonstrates how to run Greek Energy Flow analysis on a batch of instruments.
"""

import os
import logging
from datetime import datetime
from main import run_batch_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define paths
INSTRUMENTS_FILE = "data/instruments.csv"  # Your CSV file with instruments
OPTIONS_DATA_DIR = "data/options"          # Directory with options data files
PRICE_DATA_DIR = "data/prices"             # Directory with price history files
OUTPUT_DIR = "results"                     # Directory for output files

def main():
    """Run batch analysis on instruments"""
    logger.info("Starting batch analysis")
    
    # Ensure directories exist
    for directory in [OPTIONS_DATA_DIR, PRICE_DATA_DIR, OUTPUT_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Run batch analysis
    results = run_batch_analysis(
        INSTRUMENTS_FILE,
        OPTIONS_DATA_DIR,
        PRICE_DATA_DIR,
        OUTPUT_DIR
    )
    
    # Print summary
    if results:
        print("\n=== Batch Analysis Summary ===")
        for symbol, result in results.items():
            if result and "error" not in result:
                action = result.get("trade_opportunities", {}).get("suggested_action", "No action")
                regime = result.get("greek_analysis", {}).get("market_regime", {}).get("primary", "Unknown")
                print(f"{symbol}: {action} (Regime: {regime})")
            else:
                error = result.get("error", "Analysis failed") if result else "No result"
                print(f"{symbol}: ERROR - {error}")
    
    logger.info("Batch analysis complete")

if __name__ == "__main__":
    main()