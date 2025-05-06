#!/usr/bin/env python
"""
Test script for the Trading Opportunity Scanner
"""

import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Run a test scan with the opportunity scanner."""
    # Ensure directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("results/entropy_viz", exist_ok=True)
    
    # Define test parameters
    test_args = [
        "--instruments", "data/test_instruments.csv",
        "--output", f"reports/test_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        "--format", "html",
        "--min-grade", "B",
        "--direction", "All",
        "--workers", "2",
        "--limit", "5",  # Limit to 5 symbols for testing
        "--config", "config.json"
    ]
    
    # Import run_scanner and run with test args
    sys.argv = ["run_scanner.py"] + test_args
    
    try:
        # Import the module
        import run_scanner
        
        # Run the scanner
        logger.info("Starting test scan...")
        exit_code = run_scanner.main()
        
        if exit_code == 0:
            logger.info("Test scan completed successfully!")
        else:
            logger.error(f"Test scan failed with exit code {exit_code}")
            
        return exit_code
    except Exception as e:
        logger.exception(f"Error running test scan: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())