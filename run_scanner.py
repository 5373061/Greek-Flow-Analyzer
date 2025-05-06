#!/usr/bin/env python
"""
Trading Opportunity Scanner

This script scans multiple symbols to identify and rank trading opportunities
based on Greek Energy Flow and Entropy Analysis.
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta

from utils.instrument_loader import load_instruments_from_csv
from analysis.opportunity_scanner import OpportunityScanner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/scanner_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Trading Opportunity Scanner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--instruments", "-i", 
        default="data/instruments.csv",
        help="CSV file with instruments to scan"
    )
    
    parser.add_argument(
        "--output", "-o",
        default=f"reports/opportunities_{datetime.now().strftime('%Y%m%d')}.html",
        help="Output file for the report"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["html", "csv", "json", "text"],
        default="html",
        help="Output format for the report"
    )
    
    parser.add_argument(
        "--min-grade", "-g",
        choices=["A+", "A", "B+", "B", "C+", "C", "D", "F"],
        default="B",
        help="Minimum grade to include in report"
    )
    
    parser.add_argument(
        "--direction", "-d",
        choices=["Bullish", "Bearish", "All"],
        default="All",
        help="Filter by trade direction"
    )
    
    parser.add_argument(
        "--sector", "-s",
        help="Filter by sector"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=0,
        help="Limit number of symbols to scan (0 for all)"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable use of cached results"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config.json",
        help="Configuration file"
    )
    
    return parser.parse_args()

def load_config(config_file):
    """Load configuration from file."""
    import json
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Config file {config_file} not found, using defaults")
            return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Load instruments
    try:
        instruments_df = load_instruments_from_csv(args.instruments)
        
        if instruments_df is None or instruments_df.empty:
            logger.error(f"No instruments found in {args.instruments}")
            return 1
            
        # Apply filters
        if args.sector:
            instruments_df = instruments_df[instruments_df["sector"] == args.sector]
            
        # Apply limit if specified
        if args.limit > 0:
            instruments_df = instruments_df.head(args.limit)
            
        # Get list of symbols to scan
        symbols = instruments_df["symbol"].tolist()
        logger.info(f"Scanning {len(symbols)} symbols")
        
        # Initialize scanner
        scanner = OpportunityScanner(config=config, max_workers=args.workers)
        
        # Run scan
        opportunities = scanner.scan_symbols(symbols, use_cached=not args.no_cache)
        
        # Filter results
        direction = None if args.direction == "All" else args.direction
        filtered_opportunities = scanner.get_top_opportunities(
            min_grade=args.min_grade,
            direction=direction
        )
        
        # Generate report
        report = scanner.generate_report(output_format=args.format)
        
        # Save report
        output_file = scanner.save_report(args.output, args.format)
        
        if output_file:
            logger.info(f"Report saved to {output_file}")
            logger.info(f"Found {len(filtered_opportunities)} opportunities matching criteria")
            return 0
        else:
            logger.error("Failed to save report")
            return 1
            
    except Exception as e:
        logger.exception(f"Error running scanner: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

