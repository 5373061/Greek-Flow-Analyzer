"""
Greek Energy Flow Analysis Runner

This script provides a unified interface to run Greek Energy Flow analysis
in various modes (single symbol, batch, etc.)
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Configure logging
log_file = f"greek_flow_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point with unified command line interface"""
    parser = argparse.ArgumentParser(
        description="Greek Energy Flow Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single symbol
  python run_analysis.py single SPY
  
  # Analyze a single symbol with custom data files
  python run_analysis.py single AAPL --options data/AAPL_options.csv --prices data/AAPL_prices.csv
  
  # Run batch analysis on all instruments in CSV
  python run_analysis.py batch data/instruments.csv
  
  # Run batch analysis on Technology sector only
  python run_analysis.py batch data/instruments.csv --sector Technology
  
  # Run analysis on top 10 stocks by market cap
  python run_analysis.py batch data/instruments.csv --limit 10 --sort-by market_cap_numeric
        """
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Analysis mode")
    
    # Single symbol analysis
    single_parser = subparsers.add_parser("single", help="Analyze a single symbol")
    single_parser.add_argument("symbol", help="Stock symbol to analyze")
    single_parser.add_argument("--options", help="Path to options data CSV file")
    single_parser.add_argument("--prices", help="Path to price history CSV file")
    
    # Batch analysis
    batch_parser = subparsers.add_parser("batch", help="Analyze multiple symbols from a CSV file")
    batch_parser.add_argument("instruments", help="Path to CSV file containing instruments")
    batch_parser.add_argument("--options-dir", default="data/options", help="Directory containing options data CSV files")
    batch_parser.add_argument("--prices-dir", default="data/prices", help="Directory containing price history CSV files")
    batch_parser.add_argument("--sector", help="Filter by sector (e.g., 'Technology')")
    batch_parser.add_argument("--min-market-cap", type=float, help="Minimum market cap in billions (e.g., 100 for $100B)")
    batch_parser.add_argument("--max-market-cap", type=float, help="Maximum market cap in billions (e.g., 1000 for $1T)")
    batch_parser.add_argument("--limit", type=int, help="Limit number of instruments to analyze")
    batch_parser.add_argument("--sort-by", help="Column to sort by (e.g., 'market_cap_numeric')")
    batch_parser.add_argument("--descending", action="store_true", help="Sort in descending order")
    
    # Common arguments
    parser.add_argument("--output", default="results", help="Directory to save output files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Import modules here to avoid slow startup time
    logger.info("Initializing Greek Energy Flow Analysis...")
    
    try:
        from main import run_analysis, run_batch_analysis
        from utils.instrument_loader import get_instrument_list
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Make sure you're running this script from the project root directory")
        sys.exit(1)
    
    # Run appropriate analysis mode
    if args.mode == "single":
        logger.info(f"Running analysis on symbol: {args.symbol}")
        run_analysis(args.symbol, args.options, args.prices, args.output)
        
    elif args.mode == "batch":
        logger.info(f"Running batch analysis from: {args.instruments}")
        
        # Build filter criteria
        criteria = {}
        if args.sector:
            criteria['sector'] = args.sector
        
        market_cap_filter = {}
        if args.min_market_cap:
            market_cap_filter['min'] = args.min_market_cap * 1e9  # Convert billions to actual value
        if args.max_market_cap:
            market_cap_filter['max'] = args.max_market_cap * 1e9  # Convert billions to actual value
        
        if market_cap_filter:
            criteria['market_cap_numeric'] = market_cap_filter
        
        # Get filtered instrument list
        symbols = get_instrument_list(
            args.instruments,
            criteria=criteria if criteria else None,
            limit=args.limit,
            sort_by=args.sort_by,
            ascending=not args.descending
        )
        
        if not symbols:
            logger.error(f"No instruments found matching the criteria")
            return
        
        logger.info(f"Running analysis on {len(symbols)} instruments")
        run_batch_analysis(symbols, args.options_dir, args.prices_dir, args.output)
        
    else:
        parser.print_help()
    
    logger.info(f"Analysis complete. Results saved to {args.output}")
    logger.info(f"Log file: {log_file}")

if __name__ == "__main__":
    main()
