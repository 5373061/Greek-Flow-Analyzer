"""
Run Greek Energy Flow analysis using the Pipeline Manager directly.
Includes options for pattern analysis and trade recommendations.

This script provides a command-line interface for running the full analysis pipeline,
including Greek Energy Flow analysis, pattern recognition, and entropy analysis.
It can process individual tickers or lists of tickers from a CSV file.
"""

import os
import logging
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
from analysis.pipeline_manager import AnalysisPipeline
from analysis.trade_recommendations import TradeRecommendationEngine, generate_trade_recommendation
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import threading

# Import data fetching functions
from pipeline.data_pipeline import OptionsDataPipeline
import config

# Add import for pattern analyzer
try:
    from analysis.symbol_analyzer import SymbolAnalyzer
    HAS_PATTERN_ANALYZER = True
except ImportError:
    HAS_PATTERN_ANALYZER = False
    logging.warning("Pattern Analyzer module not found. Pattern analysis will be disabled.")

# Add import for ordinal pattern analyzer
try:
    from ordinal_pattern_analyzer import GreekOrdinalPatternAnalyzer
    HAS_ORDINAL_ANALYZER = True
except ImportError:
    HAS_ORDINAL_ANALYZER = False
    logging.warning("Ordinal Pattern Analyzer module not found. Ordinal pattern analysis will be disabled.")

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.datetime64):
            return str(obj)
        elif isinstance(obj, np.complex128) or isinstance(obj, np.complex64):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ma.MaskedArray):
            return [None if np.ma.is_masked(x) else x for x in obj]
        elif hasattr(obj, 'dtype') and obj.dtype.names is not None:  # Structured array
            return {name: obj[name].item() for name in obj.dtype.names}
        return super(NumpyEncoder, self).default(obj)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Import API key from config
from config import POLYGON_API_KEY, DATA_DIR, OUTPUT_DIR

def setup_logging(debug=False):
    """Configure logging for the application."""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger()

def load_tickers_from_csv(csv_path):
    """
    Load ticker symbols from a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing ticker symbols
        
    Returns:
        List of ticker symbols or empty list if file not found or no symbol column
    """
    try:
        df = pd.read_csv(csv_path)
        if 'symbol' in df.columns:
            tickers = df['symbol'].tolist()
            logger.info(f"Loaded {len(tickers)} tickers from {csv_path}")
            return tickers
        else:
            logger.error(f"CSV file {csv_path} does not contain a 'symbol' column")
            return []
    except FileNotFoundError:
        logger.error(f"Error loading tickers from {csv_path}: File not found")
        return []
    except Exception as e:
        logger.error(f"Error loading tickers from {csv_path}: {e}")
        return []

def process_ticker(symbol, api_key, data_dir, output_dir, skip_entropy, pipeline, data_pipeline):
    """
    Process a single ticker symbol.
    """
    logger.info(f"Starting process_ticker for {symbol}...")
    
    # Define file paths
    options_path = f"{data_dir}/options/{symbol}_options.csv"
    price_path = f"{data_dir}/prices/{symbol}_price.csv"
    
    # Fetch real market data using api_fetcher
    try:
        from api_fetcher import fetch_options_chain_snapshot, fetch_underlying_snapshot, preprocess_api_options_data, get_spot_price_from_snapshot
        from datetime import datetime
    except ImportError as e:
        logger.error(f"Failed to import api_fetcher: {e}")
        return {"status": "error", "message": f"Import error: {e}"}
    
    try:
        # Fetch underlying data with retry logic
        max_retries = 3
        retry_delay = 2  # seconds
        underlying_data = None
        
        logger.info(f"Fetching underlying data for {symbol}...")
        for attempt in range(max_retries):
            try:
                underlying_data = fetch_underlying_snapshot(symbol, api_key)
                if underlying_data:
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} failed for {symbol}: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to fetch underlying data for {symbol} after {max_retries} attempts: {str(e)}")
        
        if not underlying_data:
            logger.error(f"Failed to fetch underlying data for {symbol}, skipping...")
            return {"status": "error", "message": "Failed to fetch underlying data"}
        
        # Get current price
        logger.info(f"Getting current price for {symbol}...")
        current_price = get_spot_price_from_snapshot(underlying_data)
        if not current_price:
            logger.error(f"Failed to get current price for {symbol}, skipping...")
            return {"status": "error", "message": "Failed to get current price"}
        
        logger.info(f"Current price for {symbol}: ${current_price}")
        
        # Fetch options chain with retry logic
        logger.info(f"Fetching options chain for {symbol}...")
        options_data = None
        for attempt in range(max_retries):
            try:
                options_data = fetch_options_chain_snapshot(symbol, api_key)
                if options_data:
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} failed for {symbol} options: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to fetch options data for {symbol} after {max_retries} attempts: {str(e)}")
        
        if not options_data:
            logger.error(f"Failed to fetch options data for {symbol}, skipping...")
            return {"status": "error", "message": "Failed to fetch options data"}
        
        # Process options data
        logger.info(f"Processing options data for {symbol}...")
        options_df = preprocess_api_options_data(options_data, datetime.now().date())
        if options_df.empty:
            logger.error(f"No valid options data for {symbol}, skipping...")
            return {"status": "error", "message": "No valid options data"}
        
        # Add symbol column to options data
        options_df['symbol'] = symbol
        
        # Save processed data to files
        logger.info(f"Saving options data for {symbol}...")
        options_df.to_csv(options_path, index=False)
        logger.info(f"Saved options data to {options_path}")
        
        # Create a simple price dataframe with current price
        price_df = pd.DataFrame({
            'date': [datetime.now().date()],
            'open': [current_price],
            'high': [current_price],
            'low': [current_price],
            'close': [current_price],
            'volume': [0]  # We don't have volume data from the snapshot
        })
        price_df.to_csv(price_path, index=False)
        logger.info(f"Saved price data to {price_path}")
        
        # Run the full analysis
        try:
            logger.info(f"Running full analysis for {symbol}...")
            analysis_results = pipeline.run_full_analysis(
                symbol=symbol,
                options_path=options_path,
                price_path=price_path,
                output_dir=output_dir,
                skip_entropy=skip_entropy
            )
            
            if analysis_results:
                logger.info(f"Analysis complete for {symbol}")
                result = {
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    "price": current_price,
                    "options_count": len(options_df),
                    "analysis_results": analysis_results
                }
                
                # Generate trade recommendations if analysis was successful
                try:
                    logger.info(f"Generating trade recommendations for {symbol}...")
                    # Fix the parameter names to match what the function expects
                    recommendations = generate_trade_recommendation(
                        analysis_results=analysis_results.get('greek_analysis', {}),
                        entropy_data=analysis_results.get('chain_energy', {}),
                        current_price=current_price
                    )
                    result["trade_recommendations"] = recommendations
                except Exception as e:
                    logger.warning(f"Error generating trade recommendations for {symbol}: {str(e)}")
                    result["trade_recommendations"] = {"error": str(e)}
                
                # Save individual results to JSON
                result_path = os.path.join(output_dir, f"{symbol}_analysis.json")
                with open(result_path, 'w') as f:
                    json.dump(result, f, cls=NumpyEncoder, indent=2)
                logger.info(f"Saved analysis results to {result_path}")
                
                return result
            else:
                logger.error(f"Analysis failed for {symbol}")
                return {"status": "error", "message": "Analysis failed"}
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    except Exception as e:
        logger.error(f"Unexpected error processing {symbol}: {str(e)}")
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

def run_pipeline_analysis(tickers, api_key=POLYGON_API_KEY, data_dir=DATA_DIR, output_dir=OUTPUT_DIR, skip_entropy=False, max_workers=5):
    """
    Run the pipeline analysis for a list of tickers
    
    Args:
        tickers: List of ticker symbols to analyze
        api_key: Polygon API key
        data_dir: Directory for data files
        output_dir: Directory for output files
        skip_entropy: Whether to skip entropy analysis
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary with analysis results for all tickers
    """
    # Validate API key
    if not api_key or "YOUR_ACTUAL_API_KEY_HERE" in api_key:
        logger.error("Invalid API key in config.py. Please update with your actual Polygon API key.")
        return False

    # Create necessary directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "options"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "prices"), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the pipeline with config
    pipeline_config = {
        "greek_config": {
            "regime_thresholds": {
                "highVolatility": 0.3,
                "lowVolatility": 0.15,
                "strongBullish": 0.7,
                "strongBearish": -0.7,
                "neutralZone": 0.2
            }
        },
        "cache_dir": "cache",
        "POLYGON_API_KEY": api_key
    }

    # Initialize the pipeline
    pipeline = AnalysisPipeline(config=pipeline_config)

    # Initialize the data pipeline
    data_pipeline = OptionsDataPipeline(pipeline_config)

    # Process each ticker
    results = {}
    
    # Determine if we should use parallel processing
    use_parallel = len(tickers) > 1 and max_workers > 1
    
    if use_parallel:
        # Process tickers in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(
                    process_ticker, symbol, api_key, data_dir, output_dir, skip_entropy, pipeline, data_pipeline
                ): symbol for symbol in tickers
            }
            
            # Process results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results[symbol] = result
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    results[symbol] = {"status": "error", "message": str(e)}
    else:
        # Process tickers sequentially
        for symbol in tickers:
            results[symbol] = process_ticker(symbol, api_key, data_dir, output_dir, skip_entropy, pipeline, data_pipeline)

    # Save summary results
    summary_path = os.path.join(output_dir, f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)
    logger.info(f"Saved analysis summary to {summary_path}")
    
    logger.info("All analyses completed.")
    return results

def run_pattern_analysis(tickers, analysis_date=None, cache_dir="cache", output_dir=OUTPUT_DIR, use_parallel=True):
    """
    Run pattern analysis for a list of tickers
    
    Args:
        tickers: List of ticker symbols to analyze
        analysis_date: Specific date for analysis (YYYY-MM-DD)
        cache_dir: Directory for cached data
        output_dir: Directory for output files
        use_parallel: Whether to use parallel processing
        
    Returns:
        Dictionary with pattern analysis results or False if analysis failed
    """
    if not HAS_PATTERN_ANALYZER:
        logger.error("Pattern Analyzer module not available. Cannot run pattern analysis.")
        return False
    
    logger.info(f"Running pattern analysis for {len(tickers)} symbols...")
    
    try:
        # Create analyzer with provided configuration
        analyzer = SymbolAnalyzer(
            cache_dir=cache_dir or "cache",  # Default to "cache" if None
            output_dir=output_dir or OUTPUT_DIR,  # Default to OUTPUT_DIR if None
            use_parallel=use_parallel
        )
        
        # Run analysis
        results = analyzer.run_analysis(tickers, analysis_date=analysis_date)
        
        # Display summary
        if results:
            analyzer.display_summary(results)
            logger.info("Pattern analysis completed successfully")
            
            # Save results to JSON
            pattern_path = os.path.join(output_dir, f"pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(pattern_path, 'w') as f:
                json.dump(results, f, cls=NumpyEncoder, indent=2)
            logger.info(f"Saved pattern analysis to {pattern_path}")
        else:
            logger.warning("Pattern analysis returned no results")
            
        return results
    except Exception as e:
        logger.error(f"Error during pattern analysis: {e}", exc_info=True)
        return False

def run_ordinal_pattern_analysis(tickers, output_dir=OUTPUT_DIR, use_parallel=True):
    """
    Run ordinal pattern analysis for a list of tickers
    
    Args:
        tickers: List of ticker symbols to analyze
        output_dir: Directory for output files
        use_parallel: Whether to use parallel processing
        
    Returns:
        Dictionary with ordinal pattern analysis results or False if analysis failed
    """
    if not HAS_ORDINAL_ANALYZER:
        logger.error("Ordinal Pattern Analyzer module not available. Cannot run ordinal pattern analysis.")
        return False
    
    logger.info(f"Running ordinal pattern analysis for {len(tickers)} symbols...")
    
    try:
        # Initialize the analyzer
        analyzer = GreekOrdinalPatternAnalyzer(cache_dir="cache")
        
        # Run analysis
        results = analyzer.analyze_tickers(tickers, use_parallel=use_parallel)
        
        # Save results to JSON
        pattern_path = os.path.join(output_dir, f"ordinal_pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(pattern_path, 'w') as f:
            json.dump(results, f, cls=NumpyEncoder, indent=2)
        logger.info(f"Saved ordinal pattern analysis to {pattern_path}")
        
        # Generate trade recommendations based on patterns
        try:
            from analysis.trade_recommendations import generate_pattern_trade_recommendation
            
            pattern_recommendations = {}
            for symbol, pattern_data in results.items():
                recommendation = generate_pattern_trade_recommendation(
                    symbol=symbol,
                    pattern_data=pattern_data,
                    output_dir=output_dir
                )
                pattern_recommendations[symbol] = recommendation
                
            # Save pattern-based recommendations
            rec_path = os.path.join(output_dir, f"pattern_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(rec_path, 'w') as f:
                json.dump(pattern_recommendations, f, cls=NumpyEncoder, indent=2)
            logger.info(f"Saved pattern-based trade recommendations to {rec_path}")
            
        except Exception as e:
            logger.error(f"Error generating pattern-based trade recommendations: {e}", exc_info=True)
        
        return results
    except Exception as e:
        logger.error(f"Error during ordinal pattern analysis: {e}", exc_info=True)
        return False

def main():
    """Main function to run the pipeline analysis."""
    try:
        # Set up emergency timer to prevent hanging
        timer = threading.Timer(3600, lambda: os._exit(1))  # Force exit after 1 hour
        timer.daemon = True
        timer.start()
        
        # Configure logging
        setup_logging()
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Run Greek Energy Flow analysis using the Pipeline Manager")
        
        # Add arguments
        parser.add_argument("--tickers", nargs="+", help="List of ticker symbols to analyze")
        parser.add_argument("--csv", help="CSV file with ticker symbols")
        parser.add_argument("--output-dir", help="Output directory for analysis files")
        parser.add_argument("--data-dir", help="Directory for data files")
        parser.add_argument("--api-key", help="Polygon API key")
        parser.add_argument("--date", help="Analysis date (YYYY-MM-DD)")
        parser.add_argument("--cache-dir", default="cache", help="Cache directory")
        parser.add_argument("--analysis-type", choices=["greek", "pattern", "ordinal", "both", "all"], default="both", 
                          help="Type of analysis to run")
        parser.add_argument("--skip-entropy", action="store_true", help="Skip entropy analysis")
        parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of parallel workers")
        parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
        parser.add_argument("--debug-ticker", help="Run in debug mode with a single ticker")
        parser.add_argument("--track-performance", action="store_true", help="Track performance of trade recommendations")
        
        # Check if running without arguments or with debug ticker
        if len(sys.argv) == 1 or (len(sys.argv) == 3 and sys.argv[1] == "--debug-ticker"):
            # Use debug ticker if provided, otherwise use default tickers
            if len(sys.argv) == 3:
                tickers = [sys.argv[2]]
                logger.info(f"Debug mode: analyzing single ticker {tickers[0]}")
            else:
                # Use a list of default tickers instead of just AAPL
                tickers = ["AAPL", "MSFT", "QQQ", "SPY", "LULU", "TSLA", "CMG", "WYNN", "ZM", "SPOT"]
                logger.info(f"Using default tickers: {', '.join(tickers)}")
            
            # Run Greek analysis with debug logging
            logger.info("Starting pipeline analysis...")
            greek_results = run_pipeline_analysis(
                tickers=tickers,
                api_key=POLYGON_API_KEY,
                data_dir=DATA_DIR,
                output_dir=OUTPUT_DIR,
                skip_entropy=True,  # Skip entropy for faster debugging
                max_workers=5  # Use parallel processing for multiple tickers
            )
            
            logger.info("Pipeline analysis completed.")
            
            # Cancel the emergency timer
            timer.cancel()
            return 0
        
        # Parse arguments
        args = parser.parse_args()
        
        # Get tickers from args or CSV
        if args.tickers:
            tickers = args.tickers
            logger.info(f"Analyzing {len(tickers)} tickers from command line")
        else:
            tickers = load_tickers_from_csv(args.csv)
            if not tickers:
                logger.error("No tickers loaded from CSV")
                timer.cancel()
                return 1
        
        # Determine which analyses to run
        run_greek = args.analysis_type in ["greek", "both", "all"]
        run_pattern = args.analysis_type in ["pattern", "both", "all"] and HAS_PATTERN_ANALYZER
        run_ordinal = args.analysis_type in ["ordinal", "all"] and HAS_ORDINAL_ANALYZER
        
        results = {}
        
        # Run Greek Energy Flow analysis if requested
        if run_greek:
            logger.info("Running Greek Energy Flow analysis...")
            greek_results = run_pipeline_analysis(
                tickers=tickers,
                api_key=args.api_key or POLYGON_API_KEY,
                data_dir=args.data_dir or DATA_DIR,
                output_dir=args.output_dir or OUTPUT_DIR,
                skip_entropy=args.skip_entropy,
                max_workers=1 if args.no_parallel else args.max_workers
            )
            results["greek"] = greek_results
        
        # Run pattern analysis if requested
        if run_pattern:
            logger.info("Running Pattern analysis...")
            pattern_results = run_pattern_analysis(
                tickers=tickers,
                analysis_date=args.date,
                cache_dir=args.cache_dir,
                output_dir=args.output_dir or OUTPUT_DIR,
                use_parallel=not args.no_parallel
            )
            results["pattern"] = pattern_results
            
        # Run ordinal pattern analysis if requested
        if run_ordinal:
            logger.info("Running Ordinal Pattern analysis...")
            ordinal_results = run_ordinal_pattern_analysis(
                tickers=tickers,
                output_dir=args.output_dir or OUTPUT_DIR,
                use_parallel=not args.no_parallel
            )
            results["ordinal"] = ordinal_results
        
        if not run_greek and not run_pattern and not run_ordinal:
            logger.warning("No analysis was run. Check if required modules are available.")
        
        # Track performance if requested
        if args.track_performance:
            try:
                from tools.instrument_tracker import TradePerformanceTracker
                
                # Initialize tracker
                tracker = TradePerformanceTracker(results_dir=args.output_dir or OUTPUT_DIR)
                
                # Load recommendations
                recommendations_dir = args.output_dir or OUTPUT_DIR
                import glob
                for rec_file in glob.glob(os.path.join(recommendations_dir, "*recommendations*.json")):
                    try:
                        with open(rec_file, 'r') as f:
                            recommendations = json.load(f)
                        
                        # Determine strategy type from filename
                        strategy_type = "ML"
                        if "pattern" in os.path.basename(rec_file).lower():
                            strategy_type = "Pattern"
                        elif "ordinal" in os.path.basename(rec_file).lower():
                            strategy_type = "Ordinal"
                        
                        # Add recommendations to tracker
                        for symbol, recommendation in recommendations.items():
                            tracker.add_recommendation(recommendation, strategy_type)
                            
                        logger.info(f"Added recommendations from {rec_file} to performance tracker")
                    except Exception as e:
                        logger.error(f"Error loading recommendations from {rec_file}: {e}")
                
                # Generate performance report
                report_path = os.path.join(recommendations_dir, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                tracker.generate_performance_report(report_path)
                logger.info(f"Generated performance report at {report_path}")
                
            except ImportError:
                logger.warning("Trade performance tracker not available. Skipping performance tracking.")
        
        # Cancel the emergency timer
        timer.cancel()
        return 0
    
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        # Cancel the emergency timer
        timer.cancel()
        return 1

if __name__ == "__main__":
    sys.exit(main())




























