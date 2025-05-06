"""
Run the full analysis pipeline followed by market regime analysis.
This script ensures that analysis files are generated before running the regime analyzer.
"""

import os
import sys
import logging
import subprocess
import argparse
from datetime import datetime

def setup_logging(debug=False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("RegimeAnalysis")

logger = setup_logging()

def run_pipeline_analysis(tickers, output_dir="results", skip_entropy=False, analysis_type="both"):
    """Run the pipeline analysis to generate analysis files."""
    logger.info(f"Running pipeline analysis for {len(tickers)} tickers...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process tickers one by one for better error handling
    results = {}
    for ticker in tickers:
        logger.info(f"Processing {ticker}...")
        
        # Build command for single ticker
        cmd = [
            "python", "run_with_pipeline.py",
            "--tickers", ticker,
            "--output-dir", output_dir,
            "--analysis-type", analysis_type
        ]
        
        if skip_entropy:
            cmd.append("--skip-entropy")
        
        # Set environment variable to use non-interactive backend
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"  # Use non-interactive backend
        
        # Run command with timeout
        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)  # 5-minute timeout
            
            if result.returncode != 0:
                logger.error(f"Analysis failed for {ticker} with code {result.returncode}")
                logger.error(f"Error: {result.stderr}")
                results[ticker] = False
            else:
                logger.info(f"Analysis completed for {ticker}")
                results[ticker] = True
                
        except subprocess.TimeoutExpired:
            logger.error(f"Analysis for {ticker} timed out after 5 minutes")
            results[ticker] = False
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            results[ticker] = False
    
    # Check if at least some tickers were processed successfully
    success_count = sum(1 for success in results.values() if success)
    logger.info(f"Completed analysis for {success_count}/{len(tickers)} tickers")
    
    return success_count > 0

def run_regime_analysis(results_dir="results"):
    """Run the market regime analyzer."""
    logger.info("Running market regime analysis...")
    
    # Create market regime directory
    os.makedirs(os.path.join(results_dir, "market_regime"), exist_ok=True)
    
    # Build command
    cmd = [
        "python", "analysis/market_regime_analyzer.py",
        "--results-dir", results_dir,
        "--validate"
    ]
    
    # Set environment variable to use non-interactive backend
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # Use non-interactive backend
    
    # Run command
    logger.info(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        logger.error(f"Market regime analysis failed with code {result.returncode}")
        logger.error(f"Error: {result.stderr}")
        return False
    
    logger.info("Market regime analysis completed successfully")
    return True

def run_momentum_analysis(tickers, output_dir="results"):
    """Run momentum analysis for the specified tickers."""
    logger.info(f"Running momentum analysis for {len(tickers)} tickers...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process tickers one by one
    results = {}
    for ticker in tickers:
        logger.info(f"Processing momentum analysis for {ticker}...")
        
        # Build command for momentum analysis
        cmd = [
            "python", "analysis/momentum_analyzer.py",
            "--ticker", ticker,
            "--output-dir", output_dir
        ]
        
        # Set environment variable to use non-interactive backend
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"  # Use non-interactive backend
        
        # Run command
        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
            
            if result.returncode != 0:
                logger.error(f"Momentum analysis failed for {ticker} with code {result.returncode}")
                logger.error(f"Error: {result.stderr}")
                results[ticker] = False
            else:
                logger.info(f"Momentum analysis completed for {ticker}")
                results[ticker] = True
                
        except subprocess.TimeoutExpired:
            logger.error(f"Momentum analysis for {ticker} timed out after 5 minutes")
            results[ticker] = False
        except Exception as e:
            logger.error(f"Error processing momentum for {ticker}: {str(e)}")
            results[ticker] = False
    
    # Check if at least some tickers were processed successfully
    success_count = sum(1 for success in results.values() if success)
    logger.info(f"Completed momentum analysis for {success_count}/{len(tickers)} tickers")
    
    return success_count > 0

def main():
    """Main function to run the full analysis pipeline."""
    parser = argparse.ArgumentParser(description="Run full analysis pipeline with market regime analysis")
    
    # Add arguments
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "QQQ", "SPY", "LULU", "TSLA", "CMG", "WYNN", "ZM", "SPOT"],
                      help="List of ticker symbols to analyze")
    parser.add_argument("--output-dir", default="results", help="Output directory for analysis files")
    parser.add_argument("--skip-entropy", action="store_true", help="Skip entropy analysis for faster processing")
    parser.add_argument("--analysis-type", choices=["greek", "pattern", "momentum", "all"], default="all",
                      help="Type of analysis to run (default: all)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Determine which analysis type to use
    analysis_type = "both"  # Default for run_with_pipeline.py
    if args.analysis_type == "greek":
        analysis_type = "greek"
    elif args.analysis_type == "pattern":
        analysis_type = "pattern"
    elif args.analysis_type == "momentum" or args.analysis_type == "all":
        analysis_type = "both"  # "both" includes momentum in run_with_pipeline.py
    
    # Run pipeline analysis
    pipeline_success = run_pipeline_analysis(
        tickers=args.tickers,
        output_dir=args.output_dir,
        skip_entropy=args.skip_entropy,
        analysis_type=analysis_type
    )
    
    if not pipeline_success:
        logger.error("Pipeline analysis failed, aborting regime analysis")
        return 1
    
    # Run additional momentum analysis if specifically requested
    if args.analysis_type == "momentum":
        momentum_success = run_momentum_analysis(
            tickers=args.tickers,
            output_dir=args.output_dir
        )
        if not momentum_success:
            logger.warning("Momentum analysis failed for some tickers")
    
    # Run market regime analysis
    regime_success = run_regime_analysis(results_dir=args.output_dir)
    
    if not regime_success:
        logger.error("Market regime analysis failed")
        return 1
    
    logger.info("Full analysis pipeline completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
















