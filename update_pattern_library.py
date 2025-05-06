#!/usr/bin/env python
"""
Update Ordinal Pattern Library

This script updates the ordinal pattern library by analyzing recent Greek data
and identifying new patterns or updating statistics for existing patterns.

Usage:
    python update_pattern_library.py --results-dir results --pattern-dir patterns
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from ordinal_pattern_analyzer import GreekOrdinalPatternAnalyzer
from pattern_integration import integrate_with_pipeline

def setup_logging(debug=False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("PatternLibrary")

def load_greek_data(symbol, results_dir):
    """Load Greek time series data for a symbol from analysis results."""
    logger = logging.getLogger(f"PatternLibrary.{symbol}")
    
    try:
        # Look for analysis results file
        results_file = os.path.join(results_dir, f"{symbol}_analysis_results.json")
        if not os.path.exists(results_file):
            logger.warning(f"Analysis results file not found: {results_file}")
            return None
            
        # Load analysis results
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        # Check if Greek time series data is available
        if 'greek_time_series' not in results:
            logger.warning(f"No Greek time series data in results for {symbol}")
            return None
            
        # Convert to DataFrame
        greek_data = pd.DataFrame(results['greek_time_series'])
        
        # Ensure required columns exist
        required_columns = ['date', 'delta', 'gamma', 'vanna', 'charm']
        missing_columns = [col for col in required_columns if col not in greek_data.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {', '.join(missing_columns)}")
            return None
            
        # Convert date column to datetime
        if 'date' in greek_data.columns:
            greek_data['date'] = pd.to_datetime(greek_data['date'])
            
        return greek_data
        
    except Exception as e:
        logger.error(f"Error loading Greek data for {symbol}: {str(e)}")
        return None

def update_symbol_patterns(symbol, results_dir, pattern_dir):
    """Update pattern library for a single symbol."""
    logger = logging.getLogger(f"PatternLibrary.{symbol}")
    
    try:
        # Load Greek time series data
        greek_data = load_greek_data(symbol, results_dir)
        
        if greek_data is None or greek_data.empty:
            logger.warning(f"No Greek time series data found for {symbol}")
            return False
            
        if len(greek_data) < 10:  # Require at least 10 data points
            logger.warning(f"Insufficient data for {symbol}, skipping")
            return False
            
        # Initialize analyzer with appropriate window size
        window_size = min(4, len(greek_data) - 1)  # Adjust window size based on available data
        if window_size < 3:
            logger.warning(f"Not enough data points for pattern analysis (need at least 4, got {len(greek_data)})")
            return False
            
        analyzer = GreekOrdinalPatternAnalyzer(window_size=window_size)
        
        # Extract patterns
        patterns = analyzer.extract_patterns(greek_data)
        
        if not patterns or len(patterns) == 0:
            logger.warning(f"No patterns extracted for {symbol}")
            return False
            
        # Analyze pattern profitability
        analysis = analyzer.analyze_pattern_profitability(greek_data, patterns)
        
        # Build pattern library
        library = analyzer.build_pattern_library(analysis)
        
        # Save pattern library
        output_file = os.path.join(pattern_dir, f"{symbol}_patterns.json")
        analyzer.save_pattern_library(output_file)
        
        logger.info(f"Updated pattern library for {symbol}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating pattern library for {symbol}: {str(e)}")
        return False

def get_symbols_from_results(results_dir):
    """Get list of symbols that have analysis results."""
    # Look for files ending with _analysis_results.json
    results_files = [f for f in os.listdir(results_dir) if f.endswith('_analysis_results.json')]
    
    # Extract symbol names from filenames
    symbols = [f.split('_analysis_results.json')[0] for f in results_files]
    
    return symbols

def main():
    """Update pattern libraries for all symbols with analysis results."""
    parser = argparse.ArgumentParser(description='Update pattern libraries from analysis results')
    parser.add_argument('--results-dir', default='results', help='Directory containing analysis results')
    parser.add_argument('--pattern-dir', default='patterns', help='Directory to store pattern libraries')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--skip-pipeline-integration', action='store_true', help='Skip integration with pipeline')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create pattern directory if it doesn't exist
    os.makedirs(args.pattern_dir, exist_ok=True)
    
    # Get list of symbols from analysis results
    symbols = get_symbols_from_results(args.results_dir)
    
    logging.getLogger("PatternLibrary").info(f"Found {len(symbols)} symbols with analysis results")
    
    # Update pattern library for each symbol
    success_count = 0
    for symbol in symbols:
        if update_symbol_patterns(symbol, args.results_dir, args.pattern_dir):
            success_count += 1
    
    logging.getLogger("PatternLibrary").info(f"Updated pattern libraries for {success_count} out of {len(symbols)} symbols")
    
    # Create combined pattern library
    create_combined_library(args.pattern_dir)
    logging.getLogger("PatternLibrary").info("Created combined pattern library")
    
    # Try to integrate with pipeline if available and not skipped
    if not args.skip_pipeline_integration:
        try:
            from pattern_integration import integrate_with_pipeline
            from analysis.pipeline_manager import PipelineManager
            
            # Initialize pipeline manager
            pipeline_manager = PipelineManager()
            
            # Integrate patterns with pipeline
            integrate_with_pipeline(pipeline_manager)
            logging.getLogger("PatternLibrary").info("Integrated patterns with pipeline")
            
        except Exception as e:
            logging.getLogger("PatternLibrary").error(f"Error integrating patterns with pipeline: {str(e)}")
            return 1
    else:
        logging.getLogger("PatternLibrary").info("Skipping pipeline integration due to --skip-pipeline-integration flag")
    
    return 0

def create_combined_library(pattern_dir):
    """Create a combined pattern library from individual symbol libraries."""
    logger = logging.getLogger("PatternLibrary.Combined")
    
    # Initialize combined library structure
    combined_library = {
        'ITM': {},
        'ATM': {},
        'OTM': {},
        'VOL_CRUSH': {}
    }
    
    # Find all pattern files
    pattern_files = []
    for filename in os.listdir(pattern_dir):
        if filename.endswith("_patterns.json") and not filename.startswith("greek_"):
            pattern_files.append(os.path.join(pattern_dir, filename))
    
    if not pattern_files:
        logger.warning("No pattern files found to combine")
        return
    
    logger.info(f"Combining {len(pattern_files)} pattern libraries")
    
    # Process each pattern file
    for pattern_file in pattern_files:
        try:
            with open(pattern_file, 'r') as f:
                library = json.load(f)
            
            # Merge into combined library
            for moneyness, greeks in library.items():
                if moneyness not in combined_library:
                    combined_library[moneyness] = {}
                
                for greek, patterns in greeks.items():
                    if greek not in combined_library[moneyness]:
                        combined_library[moneyness][greek] = {}
                    
                    # Add each pattern
                    for pattern, stats in patterns.items():
                        if pattern not in combined_library[moneyness][greek]:
                            combined_library[moneyness][greek][pattern] = stats
                        else:
                            # Merge stats
                            existing = combined_library[moneyness][greek][pattern]
                            
                            # Update counts
                            count = existing.get('count', 0) + stats.get('count', 0)
                            wins = existing.get('wins', 0) + stats.get('wins', 0)
                            losses = existing.get('losses', 0) + stats.get('losses', 0)
                            
                            # Recalculate win rate
                            win_rate = wins / count if count > 0 else 0
                            
                            # Merge returns
                            returns = existing.get('returns', []) + stats.get('returns', [])
                            if len(returns) > 20:
                                returns = returns[-20:]  # Keep last 20
                            
                            # Calculate average return
                            avg_return = sum(returns) / len(returns) if returns else 0
                            
                            # Update combined stats
                            combined_library[moneyness][greek][pattern] = {
                                'count': count,
                                'win_rate': win_rate,
                                'avg_return': avg_return,
                                'wins': wins,
                                'losses': losses,
                                'returns': returns,
                                'expected_value': avg_return * win_rate
                            }
        
        except Exception as e:
            logger.error(f"Error processing {pattern_file}: {str(e)}")
    
    # Save combined library
    combined_file = os.path.join(pattern_dir, "greek_patterns.json")
    with open(combined_file, 'w') as f:
        json.dump(combined_library, f, indent=2)
    
    logger.info(f"Saved combined pattern library to {combined_file}")

if __name__ == "__main__":
    sys.exit(main())












