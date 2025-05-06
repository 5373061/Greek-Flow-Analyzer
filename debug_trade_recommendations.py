"""
Debug script for trade recommendations.

This script generates trade recommendations based on analysis results and saves them
in a format compatible with the dashboard.
"""

import json
import logging
import os
import sys
import argparse
from pprint import pprint
from datetime import datetime

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.trade_recommendations import generate_trade_recommendation
import config  # Import centralized configuration

# Set up logging with more detail
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for more information
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def validate_analysis_data(analysis_results, entropy_data):
    """
    Validate that the analysis data contains all required fields for trade recommendations.
    
    Args:
        analysis_results: Dictionary containing analysis results
        entropy_data: Dictionary containing entropy analysis results
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    # Check if we have the minimum required data
    if not analysis_results:
        print("Missing analysis_results")
        return False
        
    # Check for greek_analysis
    if "greek_analysis" not in analysis_results:
        print("Missing greek_analysis in analysis_results")
        return False
        
    # Check for market_regime in greek_analysis
    greek_analysis = analysis_results["greek_analysis"]
    if not greek_analysis or "market_regime" not in greek_analysis:
        print("Missing market_regime in greek_analysis")
        return False
        
    # Check for entropy data
    if not entropy_data:
        print("Missing entropy_data")
        return False
        
    # Check for metrics in entropy_data (instead of energy_state)
    if "metrics" not in entropy_data:
        print("Missing metrics in entropy_data")
        return False
        
    # All checks passed
    return True

def load_analysis_results(symbol):
    """Load analysis results for a symbol, trying different file patterns."""
    possible_paths = [
        f"results/{symbol}_analysis_results.json",
        f"results/{symbol}_analysis.json",
        f"results/{symbol}_greek_analysis.json"
    ]
    
    for path in possible_paths:
        try:
            with open(path, "r") as f:
                logging.info(f"Successfully loaded analysis from {path}")
                return json.load(f)
        except FileNotFoundError:
            continue
    
    # If we get here, none of the paths worked
    logging.error(f"Could not find analysis results for {symbol}. Please check file path.")
    available_files = [f for f in os.listdir("results") if f.endswith(".json")]
    logging.info(f"Available JSON files in results directory: {available_files}")
    sys.exit(1)

def load_options_data(symbol):
    """Load options data if available."""
    possible_paths = [
        f"data/options/{symbol}_options.json",
        f"data/{symbol}_options_chain.json"
    ]
    
    for path in possible_paths:
        try:
            with open(path, "r") as f:
                logging.info(f"Successfully loaded options data from {path}")
                return json.load(f)
        except FileNotFoundError:
            continue
    
    logging.warning(f"Could not find options data for {symbol}. Proceeding without options chain data.")
    return None

def save_dashboard_compatible_recommendation(symbol, recommendation, output_dir="results", current_price=None, analysis_results=None, entropy_data=None):
    """
    Save the recommendation in a format compatible with the dashboard.
    
    Args:
        symbol: Ticker symbol
        recommendation: Trade recommendation dictionary
        output_dir: Directory to save the file
        current_price: Current price to use if available
        analysis_results: Original analysis results for additional context
        entropy_data: Original entropy data for additional context
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get target price from recommendation or use current price
    target_price = recommendation["strategy"].get("target_price", 0.0)
    
    # If target price is invalid (-1.0), try to use current_price
    if target_price == -1.0 and current_price and current_price > 0:
        target_price = current_price
    
    # Create dashboard-compatible format
    dashboard_rec = {
        "Symbol": symbol,
        "Strategy": recommendation["strategy"]["name"],
        "Action": recommendation["strategy"].get("action", "HOLD"),
        "Entry": target_price,
        "Stop": recommendation["strategy"].get("stop_loss", target_price * 0.95),
        "Target": recommendation["strategy"].get("target_price", target_price * 1.05),
        "RiskReward": 0.0,  # Calculate if possible
        "Regime": recommendation["strategy"].get("reason", ""),
        "VolRegime": "Normal",  # Default
        "Timestamp": recommendation["timestamp"],
        "Confidence": 0.0,  # Extract from ML if available
        "Notes": "Greek Analysis: " + recommendation["strategy"].get("description", "")
    }
    
    # Add ML confidence if available
    ml_confirmation = recommendation["strategy"].get("ml_confirmation", "")
    if ml_confirmation:
        dashboard_rec["Notes"] += " " + ml_confirmation
        # Try to extract confidence percentage
        import re
        confidence_match = re.search(r"(\d+)%", ml_confirmation)
        if confidence_match:
            dashboard_rec["Confidence"] = float(confidence_match.group(1)) / 100
    
    # Add anomaly warning if available
    anomaly_warning = recommendation["strategy"].get("anomaly_warning", "")
    if anomaly_warning:
        dashboard_rec["Notes"] += " " + anomaly_warning
    
    # Calculate risk/reward if possible
    entry = dashboard_rec["Entry"]
    stop = dashboard_rec["Stop"]
    target = dashboard_rec["Target"]
    
    if entry and stop and entry != stop:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk > 0:
            dashboard_rec["RiskReward"] = round(reward / risk, 2)
    
    # Extract and add trade context if analysis results are available
    if analysis_results and entropy_data:
        trade_context = extract_trade_context(analysis_results, entropy_data, recommendation)
        dashboard_rec["TradeContext"] = trade_context
        
        # Update volatility regime from trade context
        if "volatility_regime" in trade_context:
            dashboard_rec["VolRegime"] = trade_context["volatility_regime"]
        
        # Add hold time to main recommendation
        if "hold_time_days" in trade_context:
            dashboard_rec["HoldTimeDays"] = trade_context["hold_time_days"]
    
    # Save to file
    dashboard_file = os.path.join(output_dir, f"{symbol}_trade_recommendation.json")
    with open(dashboard_file, "w") as f:
        json.dump(dashboard_rec, f, indent=2)
    
    logging.info(f"Saved dashboard-compatible recommendation to {dashboard_file}")
    return dashboard_file

def print_structure(data, indent=0, max_depth=3):
    """Print the structure of a nested dictionary or list."""
    if indent > max_depth:
        print(" " * indent + "...")
        return
        
    if isinstance(data, dict):
        for key, value in data.items():
            print(" " * indent + f"{key}:")
            print_structure(value, indent + 2, max_depth)
    elif isinstance(data, list):
        print(" " * indent + f"[list of {len(data)} items]")
        if data and indent < max_depth:
            print_structure(data[0], indent + 2, max_depth)
    else:
        print(" " * indent + str(data))

# Add this new function to get a more reliable price
def get_current_price(symbol, analysis_results):
    """
    Get the current price for a symbol, with fallbacks.
    
    Args:
        symbol: Ticker symbol
        analysis_results: Dictionary containing analysis results
        
    Returns:
        float: Current price
    """
    # Try to get price from market_data in analysis_results
    if "market_data" in analysis_results and "currentPrice" in analysis_results["market_data"]:
        price = analysis_results["market_data"]["currentPrice"]
        if 10 <= price <= 10000:  # Sanity check for reasonable stock price
            return price
            
    # Try to get from greek_analysis if available
    if "greek_analysis" in analysis_results:
        greek_data = analysis_results["greek_analysis"]
        if "price_projections" in greek_data and "current_price" in greek_data["price_projections"]:
            price = greek_data["price_projections"]["current_price"]
            if 10 <= price <= 10000:
                return price
    
    # Try to get from a price file
    try:
        price_file = f"data/{symbol}_price.json"
        with open(price_file, "r") as f:
            price_data = json.load(f)
            if "price" in price_data:
                return float(price_data["price"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
        
    # Try to get from existing trade recommendation files
    try:
        recommendation_files = [
            f"results/{symbol}_trade_recommendation.json",
            f"results/{symbol}_enhanced_recommendation.json"
        ]
        
        for rec_file in recommendation_files:
            if os.path.exists(rec_file):
                with open(rec_file, "r") as f:
                    rec_data = json.load(f)
                    # Check different possible price fields
                    for field in ["Entry", "entry", "target_price", "Target"]:
                        if field in rec_data and rec_data[field] > 0:
                            price = float(rec_data[field])
                            if 10 <= price <= 10000:  # Sanity check
                                logging.info(f"Using price {price} from {rec_file}")
                                return price
    except Exception as e:
        logging.warning(f"Error getting price from recommendation files: {e}")
        
    # If all else fails, try to fetch from an API if available
    try:
        from api_fetcher import get_current_prices_polygon
        from config import POLYGON_API_KEY
        if POLYGON_API_KEY:
            prices = get_current_prices_polygon([symbol], POLYGON_API_KEY)
            if symbol in prices:
                return prices[symbol]
    except (ImportError, Exception) as e:
        logging.warning(f"Could not fetch current price for {symbol} from API: {e}")
    
    # Use a lookup table for common stocks if all else fails
    price_lookup = {
        "AAPL": 175.50,
        "MSFT": 410.30,
        "GOOGL": 170.20,
        "AMZN": 180.50,
        "META": 480.20,
        "TSLA": 180.30,
        "NVDA": 880.50,
        "JPM": 195.40,
        "V": 275.60,
        "JNJ": 150.20,
        "SPY": 520.30,
        "QQQ": 440.20
    }
    
    if symbol in price_lookup:
        logging.info(f"Using fallback price lookup for {symbol}: {price_lookup[symbol]}")
        return price_lookup[symbol]
    
    # Return a warning and use a placeholder that's clearly wrong
    logging.warning(f"Could not determine accurate price for {symbol}. Using placeholder.")
    return -1.0  # Clearly invalid price to signal an issue

def load_tickers_from_file(file_path):
    """
    Load ticker symbols from a file (CSV or TXT).
    
    Args:
        file_path: Path to file containing tickers
        
    Returns:
        List of ticker symbols
    """
    tickers = []
    
    try:
        if file_path.endswith('.csv'):
            import csv
            with open(file_path, 'r') as f:
                csv_reader = csv.reader(f)
                # Skip header row if it contains common header names
                first_row = next(csv_reader, None)
                if first_row:
                    # Check if first row looks like a header
                    if any(header.upper() in ['SYMBOL', 'TICKER', 'NAME', 'STOCK'] for header in first_row):
                        logging.info("Skipping header row in CSV file")
                    else:
                        # If it doesn't look like a header, process it as data
                        ticker = first_row[0].strip().upper()
                        if ticker and not ticker.startswith('#'):
                            tickers.append(ticker)
                
                # Process remaining rows
                for row in csv_reader:
                    if row and isinstance(row[0], str) and row[0].strip() and not row[0].startswith('#'):
                        ticker = row[0].strip().upper()
                        # Skip common header names that might appear in data
                        if ticker not in ['SYMBOL', 'TICKER', 'NAME', 'STOCK']:
                            tickers.append(ticker)
        elif file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                # Skip header line if it looks like a header
                if lines and lines[0].strip().upper() in ['SYMBOL', 'TICKER', 'NAME', 'STOCK']:
                    lines = lines[1:]
                
                for line in lines:
                    ticker = line.strip().upper()
                    if ticker and not ticker.startswith('#'):
                        # Skip common header names that might appear in data
                        if ticker not in ['SYMBOL', 'TICKER', 'NAME', 'STOCK']:
                            tickers.append(ticker)
        else:
            logging.error(f"Unsupported file format: {file_path}")
            return []
        
        logging.info(f"Loaded {len(tickers)} tickers from {file_path}")
        return tickers
    except Exception as e:
        logging.error(f"Error loading tickers from {file_path}: {e}")
        return []

def extract_trade_context(analysis_results, entropy_data, recommendation):
    """
    Extract trade context information from analysis results and recommendation.
    
    Args:
        analysis_results: Dictionary containing analysis results
        entropy_data: Dictionary containing entropy analysis results
        recommendation: Trade recommendation dictionary
        
    Returns:
        dict: Trade context information
    """
    trade_context = {}
    
    # Extract market regime information
    if "greek_analysis" in analysis_results and "market_regime" in analysis_results["greek_analysis"]:
        market_regime = analysis_results["greek_analysis"]["market_regime"]
        trade_context["market_regime"] = market_regime
    
    # Extract energy state information
    if "metrics" in entropy_data:
        metrics = entropy_data["metrics"]
        if "energy_state" in metrics:
            trade_context["energy_state"] = metrics["energy_state"]
        if "entropy_score" in metrics:
            trade_context["entropy_score"] = metrics["entropy_score"]
    
    # Extract anomalies if available
    if "anomalies" in entropy_data and entropy_data["anomalies"]:
        trade_context["anomalies"] = entropy_data["anomalies"]
    
    # Extract Greek metrics if available
    if "greek_analysis" in analysis_results:
        greek_data = analysis_results["greek_analysis"]
        greek_metrics = {}
        
        # Look for common Greek metrics
        for greek in ["delta", "gamma", "theta", "vega", "rho", "vanna", "charm"]:
            if greek in greek_data:
                greek_metrics[greek] = greek_data[greek]
        
        if greek_metrics:
            trade_context["greek_metrics"] = greek_metrics
    
    # Extract dominant Greek if available
    if "strategy" in recommendation and "description" in recommendation["strategy"]:
        description = recommendation["strategy"]["description"]
        if "dominated" in description:
            parts = description.split("-dominated")
            if len(parts) > 1:
                trade_context["dominant_greek"] = parts[0].strip().lower()
    
    # Extract volatility context
    vol_regime = "Normal"  # Default
    if "market_regime" in analysis_results.get("greek_analysis", {}) and "volatility" in analysis_results["greek_analysis"]["market_regime"]:
        vol_regime = analysis_results["greek_analysis"]["market_regime"]["volatility"]
    trade_context["volatility_regime"] = vol_regime
    
    # Extract support/resistance if available
    if "price_projections" in analysis_results.get("greek_analysis", {}):
        price_proj = analysis_results["greek_analysis"]["price_projections"]
        if "support_levels" in price_proj:
            trade_context["support_levels"] = price_proj["support_levels"]
        if "resistance_levels" in price_proj:
            trade_context["resistance_levels"] = price_proj["resistance_levels"]
    
    # Extract trade duration recommendation
    if "strategy" in recommendation and "hold_time" in recommendation["strategy"]:
        trade_context["hold_time_days"] = recommendation["strategy"]["hold_time"]
    else:
        # Default based on market regime
        primary_regime = analysis_results.get("greek_analysis", {}).get("market_regime", {}).get("primary", "").lower()
        if "bullish" in primary_regime:
            trade_context["hold_time_days"] = 10
        elif "bearish" in primary_regime:
            trade_context["hold_time_days"] = 5
        else:
            trade_context["hold_time_days"] = 7
    
    return trade_context

def check_trade_context(symbol):
    """
    Check if trade context is included in the recommendation file.
    
    Args:
        symbol: Ticker symbol
        
    Returns:
        bool: True if trade context is included, False otherwise
    """
    try:
        rec_file = f"results/{symbol}_trade_recommendation.json"
        if os.path.exists(rec_file):
            with open(rec_file, "r") as f:
                rec_data = json.load(f)
                if "TradeContext" in rec_data:
                    print(f"Trade context found in {rec_file}:")
                    print(f"  Keys: {list(rec_data['TradeContext'].keys())}")
                    return True
                else:
                    print(f"No trade context found in {rec_file}")
                    return False
        else:
            print(f"Recommendation file {rec_file} not found")
            return False
    except Exception as e:
        print(f"Error checking trade context: {e}")
        return False

def generate_summary_report(tickers, output_dir="results"):
    """
    Generate a summary report of all trade recommendations.
    
    Args:
        tickers: List of ticker symbols
        output_dir: Directory containing recommendation files
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "recommendations": []
    }
    
    for symbol in tickers:
        try:
            rec_file = f"{output_dir}/{symbol}_trade_recommendation.json"
            if os.path.exists(rec_file):
                with open(rec_file, "r") as f:
                    rec_data = json.load(f)
                    
                    # Extract key information
                    summary_rec = {
                        "Symbol": symbol,
                        "Action": rec_data.get("Action", "HOLD"),
                        "Strategy": rec_data.get("Strategy", "Unknown"),
                        "Entry": rec_data.get("Entry", 0.0),
                        "Target": rec_data.get("Target", 0.0),
                        "Stop": rec_data.get("Stop", 0.0),
                        "RiskReward": rec_data.get("RiskReward", 0.0),
                        "Regime": rec_data.get("Regime", "Unknown"),
                        "VolRegime": rec_data.get("VolRegime", "Normal"),
                        "HoldTimeDays": rec_data.get("HoldTimeDays", 0)
                    }
                    
                    summary["recommendations"].append(summary_rec)
        except Exception as e:
            logging.warning(f"Error processing {symbol} for summary: {e}")
    
    # Save summary report
    summary_file = f"{output_dir}/trade_recommendations_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Saved summary report to {summary_file}")
    
    # Print summary table
    print("\nTrade Recommendations Summary:")
    print(f"{'Symbol':<6} {'Action':<6} {'Strategy':<20} {'Entry':<8} {'Target':<8} {'Stop':<8} {'R/R':<5} {'Regime':<15}")
    print("-" * 80)
    
    for rec in summary["recommendations"]:
        print(f"{rec['Symbol']:<6} {rec['Action']:<6} {rec['Strategy'][:18]:<20} {rec['Entry']:<8.2f} {rec['Target']:<8.2f} {rec['Stop']:<8.2f} {rec['RiskReward']:<5.2f} {rec['Regime'][:13]:<15}")
    
    return summary_file

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Debug and generate trade recommendations")
    parser.add_argument("symbol", nargs="?", default=None, help="Ticker symbol (optional)")
    parser.add_argument("--tickers-file", default="data/instruments.csv", help="File containing ticker symbols")
    parser.add_argument("--output-dir", default="results", help="Output directory for recommendations")
    parser.add_argument("--dashboard-format", action="store_true", help="Save in dashboard-compatible format", default=True)  # Always enabled
    parser.add_argument("--price", type=float, help="Override current price")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tickers to process (0 for all)")
    parser.add_argument("--check-context", action="store_true", help="Check if trade context is included in recommendations")
    parser.add_argument("--summary", action="store_true", help="Generate a summary report of all recommendations")
    args = parser.parse_args()
    
    # Determine which tickers to process
    tickers = []
    if args.symbol:
        tickers = [args.symbol.upper()]
        logging.info(f"Processing single ticker: {args.symbol}")
    else:
        tickers = load_tickers_from_file(args.tickers_file)
        if not tickers:
            logging.error(f"No tickers found in {args.tickers_file}. Using default TSLA.")
            tickers = ["TSLA"]
        elif args.limit > 0 and len(tickers) > args.limit:
            tickers = tickers[:args.limit]
            logging.info(f"Limited to first {args.limit} tickers")
    
    logging.info(f"Processing {len(tickers)} tickers: {', '.join(tickers[:5])}" + 
                (f"... and {len(tickers)-5} more" if len(tickers) > 5 else ""))
    
    success_count = 0
    error_count = 0
    
    # Process each ticker
    for symbol in tickers:
        logging.info(f"\n{'='*50}\nProcessing {symbol}\n{'='*50}")
        
        try:
            # Check if analysis results exist for this symbol
            possible_paths = [
                f"results/{symbol}_analysis_results.json",
                f"results/{symbol}_analysis.json",
                f"results/{symbol}_greek_analysis.json"
            ]
            
            analysis_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    analysis_file = path
                    break
            
            if not analysis_file:
                logging.warning(f"No analysis results found for {symbol}. Skipping.")
                error_count += 1
                continue
            
            # Load analysis results
            with open(analysis_file, "r") as f:
                analysis_results = json.load(f)
                logging.info(f"Successfully loaded analysis from {analysis_file}")
            
            # Check analysis timestamp
            timestamp = analysis_results.get("timestamp", "")
            if timestamp:
                try:
                    analysis_date = datetime.fromisoformat(timestamp.split(".")[0])
                    days_old = (datetime.now() - analysis_date).days
                    if days_old > 0:
                        logging.warning(f"Analysis is {days_old} days old. Consider refreshing.")
                except (ValueError, TypeError):
                    pass
            
            # Try to load options data
            options_data = load_options_data(symbol)
            if options_data:
                analysis_results["options_data"] = options_data
                logging.info(f"Added options data to analysis results")
            
            # Load entropy data
            entropy_data = analysis_results.get("entropy_analysis", {})
            if not entropy_data:
                logging.warning("No entropy data found in analysis results. Using empty dict.")
                entropy_data = {"metrics": {}, "anomalies": []}
            
            # Print top-level keys in analysis_results
            print(f"\nTop-level keys in {symbol} analysis_results:")
            print(list(analysis_results.keys()))
            
            # Print content of greek_analysis
            greek_analysis = analysis_results.get("greek_analysis", {})
            print(f"\nContent of {symbol} greek_analysis:")
            if greek_analysis:
                print(f"greek_analysis is not empty, keys: {list(greek_analysis.keys())}")
            else:
                print("greek_analysis is empty or not present")
            
            # Validate the data
            print(f"Validating {symbol} analysis data...")
            is_valid = validate_analysis_data(analysis_results, entropy_data)
            print(f"Data validation result: {'Valid' if is_valid else 'Invalid'}")
            
            if not is_valid:
                logging.warning(f"Skipping {symbol} due to invalid data")
                error_count += 1
                continue
            
            # Get current price with better reliability
            current_price = args.price if args.price else get_current_price(symbol, analysis_results)
            print(f"\nCurrent price for {symbol}: {current_price}")
            
            # Generate trade recommendation
            print(f"\nGenerating trade recommendation for {symbol}...")
            recommendation = generate_trade_recommendation(
                analysis_results,
                entropy_data,
                current_price
            )
            
            # Print recommendation
            print(f"\nTrade Recommendation for {symbol}:")
            print_structure(recommendation)
            
            # Save recommendation to file
            output_file = f"{args.output_dir}/{symbol}_trade_recommendation_debug.json"
            os.makedirs(args.output_dir, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(recommendation, f, indent=2)
            print(f"\nSaved {symbol} recommendation to {output_file}")
            
            # Save dashboard-compatible recommendation if requested
            if args.dashboard_format:
                dashboard_file = save_dashboard_compatible_recommendation(
                    symbol, 
                    recommendation, 
                    args.output_dir,
                    current_price,
                    analysis_results,
                    entropy_data
                )
                print(f"\nSaved dashboard-compatible recommendation to {dashboard_file}")
            
            success_count += 1
            
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}", exc_info=True)
            error_count += 1
    
    # Print summary
    logging.info(f"\n{'='*50}\nProcessing complete\n{'='*50}")
    logging.info(f"Successfully processed {success_count} of {len(tickers)} tickers")
    if error_count > 0:
        logging.warning(f"Encountered errors with {error_count} tickers")
    
    # Check trade context if requested
    if args.check_context and tickers:
        print("\nChecking trade context in recommendations...")
        for symbol in tickers[:3]:  # Check first 3 tickers
            check_trade_context(symbol)
    
    # Generate summary report
    if args.summary and tickers:
        generate_summary_report(tickers, args.output_dir)
    
    return 0 if error_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())


