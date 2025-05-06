"""
Greek Energy Flow Analysis - Main Integration Script

This script integrates all components of the Greek Energy Flow analysis system:
- Greek Energy Flow Analysis
- Momentum Analysis
- Entropy Analysis (if available)
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import os

# Import our analyzers
from greek_flow.flow import GreekEnergyFlow
from greek_flow.flow import GreekEnergyAnalyzer

# Add these imports at the top
from utils.instrument_loader import get_instrument_list
from analysis.pipeline_manager import AnalysisPipeline

# Add this import for entropy analysis
from entropy_analyzer.entropy_analyzer import EntropyAnalyzer
from entropy_analyzer.risk_manager import AdvancedRiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_options_data(symbol, data_path=None):
    """
    Load options data for a given symbol.
    If data_path is provided, load from file, otherwise generate sample data.
    """
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading options data for {symbol} from {data_path}")
        try:
            options_df = pd.read_csv(data_path)
            # Convert date strings to datetime objects
            if 'expiration' in options_df.columns:
                options_df['expiration'] = pd.to_datetime(options_df['expiration'])
            return options_df
        except Exception as e:
            logger.error(f"Error loading options data: {e}")
            return None
    else:
        logger.info(f"Generating sample options data for {symbol}")
        return generate_sample_options_data(symbol)

def generate_sample_options_data(symbol, output_file=None, count=100, trend="neutral", 
                                iv_range=(0.2, 0.5)):
    """Generate sample options data for testing.
    
    Args:
        symbol: Stock symbol
        output_file: Path to save CSV file (if None, returns DataFrame)
        count: Number of options to generate
        trend: Market trend ('bullish', 'bearish', or 'neutral')
        iv_range: Tuple of (min_iv, max_iv) for implied volatility range
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    np.random.seed(42)  # For reproducibility
    
    # Base parameters
    current_price = 100.0
    if trend == "bullish":
        current_price = 120.0  # Higher current price for bullish trend
    elif trend == "bearish":
        current_price = 80.0   # Lower current price for bearish trend
    
    # Generate strike prices around current price
    min_strike = current_price * 0.7
    max_strike = current_price * 1.3
    
    # Generate expiration dates
    today = datetime.now().date()
    expirations = [
        today + timedelta(days=30),  # 1 month
        today + timedelta(days=60),  # 2 months
        today + timedelta(days=90)   # 3 months
    ]
    
    # Generate options data
    data = []
    for _ in range(count):
        strike = np.random.uniform(min_strike, max_strike)
        expiration = np.random.choice(expirations)
        option_type = np.random.choice(["call", "put"])
        
        # Calculate days to expiration
        dte = (expiration - today).days
        
        # Generate implied volatility based on trend
        if trend == "bullish":
            # Lower IV for bullish trend
            iv = np.random.uniform(iv_range[0] * 0.8, iv_range[1] * 0.8)
        elif trend == "bearish":
            # Higher IV for bearish trend
            iv = np.random.uniform(iv_range[0] * 1.2, iv_range[1] * 1.2)
        else:
            # Neutral trend
            iv = np.random.uniform(iv_range[0], iv_range[1])
        
        # Calculate option greeks (simplified)
        if option_type == "call":
            delta = max(0.01, min(0.99, 0.5 + 0.1 * (current_price / strike - 1) * np.sqrt(30 / max(1, dte))))
        else:
            delta = max(0.01, min(0.99, 0.5 - 0.1 * (current_price / strike - 1) * np.sqrt(30 / max(1, dte))))
        
        gamma = 0.01 * np.exp(-0.5 * ((strike / current_price - 1) / 0.1) ** 2)
        theta = -0.01 * iv * strike * np.sqrt(dte / 365)
        vega = 0.1 * strike * np.sqrt(dte / 365)
        
        # Add option to data
        data.append({
            "symbol": f"{symbol}{expiration.strftime('%y%m%d')}{option_type[0].upper()}{int(strike)}",
            "underlying": symbol,
            "strike": strike,
            "expiration": expiration,
            "type": option_type,
            "impliedVolatility": iv,
            "openInterest": np.random.randint(10, 1000),
            "volume": np.random.randint(1, 500),
            "delta": delta if option_type == "call" else -delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega
        })
    
    # Create DataFrame
    options_df = pd.DataFrame(data)
    
    # If output file is specified, save to CSV
    if output_file:
        options_df.to_csv(output_file, index=False)
        return output_file
    
    return options_df

def generate_sample_price_history(symbol, output_file=None, trend="neutral", 
                                 volatility=0.2, gap_percent=None, days=90):
    """Generate sample price history with specified market conditions.
    
    Args:
        symbol: Stock symbol
        output_file: Path to save CSV file (if None, returns DataFrame)
        trend: Market trend ('bullish', 'bearish', or 'neutral')
        volatility: Daily volatility level
        gap_percent: If specified, adds a price gap of this percentage
        days: Number of days of price history to generate
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    np.random.seed(42)  # For reproducibility
    
    # Base parameters
    starting_price = 100.0
    
    # Trend parameters
    if trend == "bullish":
        daily_drift = 0.001  # Positive drift
    elif trend == "bearish":
        daily_drift = -0.001  # Negative drift
    else:  # neutral
        daily_drift = 0.0
    
    # Generate price series
    dates = pd.date_range(end=datetime.now(), periods=days)
    daily_returns = np.random.normal(daily_drift, volatility / np.sqrt(252), days)
    
    # Add gap if specified
    if gap_percent:
        # Add gap at 2/3 of the way through the series
        gap_day = int(days * 2/3)
        daily_returns[gap_day] = gap_percent
    
    # Cumulative returns
    cum_returns = np.cumprod(1 + daily_returns)
    prices = starting_price * cum_returns
    
    # Create price history DataFrame
    price_data = pd.DataFrame({
        'date': dates,
        'symbol': symbol,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days),
        'open': prices * (1 + np.random.normal(0, 0.005, days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, days)))
    })
    
    # Ensure high >= open >= close >= low
    for i in range(len(price_data)):
        price_data.loc[i, 'high'] = max(price_data.loc[i, 'high'], 
                                        price_data.loc[i, 'open'], 
                                        price_data.loc[i, 'close'])
        price_data.loc[i, 'low'] = min(price_data.loc[i, 'low'], 
                                       price_data.loc[i, 'open'], 
                                       price_data.loc[i, 'close'])
        if price_data.loc[i, 'open'] > price_data.loc[i, 'close']:
            # Bearish candle
            price_data.loc[i, 'open'] = min(price_data.loc[i, 'open'], price_data.loc[i, 'high'])
            price_data.loc[i, 'close'] = max(price_data.loc[i, 'close'], price_data.loc[i, 'low'])
        else:
            # Bullish candle
            price_data.loc[i, 'close'] = min(price_data.loc[i, 'close'], price_data.loc[i, 'high'])
            price_data.loc[i, 'open'] = max(price_data.loc[i, 'open'], price_data.loc[i, 'low'])
    
    # If output file is specified, save to CSV
    if output_file:
        price_data.to_csv(output_file, index=False)
        return output_file
    
    return price_data

def run_analysis(symbol, options_file, price_file, output_dir, skip_entropy=False, config_file=None):
    """Run the full analysis pipeline.
    
    Args:
        symbol: Stock symbol
        options_file: Path to options data CSV
        price_file: Path to price history CSV
        output_dir: Directory to save output files
        skip_entropy: Whether to skip entropy analysis
        config_file: Path to custom configuration file
    
    Returns:
        Dictionary with analysis results or None if analysis failed
    """
    import os
    import json
    import pandas as pd
    import logging
    from datetime import datetime
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load options data
        options_data = pd.read_csv(options_file)
        
        # Load price history
        price_history = pd.read_csv(price_file)
        
        # Load configuration if provided
        config = None
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        # Import components here to avoid circular imports
        try:
            from greek_flow.flow import GreekEnergyFlow, GreekEnergyAnalyzer
            from entropy_analyzer.entropy_analyzer import EntropyAnalyzer
        except ModuleNotFoundError:
            # Fallback for direct imports when running as standalone
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from greek_flow.flow import GreekEnergyFlow, GreekEnergyAnalyzer
            from entropy_analyzer.entropy_analyzer import EntropyAnalyzer
        
        # Initialize Greek Energy Flow analyzer
        greek_analyzer = GreekEnergyAnalyzer(symbol, options_data, price_history, config=config)
        
        # Run Greek analysis
        greek_results = greek_analyzer.analyze()
        
        # Calculate chain energy
        chain_energy = greek_analyzer.calculate_chain_energy()
        
        # Initialize results dictionary
        results = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "greek_analysis": greek_results,
            "chain_energy": chain_energy
        }
        
        # Run entropy analysis if not skipped
        if not skip_entropy:
            try:
                # Get entropy configuration
                entropy_config = config.get("entropy_config", {}) if config else {}
                entropy_config["visualization_dir"] = os.path.join(output_dir, "entropy_viz")
                
                # Initialize entropy analyzer
                entropy_analyzer = EntropyAnalyzer(
                    options_data, 
                    greek_analyzer.get_historical_data(),
                    config=entropy_config
                )
                
                # Run entropy analysis
                entropy_metrics = entropy_analyzer.analyze_greek_entropy()
                
                # Detect anomalies
                anomalies = entropy_analyzer.detect_anomalies()
                
                # Generate report
                report = entropy_analyzer.generate_entropy_report()
                
                # Add entropy results
                results["entropy_analysis"] = {
                    **entropy_metrics,
                    "anomalies": anomalies,
                    "report": report
                }
            except Exception as e:
                logging.error(f"Entropy analysis failed: {e}")
                results["entropy_analysis"] = {"skipped": True, "error": str(e)}
        else:
            results["entropy_analysis"] = {"skipped": True}
        
        # Save results to file
        results_file = os.path.join(output_dir, f"{symbol}_analysis_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        return None

def run_batch_analysis(symbols_or_file, options_data_dir=None, price_data_dir=None, output_dir=None, skip_entropy=False):
    """
    Run analysis on multiple instruments.
    
    Args:
        symbols_or_file: Either a list of symbols or path to CSV file containing instruments
        options_data_dir: Directory containing options data files (optional)
        price_data_dir: Directory containing price history files (optional)
        output_dir: Directory to save output files (optional)
        skip_entropy: If True, skip entropy analysis
    """
    # Determine if input is a file path or list of symbols
    if isinstance(symbols_or_file, str):
        # It's a file path
        symbols = get_instrument_list(symbols_or_file)
    else:
        # It's already a list of symbols
        symbols = symbols_or_file
    
    if not symbols:
        logger.error("No valid instruments found")
        return
    
    logger.info(f"Running batch analysis on {len(symbols)} instruments")
    
    results = {}
    for symbol in symbols:
        logger.info(f"Analyzing {symbol}...")
        
        # Determine file paths for this symbol
        options_path = None
        if options_data_dir:
            # Look for options file with pattern: SYMBOL_options*.csv
            potential_files = [
                os.path.join(options_data_dir, f) 
                for f in os.listdir(options_data_dir) 
                if f.lower().startswith(symbol.lower() + '_options') and f.endswith('.csv')
            ]
            if potential_files:
                options_path = potential_files[0]
        
        price_path = None
        if price_data_dir:
            # Look for price file with pattern: SYMBOL_price*.csv
            potential_files = [
                os.path.join(price_data_dir, f) 
                for f in os.listdir(price_data_dir) 
                if f.lower().startswith(symbol.lower() + '_price') and f.endswith('.csv')
            ]
            if potential_files:
                price_path = potential_files[0]
        
        # Run analysis for this symbol
        try:
            symbol_results = run_analysis(symbol, options_path, price_path, output_dir, skip_entropy)
            results[symbol] = symbol_results
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            results[symbol] = {"error": str(e)}
    
    # Save summary of all results
    if output_dir:
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(output_dir, f"batch_analysis_summary_{timestamp}.json")
        
        # Create a simplified summary
        summary = {}
        for symbol, result in results.items():
            if result and "error" not in result:
                summary[symbol] = {
                    "current_price": result.get("current_price"),
                    "suggested_action": result.get("trade_opportunities", {}).get("suggested_action", "No action"),
                    "market_regime": result.get("greek_analysis", {}).get("market_regime", {}).get("primary", "Unknown")
                }
            else:
                summary[symbol] = {"error": result.get("error", "Analysis failed")}
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Batch analysis summary saved to {summary_file}")
    
    return results

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Greek Energy Flow Analysis Tool")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Analysis mode")
    
    # Single symbol analysis
    single_parser = subparsers.add_parser("single", help="Analyze a single symbol")
    single_parser.add_argument("symbol", help="Stock symbol to analyze")
    single_parser.add_argument("--options", help="Path to options data CSV file")
    single_parser.add_argument("--prices", help="Path to price history CSV file")
    single_parser.add_argument("--no-entropy", action="store_true", help="Disable entropy analysis")
    
    # Batch analysis
    batch_parser = subparsers.add_parser("batch", help="Analyze multiple symbols from a CSV file")
    batch_parser.add_argument("instruments", help="Path to CSV file containing instruments")
    batch_parser.add_argument("--options-dir", help="Directory containing options data CSV files")
    batch_parser.add_argument("--prices-dir", help="Directory containing price history CSV files")
    batch_parser.add_argument("--no-entropy", action="store_true", help="Disable entropy analysis")
    
    # Common arguments
    parser.add_argument("--output", help="Directory to save output files")
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Run appropriate analysis mode
    if args.mode == "batch":
        run_batch_analysis(args.instruments, args.options_dir, args.prices_dir, args.output, 
                          skip_entropy=args.no_entropy)
    else:  # Default to single mode
        if hasattr(args, 'symbol'):
            run_analysis(args.symbol, args.options, args.prices, args.output,
                        skip_entropy=args.no_entropy)
        else:
            parser.print_help()

if __name__ == "__main__":
    main()







