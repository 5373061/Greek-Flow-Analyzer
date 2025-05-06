"""
Market Regime Analyzer

Analyzes market regimes across multiple instruments and validates them against price data.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import argparse

# Configure logging
logger = logging.getLogger(__name__)

class MarketRegimeAnalyzer:
    """Analyzes market regimes across multiple instruments."""
    
    def __init__(self, results_dir="results", data_dir="data", config=None):
        """Initialize the analyzer with configuration."""
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.config = config or {
            'regime_thresholds': {
                'highVolatility': 0.3,
                'lowVolatility': 0.15,
                'strongBullish': 0.7,
                'strongBearish': -0.7,
                'neutralZone': 0.2
            }
        }
        self.regime_data = {}
        self.price_data = {}
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.join(results_dir, "market_regime"), exist_ok=True)
        
        logger.info(f"Market Regime Analyzer initialized with results dir: {results_dir}")

    def run(self, results_dir=None, validate=False, fetch_missing=False, days=30):
        """
        Run the market regime analyzer.
        
        Args:
            results_dir: Directory containing analysis results
            validate: Whether to validate regimes against price data
            fetch_missing: Whether to fetch missing price data (ignored in simplified version)
            days: Number of days of price history to use for validation
        
        Returns:
            Dictionary with analysis results
        """
        try:
            # Use instance results_dir if not provided
            if results_dir is None:
                results_dir = self.results_dir
            
            # Load regime data from analysis files
            self.load_analysis_results(results_dir)
            logger.info(f"Loaded regime data for {len(self.regime_data)} symbols")
            
            # Generate regime summary
            summary = self.generate_regime_summary()
            logger.info("Generated regime summary")
            
            # Generate regime table
            table = self.generate_regime_table()
            logger.info("Generated regime table")
            
            # Validate regimes if requested
            validation_results = None
            if validate:
                logger.info("Validating regimes against price data")
                validation_results = self.validate_regimes(days=days)
                logger.info("Validation complete")
            
            # Return results
            results = {
                "summary": summary,
                "table": table
            }
            
            if validation_results:
                results["validation"] = validation_results
            
            return results
        
        except Exception as e:
            logger.error(f"Error in market regime analysis: {e}")
            raise

    def validate_regimes(self, days=30):
        """
        Validate regimes against price data.
        
        Args:
            days: Number of days of price history to use for validation
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating market regimes against price data")
        
        # Load price data
        self.load_price_data()
        
        # Initialize validation results
        validation_results = {
            "volatility_validation": {
                "matches": 0, 
                "total": 0, 
                "match_percentage": 0,
                "symbols": {}
            },
            "directional_validation": {
                "matches": 0, 
                "total": 0, 
                "match_percentage": 0,
                "symbols": {}
            },
            "overall_confidence": 0
        }
        
        # Validate each symbol
        for symbol, regime in self.regime_data.items():
            # Skip if no price data
            if symbol not in self.price_data:
                logger.warning(f"No price data for {symbol}, skipping validation")
                continue
            
            # Get price data
            price_data = self.price_data[symbol]
            
            # Validate volatility regime
            expected_vol = regime.get("volatility_regime", "Unknown")
            actual_vol = self._calculate_volatility_regime(price_data)
            
            # Add to volatility validation
            validation_results["volatility_validation"]["total"] += 1
            validation_results["volatility_validation"]["symbols"][symbol] = {
                "expected": expected_vol,
                "actual": actual_vol,
                "match": expected_vol == actual_vol
            }
            
            if expected_vol == actual_vol:
                validation_results["volatility_validation"]["matches"] += 1
            
            # Validate directional regime
            expected_dir = regime.get("primary_regime", "Unknown")
            actual_dir = self._calculate_directional_regime(price_data)
            
            # Add to directional validation
            validation_results["directional_validation"]["total"] += 1
            validation_results["directional_validation"]["symbols"][symbol] = {
                "expected": expected_dir,
                "actual": actual_dir,
                "match": expected_dir == actual_dir
            }
            
            if expected_dir == actual_dir:
                validation_results["directional_validation"]["matches"] += 1
        
        # Calculate match percentages
        if validation_results["volatility_validation"]["total"] > 0:
            validation_results["volatility_validation"]["match_percentage"] = (
                validation_results["volatility_validation"]["matches"] / 
                validation_results["volatility_validation"]["total"] * 100
            )
        
        if validation_results["directional_validation"]["total"] > 0:
            validation_results["directional_validation"]["match_percentage"] = (
                validation_results["directional_validation"]["matches"] / 
                validation_results["directional_validation"]["total"] * 100
            )
        
        # Calculate overall confidence
        if (validation_results["volatility_validation"]["total"] > 0 or 
            validation_results["directional_validation"]["total"] > 0):
            validation_results["overall_confidence"] = (
                (validation_results["volatility_validation"]["match_percentage"] + 
                 validation_results["directional_validation"]["match_percentage"]) / 2
            )
        
        return validation_results
    
    def _calculate_volatility_regime(self, price_data):
        """
        Calculate volatility regime based on price data.
        
        Args:
            price_data (DataFrame): Price data
            
        Returns:
            str: Volatility regime ('High', 'Normal', or 'Low')
        """
        # Calculate historical volatility
        hist_vol = self._calculate_historical_volatility(price_data)
        
        # Determine volatility regime based on thresholds
        high_vol_threshold = self.config.get('regime_thresholds', {}).get('highVolatility', 0.3)
        low_vol_threshold = self.config.get('regime_thresholds', {}).get('lowVolatility', 0.15)
        
        if hist_vol >= high_vol_threshold:
            return "High"
        elif hist_vol <= low_vol_threshold:
            return "Low"
        else:
            return "Normal"
    
    def _calculate_directional_regime(self, price_data):
        """
        Calculate directional regime based on price data.
        
        Args:
            price_data (DataFrame): Price data
            
        Returns:
            str: Directional regime ('Bullish', 'Bearish', or 'Neutral')
        """
        # Calculate momentum and trend
        momentum = self._calculate_momentum(price_data)
        trend = self._calculate_trend(price_data)
        rsi = self._calculate_rsi(price_data)
        
        # Normalize RSI to -1 to 1 scale
        rsi_normalized = (rsi - 50) / 50
        
        # Calculate composite direction score (weighted average)
        direction_score = (
            momentum * 0.4 +  # 40% weight
            trend * 0.4 +     # 40% weight
            rsi_normalized * 0.2  # 20% weight
        )
        
        # Determine regime based on thresholds
        strong_bullish = self.config.get('regime_thresholds', {}).get('strongBullish', 0.7)
        strong_bearish = self.config.get('regime_thresholds', {}).get('strongBearish', -0.7)
        neutral_zone = self.config.get('regime_thresholds', {}).get('neutralZone', 0.2)
        
        if direction_score >= strong_bullish:
            return "Bullish"
        elif direction_score > neutral_zone:
            return "Bullish"
        elif direction_score <= strong_bearish:
            return "Bearish"
        elif direction_score < -neutral_zone:
            return "Bearish"
        else:
            return "Neutral"
    
    def _calculate_historical_volatility(self, price_data, window=20):
        """Calculate historical volatility from price data."""
        try:
            # Get close prices
            if 'Close' in price_data.columns:
                close_prices = price_data['Close']
            elif 'close' in price_data.columns:
                close_prices = price_data['close']
            else:
                logger.warning("No close price column found, using first available column")
                close_prices = price_data.iloc[:, 0]
            
            # Calculate returns
            returns = close_prices.pct_change().dropna()
            
            # Calculate volatility (annualized)
            volatility = returns.std() * np.sqrt(252)
            
            return volatility
        except Exception as e:
            logger.error(f"Error calculating historical volatility: {e}")
            return 0.2  # Default to medium volatility
    
    def _calculate_momentum(self, price_data, window=20):
        """Calculate price momentum."""
        try:
            # Get close prices
            if 'Close' in price_data.columns:
                close_prices = price_data['Close']
            elif 'close' in price_data.columns:
                close_prices = price_data['close']
            else:
                logger.warning("No close price column found, using first available column")
                close_prices = price_data.iloc[:, 0]
            
            # Calculate momentum (normalized)
            if len(close_prices) >= window:
                momentum = (close_prices.iloc[-1] / close_prices.iloc[-window] - 1)
                return momentum
            else:
                return 0
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0
    
    def _calculate_trend(self, price_data, window=20):
        """Calculate price trend direction."""
        try:
            # Get close prices
            if 'Close' in price_data.columns:
                close_prices = price_data['Close']
            elif 'close' in price_data.columns:
                close_prices = price_data['close']
            else:
                logger.warning("No close price column found, using first available column")
                close_prices = price_data.iloc[:, 0]
            
            # Calculate simple moving average
            if len(close_prices) >= window:
                sma = close_prices.rolling(window=window).mean()
                # Determine trend direction
                current_price = close_prices.iloc[-1]
                current_sma = sma.iloc[-1]
                
                # Return normalized trend (-1 to 1)
                if current_price > current_sma:
                    return 1.0  # Uptrend
                else:
                    return -1.0  # Downtrend
            else:
                return 0.0  # Neutral
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return 0.0
    
    def _calculate_rsi(self, price_data, window=14):
        """Calculate Relative Strength Index."""
        try:
            # Get close prices
            if 'Close' in price_data.columns:
                close_prices = price_data['Close']
            elif 'close' in price_data.columns:
                close_prices = price_data['close']
            else:
                logger.warning("No close price column found, using first available column")
                close_prices = price_data.iloc[:, 0]
            
            # Calculate RSI
            delta = close_prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50  # Default to neutral
    
    def load_analysis_results(self, results_dir=None):
        """
        Load regime data from analysis files.
        
        Args:
            results_dir: Directory containing analysis results
            
        Returns:
            dict: Dictionary with regime data for each symbol
        """
        if results_dir is None:
            results_dir = self.results_dir
        
        logger.info(f"Loading analysis results from {results_dir}")
        
        # Initialize regime data
        self.regime_data = {}
        
        # Find all potential analysis files in the results directory
        analysis_files = []
        
        # Look for files with various naming patterns
        for file in os.listdir(results_dir):
            if file.endswith("_analysis.json") or file.endswith("_analysis_results.json") or file.endswith("_greek_analysis.json"):
                analysis_files.append(os.path.join(results_dir, file))
        
        # Check if analysis subdirectory exists and look there too
        analysis_dir = os.path.join(results_dir, "analysis")
        if os.path.exists(analysis_dir) and os.path.isdir(analysis_dir):
            for file in os.listdir(analysis_dir):
                if file.endswith("_analysis.json") or file.endswith("_analysis_results.json") or file.endswith("_greek_analysis.json"):
                    analysis_files.append(os.path.join(analysis_dir, file))
        
        if not analysis_files:
            logger.warning(f"No analysis files found in {results_dir} or its analysis subdirectory")
            return self.regime_data
        
        logger.info(f"Found {len(analysis_files)} analysis files")
        
        # Process each analysis file
        for file_path in analysis_files:
            try:
                # Extract symbol from filename
                file_name = os.path.basename(file_path)
                symbol = file_name.split('_')[0]
                
                logger.info(f"Processing {file_path} for symbol {symbol}")
                
                # Load analysis data
                with open(file_path, 'r') as f:
                    analysis = json.load(f)
                
                # Extract regime data - handle different possible structures
                market_regime = None
                
                # Try different paths to find market_regime
                if "greek_analysis" in analysis and "market_regime" in analysis["greek_analysis"]:
                    market_regime = analysis["greek_analysis"]["market_regime"]
                elif "analysis_results" in analysis and "market_regime" in analysis["analysis_results"]:
                    market_regime = analysis["analysis_results"]["market_regime"]
                elif "formatted_results" in analysis and "market_regime" in analysis.get("formatted_results", {}):
                    market_regime = analysis["formatted_results"]["market_regime"]
                elif "greek_analysis" in analysis and "formatted_results" in analysis["greek_analysis"] and "market_regime" in analysis["greek_analysis"]["formatted_results"]:
                    market_regime = analysis["greek_analysis"]["formatted_results"]["market_regime"]
                
                if market_regime:
                    # Create regime data entry
                    self.regime_data[symbol] = {
                        "primary_regime": market_regime.get("primary_label", "Unknown"),
                        "secondary_regime": market_regime.get("secondary_label", "Unknown"),
                        "volatility_regime": market_regime.get("volatility_regime", "Unknown"),
                        "dominant_greek": market_regime.get("dominant_greek", "Unknown"),
                        "greek_magnitudes": market_regime.get("greek_magnitudes", {})
                    }
                    
                    logger.info(f"Loaded regime data for {symbol}")
                else:
                    logger.warning(f"No market regime data found in {file_path}")
                    
                    # Debug: Print the structure of the file to help diagnose
                    def print_keys(d, prefix=""):
                        if isinstance(d, dict):
                            for k, v in d.items():
                                if isinstance(v, (dict, list)):
                                    logger.debug(f"{prefix}{k}: {type(v)}")
                                    if isinstance(v, dict):
                                        print_keys(v, prefix + "  ")
                                else:
                                    logger.debug(f"{prefix}{k}: {type(v)}")
                    
                    logger.debug(f"Structure of {file_path}:")
                    print_keys(analysis)
            
            except Exception as e:
                logger.error(f"Error loading analysis file {file_path}: {e}")
        
        logger.info(f"Loaded regime data for {len(self.regime_data)} symbols")
        return self.regime_data
    
    def load_price_data(self):
        """
        Load price data for each symbol in the regime data.
        
        Returns:
            dict: Dictionary with price data for each symbol
        """
        logger.info("Loading price data")
        
        # Initialize price data dictionary
        self.price_data = {}
        
        # Load price data for each symbol
        for symbol in self.regime_data.keys():
            # Check if we have price data
            price_path = os.path.join(self.data_dir, "price_history", f"{symbol}_daily.csv")
            
            if os.path.exists(price_path):
                try:
                    # Load from file
                    price_df = pd.read_csv(price_path)
                    
                    # Handle date column
                    date_col = None
                    for col in ['date', 'Date', 'timestamp', 'Timestamp']:
                        if col in price_df.columns:
                            date_col = col
                            break
                    
                    if date_col:
                        price_df[date_col] = pd.to_datetime(price_df[date_col])
                        price_df.set_index(date_col, inplace=True)
                    
                    self.price_data[symbol] = price_df
                    logger.info(f"Loaded price data for {symbol}")
                    
                except Exception as e:
                    logger.warning(f"Error loading price data for {symbol}: {e}")
            else:
                logger.warning(f"No price data file found for {symbol}")
        
        return self.price_data
    
    def generate_regime_summary(self):
        """
        Generate a summary of market regimes.
        
        Returns:
            Dictionary with regime summary
        """
        logger.info("Generating regime summary")
        
        # Initialize summary
        summary = {
            "primary_regimes": {},
            "volatility_regimes": {},
            "dominant_greeks": {},
            "total_symbols": len(self.regime_data)
        }
        
        # Count regimes
        for symbol, regime in self.regime_data.items():
            # Count primary regimes
            primary = regime.get("primary_regime", "Unknown")
            summary["primary_regimes"][primary] = summary["primary_regimes"].get(primary, 0) + 1
            
            # Count volatility regimes
            volatility = regime.get("volatility_regime", "Unknown")
            summary["volatility_regimes"][volatility] = summary["volatility_regimes"].get(volatility, 0) + 1
            
            # Count dominant greeks
            greek = regime.get("dominant_greek", "Unknown")
            summary["dominant_greeks"][greek] = summary["dominant_greeks"].get(greek, 0) + 1
        
        return summary
    
    def generate_regime_table(self):
        """
        Generate a table of market regimes.
        
        Returns:
            DataFrame with regime data
        """
        logger.info("Generating regime table")
        
        # Create table data
        table_data = []
        
        for symbol, regime in self.regime_data.items():
            table_data.append({
                "Symbol": symbol,
                "Primary Regime": regime.get("primary_regime", "Unknown"),
                "Volatility Regime": regime.get("volatility_regime", "Unknown"),
                "Dominant Greek": regime.get("dominant_greek", "Unknown")
            })
        
        # Create DataFrame
        table = pd.DataFrame(table_data)
        
        return table

def main():
    """Main function to run the market regime analyzer."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze market regimes across multiple instruments")
    parser.add_argument("--results-dir", default="results", help="Directory containing analysis results")
    parser.add_argument("--output", default="market_regime", help="Output directory for regime analysis")
    parser.add_argument("--validate", action="store_true", help="Validate regimes against price data")
    parser.add_argument("--days", type=int, default=30, help="Number of days for validation")
    args = parser.parse_args()
    
    try:
        # Load configuration
        try:
            import config
            logger.info("Loaded configuration from config.py")
        except ImportError:
            logger.warning("config.py not found, using default configuration")
        
        # Initialize analyzer
        analyzer = MarketRegimeAnalyzer()
        
        # Run analysis
        results = analyzer.run(
            results_dir=args.results_dir,
            validate=args.validate,
            days=args.days
        )
        
        # Create output directory
        output_dir = os.path.join(args.results_dir, args.output)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        with open(os.path.join(output_dir, "regime_summary.txt"), "w") as f:
            f.write(json.dumps(results["summary"], indent=2))  # Convert dict to string
        
        with open(os.path.join(output_dir, "regime_table.txt"), "w") as f:
            f.write(results["table"].to_string())  # Convert DataFrame to string
        
        # Save validation results if available
        if "validation" in results:
            with open(os.path.join(output_dir, "validation_results.json"), "w") as f:
                json.dump(results["validation"], f, indent=2)
        
        logger.info(f"Market regime analysis completed. Results saved to {output_dir}")
        return 0
    
    except Exception as e:
        logger.error(f"Error in market regime analysis: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)








