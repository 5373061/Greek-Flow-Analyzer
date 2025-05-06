"""
Pipeline Manager for Greek Energy Flow Analysis.
Coordinates the flow of data between different analysis components.
"""

import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import analysis components
try:
    from greek_flow.flow import GreekEnergyFlow, GreekEnergyAnalyzer
except ModuleNotFoundError:
    # Try relative import
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from greek_flow.flow import GreekEnergyFlow, GreekEnergyAnalyzer

try:
    from entropy_analyzer.entropy_analyzer import EntropyAnalyzer
except ModuleNotFoundError:
    # Create a dummy EntropyAnalyzer class if the module is not found
    class EntropyAnalyzer:
        def __init__(self, *args, **kwargs):
            pass
        
        def analyze_greek_entropy(self):
            return {"error": "EntropyAnalyzer module not found"}
        
        def detect_anomalies(self):
            return []
        
        def generate_entropy_report(self):
            return {"error": "EntropyAnalyzer module not found"}

logger = logging.getLogger(__name__)

class AnalysisPipeline:
    """
    Manages the flow of data between different analysis components.
    Provides a unified interface for running the full analysis pipeline.
    """
    
    def __init__(self, config=None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        self.greek_analyzer = GreekEnergyFlow(self.config.get("greek_config", {}))
        
        # Set up cache directory
        self.cache_dir = self.config.get("cache_dir", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_data(self, symbol, options_path=None, price_path=None):
        """Load and preprocess data for analysis."""
        try:
            # Load options data
            logger.info(f"Loading options data for {symbol} from {options_path}")
            options_df = pd.read_csv(options_path)
            
            # Load price history
            logger.info(f"Loading price history for {symbol} from {price_path}")
            price_history = pd.read_csv(price_path)
            
            # Validate required columns
            required_columns = ['strike', 'expiration', 'type', 'openInterest']
            missing_columns = [col for col in required_columns if col not in options_df.columns]
            if missing_columns:
                logger.error(f"Missing required columns in options data: {missing_columns}")
                return None, None, None
            
            # Create market data dictionary
            market_data = {
                'symbol': symbol,
                'currentPrice': price_history['close'].iloc[-1],
                'historicalVolatility': price_history['close'].pct_change().std() * (252 ** 0.5),
                'riskFreeRate': 0.04,  # Default risk-free rate
            }
            
            # Add implied volatility if available
            if 'impliedVolatility' in options_df.columns:
                market_data['impliedVolatility'] = options_df['impliedVolatility'].median()
            else:
                # Use historical volatility as fallback
                market_data['impliedVolatility'] = market_data['historicalVolatility']
                logger.warning(f"No impliedVolatility column found, using historical volatility")
            
            return options_df, price_history, market_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None, None, None
    
    def run_greek_analysis(self, options_df, market_data):
        """
        Run Greek Energy Flow analysis.
        
        Args:
            options_df: Options data DataFrame
            market_data: Market data dictionary
            
        Returns:
            Dictionary of Greek analysis results
        """
        logger.info("Running Greek Energy Flow analysis")
        greek_results = self.greek_analyzer.analyze_greek_profiles(options_df, market_data)
        formatted_results = GreekEnergyAnalyzer.format_results(greek_results)
        chain_energy = GreekEnergyAnalyzer.analyze_chain_energy(options_df, options_df['symbol'].iloc[0])
        
        return {
            "greek_profiles": greek_results,
            "formatted_results": formatted_results,
            "chain_energy": chain_energy
        }
    
    def run_entropy_analysis(self, options_df, output_dir=None):
        """
        Run Entropy Analysis.
        
        Args:
            options_df: Options data DataFrame
            output_dir: Directory to save output files (optional)
            
        Returns:
            Dictionary of entropy analysis results
        """
        logger.info("Running Entropy Analysis")
        
        try:
            # Configure entropy analyzer
            config = {
                "visualization_enabled": output_dir is not None,
                "visualization_dir": os.path.join(output_dir, "entropy_viz") if output_dir else None,
                "entropy_threshold_low": self.config.get("entropy_threshold_low", 30),
                "entropy_threshold_high": self.config.get("entropy_threshold_high", 70),
                "anomaly_detection_sensitivity": self.config.get("anomaly_sensitivity", 1.5)
            }

            # Create entropy analyzer
            entropy_analyzer = EntropyAnalyzer(
                options_data=options_df,
                historical_data=None,
                config=config
            )

            # Run analysis
            entropy_metrics = entropy_analyzer.analyze_greek_entropy()
            anomaly_results = entropy_analyzer.detect_anomalies()
            entropy_report = entropy_analyzer.generate_entropy_report()

            # Ensure energy_state is available for trade recommendations
            if "energy_state" in entropy_metrics and isinstance(entropy_metrics["energy_state"], dict):
                # If energy_state is a dictionary, extract the state string
                energy_state_string = entropy_metrics["energy_state"].get("state", "Unknown")
                # Add it as a top-level field for compatibility
                entropy_metrics["energy_state_string"] = energy_state_string

            return {
                "entropy_metrics": entropy_metrics,
                "anomalies": anomaly_results,
                "report": entropy_report,
                "energy_state": entropy_metrics.get("energy_state", {}),
                "energy_state_string": entropy_metrics.get("energy_state_string", "Unknown")
            }
            
        except Exception as e:
            logger.error(f"Error in Entropy Analysis: {e}")
            return {"error": str(e)}
    
    def analyze_trade_opportunities(self, chain_energy, price_history):
        """
        Analyze trade opportunities based on Greek energy and price history.
        
        Args:
            chain_energy: Chain energy analysis results
            price_history: Price history DataFrame
            
        Returns:
            Dictionary of trade opportunity analysis
        """
        logger.info("Analyzing trade opportunities")
        
        if price_history is None:
            return {"status": "No price history available"}
        
        try:
            opportunities = GreekEnergyAnalyzer.analyze_trade_opportunities(
                chain_energy, price_history
            )
            return opportunities
        except Exception as e:
            logger.error(f"Error analyzing trade opportunities: {e}")
            return {"error": str(e)}
    
    def run_chain_energy_analysis(self, options_df, symbol):
        """
        Run chain energy analysis on options data.
        
        Args:
            options_df: DataFrame containing options data
            symbol: Stock symbol
        
        Returns:
            Dictionary containing chain energy analysis results
        """
        try:
            # Basic implementation - can be expanded later
            chain_energy = {
                "symbol": symbol,
                "total_contracts": len(options_df),
                "total_open_interest": options_df["openInterest"].sum() if "openInterest" in options_df.columns else 0,
                "call_put_ratio": self._calculate_call_put_ratio(options_df),
                "energy_concentration": self._calculate_energy_concentration(options_df),
                "timestamp": datetime.now().isoformat()
            }
            return chain_energy
        except Exception as e:
            logger.error(f"Error in chain energy analysis: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _calculate_call_put_ratio(self, options_df):
        """Calculate call/put ratio from options data."""
        if "type" not in options_df.columns:
            return 1.0  # Default if type column is missing
        
        call_count = len(options_df[options_df["type"].str.lower() == "call"])
        put_count = len(options_df[options_df["type"].str.lower() == "put"])
        
        if put_count == 0:
            return float('inf')  # Avoid division by zero
        
        return call_count / put_count

    def _calculate_energy_concentration(self, options_df):
        """Calculate energy concentration from options data."""
        # Simple implementation - can be enhanced
        if "gamma" not in options_df.columns or "openInterest" not in options_df.columns:
            return 0.0
        
        # Weight gamma by open interest
        weighted_gamma = options_df["gamma"] * options_df["openInterest"]
        
        # Calculate concentration (normalized standard deviation)
        if weighted_gamma.sum() == 0:
            return 0.0
        
        normalized_gamma = weighted_gamma / weighted_gamma.sum()
        return float(normalized_gamma.std())

    def run_full_analysis(self, symbol, options_path, price_path, output_dir, skip_entropy=False):
        """Run the full analysis pipeline."""
        try:
            # Load and preprocess data
            options_df, price_history, market_data = self.load_data(symbol, options_path, price_path)
            
            # Check if data loading was successful
            if options_df is None or price_history is None or market_data is None:
                logger.error(f"Failed to load data for {symbol}")
                return None
            
            # Run Greek analysis
            greek_analysis = self.run_greek_analysis(options_df, market_data)
            if greek_analysis is None:
                logger.error(f"Greek analysis failed for {symbol}")
                return None
            
            # Run chain energy analysis
            chain_energy = self.run_chain_energy_analysis(options_df, symbol)
            
            # Run entropy analysis if not skipped
            entropy_analysis = {}
            if skip_entropy:
                entropy_analysis = {"skipped": True}
                logger.info(f"Entropy analysis skipped for {symbol}")
            else:
                entropy_analysis = self.run_entropy_analysis(options_df, output_dir)
            
            # Combine results
            results = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "market_data": market_data,
                "greek_analysis": greek_analysis,
                "chain_energy": chain_energy,
                "entropy_analysis": entropy_analysis
            }
            
            return results
        except Exception as e:
            logger.error(f"Error in run_full_analysis: {e}", exc_info=True)
            return None





