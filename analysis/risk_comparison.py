# risk_comparison.py - Compare entropy-based risk management with current RM system

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
from datetime import datetime, timedelta
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the entropy analyzer - fix the import path
from entropy_analyzer import EntropyAnalyzer  # Changed from entropy_analyzer.entropy_analyzer
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RiskComparison")

class RiskComparisonAnalyzer:
    """
    Compares entropy-based risk management with current risk management system.
    """
    
    def __init__(self, options_data=None, market_data=None, historical_data=None):
        """
        Initialize the risk comparison analyzer.
        
        Args:
            options_data: DataFrame containing options data with Greeks
            market_data: Dictionary with market data (price, volatility, etc.)
            historical_data: Optional DataFrame with historical data for comparison
        """
        self.options_data = options_data
        self.market_data = market_data if market_data else {}
        self.historical_data = historical_data
        
        # Default configuration
        self.config = {
            "output_dir": "results/risk_comparison",
            "visualization_enabled": True,
            "current_rm_stop_mult": getattr(config, "STOP_MULT", 0.5),
            "min_profit_loss_ratio": getattr(config, "MIN_PROFIT_LOSS_RATIO", 1.75)
        }
        
        # Create output directory
        os.makedirs(self.config["output_dir"], exist_ok=True)
    
    def load_data(self, options_file, market_file=None, historical_file=None):
        """
        Load data from files.
        
        Args:
            options_file: Path to options data file (CSV or JSON)
            market_file: Path to market data file (JSON)
            historical_file: Path to historical data file (CSV or JSON)
        """
        try:
            # Load options data
            if options_file.endswith('.csv'):
                self.options_data = pd.read_csv(options_file)
            elif options_file.endswith('.json'):
                with open(options_file, 'r') as f:
                    options_json = json.load(f)
                    # Convert to DataFrame (assuming it's a list of dictionaries)
                    if isinstance(options_json, list):
                        self.options_data = pd.DataFrame(options_json)
                    else:
                        # Handle nested structure if needed
                        self.options_data = pd.DataFrame(options_json.get('data', []))
            else:
                logger.error(f"Unsupported file format for options data: {options_file}")
                return False
            
            # Load market data if provided
            if market_file:
                with open(market_file, 'r') as f:
                    self.market_data = json.load(f)
            
            # Load historical data if provided
            if historical_file:
                if historical_file.endswith('.csv'):
                    self.historical_data = pd.read_csv(historical_file)
                elif historical_file.endswith('.json'):
                    with open(historical_file, 'r') as f:
                        self.historical_data = json.load(f)
                else:
                    logger.error(f"Unsupported file format for historical data: {historical_file}")
            
            logger.info(f"Loaded data: {len(self.options_data)} options records")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def calculate_current_rm_metrics(self):
        """
        Calculate risk metrics using current risk management system.
        
        Returns:
            Dictionary with current RM metrics
        """
        try:
            # Get current price
            current_price = self.market_data.get('currentPrice', 
                                               self.options_data.get('underlyingPrice', 
                                                                   self.options_data.get('spot', 100)))
            
            # Calculate stop loss based on current RM system
            stop_mult = self.config["current_rm_stop_mult"]
            
            # Get ATR if available, otherwise use a percentage of price
            atr = self.market_data.get('atr', current_price * 0.02)  # Default to 2% if not available
            
            # Calculate stop loss
            stop_loss = atr * stop_mult
            
            # Calculate risk metrics
            risk_metrics = {
                "stop_loss_amount": stop_loss,
                "stop_loss_percent": (stop_loss / current_price) * 100,
                "min_profit_target": stop_loss * self.config["min_profit_loss_ratio"],
                "min_profit_target_percent": (stop_loss * self.config["min_profit_loss_ratio"] / current_price) * 100,
                "risk_reward_ratio": self.config["min_profit_loss_ratio"]
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating current RM metrics: {e}")
            return {}
    
    def calculate_entropy_rm_metrics(self):
        """
        Calculate risk metrics using entropy-based risk management.
        
        Returns:
            Dictionary with entropy-based RM metrics
        """
        try:
            # Create entropy analyzer
            entropy_analyzer = EntropyAnalyzer(
                self.options_data,
                self.historical_data,
                config={"visualization_enabled": self.config["visualization_enabled"],
                       "visualization_dir": self.config["output_dir"]}
            )
            
            # Run entropy analysis
            entropy_metrics = entropy_analyzer.analyze_greek_entropy()
            
            # Detect anomalies
            anomalies = entropy_analyzer.detect_anomalies()
            
            # Generate full report
            report = entropy_analyzer.generate_entropy_report()
            
            # Get current price
            current_price = self.market_data.get('currentPrice', 
                                               self.options_data.get('underlyingPrice', 
                                                                   self.options_data.get('spot', 100)))
            
            # Get energy state and anomaly score
            energy_state = report.get("energy_state", {})
            anomaly_score = anomalies.get("anomaly_score", 0)
            
            # Calculate entropy-based stop loss
            base_stop_mult = self.config["current_rm_stop_mult"]
            
            # Adjust stop multiplier based on entropy state
            if "Low Entropy" in energy_state.get("state", ""):
                # Higher risk in concentrated energy state
                entropy_stop_mult = base_stop_mult * 1.5
            elif "High Entropy" in energy_state.get("state", ""):
                # Lower risk in dispersed energy state
                entropy_stop_mult = base_stop_mult * 0.8
            else:
                # Normal risk in balanced energy state
                entropy_stop_mult = base_stop_mult
            
            # Further adjust based on anomaly score
            if anomaly_score > 3:
                # High anomaly score means higher risk
                entropy_stop_mult *= 1.5
            elif anomaly_score > 1:
                # Moderate anomaly score
                entropy_stop_mult *= 1.2
            
            # Get ATR if available, otherwise use a percentage of price
            atr = self.market_data.get('atr', current_price * 0.02)  # Default to 2% if not available
            
            # Calculate entropy-based stop loss
            entropy_stop_loss = atr * entropy_stop_mult
            
            # Calculate risk metrics
            risk_metrics = {
                "stop_loss_amount": entropy_stop_loss,
                "stop_loss_percent": (entropy_stop_loss / current_price) * 100,
                "min_profit_target": entropy_stop_loss * self.config["min_profit_loss_ratio"],
                "min_profit_target_percent": (entropy_stop_loss * self.config["min_profit_loss_ratio"] / current_price) * 100,
                "risk_reward_ratio": self.config["min_profit_loss_ratio"],
                "entropy_state": energy_state.get("state", "Unknown"),
                "anomaly_score": anomaly_score,
                "stop_multiplier": entropy_stop_mult
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating entropy RM metrics: {e}")
            return {}
    
    def compare_risk_management_systems(self):
        """
        Compare current and entropy-based risk management systems.
        
        Returns:
            Dictionary with comparison results
        """
        try:
            # Get current price
            current_price = self.market_data.get('currentPrice', 
                                               self.options_data.get('underlyingPrice', 
                                                                   self.options_data.get('spot', 100)))
            
            # Calculate current risk metrics
            current_metrics = self.calculate_current_risk_metrics()
            
            # Calculate entropy-based risk metrics
            entropy_metrics = self.calculate_entropy_risk_metrics()
            
            # Generate recommendation
            recommendation = self.generate_risk_recommendation(current_metrics, entropy_metrics, current_price)
            
            # Compare metrics
            comparison = {
                "current_system": current_metrics,
                "entropy_system": entropy_metrics,
                "recommendation": recommendation,
                "comparison": {
                    "stop_loss_difference": entropy_metrics["stop_loss_amount"] - current_metrics["stop_loss_amount"],
                    "stop_loss_percent_difference": entropy_metrics["stop_loss_percent"] - current_metrics["stop_loss_percent"],
                    "risk_reward_difference": entropy_metrics["risk_reward_ratio"] - current_metrics["risk_reward_ratio"]
                }
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error in risk management comparison: {e}")
            return {
                "error": str(e),
                "current_system": {},
                "entropy_system": {},
                "recommendation": {"system": "Current", "reason": "Error in comparison"}
            }
    
    def _generate_recommendation(self, current_metrics, entropy_metrics):
        """
        Generate recommendation based on comparison.
        
        Args:
            current_metrics: Current RM metrics
            entropy_metrics: Entropy-based RM metrics
            
        Returns:
            Dictionary with recommendation
        """
        try:
            # Get energy state and anomaly score
            energy_state = entropy_metrics.get("entropy_state", "Unknown")
            anomaly_score = entropy_metrics.get("anomaly_score", 0)
            
            # Default recommendation
            recommendation = {
                "system": "Current",
                "reason": "Default system is appropriate for current market conditions",
                "position_size_adjustment": 1.0
            }
            
            # Check for high anomaly score
            if anomaly_score > 3:
                recommendation["system"] = "Entropy-based"
                recommendation["reason"] = "High anomaly score indicates unusual market conditions"
                recommendation["position_size_adjustment"] = 0.5
            
            # Check for concentrated energy state
            elif "Low Entropy" in energy_state:
                recommendation["system"] = "Entropy-based"
                recommendation["reason"] = "Concentrated energy state requires more conservative risk management"
                recommendation["position_size_adjustment"] = 0.75
            
            # Check for dispersed energy state
            elif "High Entropy" in energy_state:
                # In dispersed state, current system might be too conservative
                if entropy_metrics["stop_loss_amount"] < current_metrics["stop_loss_amount"]:
                    recommendation["system"] = "Entropy-based"
                    recommendation["reason"] = "Dispersed energy state allows for less conservative risk management"
                    recommendation["position_size_adjustment"] = 1.0
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return {"system": "Current", "reason": "Error in analysis", "position_size_adjustment": 1.0}
    
    def _visualize_comparison(self, comparison):
        """
        Create visualization of risk management comparison.
        
        Args:
            comparison: Comparison results dictionary
        """
        try:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Get metrics
            current_rm = comparison["current_rm"]
            entropy_rm = comparison["entropy_rm"]
            
            # Plot stop loss comparison
            labels = ['Current RM', 'Entropy RM']
            stop_values = [current_rm["stop_loss_amount"], entropy_rm["stop_loss_amount"]]
            target_values = [current_rm["min_profit_target"], entropy_rm["min_profit_target"]]
            
            x = np.arange(len(labels))
            width = 0.35
            
            ax1.bar(x - width/2, stop_values, width, label='Stop Loss')
            ax1.bar(x + width/2, target_values, width, label='Profit Target')
            
            ax1.set_ylabel('Amount')
            ax1.set_title('Risk Management Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels)
            ax1.legend()
            
            # Add values on bars
            for i, v in enumerate(stop_values):
                ax1.text(i - width/2, v + 0.1, f'${v:.2f}', ha='center')
            
            for i, v in enumerate(target_values):
                ax1.text(i + width/2, v + 0.1, f'${v:.2f}', ha='center')
            
            # Plot risk-reward ratio
            labels = ['Current RM', 'Entropy RM']
            rr_values = [current_rm["risk_reward_ratio"], entropy_rm["risk_reward_ratio"]]
            
            ax2.bar(x, rr_values, width, label='Risk-Reward Ratio')
            
            ax2.set_ylabel('Ratio')
            ax2.set_title('Risk-Reward Ratio Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            
            # Add values on bars
            for i, v in enumerate(rr_values):
                ax2.text(i, v + 0.1, f'{v:.2f}', ha='center')
            
            # Add recommendation as text
            recommendation = comparison.get("recommendation", {})
            plt.figtext(0.5, 0.01, 
                       f"Recommendation: Use {recommendation.get('system', 'Current')} RM system\n"
                       f"Reason: {recommendation.get('reason', 'N/A')}\n"
                       f"Position Size Adjustment: {recommendation.get('position_size_adjustment', 1.0):.2f}x",
                       ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
            
            plt.tight_layout(rect=[0, 0.08, 1, 0.95])
            
            # Save figure
            filename = os.path.join(self.config["output_dir"], "risk_comparison.png")
            plt.savefig(filename)
            plt.close()
            
            logger.info(f"Saved comparison visualization to {filename}")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
    
    def run_backtest(self, historical_trades_file):
        """
        Run backtest to compare performance of both risk management systems.
        
        Args:
            historical_trades_file: Path to CSV file with historical trades
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Load historical trades
            if not os.path.exists(historical_trades_file):
                logger.error(f"Historical trades file not found: {historical_trades_file}")
                return {}
                
            trades_df = pd.read_csv(historical_trades_file)
            
            # Check required columns
            required_cols = ['symbol', 'entry_price', 'exit_price', 'atr', 'result']
            missing_cols = [col for col in required_cols if col not in trades_df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns in trades file: {missing_cols}")
                return {}
            
            # Initialize results
            results = {
                "current_rm": {"wins": 0, "losses": 0, "total_return": 0, "max_drawdown": 0},
                "entropy_rm": {"wins": 0, "losses": 0, "total_return": 0, "max_drawdown": 0}
            }
            
            # Process each trade
            for _, trade in trades_df.iterrows():
                # Get trade details
                symbol = trade['symbol']
                entry_price = trade['entry_price']
                exit_price = trade['exit_price']
                atr = trade['atr']
                
                # Calculate current RM stop loss
                current_stop = entry_price - (atr * self.config["current_rm_stop_mult"])
                
                # Calculate entropy-based stop loss
                # This is simplified - in a real backtest, we would need historical options data
                # Here we're using a random factor to simulate entropy-based adjustment
                entropy_factor = np.random.uniform(0.8, 1.5)  # Random factor for simulation
                entropy_stop = entry_price - (atr * self.config["current_rm_stop_mult"] * entropy_factor)
                
                # Calculate results for current RM
                if exit_price > current_stop:
                    # Trade was a win or hit profit target
                    results["current_rm"]["wins"] += 1
                    results["current_rm"]["total_return"] += (exit_price - entry_price) / entry_price
                else:
                    # Trade hit stop loss
                    results["current_rm"]["losses"] += 1
                    results["current_rm"]["total_return"] += (current_stop - entry_price) / entry_price
                
                # Calculate results for entropy RM
                if exit_price > entropy_stop:
                    # Trade was a win or hit profit target
                    results["entropy_rm"]["wins"] += 1
                    results["entropy_rm"]["total_return"] += (exit_price - entry_price) / entry_price
                else:
                    # Trade hit stop loss
                    results["entropy_rm"]["losses"] += 1
                    results["entropy_rm"]["total_return"] += (entropy_stop - entry_price) / entry_price
            
            # Calculate win rates
            for system in ["current_rm", "entropy_rm"]:
                total_trades = results[system]["wins"] + results[system]["losses"]
                if total_trades > 0:
                    results[system]["win_rate"] = results[system]["wins"] / total_trades
                else:
                    results[system]["win_rate"] = 0
            
            # Visualize backtest results if enabled
            if self.config["visualization_enabled"]:
                self._visualize_backtest(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return {}
    
    def _visualize_backtest(self, results):
        """
        Create visualization of backtest results.
        
        Args:
            results: Backtest results dictionary
        """
        try:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot win rates
            labels = ['Current RM', 'Entropy RM']
            win_rates = [results["current_rm"]["win_rate"], results["entropy_rm"]["win_rate"]]
            
            ax1.bar(labels, win_rates, color=['blue', 'orange'])
            ax1.set_ylabel('Win Rate')
            ax1.set_title('Win Rate Comparison')
            
            # Plot total returns
            total_returns = [results["current_rm"]["total_return"], results["entropy_rm"]["total_return"]]
            
            ax2.bar(labels, total_returns, color=['green', 'purple'])
            ax2.set_ylabel('Total Return')
            ax2.set_title('Total Return Comparison')
            
            plt.tight_layout()
            
            # Save figure
            filename = os.path.join(self.config["output_dir"], "backtest_results.png")
            plt.savefig(filename)
            plt.close()
            
            logger.info(f"Saved backtest results visualization to {filename}")
            
        except Exception as e:
            logger.error(f"Error creating backtest visualization: {e}")


