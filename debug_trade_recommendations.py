"""
Debug Trade Recommendations Script

This script generates sample trade recommendations with the standardized trade context
for testing and debugging purposes.
"""

import os
import sys
import json
import random
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug_recommendations.log')
    ]
)

logger = logging.getLogger(__name__)

def generate_sample_recommendations(symbols: List[str], output_dir: str, count_per_symbol: int = 3):
    """
    Generate sample trade recommendations for debugging
    
    Args:
        symbols: List of ticker symbols
        output_dir: Directory to save recommendations
        count_per_symbol: Number of recommendations per symbol
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Generating {count_per_symbol} sample recommendations for {len(symbols)} symbols")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Strategy types
    strategies = ["Greek Flow", "ML Enhanced", "Ordinal Pattern", "Momentum", "Mean Reversion", "Volatility Expansion"]
    
    # Market regimes
    market_regimes = ["Bullish Trend", "Bearish Trend", "Vanna-Driven", "Charm-Driven", "Gamma Squeeze", "Volatility Expansion", "Mean Reversion"]
    
    # Energy states
    energy_states = ["Accumulation", "Distribution", "Equilibrium", "Transition", "Expansion", "Contraction"]
    
    # Dominant Greeks
    dominant_greeks = ["delta", "gamma", "vanna", "charm", "vomma"]
    
    # Generate recommendations for each symbol
    for symbol in symbols:
        for i in range(count_per_symbol):
            # Generate random price data
            current_price = random.uniform(50, 500)
            
            # Determine direction
            direction = random.choice(["BUY", "SELL"])
            
            # Calculate entry, target, and stop prices
            if direction == "BUY":
                entry_price = current_price
                target_price = entry_price * (1 + random.uniform(0.05, 0.20))
                stop_loss = entry_price * (1 - random.uniform(0.03, 0.10))
            else:
                entry_price = current_price
                target_price = entry_price * (1 - random.uniform(0.05, 0.20))
                stop_loss = entry_price * (1 + random.uniform(0.03, 0.10))
            
            # Calculate risk/reward
            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            risk_reward = round(reward / risk, 2) if risk > 0 else 0
            
            # Generate random anomalies
            anomalies = []
            if random.random() > 0.7:
                anomalies.append("Gamma Spike")
            if random.random() > 0.7:
                anomalies.append("Vanna Reversal")
            if random.random() > 0.8:
                anomalies.append("Charm Acceleration")
            
            # Create trade context
            trade_context = {
                "market_regime": {
                    "primary": random.choice(market_regimes),
                    "volatility": random.choice(["Low", "Normal", "High"]),
                    "confidence": round(random.uniform(0.5, 0.95), 2)
                },
                "volatility_regime": random.choice(["Low", "Normal", "High"]),
                "dominant_greek": random.choice(dominant_greeks),
                "energy_state": random.choice(energy_states),
                "entropy_score": round(random.uniform(0, 1), 2),
                "support_levels": [round(current_price * (1 - random.uniform(0.03, 0.15)), 2) for _ in range(2)],
                "resistance_levels": [round(current_price * (1 + random.uniform(0.03, 0.15)), 2) for _ in range(2)],
                "greek_metrics": {
                    "delta": round(random.uniform(-1.0, 1.0), 2),
                    "gamma": round(random.uniform(0, 0.2), 2),
                    "vanna": round(random.uniform(-0.2, 0.2), 2),
                    "charm": round(random.uniform(-0.1, 0.1), 2),
                    "vomma": round(random.uniform(0, 0.3), 2)
                },
                "anomalies": anomalies,
                "hold_time_days": random.randint(1, 30),
                "confidence_score": round(random.uniform(0.5, 0.95), 2)
            }
            
            # Add ML prediction for some recommendations
            if random.random() > 0.5:
                trade_context["ml_prediction"] = "Bullish" if direction == "BUY" else "Bearish"
                trade_context["ml_confidence"] = round(random.uniform(0.6, 0.9), 2)
            
            # Create trade recommendation
            recommendation = {
                "Symbol": symbol,
                "Strategy": random.choice(strategies),
                "Action": direction,
                "Entry": round(entry_price, 2),
                "Target": round(target_price, 2),
                "Stop": round(stop_loss, 2),
                "RiskReward": risk_reward,
                "Regime": trade_context["market_regime"]["primary"],
                "VolRegime": trade_context["volatility_regime"],
                "Timestamp": datetime.now().isoformat(),
                "Confidence": trade_context["confidence_score"],
                "Notes": f"Sample recommendation for {symbol}",
                "TradeContext": trade_context
            }
            
            # Save recommendation to file
            file_path = os.path.join(output_dir, f"{symbol}_trade_recommendation_{i+1}.json")
            with open(file_path, "w") as f:
                json.dump(recommendation, f, indent=2)
            
            logger.info(f"Saved sample recommendation for {symbol} to {file_path}")
    
    return True

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate sample trade recommendations for debugging")
    parser.add_argument("symbols", nargs="*", help="List of ticker symbols")
    parser.add_argument("--output-dir", default="sample_recommendations", help="Output directory for recommendations")
    parser.add_argument("--count", type=int, default=3, help="Number of recommendations per symbol")
    args = parser.parse_args()
    
    # Use default symbols if none provided
    symbols = args.symbols
    if not symbols:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        logger.info(f"No symbols provided, using defaults: {symbols}")
    
    # Generate sample recommendations
    success = generate_sample_recommendations(symbols, args.output_dir, args.count)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())





