import os
import json
from datetime import datetime
import argparse

def create_market_regime(primary_label="Bullish Trend", volatility="Normal", dominant_greek="Delta"):
    """
    Create a market regime file for the dashboard.
    
    Args:
        primary_label (str): Primary market regime label
            Options: "Bullish Trend", "Bearish Trend", "Neutral", "Vanna-Driven", "Charm-Dominated"
        volatility (str): Volatility regime
            Options: "High", "Normal", "Low"
        dominant_greek (str): Dominant Greek influencing the market
            Options: "Delta", "Gamma", "Vanna", "Charm"
    """
    # Create a market regime file
    market_regime = {
        "primary_label": primary_label,
        "secondary_label": "Secondary Classification",
        "volatility_regime": volatility,
        "dominant_greek": dominant_greek,
        "timestamp": datetime.now().isoformat()
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.join("results", "market_regime"), exist_ok=True)

    # Save to file
    with open(os.path.join("results", "market_regime", "current_regime.json"), 'w') as f:
        json.dump(market_regime, f, indent=2)

    print(f"Market regime file created successfully: {primary_label}, {volatility} volatility, {dominant_greek} dominant")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create market regime file for dashboard")
    parser.add_argument("--regime", default="Bullish Trend", 
                        choices=["Bullish Trend", "Bearish Trend", "Neutral", "Vanna-Driven", "Charm-Dominated"],
                        help="Primary market regime")
    parser.add_argument("--volatility", default="Normal", 
                        choices=["High", "Normal", "Low"],
                        help="Volatility regime")
    parser.add_argument("--greek", default="Delta", 
                        choices=["Delta", "Gamma", "Vanna", "Charm"],
                        help="Dominant Greek")
    
    args = parser.parse_args()
    create_market_regime(args.regime, args.volatility, args.greek)
