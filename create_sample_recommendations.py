#!/usr/bin/env python
"""
Create sample trade recommendations for dashboard testing
"""

import os
import json
from datetime import datetime
import random
import glob

def create_sample_recommendations(num_samples=10):
    """Create sample trade recommendations with ordinal pattern data"""
    
    # Ensure directories exist
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", "recommendations"), exist_ok=True)
    
    # Sample tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "SPY"]
    
    # Sample strategies - IMPORTANT: Use exact strategy names that the dashboard expects
    strategies = ["Ordinal", "Greek Flow", "ML Enhanced", "Momentum", "Mean Reversion"]
    
    # Sample actions
    actions = ["BUY", "SELL"]
    
    # Sample regimes
    regimes = ["Bullish Trend", "Bearish Trend", "Neutral", "Vanna-Driven", "Charm-Dominated"]
    
    # Sample volatility regimes
    vol_regimes = ["High", "Normal", "Low"]
    
    # Create recommendations
    for i in range(min(num_samples, len(tickers))):
        ticker = tickers[i]
        current_price = round(random.uniform(50, 500), 2)
        
        # Determine action and prices
        action = random.choice(actions)
        entry_price = current_price
        stop_price = round(entry_price * (0.95 if action == "BUY" else 1.05), 2)
        target_price = round(entry_price * (1.1 if action == "BUY" else 0.9), 2)
        
        # Calculate risk/reward
        risk = abs(entry_price - stop_price)
        reward = abs(target_price - entry_price)
        risk_reward = round(reward / risk, 2)
        
        # Create ordinal pattern data
        ordinal_patterns = {
            "delta": {
                "pattern": [0, 1, 2] if action == "BUY" else [2, 1, 0],
                "confidence": round(random.uniform(0.7, 0.95), 2),
                "expected_value": round(random.uniform(0.02, 0.1), 3) if action == "BUY" else round(random.uniform(-0.1, -0.02), 3)
            },
            "gamma": {
                "pattern": [1, 0, 2],
                "confidence": round(random.uniform(0.6, 0.9), 2),
                "expected_value": round(random.uniform(-0.05, 0.05), 3)
            },
            "vanna": {
                "pattern": [0, 2, 1] if action == "BUY" else [2, 0, 1],
                "confidence": round(random.uniform(0.5, 0.85), 2),
                "expected_value": round(random.uniform(0.01, 0.08), 3) if action == "BUY" else round(random.uniform(-0.08, -0.01), 3)
            }
        }
        
        # Create trade context
        trade_context = {
            "market_regime": {
                "primary": random.choice(regimes),
                "volatility": random.choice(vol_regimes),
                "confidence": round(random.uniform(0.6, 0.9), 2)
            },
            "volatility_regime": random.choice(vol_regimes),
            "dominant_greek": random.choice(["delta", "gamma", "vanna", "charm"]),
            "energy_state": random.choice(["Accumulation", "Distribution", "Equilibrium"]),
            "entropy_score": round(random.uniform(0, 1), 2),
            "support_levels": [round(current_price * 0.9, 2), round(current_price * 0.95, 2)],
            "resistance_levels": [round(current_price * 1.05, 2), round(current_price * 1.1, 2)],
            "greek_metrics": {
                "delta": round(random.uniform(-1, 1), 2),
                "gamma": round(random.uniform(0, 0.5), 3),
                "vanna": round(random.uniform(-0.5, 0.5), 3),
                "charm": round(random.uniform(-0.2, 0.2), 3),
                "vomma": round(random.uniform(0, 0.3), 3)
            },
            "anomalies": random.sample(["Delta Spike", "Gamma Squeeze", "Vanna Flip", "Charm Acceleration"], 
                                      k=random.randint(0, 3)),
            "hold_time_days": random.randint(1, 14),
            "ordinal_patterns": ordinal_patterns
        }
        
        # Ensure at least 3 recommendations use the "Ordinal" strategy
        strategy = "Ordinal" if i < 3 else random.choice(strategies)
        
        # Create recommendation
        recommendation = {
            "symbol": ticker,
            "strategy_name": strategy,
            "action": action,
            "entry_price": entry_price,
            "stop_loss": stop_price,
            "target_price": target_price,
            "rr_ratio": risk_reward,
            "rr_ratio_str": f"{risk_reward:.1f}:1",
            "market_regime": trade_context["market_regime"]["primary"],
            "volatility_regime": trade_context["volatility_regime"],
            "timestamp": datetime.now().isoformat(),
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "notes": f"Ordinal pattern analysis: {ordinal_patterns['delta']['pattern']} pattern detected",
            "trade_context": trade_context,
            "days_to_hold": trade_context["hold_time_days"],
            "risk_category": random.choice(["LOW", "MEDIUM", "HIGH"])
        }
        
        # Save recommendation in multiple formats and locations to ensure compatibility
        
        # 1. Standard location
        filename = os.path.join("results", f"{ticker}_recommendation.json")
        with open(filename, 'w') as f:
            json.dump(recommendation, f, indent=2)
        
        # 2. Recommendations subdirectory
        filename2 = os.path.join("results", "recommendations", f"{ticker}_recommendation.json")
        with open(filename2, 'w') as f:
            json.dump(recommendation, f, indent=2)
        
        # 3. Try with uppercase Strategy field (for older dashboard versions)
        recommendation_alt = recommendation.copy()
        recommendation_alt["Strategy"] = recommendation_alt.pop("strategy_name")
        filename3 = os.path.join("results", f"{ticker}_recommendation_alt.json")
        with open(filename3, 'w') as f:
            json.dump(recommendation_alt, f, indent=2)
        
        # 4. Try with different field names for entry/stop/target
        recommendation_alt2 = recommendation.copy()
        recommendation_alt2["Entry"] = recommendation_alt2.pop("entry_price")
        recommendation_alt2["Stop"] = recommendation_alt2.pop("stop_loss")
        recommendation_alt2["Target"] = recommendation_alt2.pop("target_price")
        filename4 = os.path.join("results", f"{ticker}_recommendation_alt2.json")
        with open(filename4, 'w') as f:
            json.dump(recommendation_alt2, f, indent=2)
        
        print(f"Created recommendation for {ticker}")
    
    # Create market regime summary
    create_market_regime_summary(tickers[:num_samples])
    
    print(f"Created {num_samples} sample recommendations")
    
    # Print summary of all recommendation files
    all_rec_files = glob.glob(os.path.join("results", "*_recommendation*.json"))
    print(f"Total recommendation files in results directory: {len(all_rec_files)}")

def create_market_regime_summary(tickers):
    """Create a market regime summary file"""
    regimes = {}
    for ticker in tickers:
        try:
            with open(os.path.join("results", f"{ticker}_recommendation.json"), 'r') as f:
                data = json.load(f)
                regimes[ticker] = data.get("market_regime", "Unknown")
        except:
            regimes[ticker] = "Unknown"
    
    # Count regimes
    bullish = sum(1 for r in regimes.values() if "Bullish" in r)
    bearish = sum(1 for r in regimes.values() if "Bearish" in r)
    neutral = sum(1 for r in regimes.values() if "Neutral" in r)
    
    # Determine overall bias
    if bullish > bearish:
        overall_bias = "BULLISH"
    elif bearish > bullish:
        overall_bias = "BEARISH"
    else:
        overall_bias = "NEUTRAL"
    
    # Create summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "overall_bias": overall_bias,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "regimes": regimes
    }
    
    # Save summary
    with open(os.path.join("results", "market_regime.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Also create the current_regime.json file in the expected location
    os.makedirs(os.path.join("results", "market_regime"), exist_ok=True)
    
    current_regime = {
        "primary_label": "Bullish Trend" if overall_bias == "BULLISH" else "Bearish Trend" if overall_bias == "BEARISH" else "Neutral",
        "secondary_label": "Secondary Classification",
        "volatility_regime": "Normal",
        "dominant_greek": "Delta",
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join("results", "market_regime", "current_regime.json"), 'w') as f:
        json.dump(current_regime, f, indent=2)
    
    print("Created market regime summary")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create sample trade recommendations")
    parser.add_argument("--num", type=int, default=10, help="Number of recommendations to create")
    
    args = parser.parse_args()
    create_sample_recommendations(args.num)


