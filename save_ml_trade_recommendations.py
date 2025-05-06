#!/usr/bin/env python
"""
ML Trade Recommendation Saver

This script can be imported by ML prediction scripts to save trade recommendations
in the format expected by the dashboard.
"""

import os
import json
import logging
from datetime import datetime

def save_ml_trade_recommendation(ticker, ml_predictions, trade_signals, transition, entropy_state, output_dir='results'):
    """
    Save ML-enhanced trade recommendations in a format compatible with the dashboard.
    
    Args:
        ticker (str): The stock ticker symbol
        ml_predictions (dict): ML predictions from the model
        trade_signals (dict): Generated trade signals
        transition (dict): Regime transition prediction
        entropy_state (str): Current entropy state
        output_dir (str): Directory to save the files
    
    Returns:
        tuple: Paths to the saved files (dashboard_rec_file, ml_prediction_file)
    """
    # Create the recommendations directory for dashboard
    recommendations_dir = os.path.join(output_dir, "recommendations")
    os.makedirs(recommendations_dir, exist_ok=True)
    
    # Extract action from trade signals
    entry_signal = trade_signals.get('entry', {}).get('signal', None)
    action = "HOLD"
    
    if entry_signal == "bullish":
        action = "BUY"
    elif entry_signal == "bearish":
        action = "SELL"
    
    # Extract confidence values
    entry_confidence = trade_signals.get('entry', {}).get('confidence', 0)
    primary_confidence = ml_predictions.get('primary_regime', {}).get('confidence', 0)
    confidence = max(entry_confidence, primary_confidence, 0.5)  # Use the higher confidence, but at least 0.5
    
    # Extract regime information
    primary_regime = ml_predictions.get('primary_regime', {}).get('prediction', 'Unknown')
    volatility_regime = ml_predictions.get('volatility_regime', {}).get('prediction', 'Normal')
    
    # Extract reasons for signal
    reasons = trade_signals.get('entry', {}).get('reasons', [])
    notes = "\n".join(reasons) if reasons else "Based on ML prediction"
    
    # Create dashboard-compatible format
    dashboard_rec = {
        "Symbol": ticker,
        "Strategy": "ML Enhanced",
        "Action": action,
        "Entry": 0.0,  # Would be calculated from actual price data
        "Stop": 0.0,   # Would be calculated from actual price data
        "Target": 0.0, # Would be calculated from actual price data
        "RiskReward": 0.0,
        "Regime": primary_regime,
        "VolRegime": volatility_regime,
        "Timestamp": datetime.now().isoformat(),
        "Confidence": round(confidence, 2),
        "Notes": notes
    }
    
    # Save to recommendations directory for dashboard
    dashboard_rec_file = os.path.join(recommendations_dir, f"{ticker}_recommendation.json")
    with open(dashboard_rec_file, 'w') as f:
        json.dump(dashboard_rec, f, indent=2)
    
    # Save original ML predictions and trade signals
    ml_data = {
        'ticker': ticker,
        'timestamp': datetime.now().isoformat(),
        'ml_predictions': ml_predictions,
        'trade_signals': trade_signals,
        'regime_transition': transition,
        'entropy_state': entropy_state
    }
    
    ml_prediction_file = os.path.join(output_dir, f"{ticker}_ml_prediction.json")
    with open(ml_prediction_file, 'w') as f:
        json.dump(ml_data, f, indent=2)
    
    # For backward compatibility, also save a traditional trade recommendation
    old_format_rec = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "primary_regime": primary_regime,
        "volatility_regime": volatility_regime,
        "suggested_action": action
    }
    
    trad_rec_file = os.path.join(output_dir, f"{ticker}_trade_recommendation.json")
    with open(trad_rec_file, 'w') as f:
        json.dump(old_format_rec, f, indent=2)
    
    return dashboard_rec_file, ml_prediction_file

def create_market_regime_summary(output_dir='results'):
    """
    Create a market regime summary file from all recommendation files.
    
    Args:
        output_dir (str): Directory containing the recommendation files
    
    Returns:
        str: Path to the market regime summary file
    """
    recommendations_dir = os.path.join(output_dir, "recommendations")
    if not os.path.exists(recommendations_dir):
        return None
    
    # Load all recommendation files
    recs = []
    for file in os.listdir(recommendations_dir):
        if file.endswith('_recommendation.json'):
            try:
                with open(os.path.join(recommendations_dir, file), 'r') as f:
                    data = json.load(f)
                recs.append(data)
            except Exception as e:
                logging.error(f"Error loading {file}: {e}")
    
    if not recs:
        return None
    
    # Count regimes
    regimes = {}
    regime_counts = {"Bullish": 0, "Bearish": 0, "Neutral": 0, "Unknown": 0}
    
    for rec in recs:
        ticker = rec.get('Symbol', 'Unknown')
        regime = rec.get('Regime', 'Unknown')
        regimes[ticker] = regime
        
        if "Bullish" in regime:
            regime_counts["Bullish"] += 1
        elif "Bearish" in regime:
            regime_counts["Bearish"] += 1
        elif "Neutral" in regime:
            regime_counts["Neutral"] += 1
        else:
            regime_counts["Unknown"] += 1
    
    # Determine overall bias
    overall_bias = "NEUTRAL"
    if regime_counts["Bullish"] > regime_counts["Bearish"]:
        overall_bias = "BULLISH"
    elif regime_counts["Bearish"] > regime_counts["Bullish"]:
        overall_bias = "BEARISH"
    
    # Create market regime summary
    market_regime = {
        "timestamp": datetime.now().isoformat(),
        "overall_bias": overall_bias,
        "bullish_count": regime_counts["Bullish"],
        "bearish_count": regime_counts["Bearish"],
        "neutral_count": regime_counts["Neutral"],
        "regimes": regimes
    }
    
    # Save market regime file
    regime_file = os.path.join(output_dir, "market_regime.json")
    with open(regime_file, 'w') as f:
        json.dump(market_regime, f, indent=2)
    
    return regime_file

if __name__ == "__main__":
    # Example usage if run directly
    example_ticker = "AAPL"
    example_ml_predictions = {
        "primary_regime": {"prediction": "Bullish", "confidence": 0.85},
        "volatility_regime": {"prediction": "Normal", "confidence": 0.7}
    }
    example_trade_signals = {
        "entry": {
            "signal": "bullish",
            "strength": 2,
            "confidence": 0.8,
            "reasons": ["Vanna-driven regime in low volatility environment"]
        },
        "exit": {
            "signal": None,
            "strength": 0,
            "confidence": 0,
            "reasons": []
        }
    }
    example_transition = {
        "transition_type": "unlikely",
        "transition_probability": 0.2
    }
    example_entropy_state = "Concentrated Energy (Low Entropy)"
    
    save_ml_trade_recommendation(
        example_ticker,
        example_ml_predictions,
        example_trade_signals,
        example_transition,
        example_entropy_state,
        output_dir='results'
    )
    
    # Create market regime summary
    create_market_regime_summary(output_dir='results')
