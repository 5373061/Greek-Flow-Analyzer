"""
Compatibility utilities for the dashboard.

This module provides functions to handle compatibility between different
versions of recommendation formats and trade context structures.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def standardize_recommendation(data: Dict[str, Any], file_path: str = "") -> Dict[str, Any]:
    """
    Standardize a recommendation to the dashboard-compatible format
    
    Args:
        data: Raw recommendation data
        file_path: Source file path (for logging)
        
    Returns:
        Standardized recommendation dictionary
    """
    try:
        # Create a standardized structure
        standardized = {
            "Symbol": "",
            "Strategy": "",
            "Action": "HOLD",
            "Entry": 0.0,
            "Stop": 0.0,
            "Target": 0.0,
            "RiskReward": 0.0,
            "Regime": "Unknown",
            "VolRegime": "Normal",
            "Timestamp": datetime.now().isoformat(),
            "Confidence": 0.0,
            "Notes": "",
            "TradeContext": {}
        }
        
        # Extract symbol (handle potential variations)
        symbol = data.get("Symbol", data.get("symbol", ""))
        if not symbol and "instrument" in data:
            symbol = data["instrument"]
        if not symbol and file_path:
            try:
                # Try guessing from filename (e.g., AAPL_...)
                symbol = os.path.basename(file_path).split('_')[0].upper()
            except IndexError:
                symbol = "Unknown"
        
        standardized["Symbol"] = symbol
        
        # Extract strategy
        standardized["Strategy"] = data.get("Strategy", data.get("strategy", data.get("strategy_name", "Unknown")))
        
        # Extract action/direction
        action = data.get("Action", data.get("action", data.get("direction", "HOLD")))
        # Normalize action format
        if action.upper() in ["BUY", "LONG"]:
            action = "BUY"
        elif action.upper() in ["SELL", "SHORT"]:
            action = "SELL"
        standardized["Action"] = action
        
        # Extract price information
        standardized["Entry"] = float(data.get("Entry", data.get("entry_price", data.get("entry", 0.0))))
        standardized["Target"] = float(data.get("Target", data.get("target_price", data.get("target", 0.0))))
        standardized["Stop"] = float(data.get("Stop", data.get("stop_loss", data.get("stop", 0.0))))
        
        # Extract risk/reward
        standardized["RiskReward"] = float(data.get("RiskReward", data.get("risk_reward", 0.0)))
        
        # If risk/reward is missing but we have entry, target, and stop, calculate it
        if standardized["RiskReward"] == 0.0 and standardized["Entry"] > 0:
            entry = standardized["Entry"]
            target = standardized["Target"]
            stop = standardized["Stop"]
            
            if entry != stop and entry > 0 and target > 0 and stop > 0:
                risk = abs(entry - stop)
                reward = abs(target - entry)
                if risk > 0:
                    standardized["RiskReward"] = round(reward / risk, 2)
        
        # Extract regime information
        standardized["Regime"] = data.get("Regime", data.get("regime", data.get("market_regime", "Unknown")))
        if isinstance(standardized["Regime"], dict):
            standardized["Regime"] = standardized["Regime"].get("primary", "Unknown")
            
        standardized["VolRegime"] = data.get("VolRegime", data.get("volatility_regime", "Normal"))
        
        # Extract timestamp
        timestamp = data.get("Timestamp", data.get("timestamp", datetime.now().isoformat()))
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        standardized["Timestamp"] = timestamp
        
        # Extract confidence
        standardized["Confidence"] = float(data.get("Confidence", data.get("confidence", data.get("confidence_score", 0.0))))
        
        # Extract notes
        standardized["Notes"] = data.get("Notes", data.get("notes", ""))
        
        # Extract or create trade context
        trade_context = data.get("TradeContext", data.get("trade_context", {}))
        
        # If trade context is empty, try to build it from other fields
        if not trade_context:
            trade_context = extract_trade_context_from_legacy(data)
            
        standardized["TradeContext"] = trade_context
        
        return standardized
        
    except Exception as e:
        logger.error(f"Error standardizing recommendation from {file_path}: {e}")
        return {
            "Symbol": "ERROR",
            "Strategy": "Unknown",
            "Action": "HOLD",
            "Entry": 0.0,
            "Stop": 0.0,
            "Target": 0.0,
            "RiskReward": 0.0,
            "Regime": "Unknown",
            "VolRegime": "Normal",
            "Timestamp": datetime.now().isoformat(),
            "Confidence": 0.0,
            "Notes": f"Error processing recommendation: {str(e)}",
            "TradeContext": {}
        }

def extract_trade_context_from_legacy(recommendation):
    """
    Extract trade context information from legacy recommendation formats.
    
    Args:
        recommendation (dict): The recommendation data
        
    Returns:
        dict: A standardized trade context dictionary
    """
    # Initialize empty trade context
    trade_context = {}
    
    # Extract market regime information
    market_regime = recommendation.get("market_regime", {})
    market_context = recommendation.get("market_context", {})
    
    # Build market regime structure
    if isinstance(market_regime, dict):
        trade_context["market_regime"] = market_regime
    elif isinstance(market_context, dict) and "regime" in market_context:
        # Convert from market_context format
        trade_context["market_regime"] = {
            "primary": market_context.get("regime", "Unknown"),
            "volatility": market_context.get("volatility_regime", "Normal"),
            "confidence": market_context.get("confidence", 0.5)
        }
    elif isinstance(market_regime, str):
        # Convert from string format
        trade_context["market_regime"] = {
            "primary": market_regime,
            "volatility": "Normal",
            "confidence": 0.5
        }
    
    # Extract volatility regime
    if "volatility_regime" in market_context:
        trade_context["volatility_regime"] = market_context["volatility_regime"]
    
    # Extract dominant greek if available
    if "dominant_greek" in recommendation:
        trade_context["dominant_greek"] = recommendation["dominant_greek"]
    elif "greek_analysis" in recommendation:
        # Try to determine dominant greek from analysis
        greek_analysis = recommendation["greek_analysis"]
        if isinstance(greek_analysis, dict):
            # Find the greek with the highest absolute value
            max_greek = None
            max_value = 0
            for greek, value in greek_analysis.items():
                if isinstance(value, (int, float)) and abs(value) > max_value:
                    max_greek = greek
                    max_value = abs(value)
            
            if max_greek:
                trade_context["dominant_greek"] = max_greek
    
    # Extract energy state if available
    if "energy_state" in recommendation:
        trade_context["energy_state"] = recommendation["energy_state"]
    
    # Extract entropy score if available
    if "entropy_score" in recommendation:
        trade_context["entropy_score"] = recommendation["entropy_score"]
    
    # Extract anomalies if available
    if "anomalies" in recommendation:
        trade_context["anomalies"] = recommendation["anomalies"]
    elif "greek_anomalies" in recommendation:
        trade_context["anomalies"] = recommendation["greek_anomalies"]
    
    # Extract hold time if available
    if "days_to_hold" in recommendation:
        trade_context["hold_time_days"] = recommendation["days_to_hold"]
    
    # Extract confidence score if available
    if "confidence" in recommendation:
        trade_context["confidence_score"] = recommendation["confidence"]
    
    return trade_context

def save_standardized_recommendation(recommendation: Dict[str, Any], output_dir: str) -> str:
    """
    Save a standardized recommendation to a file
    
    Args:
        recommendation: Standardized recommendation dictionary
        output_dir: Output directory
        
    Returns:
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    symbol = recommendation.get("Symbol", "Unknown")
    filename = f"{symbol}_recommendation.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(recommendation, f, indent=2)
    
    return filepath

def load_and_standardize_recommendations(directory: str) -> List[Dict[str, Any]]:
    """
    Load and standardize all recommendations from a directory
    
    Args:
        directory: Directory containing recommendation files
        
    Returns:
        List of standardized recommendations
    """
    recommendations = []
    
    if not os.path.exists(directory):
        logger.warning(f"Directory {directory} does not exist")
        return recommendations
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                standardized = standardize_recommendation(data, filepath)
                recommendations.append(standardized)
                
            except Exception as e:
                logger.error(f"Error loading recommendation from {filepath}: {e}")
    
    return recommendations

def fix_recommendation_files(input_dir: str, output_dir: str = None) -> int:
    """
    Fix all recommendation files in a directory
    
    Args:
        input_dir: Input directory containing recommendation files
        output_dir: Output directory (defaults to input_dir if None)
        
    Returns:
        Number of fixed files
    """
    if output_dir is None:
        output_dir = input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    fixed_count = 0
    
    if not os.path.exists(input_dir):
        logger.warning(f"Input directory {input_dir} does not exist")
        return fixed_count
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            try:
                with open(input_path, 'r') as f:
                    data = json.load(f)
                
                standardized = standardize_recommendation(data, input_path)
                
                output_path = os.path.join(output_dir, filename)
                with open(output_path, 'w') as f:
                    json.dump(standardized, f, indent=2)
                
                fixed_count += 1
                
            except Exception as e:
                logger.error(f"Error fixing recommendation file {input_path}: {e}")
    
    return fixed_count
