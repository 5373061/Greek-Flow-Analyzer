#!/usr/bin/env python
"""
Greek Energy Flow II - ML-Compatible Trade Recommendation Format Fix

This script fixes the format of ML-enhanced trade recommendation files to be 
compatible with the dashboard while preserving all ML-specific enhancements.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/ml_format_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("MLFormatFix")

def fix_existing_trade_recommendations(results_dir='results'):
    """Convert existing trade recommendation files to dashboard-compatible format."""
    try:
        # Find all trade recommendation files and enhanced recommendation files
        trade_files = []
        enhanced_files = []
        
        for file in os.listdir(results_dir):
            if file.endswith('_trade_recommendation.json'):
                trade_files.append(file)
            elif file.endswith('_enhanced_recommendation.json'):
                enhanced_files.append(file)
        
        logger.info(f"Found {len(trade_files)} trade recommendation files")
        logger.info(f"Found {len(enhanced_files)} enhanced recommendation files")
        
        # Create recommendations dir for dashboard
        recs_dir = os.path.join(results_dir, 'recommendations')
        os.makedirs(recs_dir, exist_ok=True)
        
        # Process each ticker's recommendations
        processed_tickers = set()
        
        # First, try to use enhanced recommendations as they contain more data
        for file in enhanced_files:
            ticker = file.split('_')[0]
            file_path = os.path.join(results_dir, file)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract relevant information from enhanced recommendation
                primary_regime = data.get('greek_analysis', {}).get('market_regime', {}).get('primary', 'Unknown')
                volatility_regime = data.get('greek_analysis', {}).get('market_regime', {}).get('volatility', 'Normal')
                dominant_greek = data.get('greek_analysis', {}).get('dominant_greek', 'Unknown')
                confidence = data.get('ml_predictions', {}).get('primary_regime', {}).get('confidence', 0.8)
                
                # Determine action
                action = "HOLD"
                if "Bullish" in primary_regime:
                    action = "BUY"
                elif "Bearish" in primary_regime:
                    action = "SELL"
                
                # Extract price levels if available
                entry_price = 0.0
                target_price = data.get('greek_analysis', {}).get('price_projections', {}).get('upside_target', 0.0)
                support_price = data.get('greek_analysis', {}).get('price_projections', {}).get('support_level', 0.0)
                
                # Calculate risk/reward if possible
                risk_reward = 0.0
                if action == "BUY" and entry_price > 0 and target_price > 0 and support_price > 0:
                    risk = entry_price - support_price if support_price < entry_price else 0
                    reward = target_price - entry_price if target_price > entry_price else 0
                    risk_reward = reward / risk if risk > 0 else 0
                
                # Create dashboard-compatible format
                dashboard_rec = {
                    "Symbol": ticker,
                    "Strategy": "ML Enhanced",
                    "Action": action,
                    "Entry": entry_price,
                    "Stop": support_price,
                    "Target": target_price,
                    "RiskReward": risk_reward,
                    "Regime": primary_regime,
                    "VolRegime": volatility_regime,
                    "Timestamp": data.get('timestamp', datetime.now().isoformat()),
                    "Confidence": confidence,
                    "Notes": f"Greek Analysis: {dominant_greek}-dominated"
                }
                
                # Save to recommendations directory
                output_file = os.path.join(recs_dir, f"{ticker}_recommendation.json")
                with open(output_file, 'w') as f:
                    json.dump(dashboard_rec, f, indent=2)
                
                processed_tickers.add(ticker)
                logger.info(f"Converted enhanced recommendation for {ticker}")
            except Exception as e:
                logger.error(f"Error converting enhanced recommendation {file}: {e}")
        
        # Process any remaining tickers using regular trade recommendations
        for file in trade_files:
            ticker = file.split('_')[0]
            
            # Skip if already processed using enhanced recommendations
            if ticker in processed_tickers:
                continue
                
            file_path = os.path.join(results_dir, file)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Create dashboard-compatible format
                dashboard_rec = {
                    "Symbol": ticker,
                    "Strategy": "Greek Flow",
                    "Action": data.get('suggested_action', 'HOLD'),
                    "Entry": 0.0,
                    "Stop": 0.0,
                    "Target": 0.0,
                    "RiskReward": 0.0,
                    "Regime": data.get('primary_regime', 'Unknown'),
                    "VolRegime": data.get('volatility_regime', 'Normal'),
                    "Timestamp": data.get('timestamp', datetime.now().isoformat()),
                    "Confidence": 0.7,
                    "Notes": "Generated from basic trade recommendation"
                }
                
                # Save to recommendations directory
                output_file = os.path.join(recs_dir, f"{ticker}_recommendation.json")
                with open(output_file, 'w') as f:
                    json.dump(dashboard_rec, f, indent=2)
                
                processed_tickers.add(ticker)
                logger.info(f"Converted basic recommendation for {ticker}")
            except Exception as e:
                logger.error(f"Error converting trade recommendation {file}: {e}")
        
        logger.info(f"Successfully converted recommendations for {len(processed_tickers)} tickers")
        
        # Create market regime summary from the converted files
        create_market_regime_summary(recs_dir, results_dir)
        
        return len(processed_tickers) > 0
    except Exception as e:
        logger.error(f"Error converting trade recommendation files: {e}")
        return False

def create_market_regime_summary(recs_dir, results_dir):
    """Create a market regime summary file for the dashboard."""
    try:
        # Load all recommendation files
        recs = []
        for file in os.listdir(recs_dir):
            if file.endswith('_recommendation.json'):
                try:
                    with open(os.path.join(recs_dir, file), 'r') as f:
                        data = json.load(f)
                    recs.append(data)
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
        
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
        regime_file = os.path.join(results_dir, "market_regime.json")
        with open(regime_file, 'w') as f:
            json.dump(market_regime, f, indent=2)
        
        logger.info(f"Created market regime summary: {overall_bias} " +
                   f"(Bullish: {regime_counts['Bullish']}, " +
                   f"Bearish: {regime_counts['Bearish']}, " +
                   f"Neutral: {regime_counts['Neutral']})")
        
        return True
    except Exception as e:
        logger.error(f"Error creating market regime summary: {e}")
        return False

def create_batch_file():
    """Create a batch file to quickly run the dashboard with the fixed files."""
    try:
        batch_content = """@echo off
echo Greek Energy Flow II - ML Dashboard with Fixed Trade Recommendations
echo ============================================================
echo.
echo This script will launch the dashboard with the fixed ML-enhanced trade recommendation files.
echo.
echo Press any key to continue...
pause > nul

python run_dashboard.py --mode dashboard --base-dir "%~dp0results"

echo.
echo Dashboard closed.
pause
"""
        
        with open('run_ml_dashboard.bat', 'w') as f:
            f.write(batch_content)
        
        logger.info("Created batch file run_ml_dashboard.bat")
        return True
    except Exception as e:
        logger.error(f"Error creating batch file: {e}")
        return False

def main():
    """Main entry point for the ML trade recommendation format fixer."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fix ML trade recommendation format for dashboard")
    parser.add_argument("--results-dir", default="results", help="Directory containing results")
    args = parser.parse_args()
    
    logger.info("ML Trade Recommendation Format Fixer starting...")
    
    # Fix existing trade recommendations
    fixed_files = fix_existing_trade_recommendations(args.results_dir)
    
    # Create batch file
    created_batch = create_batch_file()
    
    if fixed_files and created_batch:
        logger.info("""
ML trade recommendation format fix completed successfully!

To run the dashboard with the fixed files:
1. Run run_ml_dashboard.bat

The dashboard should now display your ML-enhanced recommendations correctly.
""")
    else:
        logger.warning("""
ML trade recommendation format fix completed with some issues.
Please check the log for details.
""")

if __name__ == "__main__":
    main()
