#!/usr/bin/env python
"""
Greek Energy Flow II - Unified Trade Recommendation Format

This script ensures all trade recommendation files are in a consistent format
compatible with the dashboard, regardless of their source (basic or ML-enhanced).
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/unified_format_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("UnifiedFormat")

def fix_run_dashboard_script():
    """Modify the run_dashboard.py script to generate dashboard-compatible trade recommendations."""
    try:
        # Read the run_dashboard.py file
        with open('run_dashboard.py', 'r') as f:
            content = f.read()
        
        # Find the trade recommendation generation section
        trade_rec_section = """        # Extract relevant portions for each file
        trade_recommendation = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "primary_regime": greek_results.get("market_regime", {}).get("primary", "Unknown"),
            "volatility_regime": greek_results.get("market_regime", {}).get("volatility", "Normal"),
            "suggested_action": "BUY" if "Bullish" in greek_results.get("market_regime", {}).get("primary", "") else 
                              "SELL" if "Bearish" in greek_results.get("market_regime", {}).get("primary", "") else 
                              "HOLD"
        }"""
        
        # New trade recommendation format compatible with dashboard
        new_trade_rec_section = """        # Extract relevant portions for each file
        suggested_action = "BUY" if "Bullish" in greek_results.get("market_regime", {}).get("primary", "") else \\
                          "SELL" if "Bearish" in greek_results.get("market_regime", {}).get("primary", "") else \\
                          "HOLD"
        
        # Create trade recommendation compatible with dashboard
        trade_recommendation = {
            "Symbol": ticker,
            "Strategy": "ML Enhanced" if ml_predictions else "Greek Flow",
            "Action": suggested_action,
            "Entry": 0.0,  # These would be computed in a real system
            "Stop": 0.0,
            "Target": 0.0,
            "RiskReward": 0.0,
            "Regime": greek_results.get("market_regime", {}).get("primary", "Unknown"),
            "VolRegime": greek_results.get("market_regime", {}).get("volatility", "Normal"),
            "Timestamp": datetime.now().isoformat(),
            "Confidence": ml_predictions.get("primary_regime", {}).get("confidence", 0.8) if ml_predictions else 0.7,
            "Notes": f"Greek Analysis: {greek_results.get('dominant_greek', 'Unknown')}-dominated",
            "TradeContext": {
                "market_regime": greek_results.get("market_regime", {}),
                "volatility_regime": greek_results.get("market_regime", {}).get("volatility", "Normal"),
                "hold_time_days": 7  # Default hold time
            },
            "HoldTimeDays": 7  # For backward compatibility
        }"""
        
        # Replace the section if it exists
        if trade_rec_section in content:
            new_content = content.replace(trade_rec_section, new_trade_rec_section)
            
            # Also fix the file writing section to support both formats
            old_write_section = """        # Save the additional files
        with open(trade_rec_file, 'w') as f:
            json.dump(trade_recommendation, f, indent=2)"""
            
            new_write_section = """        # Save the dashboard-compatible file
        with open(trade_rec_file, 'w') as f:
            json.dump(trade_recommendation, f, indent=2)"""
            
            new_content = new_content.replace(old_write_section, new_write_section)
            
            # Write the modified file
            with open('run_dashboard.py.new', 'w') as f:
                f.write(new_content)
            
            # Backup the original file
            os.rename('run_dashboard.py', 'run_dashboard.py.bak')
            
            # Rename the new file
            os.rename('run_dashboard.py.new', 'run_dashboard.py')
            
            logger.info("Successfully updated run_dashboard.py script to generate dashboard-compatible files")
            return True
        else:
            logger.warning("Could not find the trade recommendation section in run_dashboard.py")
            return False
    except Exception as e:
        logger.error(f"Error updating run_dashboard.py script: {e}")
        return False

def convert_to_dashboard_format(data, symbol, file_path):
    """Convert trade recommendation data to dashboard-compatible format."""
    try:
        # Check if already in dashboard format
        if "Symbol" in data and "Action" in data and "Strategy" in data:
            # Already in dashboard format, just ensure all required fields
            dashboard_rec = data.copy()
            
            # Ensure required fields
            if "TradeContext" not in dashboard_rec:
                dashboard_rec["TradeContext"] = {
                    "market_regime": {
                        "primary": dashboard_rec.get("Regime", "Unknown"),
                        "volatility": dashboard_rec.get("VolRegime", "Normal")
                    },
                    "volatility_regime": dashboard_rec.get("VolRegime", "Normal"),
                    "hold_time_days": dashboard_rec.get("HoldTimeDays", 7)
                }
            
            if "HoldTimeDays" not in dashboard_rec:
                dashboard_rec["HoldTimeDays"] = dashboard_rec.get("TradeContext", {}).get("hold_time_days", 7)
            
            return dashboard_rec
        
        # Extract basic information
        if "ticker" in data:
            symbol = data.get("ticker", symbol)
        
        # Determine action
        action = "HOLD"
        if "suggested_action" in data:
            action = data.get("suggested_action")
        elif "primary_regime" in data:
            regime = data.get("primary_regime", "")
            if "Bullish" in regime:
                action = "BUY"
            elif "Bearish" in regime:
                action = "SELL"
        
        # Determine strategy
        strategy = "Greek Flow"
        if "ml_predictions" in data or "enhanced" in os.path.basename(file_path):
            strategy = "ML Enhanced"
        
        # Extract regime information
        primary_regime = data.get("primary_regime", "Unknown")
        volatility_regime = data.get("volatility_regime", "Normal")
        
        # Create dashboard-compatible format
        dashboard_rec = {
            "Symbol": symbol,
            "Strategy": strategy,
            "Action": action,
            "Entry": data.get("entry_price", 0.0),
            "Stop": data.get("stop_price", 0.0),
            "Target": data.get("target_price", 0.0),
            "RiskReward": data.get("risk_reward", 0.0),
            "Regime": primary_regime,
            "VolRegime": volatility_regime,
            "Timestamp": data.get("timestamp", datetime.now().isoformat()),
            "Confidence": data.get("confidence", 0.7),
            "Notes": data.get("notes", "Generated from trade recommendation"),
            "TradeContext": {
                "market_regime": {
                    "primary": primary_regime,
                    "volatility": volatility_regime
                },
                "volatility_regime": volatility_regime,
                "hold_time_days": data.get("hold_time_days", 7)
            },
            "HoldTimeDays": data.get("hold_time_days", 7)
        }
        
        return dashboard_rec
    except Exception as e:
        logger.error(f"Error converting {file_path} to dashboard format: {e}")
        return None

def fix_existing_trade_recommendations(results_dir='results'):
    """Convert existing trade recommendation files to dashboard-compatible format."""
    try:
        # Find all trade recommendation files
        rec_files = []
        
        # Look for files with specific patterns
        patterns = [
            os.path.join(results_dir, "*_trade_recommendation.json"),
            os.path.join(results_dir, "*_enhanced_recommendation.json"),
            os.path.join(results_dir, "recommendations", "*.json")
        ]
        
        for pattern in patterns:
            rec_files.extend(glob.glob(pattern))
        
        logger.info(f"Found {len(rec_files)} recommendation files to process")
        
        # Create recommendations dir for dashboard
        recs_dir = os.path.join(results_dir, "recommendations")
        os.makedirs(recs_dir, exist_ok=True)
        
        # Process each file
        processed = 0
        for file_path in rec_files:
            try:
                # Extract symbol from filename
                symbol = os.path.basename(file_path).split('_')[0]
                
                # Read the file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Convert to dashboard format
                dashboard_rec = convert_to_dashboard_format(data, symbol, file_path)
                
                if dashboard_rec:
                    # Save to recommendations directory
                    output_file = os.path.join(recs_dir, f"{symbol}_recommendation.json")
                    with open(output_file, 'w') as f:
                        json.dump(dashboard_rec, f, indent=2)
                    
                    processed += 1
                    logger.info(f"Processed {file_path} -> {output_file}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Successfully processed {processed} of {len(rec_files)} files")
        
        # Create market regime summary
        create_market_regime_summary(recs_dir, results_dir)
        
        return processed > 0
    except Exception as e:
        logger.error(f"Error processing trade recommendation files: {e}")
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
echo Greek Energy Flow II - Dashboard with Unified Trade Recommendations
echo ==========================================================
echo.
echo This script will launch the dashboard with the unified trade recommendation files.
echo.
echo Press any key to continue...
pause > nul

python run_dashboard.py --mode dashboard --base-dir "%~dp0results"

echo.
echo Dashboard closed.
pause
"""
        
        with open('run_unified_dashboard.bat', 'w') as f:
            f.write(batch_content)
        
        logger.info("Created batch file run_unified_dashboard.bat")
        return True
    except Exception as e:
        logger.error(f"Error creating batch file: {e}")
        return False

def main():
    """Main entry point for the unified trade recommendation format script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Unify trade recommendation format for dashboard")
    parser.add_argument("--results-dir", default="results", help="Directory containing results")
    parser.add_argument("--update-script", action="store_true", help="Update run_dashboard.py script")
    args = parser.parse_args()
    
    logger.info("Unified Trade Recommendation Format script starting...")
    
    # Fix existing trade recommendations
    fixed_files = fix_existing_trade_recommendations(args.results_dir)
    
    # Update the run_dashboard.py script
    if args.update_script:
        updated_script = fix_run_dashboard_script()
    else:
        updated_script = False
        logger.info("Skipping script update as requested")
    
    # Create batch file
    created_batch = create_batch_file()
    
    if fixed_files and created_batch:
        logger.info("""
Unified trade recommendation format completed successfully!

To run the dashboard with the unified files:
1. Run run_unified_dashboard.bat

All trade recommendations are now in a consistent format compatible with the dashboard.
""")
    else:
        logger.warning("""
Unified trade recommendation format completed with some issues.
Please check the log for details.
""")

if __name__ == "__main__":
    main()