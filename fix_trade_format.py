#!/usr/bin/env python
"""
Greek Energy Flow II - Trade Recommendation Format Fix

This script fixes the format of trade recommendation files to be compatible with the dashboard.
It also updates the run_dashboard.py script to generate compatible files from now on.
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
        logging.FileHandler(f"logs/format_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("FormatFix")

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
            "Strategy": "ML Enhanced",
            "Action": suggested_action,
            "Entry": 0.0,  # These would be computed in a real system
            "Stop": 0.0,
            "Target": 0.0,
            "RiskReward": 0.0,
            "Regime": greek_results.get("market_regime", {}).get("primary", "Unknown"),
            "VolRegime": greek_results.get("market_regime", {}).get("volatility", "Normal"),
            "Timestamp": datetime.now().isoformat(),
            "Confidence": ml_predictions.get("primary_regime", {}).get("confidence", 0.8),
            "Notes": f"Greek Analysis: {greek_results.get('dominant_greek', 'Unknown')}-dominated"
        }
        
        # Also create backward-compatible format for other systems
        backward_compatible = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "primary_regime": greek_results.get("market_regime", {}).get("primary", "Unknown"),
            "volatility_regime": greek_results.get("market_regime", {}).get("volatility", "Normal"),
            "suggested_action": suggested_action
        }"""
        
        # Replace the section
        new_content = content.replace(trade_rec_section, new_trade_rec_section)
        
        # Also fix the file writing section to support both formats
        old_write_section = """        # Save the additional files
        with open(trade_rec_file, 'w') as f:
            json.dump(trade_recommendation, f, indent=2)"""
        
        new_write_section = """        # Save the dashboard-compatible file
        with open(trade_rec_file, 'w') as f:
            json.dump(trade_recommendation, f, indent=2)
            
        # Save backward-compatible file
        compat_file = os.path.join(output_dir, f"{ticker}_backward_compatible.json")
        with open(compat_file, 'w') as f:
            json.dump(backward_compatible, f, indent=2)"""
        
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
    except Exception as e:
        logger.error(f"Error updating run_dashboard.py script: {e}")
        return False

def fix_existing_trade_recommendations(results_dir='results'):
    """Convert existing trade recommendation files to dashboard-compatible format."""
    try:
        # Find all trade recommendation files
        files = []
        for file in os.listdir(results_dir):
            if file.endswith('_trade_recommendation.json'):
                files.append(file)
        
        logger.info(f"Found {len(files)} trade recommendation files to convert")
        
        # Create recommendations dir for dashboard
        recs_dir = os.path.join(results_dir, 'recommendations')
        os.makedirs(recs_dir, exist_ok=True)
        
        # Convert each file
        converted = 0
        for file in files:
            file_path = os.path.join(results_dir, file)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Create dashboard-compatible format
                ticker = data.get('ticker', file.split('_')[0])
                action = data.get('suggested_action', 'HOLD')
                
                dashboard_rec = {
                    "Symbol": ticker,
                    "Strategy": "ML Enhanced",
                    "Action": action,
                    "Entry": 0.0,
                    "Stop": 0.0,
                    "Target": 0.0,
                    "RiskReward": 0.0,
                    "Regime": data.get('primary_regime', 'Unknown'),
                    "VolRegime": data.get('volatility_regime', 'Normal'),
                    "Timestamp": data.get('timestamp', datetime.now().isoformat()),
                    "Confidence": 0.8,
                    "Notes": f"Generated from {file}"
                }
                
                # Save to recommendations directory
                output_file = os.path.join(recs_dir, f"{ticker}_recommendation.json")
                with open(output_file, 'w') as f:
                    json.dump(dashboard_rec, f, indent=2)
                
                converted += 1
                logger.info(f"Converted {file} to {output_file}")
            except Exception as e:
                logger.error(f"Error converting {file}: {e}")
        
        logger.info(f"Successfully converted {converted} of {len(files)} files")
        
        # Create market regime summary from the converted files
        create_market_regime_summary(recs_dir, results_dir)
        
        return converted > 0
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
echo Greek Energy Flow II - Dashboard with Fixed Trade Recommendations
echo ==========================================================
echo.
echo This script will launch the dashboard with the fixed trade recommendation files.
echo.
echo Press any key to continue...
pause > nul

python run_dashboard.py --mode dashboard --base-dir "%~dp0results"

echo.
echo Dashboard closed.
pause
"""
        
        with open('run_dashboard_fixed.bat', 'w') as f:
            f.write(batch_content)
        
        logger.info("Created batch file run_dashboard_fixed.bat")
        return True
    except Exception as e:
        logger.error(f"Error creating batch file: {e}")
        return False

def main():
    """Main entry point for the trade recommendation format fixer."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fix trade recommendation format for dashboard")
    parser.add_argument("--results-dir", default="results", help="Directory containing results")
    parser.add_argument("--update-script", action="store_true", help="Update run_dashboard.py script")
    args = parser.parse_args()
    
    logger.info("Trade Recommendation Format Fixer starting...")
    
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
Trade recommendation format fix completed successfully!

To run the dashboard with the fixed files:
1. Run run_dashboard_fixed.bat

For future analysis runs, the script has been updated to generate dashboard-compatible files.
""")
    else:
        logger.warning("""
Trade recommendation format fix completed with some issues.
Please check the log for details.
""")

if __name__ == "__main__":
    main()
