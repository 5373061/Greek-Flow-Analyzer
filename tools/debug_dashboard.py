#!/usr/bin/env python
"""
Debug utility for the trade dashboard
"""

import os
import json
import glob
import argparse

def check_recommendation_files():
    """Check if recommendation files exist and have the correct format"""
    print("Checking recommendation files...")
    
    # Check if any recommendation files exist
    rec_files = glob.glob(os.path.join("results", "*_recommendation.json"))
    if not rec_files:
        print("ERROR: No recommendation files found in results directory")
        return False
    
    print(f"Found {len(rec_files)} recommendation files")
    
    # Check a sample file
    sample_file = rec_files[0]
    print(f"Checking sample file: {sample_file}")
    
    try:
        with open(sample_file, 'r') as f:
            data = json.load(f)
        
        # Check required fields
        required_fields = ["symbol", "strategy_name", "entry_price", "stop_loss", "target_price", "risk_category"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            print(f"ERROR: Missing required fields in {sample_file}: {missing_fields}")
            print("Current fields:", list(data.keys()))
            return False
        
        print("Recommendation file format looks correct")
        return True
    
    except Exception as e:
        print(f"ERROR: Failed to parse {sample_file}: {e}")
        return False

def check_market_regime_file():
    """Check if market regime file exists and has the correct format"""
    print("\nChecking market regime file...")
    
    # Check primary location
    regime_file = os.path.join("results", "market_regime", "current_regime.json")
    if not os.path.exists(regime_file):
        print(f"ERROR: Market regime file not found at {regime_file}")
        return False
    
    try:
        with open(regime_file, 'r') as f:
            data = json.load(f)
        
        # Check required fields
        required_fields = ["primary_label", "volatility_regime"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            print(f"ERROR: Missing required fields in {regime_file}: {missing_fields}")
            print("Current fields:", list(data.keys()))
            return False
        
        print("Market regime file format looks correct")
        return True
    
    except Exception as e:
        print(f"ERROR: Failed to parse {regime_file}: {e}")
        return False

def fix_recommendation_files():
    """Fix common issues with recommendation files"""
    print("\nFixing recommendation files...")
    
    rec_files = glob.glob(os.path.join("results", "*_recommendation.json"))
    if not rec_files:
        print("No recommendation files found to fix")
        return
    
    for file in rec_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            # Fix field names
            field_mappings = {
                "Symbol": "symbol",
                "Strategy": "strategy_name",
                "Action": "action",
                "Entry": "entry_price",
                "Stop": "stop_loss",
                "Target": "target_price",
                "RiskReward": "rr_ratio",
                "Regime": "market_regime",
                "VolRegime": "volatility_regime",
                "Risk": "risk_category",
                "TradeContext": "trade_context"
            }
            
            fixed_data = {}
            for old_key, new_key in field_mappings.items():
                if old_key in data:
                    fixed_data[new_key] = data[old_key]
                elif new_key in data:
                    fixed_data[new_key] = data[new_key]
            
            # Add any missing fields
            if "rr_ratio" in fixed_data and "rr_ratio_str" not in fixed_data:
                fixed_data["rr_ratio_str"] = f"{fixed_data['rr_ratio']:.1f}:1"
            
            # Save fixed data
            with open(file, 'w') as f:
                json.dump(fixed_data, f, indent=2)
            
            print(f"Fixed {file}")
        
        except Exception as e:
            print(f"ERROR: Failed to fix {file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug utility for the trade dashboard")
    parser.add_argument("--check", action="store_true", help="Check recommendation and market regime files")
    parser.add_argument("--fix", action="store_true", help="Fix common issues with recommendation files")
    
    args = parser.parse_args()
    
    if args.check or not (args.check or args.fix):
        check_recommendation_files()
        check_market_regime_file()
    
    if args.fix:
        fix_recommendation_files()