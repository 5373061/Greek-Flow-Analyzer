#!/usr/bin/env python
"""
Debug the dashboard's recommendation loading process
"""

import os
import glob
import json
import sys

def debug_recommendation_files():
    """Check recommendation files and their locations"""
    print("=== Recommendation File Debug ===")
    
    # Check all possible locations
    locations = [
        "results",
        "results/recommendations",
        "output",
        "output/recommendations",
        "data/recommendations"
    ]
    
    for location in locations:
        if os.path.exists(location):
            files = glob.glob(os.path.join(location, "*_recommendation.json"))
            print(f"\nLocation: {location}")
            print(f"Found {len(files)} recommendation files")
            
            if files:
                # Show a sample
                sample_file = files[0]
                print(f"Sample file: {sample_file}")
                try:
                    with open(sample_file, 'r') as f:
                        data = json.load(f)
                    print(f"Keys: {list(data.keys())}")
                    print(f"Symbol: {data.get('symbol', 'NOT FOUND')}")
                    print(f"Strategy: {data.get('strategy_name', data.get('Strategy', 'NOT FOUND'))}")
                except Exception as e:
                    print(f"Error reading file: {e}")
        else:
            print(f"\nLocation: {location} (does not exist)")
    
    # Try to import the dashboard module to check its loading logic
    print("\n=== Dashboard Module Import Test ===")
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from tools.trade_dashboard import IntegratedDashboard
        print("Successfully imported IntegratedDashboard")
        
        # Check if we can access the load_recommendations method
        if hasattr(IntegratedDashboard, 'load_recommendations'):
            print("Dashboard has load_recommendations method")
        else:
            print("Dashboard does NOT have load_recommendations method")
            
        # Check other potential method names
        for method_name in ['load_trade_recommendations', 'load_trades', 'refresh_recommendations']:
            if hasattr(IntegratedDashboard, method_name):
                print(f"Dashboard has {method_name} method")
    except Exception as e:
        print(f"Error importing dashboard: {e}")
    
    # Create a test recommendation in all possible locations
    print("\n=== Creating Test Recommendations ===")
    test_recommendation = {
        "symbol": "TEST",
        "strategy_name": "Ordinal",
        "action": "BUY",
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "target_price": 110.0,
        "rr_ratio": 2.0,
        "rr_ratio_str": "2.0:1",
        "market_regime": "Bullish Trend",
        "volatility_regime": "Normal",
        "timestamp": "2023-01-01T12:00:00",
        "confidence": 0.8,
        "notes": "Test recommendation",
        "risk_category": "MEDIUM"
    }
    
    for location in locations:
        os.makedirs(location, exist_ok=True)
        file_path = os.path.join(location, "TEST_recommendation.json")
        try:
            with open(file_path, 'w') as f:
                json.dump(test_recommendation, f, indent=2)
            print(f"Created test recommendation in {file_path}")
        except Exception as e:
            print(f"Error creating test file in {location}: {e}")

if __name__ == "__main__":
    debug_recommendation_files()