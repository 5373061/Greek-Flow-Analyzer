"""
Market Regime Extraction Helper for Trade Dashboard

This script adds the missing extract_market_regime_from_recommendations method
to the IntegratedDashboard class in trade_dashboard.py
"""

import os
import json
import logging
from datetime import datetime

logger = logging.getLogger("MarketRegimeExtraction")

def add_extract_method():
    """
    This function adds the missing extract_market_regime_from_recommendations method
    by appending it to the trade_dashboard.py file.
    """
    # Dashboard file path
    dashboard_path = "D:\\python projects\\Greek Energy Flow II\\tools\\trade_dashboard.py"
    
    # Method to add
    method_code = """
    def extract_market_regime_from_recommendations(self):
        \"\"\"
        Extract market regime data from recommendations and create a market_regime.json file.
        \"\"\"
        try:
            logger.info("Extracting market regime from recommendations...")
            
            if not self.recommendations:
                logger.warning("No recommendations available to extract market regime")
                return
            
            # Extract regimes from recommendations
            regimes = {}
            regime_counts = {"Bullish": 0, "Bearish": 0, "Neutral": 0, "Unknown": 0}
            
            for rec in self.recommendations:
                ticker = rec.get("Symbol", "")
                if not ticker:
                    continue
                    
                # Get regime from recommendation
                regime = rec.get("Regime", "Unknown")
                regimes[ticker] = regime
                
                # Count by regime type
                if "Bull" in regime:
                    regime_counts["Bullish"] += 1
                elif "Bear" in regime:
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
            regime_file = os.path.join(self.base_dir, "results", "market_regime.json")
            os.makedirs(os.path.join(self.base_dir, "results"), exist_ok=True)
            
            with open(regime_file, 'w') as f:
                json.dump(market_regime, f, indent=2)
            
            logger.info(f"Created market regime summary: {overall_bias} " +
                       f"(Bullish: {regime_counts['Bullish']}, " +
                       f"Bearish: {regime_counts['Bearish']}, " +
                       f"Neutral: {regime_counts['Neutral']})")
                       
            # Store market regime data for dashboard use
            self.market_regime = market_regime
            
        except Exception as e:
            logger.error(f"Error extracting market regime from recommendations: {e}")
    """
    
    try:
        # Create a backup of the original file
        with open(dashboard_path, 'r') as f:
            dashboard_code = f.read()
        
        # Create backup
        backup_path = dashboard_path + ".bak"
        with open(backup_path, 'w') as f:
            f.write(dashboard_code)
        
        # Check if method already exists
        if "def extract_market_regime_from_recommendations" in dashboard_code:
            print("Method already exists in the dashboard file")
            return False
        
        # Find a good place to insert the method - after extract_energy_and_reset_from_recommendations
        if "def extract_energy_and_reset_from_recommendations" in dashboard_code:
            # Split at that method and add our new method after it
            parts = dashboard_code.split("def extract_energy_and_reset_from_recommendations")
            
            # Find the end of that method
            method_part = parts[1]
            lines = method_part.split("\n")
            
            # Find the next method or the end of the class
            end_index = 0
            in_method = True
            indent_level = 0  # Expected indentation for class methods
            
            for i, line in enumerate(lines):
                if i == 0:  # First line is the method definition
                    continue
                    
                # Check if we've found a new method or we're back at class level
                if line.strip().startswith("def ") and line.startswith(" " * indent_level):
                    end_index = i
                    break
            
            if end_index > 0:
                # Insert our method before the next method
                new_dashboard_code = parts[0] + "def extract_energy_and_reset_from_recommendations" + "\n".join(lines[:end_index]) + method_code + "\n".join(lines[end_index:])
                
                # Write the modified file
                with open(dashboard_path, 'w') as f:
                    f.write(new_dashboard_code)
                    
                print("Successfully added the missing method to the dashboard!")
                return True
        
        # If we couldn't find a good insertion point, append to file
        with open(dashboard_path, 'a') as f:
            f.write("\n\n# Added by fix script\n" + method_code)
            
        print("Added method to the end of the dashboard file")
        return True
        
    except Exception as e:
        print(f"Error adding method to dashboard: {e}")
        return False

if __name__ == "__main__":
    print("Adding missing extract_market_regime_from_recommendations method to trade_dashboard.py")
    success = add_extract_method()
    if success:
        print("Success! You can now run the dashboard.")
    else:
        print("Failed to add the method. Please add it manually.")
