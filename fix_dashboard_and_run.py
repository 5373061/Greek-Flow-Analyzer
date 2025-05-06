#!/usr/bin/env python
"""
Fix Dashboard and Run

This script fixes the dashboard implementation and runs it.
"""

import os
import sys
import logging
import tkinter as tk
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/dashboard_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DashboardFixer")

def fix_dashboard():
    """Fix the dashboard implementation."""
    # Check if trade_dashboard.py exists
    dashboard_path = os.path.join("tools", "trade_dashboard.py")
    if not os.path.exists(dashboard_path):
        logger.error(f"Dashboard file not found: {dashboard_path}")
        return False
    
    # Create backup
    backup_path = f"{dashboard_path}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    try:
        import shutil
        shutil.copy2(dashboard_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return False
    
    # Read the dashboard file
    try:
        with open(dashboard_path, "r") as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Failed to read dashboard file: {e}")
        return False
    
    # Check if extract_market_regime_from_recommendations method exists
    if "def extract_market_regime_from_recommendations" not in content:
        logger.info("Adding extract_market_regime_from_recommendations method")
        
        # Add the method
        method_code = """
    def extract_market_regime_from_recommendations(self):
        \"\"\"
        Extract market regime information from loaded recommendations.
        
        Returns:
            dict: Market regime information
        \"\"\"
        # Default regime information
        regime_info = {
            "primary": "Neutral",
            "volatility": "Normal",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Try to extract from recommendations
        if self.recommendations:
            # Count regimes
            regime_counts = {}
            vol_regime_counts = {}
            
            for rec in self.recommendations:
                # Extract primary regime
                regime = rec.get("Regime", "")
                if regime:
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
                # Extract volatility regime
                vol_regime = rec.get("VolRegime", "")
                if vol_regime:
                    vol_regime_counts[vol_regime] = vol_regime_counts.get(vol_regime, 0) + 1
                
                # Extract from TradeContext if available
                trade_context = rec.get("TradeContext", {})
                if trade_context:
                    market_regime = trade_context.get("market_regime", "")
                    if market_regime:
                        regime_counts[market_regime] = regime_counts.get(market_regime, 0) + 1
                    
                    vol_regime = trade_context.get("volatility_regime", "")
                    if vol_regime:
                        vol_regime_counts[vol_regime] = vol_regime_counts.get(vol_regime, 0) + 1
            
            # Find most common regimes
            if regime_counts:
                regime_info["primary"] = max(regime_counts.items(), key=lambda x: x[1])[0]
            
            if vol_regime_counts:
                regime_info["volatility"] = max(vol_regime_counts.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Inferred market regimes from recommendations: {regime_info}")
        return regime_info
"""
        
        # Find a good place to insert the method (before the last method)
        import re
        last_method_match = re.search(r"    def [a-zA-Z_]+\([^)]*\):[^\n]*\n", content)
        if last_method_match:
            insert_pos = last_method_match.start()
            content = content[:insert_pos] + method_code + content[insert_pos:]
        else:
            # Append to the end of the class
            content += method_code
    
    # Check if refresh_data method exists
    if "def refresh_data" not in content:
        logger.info("Adding refresh_data method")
        
        # Add the method
        method_code = """
    def refresh_data(self):
        \"\"\"Refresh all data from source files.\"\"\"
        logger.info("Refreshing dashboard data...")
        
        # Reload recommendations
        self.load_recommendations()
        
        # Reload market regimes
        self.load_market_regimes()
        
        # Update UI
        self.update_recommendation_list()
        
        # Update status
        self.update_status("Data refreshed successfully", "info")
        
        logger.info("Data refresh complete")
"""
        
        # Find a good place to insert the method (before the last method)
        import re
        last_method_match = re.search(r"    def [a-zA-Z_]+\([^)]*\):[^\n]*\n", content)
        if last_method_match:
            insert_pos = last_method_match.start()
            content = content[:insert_pos] + method_code + content[insert_pos:]
        else:
            # Append to the end of the class
            content += method_code
    
    # Check if results_dir is initialized in __init__
    init_match = re.search(r"def __init__\([^)]*\):[^#]*?self\.base_dir = [^\n]+", content, re.DOTALL)
    if init_match and "self.results_dir" not in init_match.group(0):
        logger.info("Adding results_dir initialization to __init__")
        
        # Add the initialization
        init_code = init_match.group(0)
        new_init_code = init_code + "\n        # Set results directory\n        self.results_dir = os.path.join(self.base_dir, \"results\")"
        content = content.replace(init_code, new_init_code)
    
    # Write the updated content
    try:
        with open(dashboard_path, "w") as f:
            f.write(content)
        logger.info(f"Updated dashboard file: {dashboard_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write dashboard file: {e}")
        return False

def run_dashboard():
    """Run the dashboard."""
    logger.info("Running dashboard...")
    
    # Import the dashboard
    try:
        from tools.trade_dashboard import IntegratedDashboard
        logger.info("Successfully imported IntegratedDashboard")
    except ImportError as e:
        logger.error(f"Failed to import IntegratedDashboard: {e}")
        return False
    
    # Create Tkinter root
    root = tk.Tk()
    root.title("Greek Energy Flow Dashboard")
    root.geometry("1200x800")
    
    # Create dashboard
    try:
        dashboard = IntegratedDashboard(root)
        logger.info("Created dashboard")
    except Exception as e:
        logger.error(f"Failed to create dashboard: {e}")
        return False
    
    # Run the main loop
    try:
        root.mainloop()
        return True
    except Exception as e:
        logger.error(f"Error in dashboard main loop: {e}")
        return False

def main():
    """Main function."""
    logger.info("Starting dashboard fixer and runner")
    
    # Fix the dashboard
    if not fix_dashboard():
        logger.error("Failed to fix dashboard")
        return 1
    
    # Run the dashboard
    if not run_dashboard():
        logger.error("Failed to run dashboard")
        return 1
    
    logger.info("Dashboard fixed and run successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())