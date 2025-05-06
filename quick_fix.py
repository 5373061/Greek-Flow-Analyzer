#!/usr/bin/env python
"""
Quick fix script to comment out the problematic line in trade_dashboard.py
"""

import os
import shutil
from datetime import datetime

# Dashboard file path
dashboard_path = "D:\\python projects\\Greek Energy Flow II\\tools\\trade_dashboard.py"

# Create backup
backup_path = f"{dashboard_path}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
shutil.copy2(dashboard_path, backup_path)
print(f"Created backup at {backup_path}")

# Read the file
with open(dashboard_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the problematic line
fixed_content = content.replace(
    "            self.extract_market_regime_from_recommendations()",
    "            # Line commented out by quick_fix.py\n            # self.extract_market_regime_from_recommendations()"
)

# Add the empty method at the end of the file
if "def extract_market_regime_from_recommendations(self)" not in fixed_content:
    fixed_content += """
    def extract_market_regime_from_recommendations(self):
        \"\"\"
        Placeholder method added by quick_fix.py
        This method would normally extract market regime data from recommendations.
        \"\"\"
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Placeholder extract_market_regime_from_recommendations method called")
        pass
"""

# Write the modified content back
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print("Fixed the dashboard file.")
print("You can now run the dashboard again.")
