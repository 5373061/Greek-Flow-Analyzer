#!/usr/bin/env python
"""
Fix the dashboard canvas issue
"""

import os
import re

def fix_dashboard_canvas():
    """Fix the canvas attribute error in the dashboard"""
    dashboard_path = os.path.join("tools", "trade_dashboard.py")
    
    if not os.path.exists(dashboard_path):
        print(f"Error: Dashboard file not found at {dashboard_path}")
        return False
    
    # Create a backup
    backup_path = f"{dashboard_path}.bak.{os.path.getmtime(dashboard_path):.0f}"
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_content)
    
    print(f"Created backup at {backup_path}")
    
    # Fix the clear_details_view method
    clear_details_pattern = r"def clear_details_view\(self\):(.*?)if self\.canvas: self\.canvas\.get_tk_widget\(\)\.destroy\(\)"
    clear_details_replacement = r"def clear_details_view(self):\1if hasattr(self, 'canvas') and self.canvas: self.canvas.get_tk_widget().destroy()"
    
    modified_content = re.sub(clear_details_pattern, clear_details_replacement, original_content, flags=re.DOTALL)
    
    # Fix the update_price_chart method
    update_chart_pattern = r"def update_price_chart\(self, rec\):(.*?)if self\.canvas:"
    update_chart_replacement = r"def update_price_chart(self, rec):\1if hasattr(self, 'canvas') and self.canvas:"
    
    modified_content = re.sub(update_chart_pattern, update_chart_replacement, modified_content, flags=re.DOTALL)
    
    # Add canvas initialization in __init__ if it doesn't exist
    init_pattern = r"def __init__\(self, root\):(.*?)# Initialize UI components"
    canvas_init = "\n        # Initialize canvas attribute\n        self.canvas = None\n"
    init_replacement = r"def __init__(self, root):\1" + canvas_init + r"# Initialize UI components"
    
    modified_content = re.sub(init_pattern, init_replacement, modified_content, flags=re.DOTALL)
    
    # Write the modified content
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"Fixed canvas attribute error in {dashboard_path}")
    return True

if __name__ == "__main__":
    if fix_dashboard_canvas():
        print("\nFix applied successfully. Please restart the dashboard.")
    else:
        print("\nFailed to apply fix.")