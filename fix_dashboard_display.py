"""
Fix Dashboard Display Script

This script fixes common issues with the dashboard display and ensures
that all recommendation files are in the correct format.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dashboard_fix.log')
    ]
)

logger = logging.getLogger(__name__)

def fix_recommendation_files(args):
    """Fix recommendation files"""
    try:
        from utils.dashboard_compatibility import fix_recommendation_files
        
        input_dir = args.input_dir
        output_dir = args.output_dir or input_dir
        
        logger.info(f"Fixing recommendation files in {input_dir}")
        fixed_count = fix_recommendation_files(input_dir, output_dir)
        
        logger.info(f"Fixed {fixed_count} recommendation files")
        return fixed_count > 0
        
    except ImportError:
        logger.error("Failed to import dashboard_compatibility module")
        return False

def check_dashboard_display():
    """Check dashboard display issues"""
    dashboard_path = os.path.join("tools", "trade_dashboard.py")
    
    if not os.path.exists(dashboard_path):
        logger.error(f"Dashboard file not found at {dashboard_path}")
        return False
    
    logger.info("Dashboard file found, checking for display issues")
    
    # Check for common issues in the dashboard file
    with open(dashboard_path, 'r') as f:
        content = f.read()
    
    issues = []
    
    # Check for missing trade context handling
    if "TradeContext" not in content:
        issues.append("Dashboard may not handle TradeContext properly")
    
    # Check for proper symbol extraction
    if "symbol = data.get(\"Symbol\", data.get(\"symbol\"" not in content:
        issues.append("Dashboard may not extract symbol correctly")
    
    # Check for proper action extraction
    if "action = data.get(\"Action\", data.get(\"action\"" not in content:
        issues.append("Dashboard may not extract action correctly")
    
    if issues:
        logger.warning("Potential dashboard display issues found:")
        for issue in issues:
            logger.warning(f"- {issue}")
    else:
        logger.info("No obvious dashboard display issues found")
    
    return True

def create_sample_recommendations(args):
    """Create sample recommendations for testing"""
    try:
        # Try to import the debug_trade_recommendations module
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from debug_trade_recommendations import generate_sample_recommendations
        
        symbols = args.symbols or ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        output_dir = args.output_dir or "sample_recommendations"
        count = args.count or 3
        
        logger.info(f"Generating {count} sample recommendations for {len(symbols)} symbols")
        success = generate_sample_recommendations(symbols, output_dir, count)
        
        if success:
            logger.info(f"Successfully generated sample recommendations in {output_dir}")
        else:
            logger.error("Failed to generate sample recommendations")
        
        return success
        
    except ImportError:
        logger.error("Failed to import debug_trade_recommendations module")
        return False

def check_trade_context(args):
    """Check if trade context is included in recommendation files"""
    input_dir = args.input_dir
    
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist")
        return False
    
    logger.info(f"Checking trade context in {input_dir}")
    
    files_with_context = 0
    files_without_context = 0
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(input_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                if "TradeContext" in data and data["TradeContext"]:
                    files_with_context += 1
                    logger.debug(f"{filename} has trade context")
                else:
                    files_without_context += 1
                    logger.warning(f"{filename} does not have trade context")
                    
            except Exception as e:
                logger.error(f"Error checking {filepath}: {e}")
    
    logger.info(f"Found {files_with_context} files with trade context and {files_without_context} files without")
    
    return files_with_context > 0

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Fix dashboard display issues")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Fix recommendation files command
    fix_parser = subparsers.add_parser("fix", help="Fix recommendation files")
    fix_parser.add_argument("--input-dir", default="results", help="Input directory containing recommendation files")
    fix_parser.add_argument("--output-dir", default=None, help="Output directory (defaults to input directory)")
    
    # Check dashboard display command
    check_parser = subparsers.add_parser("check", help="Check dashboard display issues")
    
    # Create sample recommendations command
    sample_parser = subparsers.add_parser("sample", help="Create sample recommendations for testing")
    sample_parser.add_argument("--symbols", nargs="+", help="List of ticker symbols")
    sample_parser.add_argument("--output-dir", default=None, help="Output directory")
    sample_parser.add_argument("--count", type=int, help="Number of recommendations per symbol")
    
    # Check trade context command
    context_parser = subparsers.add_parser("context", help="Check if trade context is included in recommendation files")
    context_parser.add_argument("--input-dir", default="results", help="Input directory containing recommendation files")
    
    args = parser.parse_args()
    
    if args.command == "fix":
        if not fix_recommendation_files(args):
            return 1
    elif args.command == "check":
        if not check_dashboard_display():
            return 1
    elif args.command == "sample":
        if not create_sample_recommendations(args):
            return 1
    elif args.command == "context":
        if not check_trade_context(args):
            return 1
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())