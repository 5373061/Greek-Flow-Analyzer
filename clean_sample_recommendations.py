#!/usr/bin/env python
"""
Clean only sample recommendations without affecting other trade recommendation files
"""

import os
import glob
import json
import argparse

def clean_sample_recommendations(dry_run=True):
    """
    Remove only sample recommendations created by create_sample_recommendations.py
    
    Args:
        dry_run (bool): If True, only print what would be deleted without actually deleting
    """
    # Sample tickers used by create_sample_recommendations.py
    sample_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "SPY"]
    
    # Find all recommendation files
    results_dir = "results"
    recommendation_files = []
    
    # Check main results directory
    for ticker in sample_tickers:
        pattern = os.path.join(results_dir, f"{ticker}_recommendation.json")
        recommendation_files.extend(glob.glob(pattern))
    
    # Check recommendations subdirectory
    recommendations_dir = os.path.join(results_dir, "recommendations")
    if os.path.exists(recommendations_dir):
        for ticker in sample_tickers:
            pattern = os.path.join(recommendations_dir, f"{ticker}_recommendation.json")
            recommendation_files.extend(glob.glob(pattern))
    
    # Process each file
    deleted_count = 0
    for file_path in recommendation_files:
        try:
            # Check if this is a sample file by looking at its content
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if this looks like a sample file
            is_sample = False
            if "notes" in data and "Ordinal pattern analysis" in data.get("notes", ""):
                is_sample = True
            elif "trade_context" in data and "ordinal_patterns" in data.get("trade_context", {}):
                is_sample = True
            
            if is_sample:
                if dry_run:
                    print(f"Would delete: {file_path}")
                else:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                deleted_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if dry_run:
        print(f"Would delete {deleted_count} sample recommendation files")
    else:
        print(f"Deleted {deleted_count} sample recommendation files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean sample recommendations")
    parser.add_argument("--force", action="store_true", help="Actually delete files (default is dry run)")
    
    args = parser.parse_args()
    
    clean_sample_recommendations(dry_run=not args.force)