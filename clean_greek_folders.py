#!/usr/bin/env python3
"""
Script to clean out all Greek-related folders and files from the project.
"""

import os
import sys
import shutil
import logging
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def clean_greek_folders():
    """Remove all Greek-related folders and files."""
    # List of Greek-related folders and patterns to remove
    greek_patterns = [
        "greek_flow",
        "analysis/greek_*",
        "**/greek_*",
        "tests/test_greek_*",
        "Greek_Energy_FlowII.py",
        "interp.py"
    ]
    
    removed_count = 0
    
    for pattern in greek_patterns:
        # Find all matching paths
        matching_paths = glob.glob(os.path.join(ROOT_DIR, pattern), recursive=True)
        
        for path in matching_paths:
            logger.info(f"Removing: {path}")
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                removed_count += 1
            except Exception as e:
                logger.error(f"Failed to remove {path}: {e}")
    
    # Also remove config.json which contains Greek-related configurations
    config_path = os.path.join(ROOT_DIR, "config.json")
    if os.path.exists(config_path):
        try:
            logger.info(f"Removing: {config_path}")
            os.remove(config_path)
            removed_count += 1
        except Exception as e:
            logger.error(f"Failed to remove {config_path}: {e}")
    
    logger.info(f"Removed {removed_count} Greek-related folders and files")
    return removed_count

if __name__ == "__main__":
    try:
        print(f"Running from: {os.getcwd()}")
        print(f"Script location: {os.path.abspath(__file__)}")
        clean_greek_folders()
    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)
        sys.exit(1)
