#!/usr/bin/env python3
"""
Script to update imports across the codebase to use the centralized configuration.
"""

import os
import re
import glob
import logging
import chardet

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def detect_encoding(file_path):
    """Detect file encoding."""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # Read up to 10KB to detect encoding
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'  # Default to utf-8 if detection fails

def update_imports():
    """Update imports across the codebase."""
    # Find all Python files
    python_files = glob.glob(os.path.join(ROOT_DIR, "**/*.py"), recursive=True)
    
    # Skip this script and the config.py file
    python_files = [f for f in python_files if not f.endswith("update_imports.py") and not f.endswith("greek_flow/config.py")]
    
    # Patterns to search for
    patterns = [
        (r"from greek_flow\.flow import DEFAULT_CONFIG", "from greek_flow.config import get_config, DEFAULT_CONFIG"),
        (r"from Greek_Energy_FlowII import DEFAULT_CONFIG", "from greek_flow.config import get_config, DEFAULT_CONFIG"),
        (r"from \.flow import DEFAULT_CONFIG", "from .config import get_config, DEFAULT_CONFIG"),
        (r"DEFAULT_CONFIG = \{[^}]+\}", "# Using centralized configuration from greek_flow.config"),
        (r"config = config if config is not None else DEFAULT_CONFIG", "config = config if config is not None else get_config()"),
    ]
    
    # Update each file
    for file_path in python_files:
        try:
            # Detect file encoding
            encoding = detect_encoding(file_path)
            
            # Read file content with detected encoding
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            # Check if the file imports or defines DEFAULT_CONFIG
            if "DEFAULT_CONFIG" in content:
                # Apply patterns
                modified = False
                for pattern, replacement in patterns:
                    if re.search(pattern, content):
                        content = re.sub(pattern, replacement, content)
                        modified = True
                
                # Save changes if modified
                if modified:
                    with open(file_path, 'w', encoding=encoding) as f:
                        f.write(content)
                    logger.info(f"Updated imports in {file_path}")
        except Exception as e:
            logger.error(f"Failed to update imports in {file_path}: {e}")
    
    logger.info("Import updates completed")

if __name__ == "__main__":
    # Check if chardet is installed
    try:
        import chardet
    except ImportError:
        logger.error("chardet package is required. Please install it with 'pip install chardet'")
        import sys
        sys.exit(1)
    
    update_imports()
