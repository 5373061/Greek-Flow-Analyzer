#!/usr/bin/env python3
"""
Simple script to clean out the .venv folder.
"""

import os
import sys
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def clean_venv():
    """Remove the .venv directory."""
    venv_path = os.path.join(ROOT_DIR, ".venv")
    
    if os.path.exists(venv_path):
        logger.info(f"Removing .venv directory: {venv_path}")
        try:
            shutil.rmtree(venv_path)
            logger.info("Successfully removed .venv directory")
            return True
        except Exception as e:
            logger.error(f"Failed to remove .venv directory: {e}")
            return False
    else:
        logger.info("No .venv directory found")
        return False

def create_gitignore_for_venv():
    """Create or update .gitignore to ignore .venv directory."""
    gitignore_path = os.path.join(ROOT_DIR, ".gitignore")
    venv_ignore_line = ".venv/"
    
    # Check if .gitignore exists
    if os.path.exists(gitignore_path):
        # Read existing content
        with open(gitignore_path, 'r') as f:
            content = f.read()
        
        # Check if .venv is already ignored
        if venv_ignore_line not in content:
            # Append .venv ignore line
            with open(gitignore_path, 'a') as f:
                f.write(f"\n# Virtual environment\n{venv_ignore_line}\n")
            logger.info("Updated .gitignore to ignore .venv directory")
    else:
        # Create new .gitignore
        with open(gitignore_path, 'w') as f:
            f.write(f"# Virtual environment\n{venv_ignore_line}\n")
        logger.info("Created .gitignore to ignore .venv directory")

if __name__ == "__main__":
    try:
        print(f"Running from: {os.getcwd()}")
        print(f"Script location: {os.path.abspath(__file__)}")
        
        # Clean the virtual environment
        clean_venv()
        
        # Update .gitignore to prevent accidental commits of .venv
        create_gitignore_for_venv()
        
        print("\nVirtual environment cleanup complete.")
        print("If you need to recreate the virtual environment in the future, run:")
        print("python -m venv .venv")
        print("\nTo install required packages globally (recommended for your setup):")
        print("pip install -r requirements.txt")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)
        sys.exit(1)
