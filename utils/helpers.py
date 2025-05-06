# utils/helpers.py
import os
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

def ensure_directory(directory_path):
    """
    Ensure a directory exists; create it if it doesn't.
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        str: Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)
    return directory_path

def format_timestamp(dt=None):
    """
    Format a datetime object as a string.
    
    Args:
        dt (datetime, optional): Datetime to format. Defaults to current time.
        
    Returns:
        str: Formatted timestamp
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d_%H-%M-%S")

def parse_date(date_str):
    """
    Parse a date string into a datetime object.
    
    Args:
        date_str (str): Date string to parse
        
    Returns:
        datetime: Parsed datetime object
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        try:
            return datetime.strptime(date_str, "%Y/%m/%d")
        except ValueError:
            try:
                return datetime.strptime(date_str, "%m/%d/%Y")
            except ValueError:
                logger.error(f"Could not parse date: {date_str}")
                return None