# config.py
import os
import sys
import logging
from pathlib import Path

# --- Required Packages ---
# This section lists all required packages for the project
# If any are missing, they will be reported but won't crash the program
REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "matplotlib",
    "scipy",
    "joblib",
    "pickle",
    "requests",
    "chardet"
]

# Check for required packages
missing_packages = []
for package in REQUIRED_PACKAGES:
    try:
        if package == "pickle":  # pickle is built-in
            import pickle
        elif package == "joblib":
            import joblib
        elif package == "pandas":
            import pandas as pd
        elif package == "numpy":
            import numpy as np
        elif package == "matplotlib":
            import matplotlib.pyplot as plt
        elif package == "scipy":
            import scipy
        elif package == "requests":
            import requests
        elif package == "chardet":
            import chardet
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"⚠️ WARNING: Missing required packages: {', '.join(missing_packages)}")
    print("Please install them using: pip install " + " ".join(missing_packages))

# --- Polygon API Configuration ---
POLYGON_API_KEY = "rNX4MlGs7zNvpT2RETtpYWcFZgTXpccV"  # Replace with your actual API key
PRICE_HISTORY_YEARS = 5

# --- Paths ---
DATA_DIR = r'D:\python projects\Greek Energy Flow II\data'
OUTPUT_DIR = r'D:\python projects\Greek Energy Flow II\output'

# --- Strategy Parameters ---
STOP_MULT = 0.5 # Used ONLY for calculating historical R-multiples & maybe feature, NOT for live filtering anymore
COST_PER_TRADE_R = 0.05 # Historical cost analysis

# --- Machine Learning Parameters ---
ML_PROB_THRESHOLD = 0.65 # Base probability threshold
MIN_HISTORICAL_ROWS = 50

# --- Feature Definitions ---
BASE_FEATURES = ["RewardToRisk", "RSI", "VolSpike", "IV_Pctl"]
OPTIONS_FEATURES = ["IV_RSI_Divergence_Score", "Divergence_Zone", "Vol_Adjusted_RSI"]
ALL_ML_FEATURES = BASE_FEATURES + OPTIONS_FEATURES

# --- Analysis & Robustness Parameters ---
N_SLICES = 5
N_FOLDS = 5

# --- Live Signal Parameters ---
LIVE_SIGNAL_LOOKBACK = 5        # How many recent bars to check for live signals
MIN_PROFIT_LOSS_RATIO = 1.75  # Require Potential $ Profit / Potential $ Risk >= 2.0 <-- NEW/REVISED PARAMETER
# MIN_POTENTIAL_PROFIT_DOLLAR = 0.50 # <-- Keep commented out or remove

# --- Email Notification Configuration ---
SEND_EMAIL_NOTIFICATIONS = True
EMAIL_SENDER_ADDRESS = "mrs.carteblanche@gmail.com"
EMAIL_SENDER_PASSWORD = "YOUR_APP_PASSWORD" # Use App Password for Gmail
EMAIL_RECIPIENT_ADDRESS = "mrs.carteblanche@example.com"
EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587

# --- Sanity Check Paths ---
if not POLYGON_API_KEY or "YOUR_API_KEY" in POLYGON_API_KEY:
     print("❌ CONFIG ERROR: POLYGON_API_KEY is missing or not replaced in config.py!")
if not os.path.isdir(DATA_DIR):
    print(f"❌ CONFIG ERROR: Data directory not found: {DATA_DIR}")
if not os.path.isdir(OUTPUT_DIR):
     print(f"✅ CONFIG INFO: Output directory not found, will be created: {OUTPUT_DIR}")

# Define minimum risk offset used in stop calculation (pipeline_processor.py)
# This ensures the denominator in Profit/Loss ratio is not zero/tiny
MIN_RISK_PRICE_OFFSET_STOP_CALC = 0.02 # Used only to ensure stop isn't AT the entry

# --- Default Configuration ---
DEFAULT_CONFIG = {
    'regime_thresholds': {
        'highVolatility': 0.3,
        'lowVolatility': 0.15,
        'strongBullish': 0.7,
        'strongBearish': -0.7,
        'neutralZone': 0.2
    },
    'reset_factors': {
        'gammaFlip': 0.35,
        'vannaPeak': 0.25,
        'charmCluster': 0.15,
        'timeDecay': 0.10
    },
    'price_projection': {
        'range_percent': 0.15,
        'steps': 30
    }
}

def get_config():
    """
    Get the configuration dictionary.
    
    Returns:
        dict: The configuration dictionary with default values.
    """
    return DEFAULT_CONFIG.copy()


