"""
Standalone test for the ML Trade Executor component.

This script tests the ML Trade Executor in isolation to ensure it can generate
trade orders when provided with appropriate test data.
"""

import os
import sys
import json
import logging
from datetime import datetime
import numpy as np

# Import ML modules
from models.ml.trade_executor import MLTradeExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sample_data():
    """Load a sample of analysis data for testing."""
    # Check if we have any analysis files
    results_dir = 'results'
    if not os.path.exists(results_dir):
        logger.error(f"Results directory {results_dir} not found")
        return None
    
    # Find analysis files
    analysis_files = [f for f in os.listdir(results_dir) if f.endswith('_analysis.json')]
    
    if not analysis_files:
        logger.error("No analysis files found")
        return None
    
    logger.info(f"Found {len(analysis_files)} analysis files")
    
    # Load first analysis file
    sample_file = os.path.join(results_dir, analysis_files[0])
    ticker = analysis_files[0].split('_')[0]
    
    try:
        with open(sample_file, 'r') as f:
            analysis_data = json.load(f)
        
        logger.info(f"Loaded analysis data for {ticker}")
        return ticker, analysis_data
    
    except Exception as e:
        logger.error(f"Error loading analysis data: {str(e)}")
        return None, None

def test_trade_executor():
    """Test the ML Trade Executor with test mode enabled."""
    logger.info("Testing ML Trade Executor with test mode")
    
    # Initialize executor with test mode explicitly enabled
    executor = MLTradeExecutor(test_mode=True)
    logger.info("Created MLTradeExecutor with test_mode=True")
    
    # Load sample data
    ticker, analysis_data = load_sample_data()
    
    if not ticker or not analysis_data:
        logger.error("Failed to load sample data")
        return False
    
    # Use a sample price
    price = 100.0
    if 'price' in analysis_data:
        price = analysis_data['price']
    
    # Create a test ML prediction with strong signals and favorable risk metrics
    test_prediction = {
        'ml_predictions': {
            'primary_regime': {'prediction': 'Vanna-Driven', 'confidence': 0.85},
            'volatility_regime': {'prediction': 'High', 'confidence': 0.75},
            'dominant_greek': {'prediction': 'Vanna', 'confidence': 0.8}
        },
        'trade_signals': {
            'entry': {
                'signal': 'bullish',
                'strength': 5,  # High strength
                'confidence': 0.9,  # High confidence
                'reasons': ['Strong vanna flow', 'Bullish regime']
            }
        },
        'risk_metrics': {
            'risk_reward_ratio': 3.0,  # Very favorable
            'stop_loss_price': price * 0.95,
            'take_profit_price': price * 1.15
        }
    }
    
    # Save test prediction
    os.makedirs('results/ml_predictions', exist_ok=True)
    prediction_path = os.path.join('results/ml_predictions', f"{ticker}_ml_prediction.json")
    with open(prediction_path, 'w') as f:
        json.dump(test_prediction, f, indent=2)
    logger.info(f"Saved test prediction to {prediction_path}")
    
    # Generate entry order
    entry_order = executor.generate_entry_order(ticker, price, test_prediction, analysis_data)
    
    if entry_order:
        logger.info(f"Successfully generated entry order for {ticker}")
        logger.info(f"Order details: {json.dumps(entry_order, indent=2)}")
        return True
    else:
        logger.error(f"Failed to generate entry order for {ticker}")
        return False

def main():
    """Main test function."""
    logger.info("Starting ML Trade Executor test")
    
    # Run test
    success = test_trade_executor()
    
    # Print result
    if success:
        logger.info("TEST PASSED: ML Trade Executor generated an entry order successfully")
        return 0
    else:
        logger.error("TEST FAILED: ML Trade Executor did not generate an entry order")
        return 1

if __name__ == "__main__":
    sys.exit(main())