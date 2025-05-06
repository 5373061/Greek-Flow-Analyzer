"""
Test script for verifying the ML model updates to fix feature mismatch issues.

This script:
1. Loads the trained models
2. Makes predictions for all tickers
3. Logs any feature mismatch issues

Usage:
    python test_ml_update.py
"""

import os
import sys
import logging
import json
from datetime import datetime

# Import ML modules
from models.ml.regime_classifier import GreekRegimeClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"ml_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_latest_model(model_dir="models/ml", target="primary_regime"):
    """Find the latest model for a target."""
    models = []
    for file in os.listdir(model_dir):
        if file.endswith(".pkl") and target in file:
            models.append(os.path.join(model_dir, file))
    
    if not models:
        return None
    
    # Return the most recently modified model
    return max(models, key=os.path.getmtime)

def load_analysis_files(results_dir="results"):
    """Load all analysis files."""
    analysis_data = {}
    
    for file in os.listdir(results_dir):
        if file.endswith("_analysis.json"):
            try:
                symbol = file.split("_")[0]
                with open(os.path.join(results_dir, file), "r") as f:
                    analysis_data[symbol] = json.load(f)
                logger.info(f"Loaded analysis data for {symbol}")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
    
    return analysis_data

def test_predictions(model_path, analysis_data):
    """Test making predictions with the model."""
    logger.info(f"Testing predictions with model {model_path}")
    
    try:
        # Load the model
        classifier = GreekRegimeClassifier()
        classifier.load_model(model_path)
        
        # Make predictions
        predictions = classifier.predict(analysis_data)
        
        if predictions:
            logger.info(f"Successfully made predictions for {len(predictions)} tickers")
            
            # Log the predictions
            for symbol, prediction in predictions.items():
                pred_text = prediction.get('prediction', 'N/A')
                confidence = prediction.get('confidence', 0.0)
                logger.info(f"  {symbol}: {pred_text} (confidence: {confidence:.2f})")
            
            return True
        else:
            logger.error("Failed to make predictions")
            return False
    
    except Exception as e:
        logger.error(f"Error testing predictions: {e}")
        return False

def main():
    """Main test function."""
    logger.info("Starting ML update test")
    
    # Find the latest models
    primary_model = find_latest_model(target="primary_regime")
    volatility_model = find_latest_model(target="volatility_regime")
    greek_model = find_latest_model(target="dominant_greek")
    
    if not all([primary_model, volatility_model, greek_model]):
        logger.error("Could not find all required models")
        return 1
    
    logger.info(f"Found models:")
    logger.info(f"  Primary regime: {os.path.basename(primary_model)}")
    logger.info(f"  Volatility regime: {os.path.basename(volatility_model)}")
    logger.info(f"  Dominant Greek: {os.path.basename(greek_model)}")
    
    # Load analysis data
    analysis_data = load_analysis_files()
    
    if not analysis_data:
        logger.error("No analysis data found")
        return 1
    
    logger.info(f"Loaded {len(analysis_data)} analysis files")
    
    # Test each model
    success = []
    
    # Test primary regime model
    logger.info("\nTesting primary regime model:")
    success.append(test_predictions(primary_model, analysis_data))
    
    # Test volatility regime model
    logger.info("\nTesting volatility regime model:")
    success.append(test_predictions(volatility_model, analysis_data))
    
    # Test dominant Greek model
    logger.info("\nTesting dominant Greek model:")
    success.append(test_predictions(greek_model, analysis_data))
    
    # Check if all tests passed
    if all(success):
        logger.info("\nAll tests PASSED! The ML fix is working correctly.")
        return 0
    else:
        logger.error("\nSome tests FAILED. The ML fix may need adjustments.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
