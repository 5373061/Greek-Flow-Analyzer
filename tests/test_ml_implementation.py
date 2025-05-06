"""
Test script for the ML implementation in Greek Energy Flow II

This script performs a quick test of the ML implementation by:
1. Training a model on sample data
2. Making predictions for a ticker
3. Generating a sample trade recommendation

Usage:
    python test_ml_implementation.py
"""

import os
import sys
import json
import logging
from datetime import datetime
import numpy as np

# Import ML modules
from models.ml.regime_classifier import GreekRegimeClassifier
from models.ml.trade_executor import MLTradeExecutor, MLEnhancedAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_dirs():
    """Create necessary directories for testing."""
    os.makedirs('models/ml', exist_ok=True)
    os.makedirs('results/ml_predictions', exist_ok=True)
    os.makedirs('results/ml_trades', exist_ok=True)

def load_sample_analysis(results_dir='results'):
    """
    Load sample analysis data for testing.
    
    Returns:
        Dictionary with ticker -> analysis data
    """
    analysis_data = {}
    
    # Find analysis files
    analysis_files = []
    for file in os.listdir(results_dir):
        if file.endswith('_analysis.json'):
            analysis_files.append(os.path.join(results_dir, file))
    
    if not analysis_files:
        logger.warning(f"No analysis files found in {results_dir}")
        return {}
    
    logger.info(f"Found {len(analysis_files)} analysis files")
    
    # Load each analysis file
    for file_path in analysis_files:
        try:
            # Extract symbol from filename
            file_name = os.path.basename(file_path)
            symbol = file_name.split('_')[0]
            
            # Load analysis data
            with open(file_path, 'r') as f:
                analysis = json.load(f)
            
            analysis_data[symbol] = analysis
            logger.info(f"Loaded analysis data for {symbol}")
            
            # Limit to 5 symbols for testing
            if len(analysis_data) >= 5:
                break
                
        except Exception as e:
            logger.error(f"Error loading analysis file {file_path}: {str(e)}")
    
    return analysis_data

def test_ml_classifier():
    """Test the ML classifier."""
    logger.info("Testing ML classifier")
    
    # Load sample analysis data
    analysis_data = load_sample_analysis()
    
    if not analysis_data:
        logger.error("No analysis data available for testing")
        return False
    
    try:
        # Initialize classifier
        classifier = GreekRegimeClassifier(model_type='randomforest')
        
        # Extract features
        features_df = classifier.prepare_features(analysis_data)
        
        # Extract labels
        labels_df = classifier.extract_labels(analysis_data)
        
        logger.info(f"Extracted features for {len(features_df)} samples")
        logger.info(f"Extracted labels for {len(labels_df)} samples")
        
        # Train a simple model (for testing only)
        if len(features_df) > 0 and len(labels_df) > 0:
            logger.info("Training test model")
            classifier.build_model()
            
            # Select feature matrix and target
            if 'symbol' in features_df.columns:
                X = features_df.drop('symbol', axis=1)
            else:
                X = features_df
            
            # Clean features - replace inf values with NaN and then fill with large values
            X = X.replace([np.inf, -np.inf], np.nan)
            # Fill NaN values with column means or zeros if all values are NaN
            for col in X.columns:
                if X[col].isna().all():
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(X[col].mean())
            
            y = labels_df['primary_regime']
            
            # Simple train (no validation split for this test)
            classifier.model.fit(X, y)
            
            # Make predictions
            predictions = classifier.model.predict(X)
            
            logger.info(f"Made {len(predictions)} predictions")
            
            # Save test model
            model_path = os.path.join('models/ml', 'test_model.pkl')
            
            with open(model_path, 'wb') as f:
                import pickle
                pickle.dump(classifier.model, f)
            
            logger.info(f"Saved test model to {model_path}")
            
            return True
        else:
            logger.error("Insufficient data for training")
            return False
        
    except Exception as e:
        logger.error(f"Error testing ML classifier: {str(e)}")
        return False

def test_trade_executor():
    """Test the ML Trade Executor with test data."""
    logger.info("Testing ML Trade Executor")
    
    # Initialize executor with test mode explicitly enabled
    executor = MLTradeExecutor(test_mode=True)
    logger.info("Created MLTradeExecutor with test_mode=True")
    
    # Load sample data
    analysis_data = load_sample_analysis()
    
    if not analysis_data:
        logger.error("No analysis data available for testing")
        return False
    
    # Test with first ticker
    ticker = list(analysis_data.keys())[0]
    price = analysis_data[ticker].get('price', 100.0)
    
    # Create a test ML prediction with strong signals
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
        # Add test risk metrics to bypass risk-reward check
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
    entry_order = executor.generate_entry_order(ticker, price, test_prediction, analysis_data[ticker])
    
    if entry_order:
        logger.info(f"Successfully generated entry order for {ticker}")
        logger.info(f"Order details: {json.dumps(entry_order, indent=2)}")
        return True
    else:
        logger.error(f"Failed to generate entry order for {ticker}")
        return False

def test_ml_enhanced_analyzer():
    """Test the ML Enhanced Analyzer with test data."""
    logger.info("Testing ML enhanced analyzer")
    
    # Initialize executor with test mode
    executor = MLTradeExecutor(test_mode=True)
    
    # Initialize analyzer with executor
    analyzer = MLEnhancedAnalyzer(executor=executor)
    logger.info("Initialized ML-enhanced analyzer")
    
    # Load sample analysis data
    analysis_data = load_sample_analysis()
    
    if not analysis_data:
        logger.error("No analysis data available for testing")
        return False
    
    # Test with first ticker
    ticker = list(analysis_data.keys())[0]
    price = analysis_data[ticker].get('price', 100.0)
    
    # Ensure ML prediction exists
    prediction_path = os.path.join('results/ml_predictions', f"{ticker}_ml_prediction.json")
    if not os.path.exists(prediction_path):
        logger.warning(f"No ML prediction file found for {ticker}, running test_trade_executor first")
        test_trade_executor()
    
    # Analyze ticker
    enhanced_analysis = analyzer.analyze_ticker(ticker, price)
    
    if enhanced_analysis and enhanced_analysis.get('recommendation'):
        logger.info(f"Generated enhanced analysis for {ticker}")
        logger.info(f"Recommendation action: {enhanced_analysis['recommendation']['action']}")
        logger.info(f"Confidence: {enhanced_analysis['recommendation']['confidence']}")
        
        # Save enhanced analysis
        output_path = os.path.join('results', f"{ticker}_enhanced_recommendation.json")
        with open(output_path, 'w') as f:
            json.dump(enhanced_analysis, f, indent=2)
        logger.info(f"Saved enhanced analysis to {output_path}")
        
        return True
    else:
        logger.error(f"Failed to generate enhanced analysis for {ticker}")
        return False

def main():
    """Main test function."""
    logger.info("Starting ML implementation tests")
    
    # Create test directories
    create_test_dirs()
    
    # Test components
    classifier_success = test_ml_classifier()
    executor_success = test_trade_executor()
    analyzer_success = test_ml_enhanced_analyzer()
    
    # Print summary
    logger.info("\nTest Summary:")
    logger.info(f"ML Classifier: {'PASSED' if classifier_success else 'FAILED'}")
    logger.info(f"ML Trade Executor: {'PASSED' if executor_success else 'FAILED'}")
    logger.info(f"ML Enhanced Analyzer: {'PASSED' if analyzer_success else 'FAILED'}")
    
    if classifier_success and executor_success and analyzer_success:
        logger.info("\nAll tests PASSED!")
        return 0
    else:
        logger.error("\nSome tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())





