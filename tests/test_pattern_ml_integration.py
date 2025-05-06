import unittest
import numpy as np
import os
import sys
import tempfile
import logging
from datetime import datetime

# Add the project root to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the PatternRecognizer and PatternMLIntegrator
from analysis.pattern_analyzer import PatternRecognizer
from analysis.pattern_ml_integrator import PatternMLIntegrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPatternMLIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test environment with temporary directory for models"""
        self.temp_dir = tempfile.mkdtemp()
        self.recognizer = PatternRecognizer(models_dir=self.temp_dir)
        
        # Create sample data for testing
        self.greek_results = self._create_sample_greek_results()
        self.momentum_data = self._create_sample_momentum_data()
        self.sequence_data = np.random.rand(1, 5, 10)  # Sample sequence data
        
        # Create training data
        self.training_data = self._create_training_data()
    
    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_sample_greek_results(self):
        """Create sample Greek results for testing"""
        return {
            "delta_exposure": 0.75,
            "gamma_exposure": 0.25,
            "vanna_exposure": -0.15,
            "charm_exposure": 0.05,
            "market_regime": "BULLISH",
            "energy_direction": "UP",
            "anomalies": [
                {"type": "GAMMA_SPIKE", "strength": 0.8},
                {"type": "VANNA_SURGE", "strength": 0.6}
            ]
        }
    
    def _create_sample_momentum_data(self):
        """Create sample momentum data for testing"""
        return {
            "momentum_score": 0.7,
            "energy_direction": "ACCELERATING",
            "volatility": 0.2,
            "volatility_trend": 0.1
        }
    
    def _create_training_data(self):
        """Create sample training data for testing"""
        patterns = ["BULLISH_MOMENTUM", "BEARISH_MOMENTUM", "CONSOLIDATION", 
                   "BREAKOUT_IMMINENT", "REVERSAL_SIGNAL", "COMPLEX"]
        
        training_data = []
        
        # Create 10 samples for each pattern
        for pattern in patterns:
            for _ in range(10):
                # Create variations of Greek results
                greek_results = self._create_sample_greek_results()
                
                # Modify based on pattern
                if pattern == "BULLISH_MOMENTUM":
                    greek_results["market_regime"] = "BULLISH"
                    greek_results["energy_direction"] = "UP"
                elif pattern == "BEARISH_MOMENTUM":
                    greek_results["market_regime"] = "BEARISH"
                    greek_results["energy_direction"] = "DOWN"
                    greek_results["delta_exposure"] = -0.75
                elif pattern == "CONSOLIDATION":
                    greek_results["market_regime"] = "NEUTRAL"
                    greek_results["energy_direction"] = "FLAT"
                elif pattern == "BREAKOUT_IMMINENT":
                    greek_results["market_regime"] = "NEUTRAL"
                    greek_results["energy_direction"] = "BUILDING"
                elif pattern == "REVERSAL_SIGNAL":
                    greek_results["market_regime"] = "BULLISH"
                    greek_results["energy_direction"] = "REVERSING"
                
                # Create variations of momentum data
                momentum_data = self._create_sample_momentum_data()
                
                # Add some random variation
                greek_results["delta_exposure"] *= (1 + np.random.normal(0, 0.1))
                greek_results["gamma_exposure"] *= (1 + np.random.normal(0, 0.1))
                momentum_data["momentum_score"] *= (1 + np.random.normal(0, 0.1))
                
                # Add to training data
                training_data.append({
                    "greek_results": greek_results,
                    "momentum_data": momentum_data,
                    "pattern": pattern
                })
        
        return training_data
    
    def test_feature_extraction(self):
        """Test the enhanced feature extraction method"""
        # Extract features
        features = self.recognizer._extract_features(self.greek_results, self.momentum_data)
        
        # Check that features are extracted correctly
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 5)  # Should have multiple features
        
        # Check that features are normalized
        for feature in features:
            self.assertIsInstance(feature, float)
            # Most features should be in range [-1, 1] after normalization
            # But some derived features might be outside this range
            if not (-10 <= feature <= 10):
                logger.warning(f"Feature value outside expected range: {feature}")
        
        logger.info(f"Extracted {len(features)} features")
    
    def test_train_model(self):
        """Test training models with the enhanced feature extraction"""
        # Skip if no ML integrator
        if not hasattr(self.recognizer, 'ml_integrator') or self.recognizer.ml_integrator is None:
            self.skipTest("ML integrator not available")
        
        # Train LightGBM model
        success = self.recognizer.train_model(self.training_data, model_type='lightgbm')
        
        # Check that training was successful
        self.assertTrue(success)
        self.assertIsNotNone(self.recognizer.model_path)
        self.assertTrue(os.path.exists(self.recognizer.model_path))
        
        logger.info(f"Trained model saved to {self.recognizer.model_path}")
        
        # Check that model was loaded
        self.assertIn(os.path.basename(self.recognizer.model_path).split('.')[0], 
                     self.recognizer.ml_integrator.models)
    
    def test_predict_pattern_ensemble(self):
        """Test ensemble prediction with multiple models"""
        # Skip if no ML integrator
        if not hasattr(self.recognizer, 'ml_integrator') or self.recognizer.ml_integrator is None:
            self.skipTest("ML integrator not available")
        
        # Train multiple models
        self.recognizer.train_model(self.training_data, model_type='lightgbm')
        self.recognizer.train_model(self.training_data, model_type='xgboost')
        
        # Check that models were loaded
        self.assertGreaterEqual(len(self.recognizer.ml_integrator.models), 1)
        
        # Test ensemble prediction
        prediction = self.recognizer.predict_pattern_ensemble(self.greek_results, self.momentum_data)
        
        # Check prediction
        self.assertIsNotNone(prediction)
        self.assertIn("pattern", prediction)
        self.assertIn("confidence", prediction)
        self.assertIn("description", prediction)
        self.assertIn("ensemble_info", prediction)
        
        # Check ensemble info
        ensemble_info = prediction["ensemble_info"]
        self.assertIn("models_used", ensemble_info)
        self.assertIn("supporting_models", ensemble_info)
        self.assertIn("vote_distribution", ensemble_info)
        
        logger.info(f"Ensemble prediction: {prediction['pattern']} with confidence {prediction['confidence']:.2f}")
        logger.info(f"Models used: {ensemble_info['models_used']}")
    
    def test_model_diagnostics(self):
        """Test model diagnostics functionality"""
        # Skip if no ML integrator
        if not hasattr(self.recognizer, 'ml_integrator') or self.recognizer.ml_integrator is None:
            self.skipTest("ML integrator not available")
        
        # Train a model
        self.recognizer.train_model(self.training_data, model_type='lightgbm')
        
        # Get model diagnostics
        diagnostics = self.recognizer.get_model_diagnostics()
        
        # Check diagnostics
        self.assertIsNotNone(diagnostics)
        self.assertIsInstance(diagnostics, dict)
        
        # Should have at least one model
        self.assertGreaterEqual(len(diagnostics), 1)
        
        # Check first model's diagnostics
        first_model = list(diagnostics.keys())[0]
        model_diag = diagnostics[first_model]
        
        self.assertIn("type", model_diag)
        
        logger.info(f"Model diagnostics: {diagnostics}")
    
    def test_predict_with_trained_model(self):
        """Test prediction with a trained model"""
        # Skip if no ML integrator
        if not hasattr(self.recognizer, 'ml_integrator') or self.recognizer.ml_integrator is None:
            self.skipTest("ML integrator not available")
        
        # Train a model
        self.recognizer.train_model(self.training_data, model_type='lightgbm')
        
        # Test prediction
        prediction = self.recognizer.predict_pattern(self.greek_results, self.momentum_data, use_ml=True)
        
        # Check prediction
        self.assertIsNotNone(prediction)
        self.assertIn("pattern", prediction)
        self.assertIn("confidence", prediction)
        self.assertIn("description", prediction)
        
        logger.info(f"Prediction with trained model: {prediction['pattern']} with confidence {prediction['confidence']:.2f}")
        
        # Test prediction with ML disabled
        rule_prediction = self.recognizer.predict_pattern(self.greek_results, self.momentum_data, use_ml=False)
        
        # Check rule-based prediction
        self.assertIsNotNone(rule_prediction)
        self.assertIn("pattern", rule_prediction)
        
        logger.info(f"Rule-based prediction: {rule_prediction['pattern']}")

if __name__ == "__main__":
    unittest.main()