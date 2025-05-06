import unittest
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta
import pytest
import sys

# Add the project root to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the PatternRecognizer
from analysis.pattern_analyzer import PatternRecognizer

class TestPatternRecognizerReal(unittest.TestCase):
    def setUp(self):
        """Initialize test environment with real data"""
        # Initialize pattern recognizer
        self.recognizer = PatternRecognizer()
        
        # Generate sample Greek results and momentum data
        self.greek_results = self._generate_greek_results()
        self.momentum_data = self._generate_momentum_data()
    
    def _generate_greek_results(self):
        """Generate sample Greek analysis results"""
        return {
            "reset_points": [
                {"price": 450, "strength": 0.8, "type": "gamma_flip"},
                {"price": 460, "strength": 0.6, "type": "vanna_peak"}
            ],
            "energy_levels": [
                {"level": 445, "strength": 0.7},
                {"level": 455, "strength": 0.9}
            ],
            "market_regime": "VANNA_DOMINATED",
            "anomalies": [
                {"type": "VOLATILITY_DIVERGENCE", "strength": 0.7}
            ],
            "gamma_profile": {
                "strikes": [440, 445, 450, 455, 460],
                "values": [0.1, 0.3, 0.5, 0.3, 0.1]
            },
            "delta_profile": {
                "strikes": [440, 445, 450, 455, 460],
                "values": [-0.8, -0.6, 0.0, 0.6, 0.8]
            }
        }
    
    def _generate_momentum_data(self):
        """Generate sample momentum data"""
        return {
            "energy_direction": "UP",
            "energy_gradient": 0.2,
            "momentum_score": 0.75,
            "price_trend": "BULLISH",
            "volatility_trend": "INCREASING"
        }
    
    def test_pattern_prediction_with_real_data(self):
        """Test pattern prediction with real data"""
        # Predict pattern
        prediction = self.recognizer.predict_pattern(self.greek_results, self.momentum_data)
        
        # Verify prediction
        self.assertIsNotNone(prediction, "Should return a prediction")
        self.assertIn("pattern", prediction, "Prediction should include pattern")
        self.assertIn("confidence", prediction, "Prediction should include confidence")
        self.assertIn("description", prediction, "Prediction should include description")
        
        # Print prediction
        print("\nPattern prediction with real data:")
        print(f"  Pattern: {prediction['pattern']}")
        print(f"  Confidence: {prediction['confidence']:.2f}")
        print(f"  Description: {prediction['description']}")
        
        # Check that confidence is reasonable
        self.assertGreaterEqual(prediction['confidence'], 0.5, "Confidence should be at least 0.5")
        self.assertLessEqual(prediction['confidence'], 1.0, "Confidence should be at most 1.0")
    
    def test_pattern_stability(self):
        """Test that pattern prediction is stable with small variations in data"""
        # Make a copy with small variations
        greek_results_var = self.greek_results.copy()
        
        # Add small variations to reset points
        if "reset_points" in greek_results_var:
            for point in greek_results_var["reset_points"]:
                point["strength"] *= (1 + np.random.normal(0, 0.05))
        
        # Predict patterns with both datasets
        prediction1 = self.recognizer.predict_pattern(self.greek_results, self.momentum_data)
        prediction2 = self.recognizer.predict_pattern(greek_results_var, self.momentum_data)
        
        # Check that predictions are the same or similar
        self.assertEqual(prediction1["pattern"], prediction2["pattern"], 
                         "Pattern should be stable with small variations")
        
        # Check that confidence is similar
        confidence_diff = abs(prediction1["confidence"] - prediction2["confidence"])
        self.assertLess(confidence_diff, 0.1, 
                        "Confidence should be similar with small variations")
    
    def test_pattern_with_missing_data(self):
        """Test pattern prediction with missing data"""
        # Test with missing Greek results
        prediction1 = self.recognizer.predict_pattern(None, self.momentum_data)
        
        # Verify prediction with missing Greek results
        self.assertIsNotNone(prediction1, "Should return a prediction even with missing data")
        self.assertIn("pattern", prediction1, "Prediction should include pattern")
        self.assertIn("confidence", prediction1, "Prediction should include confidence")
        
        # Test with missing momentum data
        prediction2 = self.recognizer.predict_pattern(self.greek_results, None)
        
        # Verify prediction with missing momentum data
        self.assertIsNotNone(prediction2, "Should return a prediction even with missing data")
        self.assertIn("pattern", prediction2, "Prediction should include pattern")
        self.assertIn("confidence", prediction2, "Prediction should include confidence")
        
        # Print predictions
        print("\nPattern prediction with missing data:")
        print(f"  Missing Greek results: {prediction1['pattern']} (confidence: {prediction1['confidence']:.2f})")
        print(f"  Missing momentum data: {prediction2['pattern']} (confidence: {prediction2['confidence']:.2f})")

    def test_pattern_with_extreme_values(self):
        """Test pattern prediction with extreme values in the data"""
        # Create a copy of Greek results with extreme values
        extreme_greek = self.greek_results.copy()
        
        # Modify with extreme values
        if "reset_points" in extreme_greek:
            for point in extreme_greek["reset_points"]:
                point["strength"] = 0.99  # Very high strength
        
        if "energy_levels" in extreme_greek:
            for level in extreme_greek["energy_levels"]:
                level["strength"] = 0.95  # Very high strength
        
        # Create extreme momentum data
        extreme_momentum = {
            "energy_direction": "UP",
            "energy_gradient": 0.9,  # Very steep gradient
            "momentum_score": 0.95,  # Very high momentum
            "price_trend": "STRONGLY_BULLISH",
            "volatility_trend": "RAPIDLY_INCREASING"
        }
        
        # Predict pattern with extreme values
        prediction = self.recognizer.predict_pattern(extreme_greek, extreme_momentum)
        
        # Verify prediction
        self.assertIsNotNone(prediction, "Should return a prediction")
        self.assertIn("pattern", prediction, "Prediction should include pattern")
        self.assertIn("confidence", prediction, "Prediction should include confidence")
        
        # Check that confidence is high for extreme values
        self.assertGreaterEqual(prediction['confidence'], 0.8, 
                               "Confidence should be high for extreme values")
        
        # Print prediction
        print("\nPattern prediction with extreme values:")
        print(f"  Pattern: {prediction['pattern']}")
        print(f"  Confidence: {prediction['confidence']:.2f}")
        print(f"  Description: {prediction['description']}")

    def test_pattern_with_contradictory_data(self):
        """Test pattern prediction with contradictory signals in the data"""
        # Create contradictory Greek results (mixed signals)
        contradictory_greek = self.greek_results.copy()
        
        # Set contradictory market regime
        contradictory_greek["market_regime"] = "MIXED_SIGNALS"
        
        # Add contradictory anomalies
        contradictory_greek["anomalies"] = [
            {"type": "VOLATILITY_DIVERGENCE", "strength": 0.7},
            {"type": "PRICE_MOMENTUM_DIVERGENCE", "strength": 0.6},
            {"type": "GAMMA_DELTA_DIVERGENCE", "strength": 0.5}
        ]
        
        # Create contradictory momentum data
        contradictory_momentum = {
            "energy_direction": "NEUTRAL",
            "energy_gradient": 0.0,  # Flat gradient
            "momentum_score": 0.5,   # Neutral momentum
            "price_trend": "SIDEWAYS",
            "volatility_trend": "STABLE"
        }
        
        # Predict pattern with contradictory data
        prediction = self.recognizer.predict_pattern(contradictory_greek, contradictory_momentum)
        
        # Verify prediction
        self.assertIsNotNone(prediction, "Should return a prediction")
        self.assertIn("pattern", prediction, "Prediction should include pattern")
        self.assertIn("confidence", prediction, "Prediction should include confidence")
        
        # Check that confidence is lower for contradictory data
        self.assertLessEqual(prediction['confidence'], 0.7, 
                            "Confidence should be lower for contradictory data")
        
        # Print prediction
        print("\nPattern prediction with contradictory data:")
        print(f"  Pattern: {prediction['pattern']}")
        print(f"  Confidence: {prediction['confidence']:.2f}")
        print(f"  Description: {prediction['description']}")

if __name__ == "__main__":
    unittest.main()





