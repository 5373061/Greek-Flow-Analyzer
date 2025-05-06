"""
test_cross_greek_patterns.py - Tests for the cross-Greek pattern analyzer
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the modules
try:
    from ordinal_pattern_analyzer import GreekOrdinalPatternAnalyzer
    from cross_greek_patterns import CrossGreekPatternAnalyzer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Current sys.path: {sys.path}")
    raise

class TestCrossGreekPatternAnalyzer(unittest.TestCase):
    """
    Unit tests for the CrossGreekPatternAnalyzer class.
    """
    
    def setUp(self):
        """Set up test environment before each test case."""
        # Create test data
        self.test_data = pd.DataFrame({
            'norm_delta': [0.2, 0.25, 0.3, 0.28, 0.35, 0.4, 0.38, 0.42, 0.45, 0.5],
            'norm_gamma': [0.5, 0.53, 0.48, 0.45, 0.4, 0.38, 0.35, 0.3, 0.25, 0.2],
            'norm_vanna': [0.3, 0.32, 0.35, 0.4, 0.38, 0.35, 0.32, 0.29, 0.25, 0.2],
            'norm_charm': [0.1, 0.12, 0.15, 0.18, 0.22, 0.25, 0.28, 0.3, 0.28, 0.25],
            'price': [100, 102, 105, 104, 106, 109, 107, 110, 112, 115],
            'delta': [0.5, 0.52, 0.55, 0.53, 0.57, 0.6, 0.58, 0.62, 0.65, 0.7]
        })
        
        # Initialize analyzers with test parameters
        self.base_analyzer = GreekOrdinalPatternAnalyzer(
            window_size=3,
            min_occurrences=2,
            min_confidence=0.5
        )
        
        self.cross_analyzer = CrossGreekPatternAnalyzer(self.base_analyzer)
    
    def test_analyze_cross_greek_patterns(self):
        """Test analyzing cross-Greek pattern relationships."""
        cross_patterns = self.cross_analyzer.analyze_cross_greek_patterns(
            self.test_data, forward_period=2
        )
        
        # Check that cross patterns were created
        self.assertTrue(len(cross_patterns) > 0)
        
        # Check expected structure
        for pair_key, pattern_pairs in cross_patterns.items():
            # The pair key should be in format "greek1_greek2"
            self.assertTrue('_' in pair_key)
            greek1, greek2 = pair_key.split('_')
            self.assertIn(greek1, ['delta', 'gamma', 'vanna', 'charm', 'theta', 'vega'])
            self.assertIn(greek2, ['delta', 'gamma', 'vanna', 'charm', 'theta', 'vega'])
            
            # Pattern pairs should have counts
            for pattern_pair, count in pattern_pairs.items():
                self.assertIsInstance(pattern_pair, tuple)
                self.assertEqual(len(pattern_pair), 2)
                self.assertIsInstance(pattern_pair[0], tuple)  # First pattern
                self.assertIsInstance(pattern_pair[1], tuple)  # Second pattern
                self.assertIsInstance(count, int)              # Occurrence count
                self.assertGreaterEqual(count, 1)              # Should occur at least once
    
    def test_find_predictive_relationships(self):
        """Test finding predictive relationships between Greeks."""
        # First analyze cross-Greek patterns
        self.cross_analyzer.analyze_cross_greek_patterns(
            self.test_data, forward_period=2
        )
        
        # Find predictive relationships
        predictive = self.cross_analyzer.find_predictive_relationships(min_occurrences=1)
        
        # Check that predictive relationships were found
        self.assertTrue(len(predictive) > 0)
        
        # Check structure of predictive relationships
        for pair_key, relationships in predictive.items():
            for rel in relationships:
                expected_keys = ['source_pattern', 'target_pattern', 
                                'source_description', 'target_description', 
                                'occurrences']
                for key in expected_keys:
                    self.assertIn(key, rel)
                
                # Check that descriptions are strings
                self.assertIsInstance(rel['source_description'], str)
                self.assertIsInstance(rel['target_description'], str)
                
                # Check that occurrences is at least 1
                self.assertGreaterEqual(rel['occurrences'], 1)
    
    def test_enhance_recommendation_with_cross_patterns(self):
        """Test enhancing a trade recommendation with cross-Greek insights."""
        # First analyze cross-Greek patterns
        self.cross_analyzer.analyze_cross_greek_patterns(
            self.test_data, forward_period=2
        )
        
        # Create a mock recommendation
        recommendation = {
            'symbol': 'AAPL',
            'current_price': 105,
            'action': 'WAIT',
            'strategy': 'MONITOR_FOR_CLEARER_SIGNALS',
            'confidence': 0.6,
            'option_selection': {
                'atm_strike': 105,
                'otm_strike': 110
            }
        }
        
        # Use last 3 data points as current data
        current_data = self.test_data.iloc[-3:].reset_index(drop=True)
        
        # Enhance the recommendation
        enhanced = self.cross_analyzer.enhance_recommendation_with_cross_patterns(
            recommendation, current_data
        )
        
        # Check that enhanced recommendation is a copy
        self.assertIsNot(enhanced, recommendation)
        
        # Check that original fields are preserved
        self.assertEqual(enhanced['symbol'], recommendation['symbol'])
        self.assertEqual(enhanced['current_price'], recommendation['current_price'])
        
        # If cross-Greek insights were added, check their structure
        if 'cross_greek_insights' in enhanced:
            for insight in enhanced['cross_greek_insights']:
                expected_keys = ['source_greek', 'target_greek', 
                               'source_pattern', 'target_pattern', 
                               'confidence']
                for key in expected_keys:
                    self.assertIn(key, insight)
                
                # Check that confidence is between 0 and 1
                self.assertGreaterEqual(insight['confidence'], 0)
                self.assertLessEqual(insight['confidence'], 1)
    
    def test_determine_moneyness(self):
        """Test determining moneyness from a recommendation."""
        # ITM case
        itm_rec = {
            'current_price': 100,
            'option_selection': {
                'atm_strike': 90
            }
        }
        self.assertEqual(self.cross_analyzer._determine_moneyness(itm_rec), 'ITM')
        
        # ATM case
        atm_rec = {
            'current_price': 100,
            'option_selection': {
                'atm_strike': 100
            }
        }
        self.assertEqual(self.cross_analyzer._determine_moneyness(atm_rec), 'ATM')
        
        # OTM case
        otm_rec = {
            'current_price': 100,
            'option_selection': {
                'atm_strike': 110
            }
        }
        self.assertEqual(self.cross_analyzer._determine_moneyness(otm_rec), 'OTM')
        
        # Default case
        default_rec = {}
        self.assertEqual(self.cross_analyzer._determine_moneyness(default_rec), 'ATM')

    def test_cross_greek_pattern_prediction(self):
        """Test prediction of patterns across different Greeks."""
        # Analyze cross-Greek patterns
        cross_patterns = self.cross_analyzer.analyze_cross_greek_patterns(
            self.test_data, forward_period=2
        )
        
        # Find predictive relationships
        predictive = self.cross_analyzer.find_predictive_relationships(min_occurrences=1)
        
        # Verify that predictive relationships were found
        self.assertTrue(len(predictive) > 0, "No predictive relationships found")
        
        # Create a recommendation
        recommendation = {
            'action': 'WAIT',
            'confidence': 0.6,
            'strategy': 'MONITOR_FOR_CLEARER_SIGNALS'
        }
        
        # Enhance recommendation with cross-Greek insights
        enhanced = self.cross_analyzer.enhance_recommendation_with_cross_patterns(
            recommendation, self.test_data.tail(3)
        )
        
        # Verify that recommendation was enhanced
        self.assertIn('cross_greek_insights', enhanced, "Recommendation not enhanced with cross-Greek insights")
        
        # Verify that confidence was adjusted
        self.assertNotEqual(enhanced['confidence'], recommendation['confidence'], 
                           "Confidence not adjusted based on cross-Greek insights")

if __name__ == '__main__':
    unittest.main()

