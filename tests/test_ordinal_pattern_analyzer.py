"""
test_ordinal_pattern_analyzer.py - Unit tests for the ordinal pattern analyzer module
"""

import unittest
import pandas as pd
import numpy as np
import os
import json
import tempfile
import shutil
import sys
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ordinal_pattern_analyzer import GreekOrdinalPatternAnalyzer, PatternMetrics

class TestGreekOrdinalPatternAnalyzer(unittest.TestCase):
    """
    Unit tests for the GreekOrdinalPatternAnalyzer class.
    """
    
    def setUp(self):
        """Set up test environment before each test case."""
        # Create test data with all required Greek metrics and enough data points
        n_points = 20
        
        # Create delta values that ensure enough data points in each moneyness category
        # We need at least 3 consecutive points in each category (for window_size=3)
        delta_values = []
        
        # Add 5 ITM points (delta > 0.7)
        delta_values.extend([0.85, 0.82, 0.80, 0.78, 0.75])
        
        # Add 5 ATM points (0.4 < delta < 0.6)
        delta_values.extend([0.55, 0.52, 0.50, 0.48, 0.45])
        
        # Add 5 OTM points (delta < 0.3)
        delta_values.extend([0.25, 0.22, 0.20, 0.18, 0.15])
        
        # Add 5 mixed points
        delta_values.extend([0.65, 0.60, 0.55, 0.45, 0.35])
        
        # Create vega_change values for VOL_CRUSH (all < -0.2)
        vega_change_values = []
        
        # Add 5 VOL_CRUSH points (vega_change < -0.2)
        vega_change_values.extend([-0.30, -0.28, -0.25, -0.23, -0.21])
        
        # Add 15 non-VOL_CRUSH points
        vega_change_values.extend(np.linspace(-0.19, -0.05, 15))
        
        # Create base data with increasing/decreasing patterns
        self.test_data = pd.DataFrame({
            # Add normalized Greek metrics (these are what the analyzer looks for first)
            'norm_delta': np.linspace(0.2, 0.8, n_points),
            'norm_gamma': np.linspace(0.5, 0.1, n_points),
            'norm_theta': np.linspace(0.1, 0.6, n_points),
            'norm_vega': np.linspace(0.6, 0.1, n_points),
            'norm_vanna': np.concatenate([np.linspace(0.3, 0.5, n_points//2), np.linspace(0.5, 0.2, n_points//2)]),
            'norm_charm': np.concatenate([np.linspace(0.1, 0.3, n_points//2), np.linspace(0.3, 0.1, n_points//2)]),
            
            # Add price data for profitability analysis
            'price': np.linspace(100, 120, n_points),
            
            # Add raw Greek metrics (these are used if normalized versions aren't found)
            'delta': delta_values,
            'gamma': np.linspace(0.05, 0.01, n_points),
            'theta': np.linspace(-0.1, -0.6, n_points),
            'vega': np.linspace(0.6, 0.1, n_points),
            'vanna': np.linspace(0.03, 0.01, n_points),
            'charm': np.linspace(0.01, 0.03, n_points),
            
            # Add columns needed for VOL_CRUSH filtering
            'vega_prev': np.linspace(0.7, 0.2, n_points),
            'vega_change': vega_change_values
        })
        
        # Initialize analyzer with test parameters
        self.analyzer = GreekOrdinalPatternAnalyzer(
            window_size=3,
            min_occurrences=2,
            min_confidence=0.5,
            top_patterns=2
        )
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after each test case."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_extract_patterns(self):
        """Test extraction of ordinal patterns."""
        # No need to add columns since they're already in the test_data from setUp
        patterns = self.analyzer.extract_patterns(self.test_data)
        
        # Check that patterns were extracted for each Greek
        self.assertTrue('delta' in patterns)
        self.assertTrue('gamma' in patterns)
        self.assertTrue('vanna' in patterns)
        self.assertTrue('charm' in patterns)
        self.assertTrue('theta' in patterns)
        self.assertTrue('vega' in patterns)
        
        # Check that at least one pattern was extracted for each Greek
        for greek, pattern_list in patterns.items():
            self.assertGreater(len(pattern_list), 0, f"No patterns extracted for {greek}")
        
        # Check pattern format (index, tuple of pattern)
        for pattern_list in patterns.values():
            for entry in pattern_list:
                self.assertEqual(len(entry), 2)
                self.assertIsInstance(entry[0], int)  # Index
                self.assertIsInstance(entry[1], tuple)  # Pattern

    def test_extract_patterns_with_moneyness(self):
        """Test extraction of ordinal patterns with moneyness filtering."""
        # Test extraction with each moneyness filter
        for moneyness in ['ITM', 'ATM', 'OTM', 'VOL_CRUSH']:
            patterns = self.analyzer.extract_patterns(self.test_data, moneyness_filter=moneyness)
            
            # Check that patterns were extracted
            self.assertIsInstance(patterns, dict)
            
            # At least some Greeks should have patterns
            has_patterns = False
            for greek, pattern_list in patterns.items():
                if len(pattern_list) > 0:
                    has_patterns = True
                    break
            
            self.assertTrue(has_patterns, f"No patterns extracted for any Greek with {moneyness} filter")
    
    def test_filter_by_moneyness(self):
        """Test filtering data by moneyness."""
        # Test ITM filter
        itm_data = self.analyzer._filter_by_moneyness(self.test_data, 'ITM')
        self.assertTrue(all(abs(d) > 0.7 for d in itm_data['delta']))
        
        # Test ATM filter
        atm_data = self.analyzer._filter_by_moneyness(self.test_data, 'ATM')
        self.assertTrue(all((abs(d) > 0.4) & (abs(d) < 0.6) for d in atm_data['delta']))
        
        # Test OTM filter
        otm_data = self.analyzer._filter_by_moneyness(self.test_data, 'OTM')
        self.assertTrue(all(abs(d) < 0.3 for d in otm_data['delta']))
    
    def test_analyze_pattern_profitability(self):
        """Test analyzing pattern profitability."""
        patterns = self.analyzer.extract_patterns(self.test_data)
        analysis = self.analyzer.analyze_pattern_profitability(
            self.test_data, patterns, forward_periods=[2, 3]
        )
        
        # Check that analysis contains entries for each Greek
        for greek in patterns.keys():
            self.assertIn(greek, analysis)
        
        # Check that at least one pattern is analyzed
        total_patterns = sum(len(patterns_dict) for patterns_dict in analysis.values())
        self.assertGreater(total_patterns, 0)
        
        # Check analysis format for a pattern
        for greek, patterns_dict in analysis.items():
            if patterns_dict:  # If there are patterns for this Greek
                pattern = next(iter(patterns_dict))
                pattern_analysis = patterns_dict[pattern]
                
                # Check that the analysis contains the requested forward periods
                self.assertIn(2, pattern_analysis)
                self.assertIn(3, pattern_analysis)
                
                # Check that the period analysis contains the expected keys
                period_analysis = pattern_analysis[2]
                expected_keys = ['count', 'win_rate', 'avg_return', 'median_return', 
                                'max_return', 'min_return', 'expected_value', 
                                'wins', 'losses']
                for key in expected_keys:
                    self.assertIn(key, period_analysis)
    
    def test_build_pattern_library(self):
        """Test building a pattern library."""
        patterns = self.analyzer.extract_patterns(self.test_data)
        analysis = self.analyzer.analyze_pattern_profitability(self.test_data, patterns)
        library = self.analyzer.build_pattern_library(analysis, 'ATM', forward_period=3)
        
        # Check that library contains entries for Greeks
        for greek in patterns.keys():
            self.assertIn(greek, library)
        
        # Check that library is stored in the analyzer's pattern_library
        self.assertIn('ATM', self.analyzer.pattern_library)
        self.assertEqual(library, self.analyzer.pattern_library['ATM'])
    
    def test_save_load_pattern_library(self):
        """Test saving and loading the pattern library."""
        # Create a pattern library
        patterns = self.analyzer.extract_patterns(self.test_data)
        analysis = self.analyzer.analyze_pattern_profitability(self.test_data, patterns)
        original_library = self.analyzer.build_pattern_library(analysis, 'ATM')
        
        # Save the library
        filepath = os.path.join(self.temp_dir, 'test_library.json')
        self.analyzer.save_pattern_library(filepath)
        
        # Verify file exists
        self.assertTrue(os.path.exists(filepath))
        
        # Create a new analyzer
        new_analyzer = GreekOrdinalPatternAnalyzer()
        
        # Load the library into the new analyzer
        new_analyzer.load_pattern_library(filepath)
        
        # Verify that loaded library matches original
        self.assertEqual(
            len(new_analyzer.pattern_library['ATM']), 
            len(self.analyzer.pattern_library['ATM'])
        )
        
        # Check that the pattern keys were correctly converted from strings back to tuples
        for greek in original_library.keys():
            if greek in new_analyzer.pattern_library['ATM']:
                # Check that all original patterns are in the loaded library
                for pattern in original_library[greek].keys():
                    self.assertIn(pattern, new_analyzer.pattern_library['ATM'][greek])
    
    def test_recognize_current_patterns(self):
        """Test recognizing patterns in current data."""
        # Create a pattern library
        patterns = self.analyzer.extract_patterns(self.test_data)
        analysis = self.analyzer.analyze_pattern_profitability(self.test_data, patterns)
        self.analyzer.build_pattern_library(analysis, 'ATM')
        
        # Use a subset of the data as current data
        current_data = self.test_data.iloc[5:8].reset_index(drop=True)
        recognized = self.analyzer.recognize_current_patterns(current_data)
        
        # Check that recognized patterns dictionary has the expected structure
        self.assertIn('ATM', recognized)
        
        # If patterns were recognized, check their format
        for moneyness, greek_patterns in recognized.items():
            for greek, pattern_info in greek_patterns.items():
                self.assertIn('pattern', pattern_info)
                self.assertIn('description', pattern_info)
                self.assertIn('stats', pattern_info)
    
    def test_enhance_trade_recommendation(self):
        """Test enhancing a trade recommendation with pattern analysis."""
        # Create a pattern library
        patterns = self.analyzer.extract_patterns(self.test_data)
        analysis = self.analyzer.analyze_pattern_profitability(self.test_data, patterns)
        self.analyzer.build_pattern_library(analysis, 'ATM')
        
        # Create current data and recognize patterns
        current_data = self.test_data.iloc[5:8].reset_index(drop=True)
        recognized = self.analyzer.recognize_current_patterns(current_data)
        
        # Create a mock trade recommendation
        recommendation = {
            'symbol': 'AAPL',
            'current_price': 108,
            'action': 'WAIT',
            'strategy': 'MONITOR_FOR_CLEARER_SIGNALS',
            'confidence': 0.6,
            'option_selection': {
                'atm_strike': 110,
                'otm_strike': 115
            }
        }
        
        # Enhance the recommendation
        enhanced = self.analyzer.enhance_trade_recommendation(recommendation, recognized)
        
        # Check that the enhanced recommendation is a copy
        self.assertIsNot(enhanced, recommendation)
        
        # Check that the original fields are preserved
        self.assertEqual(enhanced['symbol'], recommendation['symbol'])
        self.assertEqual(enhanced['current_price'], recommendation['current_price'])
        
        # Check for pattern enhancement flag
        if any(greek_patterns for greek_patterns in recognized.values()):
            if enhanced.get('pattern_enhanced', False):
                self.assertIn('confidence', enhanced)
                self.assertIn('supporting_patterns', enhanced)

    def test_pattern_extraction_by_moneyness(self):
        """Test pattern extraction for each moneyness category and identify issues."""
        # Set up a test logger to capture output
        test_logger = logging.getLogger('test_pattern_extraction')
        test_logger.setLevel(logging.INFO)
        
        # Store original logger
        original_logger = self.analyzer.logger
        self.analyzer.logger = test_logger
        
        for moneyness in ['ITM', 'ATM', 'OTM', 'VOL_CRUSH']:
            # First check if we have enough data after filtering
            filtered_data = self.analyzer._filter_by_moneyness(self.test_data, moneyness)
            print(f"\nTesting {moneyness} filter with {len(filtered_data)} data points")
            
            if len(filtered_data) < self.analyzer.window_size:
                print(f"  Not enough data for {moneyness} after filtering")
                continue
            
            # Check each Greek column
            for greek in self.analyzer.greeks:
                col_name = f'norm_{greek}' if f'norm_{greek}' in filtered_data.columns else greek
                
                if col_name not in filtered_data.columns:
                    print(f"  Column {col_name} not found for {moneyness}")
                    continue
                    
                # Check for NaN values
                valid_data = filtered_data[col_name].dropna()
                if len(valid_data) < self.analyzer.window_size:
                    print(f"  Not enough non-NaN values for {greek} in {moneyness} (got {len(valid_data)})")
                    continue
                    
                print(f"  {greek} has {len(valid_data)} valid data points in {moneyness}")
            
            # Try to extract patterns
            patterns = self.analyzer.extract_patterns(self.test_data, moneyness_filter=moneyness)
            
            # Check if patterns were extracted
            pattern_count = sum(len(p) for p in patterns.values())
            print(f"  Extracted {pattern_count} patterns for {moneyness}")
            
            # Check each Greek
            for greek in self.analyzer.greeks:
                if greek in patterns:
                    print(f"    {greek}: {len(patterns[greek])} patterns")
                else:
                    print(f"    {greek}: No patterns")
        
        # Restore original logger
        self.analyzer.logger = original_logger

    def test_moneyness_filtering_data_sufficiency(self):
        """Test that each moneyness filter has sufficient data for pattern extraction."""
        # Test each moneyness filter individually
        for moneyness in ['ITM', 'ATM', 'OTM', 'VOL_CRUSH']:
            filtered_data = self.analyzer._filter_by_moneyness(self.test_data, moneyness)
            
            # Log the amount of data after filtering
            print(f"\nMoneyness filter '{moneyness}' returned {len(filtered_data)} data points")
            
            # Check if there's enough data for the window size
            self.assertGreaterEqual(
                len(filtered_data), 
                self.analyzer.window_size,
                f"Insufficient data after {moneyness} filtering (got {len(filtered_data)}, need {self.analyzer.window_size})"
            )

    def test_recognize_current_patterns_with_library(self):
        """Test recognizing patterns with a pre-built library."""
        # Set up a test logger to capture output
        test_logger = logging.getLogger('test_recognize_patterns')
        test_logger.setLevel(logging.INFO)
        
        # Store original logger
        original_logger = self.analyzer.logger
        self.analyzer.logger = test_logger
        
        # First build a pattern library
        patterns = self.analyzer.extract_patterns(self.test_data)
        analysis = self.analyzer.analyze_pattern_profitability(self.test_data, patterns)
        
        # Build library for all moneyness categories
        for moneyness in ['ITM', 'ATM', 'OTM', 'VOL_CRUSH']:
            self.analyzer.build_pattern_library(analysis, moneyness)
        
        # Now test recognition with a subset of data for each moneyness
        for moneyness in ['ITM', 'ATM', 'OTM', 'VOL_CRUSH']:
            # Filter data by moneyness
            filtered_data = self.analyzer._filter_by_moneyness(self.test_data, moneyness)
            
            if len(filtered_data) < self.analyzer.window_size:
                print(f"Skipping {moneyness} - not enough data")
                continue
                
            # Use the last window_size data points
            current_data = filtered_data.iloc[-self.analyzer.window_size:].reset_index(drop=True)
            
            # Recognize patterns
            recognized = self.analyzer.recognize_current_patterns(current_data)
            
            # Check results
            print(f"\nRecognized patterns for {moneyness}:")
            for category, patterns in recognized.items():
                if patterns:
                    print(f"  {category}: {len(patterns)} patterns")
                else:
                    print(f"  {category}: No patterns")
        
        # Restore original logger
        self.analyzer.logger = original_logger


class TestPatternMetrics(unittest.TestCase):
    """
    Unit tests for the PatternMetrics utility class.
    """
    
    def test_calculate_pattern_entropy(self):
        """Test calculating pattern entropy."""
        # Create pattern frequencies
        pattern_frequencies = {
            (0, 1, 2): 5,
            (0, 2, 1): 3,
            (1, 0, 2): 2,
            (2, 1, 0): 1
        }
        
        # Calculate entropy
        entropy = PatternMetrics.calculate_pattern_entropy(pattern_frequencies)
        
        # Entropy should be positive
        self.assertGreater(entropy, 0)
        
        # Test with empty frequencies
        empty_entropy = PatternMetrics.calculate_pattern_entropy({})
        self.assertEqual(empty_entropy, 0.0)
    
    def test_calculate_pattern_complexity(self):
        """Test calculating pattern complexity."""
        # Create pattern sequence
        patterns = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (0, 1, 2), (2, 1, 0)]
        
        # Calculate complexity
        complexity = PatternMetrics.calculate_pattern_complexity(patterns)
        
        # Complexity should be between 0 and 1
        self.assertGreaterEqual(complexity, 0)
        self.assertLessEqual(complexity, 1)
        
        # Test with empty patterns
        empty_complexity = PatternMetrics.calculate_pattern_complexity([])
        self.assertEqual(empty_complexity, 0.0)


if __name__ == '__main__':
    unittest.main()


