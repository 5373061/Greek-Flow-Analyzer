import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import logging
from datetime import datetime, timedelta
import pytest
import sys

# Try to import the SymbolAnalyzer
try:
    from analysis.symbol_analyzer import SymbolAnalyzer
except ImportError:
    # Handle import error for CI environments
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.symbol_analyzer import SymbolAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSymbolAnalyzer(unittest.TestCase):
    """Test cases for the SymbolAnalyzer class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Test symbol
        self.symbol = "TEST"
        
        # Create analyzer instance
        self.analyzer = SymbolAnalyzer(
            cache_dir=self.temp_dir,
            output_dir=self.output_dir,
            use_parallel=False
        )
        
        # Generate test data
        self.options_data = self._generate_test_options_data()
        self.market_data = self._generate_test_market_data()
        
    def tearDown(self):
        """Clean up after tests"""
        # Remove temporary directory and files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _generate_test_options_data(self):
        """Generate synthetic options data for testing"""
        # Create a DataFrame with synthetic data
        np.random.seed(42)
        
        # Generate dates - use datetime objects consistently
        today = datetime.now()
        expirations = [today + timedelta(days=days) for days in [7, 14, 30, 60, 90]]
        
        # Generate strikes around current price
        current_price = 100
        strikes = np.linspace(80, 120, 9)
        
        # Create data rows
        rows = []
        for exp in expirations:
            for strike in strikes:
                # Add call option
                call = {
                    'symbol': f"{self.symbol}{exp.strftime('%y%m%d')}C{int(strike)}000",
                    'underlying': self.symbol,
                    'strike': strike,
                    'expiration': exp,  # Use datetime object
                    'type': 'call',
                    'bid': max(0, current_price - strike + np.random.uniform(0, 5)),
                    'ask': max(0, current_price - strike + np.random.uniform(5, 10)),
                    'impliedVolatility': np.random.uniform(0.2, 0.5),
                    'openInterest': int(np.random.uniform(10, 1000)),
                    'volume': int(np.random.uniform(1, 500)),
                    'delta': np.random.uniform(0, 1) if strike < current_price else np.random.uniform(0, 0.5),
                    'gamma': np.random.uniform(0, 0.1),
                    'theta': np.random.uniform(-1, 0),
                    'vega': np.random.uniform(0, 1),
                }
                rows.append(call)
                
                # Add put option
                put = {
                    'symbol': f"{self.symbol}{exp.strftime('%y%m%d')}P{int(strike)}000",
                    'underlying': self.symbol,
                    'strike': strike,
                    'expiration': exp,  # Use datetime object
                    'type': 'put',
                    'bid': max(0, strike - current_price + np.random.uniform(0, 5)),
                    'ask': max(0, strike - current_price + np.random.uniform(5, 10)),
                    'impliedVolatility': np.random.uniform(0.2, 0.5),
                    'openInterest': int(np.random.uniform(10, 1000)),
                    'volume': int(np.random.uniform(1, 500)),
                    'delta': np.random.uniform(-1, 0) if strike > current_price else np.random.uniform(-0.5, 0),
                    'gamma': np.random.uniform(0, 0.1),
                    'theta': np.random.uniform(-1, 0),
                    'vega': np.random.uniform(0, 1),
                }
                rows.append(put)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Ensure date columns are datetime objects
        df['date'] = datetime.now()  # Add date column
        
        return df
    
    def _generate_test_market_data(self):
        """Generate market data for testing"""
        return {
            'symbol': self.symbol,
            'currentPrice': 100.0,
            'previousClose': 99.0,
            'volume': 1000000,
            'historicalVolatility': 0.25,
            'impliedVolatility': 0.3,
            'riskFreeRate': 0.03,
            'analysis_date': datetime.now(),  # Use datetime object
            'timestamp': datetime.now().isoformat()
        }
    
    def test_initialization(self):
        """Test SymbolAnalyzer initialization"""
        # Check that analyzer was initialized correctly
        self.assertEqual(self.analyzer.cache_dir, self.temp_dir)
        self.assertEqual(self.analyzer.output_dir, self.output_dir)
        self.assertFalse(self.analyzer.use_parallel)
    
    def test_process_greeks(self):
        """Test the _process_greeks method"""
        # Mock the GreekEnergyAnalyzer.analyze method to avoid date issues
        from unittest.mock import patch
        with patch('greek_flow.flow.GreekEnergyAnalyzer.analyze') as mock_analyze:
            # Set up the mock to return a valid result
            mock_analyze.return_value = {
                'market_regime': {'primary_label': 'Test'},
                'reset_points': [],
                'energy_levels': []
            }
            
            # Call the method
            greek_results = self.analyzer._process_greeks(
                self.options_data, 
                self.market_data, 
                self.symbol
            )
            
            # Check that results were returned
            self.assertIsNotNone(greek_results, "Greek analysis should return results")
            
            # Check basic structure of results
            self.assertIn('symbol', greek_results, "Results should include symbol")
            self.assertEqual(greek_results['symbol'], self.symbol, "Symbol should match")
    
    def test_process_greeks_empty_data(self):
        """Test _process_greeks with empty DataFrame"""
        # Test with empty DataFrame
        options_df = pd.DataFrame()
        expected_log = "No valid options data for Greek analysis"
        
        # Call the method with the test data
        with self.assertLogs(level='INFO') as log:
            greek_results = self.analyzer._process_greeks(
                options_df, 
                self.market_data, 
                self.symbol
            )
            
            # Check logs contain expected message
            self.assertTrue(any(expected_log in msg for msg in log.output))
            
            # Results should be None for empty data
            self.assertIsNone(greek_results)
    
    def test_process_greeks_missing_columns(self):
        """Test _process_greeks with missing columns"""
        # Test with missing columns
        options_df = pd.DataFrame({
            'symbol': [f"{self.symbol}DUMMY"],
            'strike': [100.0]
        })
        expected_log = "Missing required columns for Greek analysis"
        
        # Call the method with the test data
        with self.assertLogs(level='INFO') as log:
            greek_results = self.analyzer._process_greeks(
                options_df, 
                self.market_data, 
                self.symbol
            )
            
            # Check logs contain expected message
            self.assertTrue(any(expected_log in msg for msg in log.output))
    
    def test_process_greeks_valid_data(self):
        """Test _process_greeks with valid data"""
        # Test with valid data
        options_df = self._generate_test_options_data()
        expected_log = "Greek analysis completed"
        
        # Call the method with the test data
        with self.assertLogs(level='INFO') as log:
            greek_results = self.analyzer._process_greeks(
                options_df, 
                self.market_data, 
                self.symbol
            )
            
            # Check logs contain expected message
            self.assertTrue(any(expected_log in msg for msg in log.output))
            
            # Check results structure for valid data
            self.assertIsNotNone(greek_results)
            self.assertIn('symbol', greek_results)
            self.assertEqual(greek_results['symbol'], self.symbol)

    def test_greek_results_structure(self):
        """Test the structure of Greek analysis results in detail"""
        # Mock the GreekEnergyAnalyzer.analyze method
        from unittest.mock import patch
        with patch('greek_flow.flow.GreekEnergyAnalyzer.analyze') as mock_analyze:
            # Set up the mock to return a detailed result
            mock_analyze.return_value = {
                'market_regime': {
                    'primary_label': 'Bullish',
                    'confidence': 0.85,
                    'secondary_label': 'Volatile'
                },
                'reset_points': [
                    {'price': 95.0, 'strength': 0.7},
                    {'price': 105.0, 'strength': 0.8}
                ],
                'energy_levels': [
                    {'price': 90.0, 'energy': 0.5},
                    {'price': 100.0, 'energy': 0.9}
                ]
            }
            
            # Call the method
            greek_results = self.analyzer._process_greeks(
                self._generate_test_options_data(), 
                self.market_data, 
                self.symbol
            )
            
            # Detailed structure assertions
            self.assertIsNotNone(greek_results)
            self.assertIn('symbol', greek_results)
            self.assertIn('options_count', greek_results)
            self.assertIn('market_regime', greek_results)
            self.assertIn('reset_points', greek_results)
            self.assertIn('energy_levels', greek_results)
            
            # Check specific values
            self.assertEqual(greek_results['symbol'], self.symbol)
            self.assertEqual(greek_results['options_count'], 90)  # Based on test data generation
            
            # Check nested structure
            self.assertIn('primary_label', greek_results['market_regime'])
            self.assertEqual(greek_results['market_regime']['primary_label'], 'Bullish')
            
            # Check array structures
            self.assertTrue(isinstance(greek_results['reset_points'], list))
            self.assertTrue(isinstance(greek_results['energy_levels'], list))

# This will only run when using pytest directly
if __name__ != '__main__':
    # Parameterized test using pytest
    @pytest.mark.parametrize("test_data", [
        {"empty": True, "missing_cols": False},
        {"empty": False, "missing_cols": True},
        {"empty": False, "missing_cols": False}
    ])
    def test_process_greeks_scenarios(test_data):
        """Test _process_greeks with different data scenarios using parameterization"""
        # Create test instance
        test_instance = TestSymbolAnalyzer()
        test_instance.setUp()
        
        try:
            if test_data["empty"]:
                # Test with empty DataFrame
                options_df = pd.DataFrame()
                expected_log = "No valid options data for Greek analysis"
            elif test_data["missing_cols"]:
                # Test with missing columns
                options_df = pd.DataFrame({
                    'symbol': [f"{test_instance.symbol}DUMMY"],
                    'strike': [100.0]
                })
                expected_log = "Missing required columns for Greek analysis"
            else:
                # Test with valid data
                options_df = test_instance._generate_test_options_data()
                expected_log = "Greek analysis completed"
            
            # Call the method with the test data
            with test_instance.assertLogs(level='INFO') as log:
                greek_results = test_instance.analyzer._process_greeks(
                    options_df, 
                    test_instance.market_data, 
                    test_instance.symbol
                )
                
                # Check logs contain expected message
                test_instance.assertTrue(any(expected_log in msg for msg in log.output))
                
                # Check results structure for valid data
                if not test_data["empty"] and not test_data["missing_cols"]:
                    test_instance.assertIsNotNone(greek_results)
                    test_instance.assertIn('symbol', greek_results)
                    test_instance.assertEqual(greek_results['symbol'], test_instance.symbol)
        finally:
            test_instance.tearDown()

if __name__ == '__main__':
    unittest.main()





