import unittest
import sys
from pathlib import Path
import logging
import os
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import json
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.data_pipeline import OptionsDataPipeline
import config

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Convert config module to dictionary for the pipeline
        config_dict = {
            'POLYGON_API_KEY': getattr(config, 'POLYGON_API_KEY', None),
            'API_VERSION': getattr(config, 'API_VERSION', 'v2'),
            'API_BASE_URL': getattr(config, 'API_BASE_URL', 'https://api.polygon.io')
        }
        self.pipeline = OptionsDataPipeline(config_dict)
        self.test_symbol = "SPY"
        
        # Create mock data for testing
        # Raw API response format
        self.mock_underlying_data_raw = {
            'ticker': 'SPY',
            'min': {'c': 450.25},  # Last price
            'prevDay': {'v': 50000000},  # Volume
            'day': {'o': 448.50, 'h': 452.75, 'l': 447.80, 'c': 450.25},
            'lastTrade': {'p': 450.25, 't': 1625097600000},
            'lastQuote': {'p': 450.30, 'S': 100, 'P': 450.20, 's': 200}
        }
        
        # Processed format that pipeline returns
        self.mock_underlying_data = {
            'ticker': 'SPY',
            'last': 450.25,
            'volume': 50000000
        }
        
        # Create mock options data
        self.mock_options_data = []
        for i in range(10):
            strike = 440 + i * 5
            is_call = i % 2 == 0
            option_type = 'call' if is_call else 'put'
            delta = 0.5 + (i * 0.05) if is_call else -0.5 - (i * 0.05)
            
            option = {
                'details': {
                    'strike_price': strike,
                    'expiration_date': '2023-12-15',
                    'contract_type': option_type.upper()
                },
                'day': {
                    'o': 5.25, 'h': 5.75, 'l': 5.10, 'c': 5.50, 'v': 1000
                },
                'greeks': {
                    'delta': delta,
                    'gamma': 0.05,
                    'theta': -0.10,
                    'vega': 0.15
                },
                'implied_volatility': 0.25,
                'open_interest': 500
            }
            self.mock_options_data.append(option)
    
    @patch('pipeline.data_pipeline.fetch_underlying_snapshot')
    @patch('pipeline.data_pipeline.fetch_options_chain_snapshot')
    def test_fetch_symbol_data_mocked(self, mock_fetch_options, mock_fetch_underlying):
        """Test fetching data with mocked API responses"""
        # Setup mocks with raw API response format
        mock_fetch_underlying.return_value = self.mock_underlying_data_raw
        mock_fetch_options.return_value = self.mock_options_data
        
        # Call the method
        underlying, options = self.pipeline.fetch_symbol_data(self.test_symbol)
        
        # Verify mocks were called
        mock_fetch_underlying.assert_called_once_with(self.test_symbol, self.pipeline.config.get('POLYGON_API_KEY'))
        mock_fetch_options.assert_called_once_with(self.test_symbol, self.pipeline.config.get('POLYGON_API_KEY'))
        
        # Verify results match the expected processed format
        self.assertEqual(underlying, self.mock_underlying_data)
        self.assertEqual(options, self.mock_options_data)
    
    @patch('pipeline.data_pipeline.fetch_underlying_snapshot')
    @patch('pipeline.data_pipeline.fetch_options_chain_snapshot')
    def test_prepare_analysis_data_mocked(self, mock_fetch_options, mock_fetch_underlying):
        """Test data preparation with mocked API responses"""
        # Setup mocks with raw API response format
        mock_fetch_underlying.return_value = self.mock_underlying_data_raw
        mock_fetch_options.return_value = self.mock_options_data
        
        # Call the method
        df = self.pipeline.prepare_analysis_data(self.test_symbol)
        
        # Verify results
        self.assertIsNotNone(df)
        self.assertEqual(len(df), len(self.mock_options_data))
        
        # Check columns
        expected_cols = [
            'strike', 'expiration', 'type', 'openInterest', 
            'implied_volatility', 'underlying_price', 
            'underlying_ticker', 'underlying_volume'
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)
        
        # Check values
        self.assertEqual(df['underlying_ticker'].iloc[0], 'SPY')
        self.assertEqual(df['underlying_price'].iloc[0], 450.25)
        self.assertEqual(df['underlying_volume'].iloc[0], 50000000)
    
    @patch('pipeline.data_pipeline.fetch_underlying_snapshot')
    def test_invalid_symbol_mocked(self, mock_fetch_underlying):
        """Test behavior with invalid symbol using mocks"""
        # Setup mock to return None for invalid symbol
        mock_fetch_underlying.return_value = None
        
        # Call the method
        underlying, options = self.pipeline.fetch_symbol_data("INVALID")
        
        # Verify results
        self.assertIsNone(underlying)
        self.assertIsNone(options)
        
        # Verify mock was called
        mock_fetch_underlying.assert_called_once_with("INVALID", self.pipeline.config.get('POLYGON_API_KEY'))
    
    @patch('pipeline.data_pipeline.fetch_underlying_snapshot')
    @patch('pipeline.data_pipeline.fetch_options_chain_snapshot')
    def test_error_handling_mocked(self, mock_fetch_options, mock_fetch_underlying):
        """Test error handling with mocked API errors"""
        # Setup mocks to simulate API errors
        mock_fetch_underlying.side_effect = Exception("API Error")
        
        # Call the method
        underlying, options = self.pipeline.fetch_symbol_data(self.test_symbol)
        
        # Verify results
        self.assertIsNone(underlying)
        self.assertIsNone(options)
        
        # Verify mock was called
        mock_fetch_underlying.assert_called_once_with(self.test_symbol, self.pipeline.config.get('POLYGON_API_KEY'))
        mock_fetch_options.assert_not_called()  # Should not be called if underlying fetch fails

    def test_fetch_symbol_data(self):
        """Test fetching both underlying and options data"""
        try:
            underlying, options = self.pipeline.fetch_symbol_data(self.test_symbol)
            
            # Test underlying data
            self.assertIsNotNone(underlying, "Underlying data should not be None")
            self.assertIsInstance(underlying, dict, "Underlying data should be a dictionary")
            self.assertIn('ticker', underlying, "Underlying data should contain ticker")
            
            # Test options data
            self.assertIsNotNone(options, "Options data should not be None")
            self.assertIsInstance(options, list, "Options data should be a list")
            self.assertGreater(len(options), 0, "Should have at least one options contract")
            
        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")

    def test_prepare_analysis_data(self):
        """Test data preparation for analysis"""
        try:
            df = self.pipeline.prepare_analysis_data(self.test_symbol)
            
            self.assertIsNotNone(df, "Analysis DataFrame should not be None")
            self.assertFalse(df.empty, "Analysis DataFrame should not be empty")
            
            # Check all expected columns
            expected_cols = [
                'strike', 'expiration', 'type', 'openInterest', 
                'implied_volatility', 'underlying_price', 
                'underlying_ticker', 'underlying_volume'
            ]
            for col in expected_cols:
                self.assertIn(col, df.columns, f"Missing expected column: {col}")
            
            print(f"Available columns: {list(df.columns)}")
            
            # Data quality checks
            self.assertTrue(all(df['strike'] > 0), "Strike prices should be positive")
            self.assertTrue(all(df['underlying_price'] > 0), "Underlying prices should be positive")
            self.assertTrue(all(df['implied_volatility'] > 0), "Implied volatility should be positive")
            self.assertTrue(all(df['openInterest'] >= 0), "Open interest should be non-negative")
            
            # Check option types
            self.assertTrue(all(df['type'].isin(['call', 'put'])), "Option types should be 'call' or 'put'")
            
            # Check expiration dates are in the future
            today = pd.Timestamp.today().normalize()
            self.assertTrue(all(df['expiration'] >= today), "Expiration dates should be in the future")
            
        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")
    
    def test_invalid_symbol(self):
        """Test behavior with invalid symbol"""
        invalid_symbol = "INVALID123XYZ"
        
        # Test fetch_symbol_data with invalid symbol
        underlying, options = self.pipeline.fetch_symbol_data(invalid_symbol)
        self.assertIsNone(underlying, "Underlying data should be None for invalid symbol")
        self.assertIsNone(options, "Options data should be None for invalid symbol")
        
        # Test prepare_analysis_data with invalid symbol
        df = self.pipeline.prepare_analysis_data(invalid_symbol)
        self.assertIsNone(df, "DataFrame should be None for invalid symbol")
    
    def test_calculate_greeks(self):
        """Test Greek calculations"""
        # First get some data to work with
        df = self.pipeline.prepare_analysis_data(self.test_symbol)
        
        if df is not None and not df.empty:
            # Test Greek calculations
            greeks_df = self.pipeline.calculate_greeks(df)
            
            self.assertIsNotNone(greeks_df, "Greeks DataFrame should not be None")
            self.assertFalse(greeks_df.empty, "Greeks DataFrame should not be empty")
            
            # Check for Greek columns
            greek_cols = ['delta', 'gamma', 'theta', 'vega']
            for col in greek_cols:
                self.assertIn(col, greeks_df.columns, f"Missing Greek column: {col}")
            
            # Check that Greeks are calculated for all rows
            self.assertEqual(len(greeks_df), len(df), "Greeks should be calculated for all options")
            
            # Check that Greeks are within reasonable ranges
            self.assertTrue(all(greeks_df['delta'].between(-1, 1)), "Delta should be between -1 and 1")
            self.assertTrue(all(greeks_df['gamma'] >= 0), "Gamma should be non-negative")
        else:
            self.skipTest("Skipping Greek calculations test due to no data")
    
    def test_error_handling(self):
        """Test error handling in the pipeline"""
        # Create a pipeline with invalid API key
        invalid_config = {
            'POLYGON_API_KEY': 'invalid_key',
            'API_VERSION': 'v2',
            'API_BASE_URL': 'https://api.polygon.io'
        }
        invalid_pipeline = OptionsDataPipeline(invalid_config)
        
        # Test fetch_symbol_data with invalid API key
        underlying, options = invalid_pipeline.fetch_symbol_data(self.test_symbol)
        self.assertIsNone(underlying, "Underlying data should be None with invalid API key")
        self.assertIsNone(options, "Options data should be None with invalid API key")
        
        # Test prepare_analysis_data with invalid API key
        df = invalid_pipeline.prepare_analysis_data(self.test_symbol)
        self.assertIsNone(df, "DataFrame should be None with invalid API key")

    @patch('pipeline.data_pipeline.fetch_underlying_snapshot')
    def test_underlying_data_transformation(self, mock_fetch_underlying):
        """Test that underlying data is correctly transformed"""
        # Setup mock with raw API response format
        mock_fetch_underlying.return_value = self.mock_underlying_data_raw
        
        # Call the method
        underlying, _ = self.pipeline.fetch_symbol_data(self.test_symbol)
        
        # Verify the transformation
        self.assertEqual(underlying['ticker'], 'SPY')
        self.assertEqual(underlying['last'], 450.25)
        self.assertEqual(underlying['volume'], 50000000)
        
        # Check that the raw fields are not present
        self.assertNotIn('min', underlying)
        self.assertNotIn('prevDay', underlying)
        self.assertNotIn('day', underlying)
        self.assertNotIn('lastTrade', underlying)
        self.assertNotIn('lastQuote', underlying)

if __name__ == '__main__':
    unittest.main(verbosity=2)
