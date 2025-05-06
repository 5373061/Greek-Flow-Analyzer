import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pipeline.data_pipeline import OptionsDataPipeline
import config

class TestOptionsAnalysis(unittest.TestCase):
    def setUp(self):
        self.pipeline = OptionsDataPipeline(config)
        # Base test data
        self.sample_options = pd.DataFrame({
            'strike': [440, 450, 460],
            'expiration': [datetime.now() + timedelta(days=30)] * 3,
            'openInterest': [100, 200, 150],
            'type': ['call', 'call', 'put'],
            'underlying_price': [450.0] * 3,
            'underlying_ticker': ['SPY'] * 3,
            'underlying_volume': [1000000] * 3,
            'fetch_time': [datetime.now()] * 3
        })
        
    def test_greek_calculations(self):
        """Test basic Greek energy calculations"""
        analysis_df = self.pipeline.calculate_greeks(self.sample_options)
        self.assertIsNotNone(analysis_df, "Greek calculations failed")
        
        # Column validation
        required_columns = [
            'delta', 'gamma', 'theta', 'vega',
            'gamma_contribution', 'delta_weight'
        ]
        for col in required_columns:
            self.assertIn(col, analysis_df.columns)
            
        # Value validation
        self.assertTrue(all(analysis_df['gamma'] > 0), "Gamma should be positive")
        self.assertTrue(all(analysis_df['gamma_contribution'] <= 1.0), "Gamma contribution should be <= 1")
        expected_delta_weights = analysis_df['delta'] * analysis_df['openInterest']
        pd.testing.assert_series_equal(
            analysis_df['delta_weight'],
            expected_delta_weights,
            check_names=False
        )
        
    def test_edge_cases(self):
        """Test edge cases for Greek calculations"""
        # Deep ITM call
        itm_call = self.sample_options.copy()
        itm_call.loc[0, 'strike'] = 400.0
        itm_df = self.pipeline.calculate_greeks(itm_call)
        self.assertGreater(itm_df.iloc[0]['delta'], 0.9, "Deep ITM call delta should approach 1")
        
        # Deep OTM put
        otm_put = self.sample_options.copy()
        otm_put.loc[2, 'strike'] = 550.0
        otm_df = self.pipeline.calculate_greeks(otm_put)
        self.assertLess(otm_df.iloc[2]['delta'], -0.9, "Deep OTM put delta should approach -1")
        
    def test_empty_data(self):
        """Test handling of empty DataFrames"""
        empty_df = pd.DataFrame(columns=self.sample_options.columns)
        result = self.pipeline.calculate_greeks(empty_df)
        self.assertIsNone(result, "Should handle empty DataFrame gracefully")

if __name__ == '__main__':
    unittest.main()