import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import os
import json
from unittest.mock import patch, MagicMock
import logging

sys.path.append(str(Path(__file__).parent.parent))

from analysis.pipeline_manager import AnalysisPipeline

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

class TestAnalysisPipeline(unittest.TestCase):
    """Test cases for the AnalysisPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "greek_config": {
                "regime_thresholds": {
                    "highVolatility": 0.3,
                    "lowVolatility": 0.15,
                    "strongBullish": 0.7,
                    "strongBearish": -0.7,
                    "neutralZone": 0.2
                }
            },
            "cache_dir": "test_cache",
            "entropy_threshold_low": 30,
            "entropy_threshold_high": 70,
            "anomaly_sensitivity": 1.5
        }
        
        # Create test directory
        os.makedirs(self.config["cache_dir"], exist_ok=True)
        
        # Create pipeline instance
        self.pipeline = AnalysisPipeline(self.config)
        
        # Create test data
        self.test_symbol = "TEST"
        self.test_options_df = self._create_test_options_data()
        self.test_price_history = self._create_test_price_history()
        self.test_market_data = {
            'symbol': self.test_symbol,
            'currentPrice': 100.0,
            'historicalVolatility': 0.2,
            'impliedVolatility': 0.25,
            'riskFreeRate': 0.04
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test cache directory
        import shutil
        if os.path.exists(self.config["cache_dir"]):
            shutil.rmtree(self.config["cache_dir"])
    
    def _create_test_options_data(self):
        """Create test options data."""
        # Create a simple options DataFrame
        data = {
            'strike': [90, 95, 100, 105, 110],
            'expiration': pd.date_range(start='2023-12-01', periods=5),
            'type': ['call', 'call', 'call', 'put', 'put'],
            'openInterest': [100, 200, 300, 250, 150],
            'impliedVolatility': [0.2, 0.22, 0.25, 0.23, 0.21],
            'delta': [0.7, 0.6, 0.5, -0.5, -0.6],
            'gamma': [0.05, 0.06, 0.07, 0.06, 0.05],
            'theta': [-0.1, -0.12, -0.15, -0.13, -0.11],
            'vega': [0.2, 0.25, 0.3, 0.25, 0.2],
            'symbol': [self.test_symbol] * 5
        }
        return pd.DataFrame(data)
    
    def _create_test_price_history(self):
        """Create test price history data."""
        # Create a simple price history DataFrame
        data = {
            'date': pd.date_range(start='2023-01-01', periods=30),
            'open': np.linspace(90, 110, 30),
            'high': np.linspace(92, 112, 30),
            'low': np.linspace(89, 109, 30),
            'close': np.linspace(91, 111, 30),
            'volume': np.random.randint(1000, 10000, 30)
        }
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline)
        self.assertEqual(self.pipeline.config, self.config)
        self.assertIsNotNone(self.pipeline.greek_analyzer)
        self.assertEqual(self.pipeline.cache_dir, self.config["cache_dir"])
    
    def test_load_data(self):
        """Test data loading functionality."""
        # Save test data to files
        options_path = os.path.join(self.config["cache_dir"], f"{self.test_symbol}_options.csv")
        price_path = os.path.join(self.config["cache_dir"], f"{self.test_symbol}_prices.csv")
        
        self.test_options_df.to_csv(options_path, index=False)
        self.test_price_history.to_csv(price_path, index=False)
        
        # Test load_data method
        options_df, price_history, market_data = self.pipeline.load_data(
            self.test_symbol, options_path, price_path
        )
        
        # Verify results
        self.assertIsNotNone(options_df)
        self.assertIsNotNone(price_history)
        self.assertIsNotNone(market_data)
        self.assertEqual(market_data['symbol'], self.test_symbol)
        self.assertAlmostEqual(market_data['currentPrice'], self.test_price_history['close'].iloc[-1])
    
    @patch('analysis.pipeline_manager.GreekEnergyAnalyzer')
    def test_run_greek_analysis(self, mock_analyzer):
        """Test Greek analysis functionality."""
        # Create a mock for the greek_analyzer instance
        mock_greek_analyzer = MagicMock()
        mock_greek_analyzer.analyze_greek_profiles.return_value = {'test': 'results'}
        
        # Replace the real greek_analyzer with our mock
        self.pipeline.greek_analyzer = mock_greek_analyzer
        
        # Setup static method mocks
        mock_analyzer.format_results.return_value = {'formatted': 'results'}
        mock_analyzer.analyze_chain_energy.return_value = {'chain': 'energy'}
        
        # Test run_greek_analysis method
        results = self.pipeline.run_greek_analysis(self.test_options_df, self.test_market_data)
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertEqual(results['greek_profiles'], {'test': 'results'})
        self.assertEqual(results['formatted_results'], {'formatted': 'results'})
        self.assertEqual(results['chain_energy'], {'chain': 'energy'})
        
        # Verify mocks were called
        mock_greek_analyzer.analyze_greek_profiles.assert_called_once_with(
            self.test_options_df, self.test_market_data
        )
        mock_analyzer.format_results.assert_called_once_with({'test': 'results'})
        mock_analyzer.analyze_chain_energy.assert_called_once()
    
    @patch('analysis.pipeline_manager.EntropyAnalyzer')
    def test_run_entropy_analysis(self, mock_entropy_analyzer):
        """Test entropy analysis functionality."""
        # Setup mock
        mock_instance = mock_entropy_analyzer.return_value
        mock_instance.analyze_greek_entropy.return_value = {'entropy': 'metrics'}
        mock_instance.detect_anomalies.return_value = ['anomaly1', 'anomaly2']
        mock_instance.generate_entropy_report.return_value = {'report': 'data'}
        
        # Test run_entropy_analysis method
        results = self.pipeline.run_entropy_analysis(self.test_options_df, "test_output")
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertEqual(results['entropy_metrics'], {'entropy': 'metrics'})
        self.assertEqual(results['anomalies'], ['anomaly1', 'anomaly2'])
        self.assertEqual(results['report'], {'report': 'data'})
        
        # Verify mock was called
        mock_instance.analyze_greek_entropy.assert_called_once()
        mock_instance.detect_anomalies.assert_called_once()
        mock_instance.generate_entropy_report.assert_called_once()
    
    @patch('analysis.pipeline_manager.GreekEnergyAnalyzer')
    def test_analyze_trade_opportunities(self, mock_analyzer):
        """Test trade opportunities analysis."""
        # Setup mock
        mock_analyzer.analyze_trade_opportunities.return_value = {'opportunities': 'data'}
        
        # Test analyze_trade_opportunities method
        results = self.pipeline.analyze_trade_opportunities({'chain': 'energy'}, self.test_price_history)
        
        # Verify results
        self.assertEqual(results, {'opportunities': 'data'})
        
        # Verify mock was called
        mock_analyzer.analyze_trade_opportunities.assert_called_once_with(
            {'chain': 'energy'}, self.test_price_history
        )
    
    def test_run_chain_energy_analysis(self):
        """Test chain energy analysis."""
        # Test run_chain_energy_analysis method
        results = self.pipeline.run_chain_energy_analysis(self.test_options_df, self.test_symbol)
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertEqual(results['symbol'], self.test_symbol)
        self.assertEqual(results['total_contracts'], len(self.test_options_df))
        self.assertEqual(results['total_open_interest'], self.test_options_df['openInterest'].sum())
    
    @patch('analysis.pipeline_manager.AnalysisPipeline.load_data')
    @patch('analysis.pipeline_manager.AnalysisPipeline.run_greek_analysis')
    @patch('analysis.pipeline_manager.AnalysisPipeline.run_chain_energy_analysis')
    @patch('analysis.pipeline_manager.AnalysisPipeline.run_entropy_analysis')
    def test_run_full_analysis(self, mock_entropy, mock_chain, mock_greek, mock_load):
        """Test full analysis pipeline."""
        # Setup mocks
        mock_load.return_value = (self.test_options_df, self.test_price_history, self.test_market_data)
        mock_greek.return_value = {'greek': 'analysis'}
        mock_chain.return_value = {'chain': 'energy'}
        mock_entropy.return_value = {'entropy': 'analysis'}
        
        # Test run_full_analysis method
        results = self.pipeline.run_full_analysis(
            self.test_symbol, "options.csv", "prices.csv", "output_dir"
        )
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertEqual(results['symbol'], self.test_symbol)
        self.assertEqual(results['greek_analysis'], {'greek': 'analysis'})
        self.assertEqual(results['chain_energy'], {'chain': 'energy'})
        self.assertEqual(results['entropy_analysis'], {'entropy': 'analysis'})
        
        # Verify mocks were called
        mock_load.assert_called_once()
        mock_greek.assert_called_once()
        mock_chain.assert_called_once()
        mock_entropy.assert_called_once()
    
    def test_run_full_analysis_with_skip_entropy(self):
        """Test full analysis with entropy analysis skipped."""
        # Setup test data files
        options_path = os.path.join(self.config["cache_dir"], f"{self.test_symbol}_options.csv")
        price_path = os.path.join(self.config["cache_dir"], f"{self.test_symbol}_prices.csv")
        output_dir = os.path.join(self.config["cache_dir"], "output")
        os.makedirs(output_dir, exist_ok=True)
        
        self.test_options_df.to_csv(options_path, index=False)
        self.test_price_history.to_csv(price_path, index=False)
        
        # Test run_full_analysis method with skip_entropy=True
        with patch('analysis.pipeline_manager.GreekEnergyFlow') as mock_flow:
            mock_flow_instance = mock_flow.return_value
            mock_flow_instance.analyze_greek_profiles.return_value = {'test': 'results'}
            
            results = self.pipeline.run_full_analysis(
                self.test_symbol, options_path, price_path, output_dir, skip_entropy=True
            )
            
            # Verify entropy analysis was skipped
            self.assertIsNotNone(results)
            self.assertEqual(results['entropy_analysis'], {'skipped': True})

    def test_run_full_analysis_integration(self):
        """Test full analysis pipeline with real components."""
        # Save test data to files
        options_path = os.path.join(self.config["cache_dir"], f"{self.test_symbol}_options.csv")
        price_path = os.path.join(self.config["cache_dir"], f"{self.test_symbol}_prices.csv")
        output_dir = os.path.join(self.config["cache_dir"], "output")
        os.makedirs(output_dir, exist_ok=True)
        
        self.test_options_df.to_csv(options_path, index=False)
        self.test_price_history.to_csv(price_path, index=False)
        
        # Run the full analysis
        with patch('analysis.pipeline_manager.GreekEnergyFlow') as mock_flow:
            # Setup the mock to return valid results
            mock_instance = mock_flow.return_value
            mock_instance.analyze_greek_profiles.return_value = {
                'delta_exposure': 0.5,
                'gamma_exposure': 0.2,
                'vega_exposure': 0.3,
                'theta_exposure': -0.1
            }
            
            # Run the full analysis
            results = self.pipeline.run_full_analysis(
                self.test_symbol, options_path, price_path, output_dir
            )
            
            # Verify results
            self.assertIsNotNone(results)
            self.assertEqual(results['symbol'], self.test_symbol)
            self.assertIn('greek_analysis', results)
            self.assertIn('chain_energy', results)
            self.assertIn('entropy_analysis', results)
            self.assertIn('market_data', results)
            self.assertEqual(results['market_data']['symbol'], self.test_symbol)

if __name__ == '__main__':
    unittest.main()





