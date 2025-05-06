import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import logging

sys.path.append(str(Path(__file__).parent.parent))

from entropy_analyzer.integration_after_greeks import (
    run_entropy_analysis,
    run_advanced_risk_management,
    integrate_with_analysis_pipeline
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

class TestEntropyIntegration(unittest.TestCase):
    """Test cases for the entropy analyzer integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        self.test_symbol = "TEST"
        self.test_options_df = self._create_test_options_data()
        self.test_greek_data = {
            'delta_exposure': 0.5,
            'gamma_exposure': 0.2,
            'vega_exposure': 0.3,
            'theta_exposure': -0.1,
            'profiles': {
                'delta': {'bullish': 0.7},
                'gamma': {'neutral': 0.5},
                'vega': {'positive': 0.6}
            }
        }
        self.test_config = {
            'risk_thresholds': {
                'high_delta': 0.8,
                'high_gamma': 0.3,
                'high_vega': 0.4
            }
        }
        self.test_risk_metrics = {
            'basic_risk_score': 65,
            'max_loss': 1000
        }
    
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
    
    @patch('entropy_analyzer.integration_after_greeks.EntropyAnalyzer')
    def test_run_entropy_analysis(self, mock_entropy_analyzer):
        """Test entropy analysis functionality."""
        # Setup mock
        mock_instance = mock_entropy_analyzer.return_value
        mock_instance.analyze_greek_entropy.return_value = {'entropy': 'metrics'}
        mock_instance.detect_anomalies.return_value = ['anomaly1', 'anomaly2']
        
        # Test run_entropy_analysis method
        results = run_entropy_analysis(self.test_options_df, self.test_symbol)
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertEqual(results['entropy'], 'metrics')
        self.assertEqual(results['anomalies'], ['anomaly1', 'anomaly2'])
        
        # Verify mock was called
        mock_instance.analyze_greek_entropy.assert_called_once()
        mock_instance.detect_anomalies.assert_called_once()
    
    @patch('entropy_analyzer.integration_after_greeks.AdvancedRiskManager')
    def test_run_advanced_risk_management(self, mock_risk_manager):
        """Test advanced risk management functionality."""
        # Setup mock
        mock_instance = mock_risk_manager.return_value
        mock_instance.generate_risk_management_plan.return_value = {'risk': 'plan'}
        
        # Test run_advanced_risk_management method
        entropy_data = {'entropy': 'metrics', 'anomalies': ['anomaly1']}
        results = run_advanced_risk_management(
            self.test_greek_data, entropy_data, 100.0, self.test_config
        )
        
        # Verify results
        self.assertEqual(results, {'risk': 'plan'})
        
        # Verify mock was called
        mock_instance.generate_risk_management_plan.assert_called_once()
    
    @patch('entropy_analyzer.integration_after_greeks.run_entropy_analysis')
    @patch('entropy_analyzer.integration_after_greeks.run_advanced_risk_management')
    def test_integrate_with_analysis_pipeline(self, mock_risk, mock_entropy):
        """Test integration with analysis pipeline."""
        # Setup mocks
        mock_entropy.return_value = {'entropy': 'metrics', 'energy_state': {'state': 'BULLISH'}}
        mock_risk.return_value = {'risk': 'plan', 'adaptive_exits': [
            {'type': 'Stop Loss', 'condition': 'Price < 90', 'reason': 'Protect capital'}
        ]}
        
        # Test integrate_with_analysis_pipeline method
        entropy_data, risk_metrics = integrate_with_analysis_pipeline(
            self.test_options_df,
            self.test_greek_data,
            100.0,
            self.test_config,
            self.test_risk_metrics,
            self.test_symbol
        )
        
        # Verify results
        self.assertIsNotNone(entropy_data)
        self.assertEqual(entropy_data['entropy'], 'metrics')
        self.assertEqual(entropy_data['energy_state_string'], 'BULLISH')
        
        self.assertIsNotNone(risk_metrics)
        self.assertEqual(risk_metrics['basic_risk_score'], 65)
        self.assertEqual(risk_metrics['risk'], 'plan')
        
        # Verify mocks were called
        mock_entropy.assert_called_once_with(self.test_options_df, self.test_symbol)
        mock_risk.assert_called_once()

if __name__ == '__main__':
    unittest.main()