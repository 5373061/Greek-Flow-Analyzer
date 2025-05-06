"""
Unit tests for the Greek Energy Analyzer component.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
from unittest.mock import patch, MagicMock

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the component to test
from greek_flow.flow import GreekEnergyAnalyzer

class TestGreekEnergyAnalyzer(unittest.TestCase):
    """Test the GreekEnergyAnalyzer class."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample options data
        self.options_data = pd.DataFrame({
            "symbol": ["TEST"] * 10,
            "strike": [90, 95, 100, 105, 110, 90, 95, 100, 105, 110],
            "expiration": [datetime.now() + timedelta(days=30)] * 10,
            "type": ["call"] * 5 + ["put"] * 5,
            "openInterest": [100, 200, 300, 200, 100, 100, 200, 300, 200, 100],
            "impliedVolatility": [0.3, 0.25, 0.2, 0.25, 0.3, 0.3, 0.25, 0.2, 0.25, 0.3],
            "delta": [0.8, 0.6, 0.5, 0.4, 0.2, -0.2, -0.4, -0.5, -0.6, -0.8],
            "gamma": [0.02, 0.04, 0.05, 0.04, 0.02, 0.02, 0.04, 0.05, 0.04, 0.02],
            "theta": [-0.01, -0.02, -0.03, -0.02, -0.01, -0.01, -0.02, -0.03, -0.02, -0.01],
            "vega": [0.1, 0.2, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.2, 0.1]
        })
        
        # Create sample price history
        self.price_history = pd.DataFrame({
            "date": pd.date_range(start=datetime.now() - timedelta(days=30), periods=30),
            "open": np.linspace(95, 100, 30),
            "high": np.linspace(97, 102, 30),
            "low": np.linspace(93, 98, 30),
            "close": np.linspace(95, 100, 30),
            "volume": np.random.randint(1000, 10000, 30)
        })
        
        # Create sample analysis results
        self.analysis_results = {
            "market_regime": {
                "primary_label": "Bullish Trend",
                "secondary_label": "High Volatility",
                "regime_score": 0.75,
                "confidence": 0.8
            },
            "energy_levels": [
                {
                    "price": 100,
                    "type": "Support",
                    "direction": "Upward",
                    "strength": 0.8,
                    "description": "Strong support level"
                },
                {
                    "price": 105,
                    "type": "Resistance",
                    "direction": "Neutral",
                    "strength": 0.6,
                    "description": "Moderate resistance level"
                }
            ],
            "reset_points": [
                {
                    "price": 95,
                    "significance": 0.7,
                    "factors": {"gammaFlip": 0.8, "vannaPeak": 0.6},
                    "description": "Potential reset point"
                }
            ],
            "greek_anomalies": [
                {
                    "type": "Gamma Spike",
                    "severity": 0.9,
                    "description": "Unusually high gamma at 100",
                    "implication": "Potential for rapid price movement"
                }
            ]
        }
        
        # Create an instance of the analyzer for instance method tests
        # Note: GreekEnergyAnalyzer doesn't take batch_size parameter
        self.analyzer = GreekEnergyAnalyzer()
        
        # Create a spot price for report generation
        self.spot_price = 100.0
    
    def test_format_results(self):
        """Test formatting of analysis results."""
        # The format_results method only takes one argument (analysis_results)
        formatted = GreekEnergyAnalyzer.format_results(self.analysis_results)
        
        # Check that the formatted results contain the expected keys
        self.assertIn("market_regime", formatted)
        self.assertIn("reset_points", formatted)
        self.assertIn("energy_levels", formatted)
        
        # Check that the market regime is formatted correctly
        self.assertEqual(formatted["market_regime"]["primary_label"], "Bullish Trend")
        self.assertEqual(formatted["market_regime"]["secondary_label"], "High Volatility")
        
        # Check that energy levels are formatted correctly
        self.assertEqual(len(formatted["energy_levels"]), 2)
        self.assertEqual(formatted["energy_levels"][0]["price"], 100)
        self.assertEqual(formatted["energy_levels"][0]["type"], "Support")
        
        # Check that reset points are formatted correctly
        self.assertEqual(len(formatted["reset_points"]), 1)
        self.assertEqual(formatted["reset_points"][0]["price"], 95)
        self.assertAlmostEqual(formatted["reset_points"][0]["significance"], 0.7)
    
    def test_analyze_chain_energy(self):
        """Test chain energy analysis."""
        chain_energy = GreekEnergyAnalyzer.analyze_chain_energy(self.options_data, "TEST")
        
        # Check that basic structure is present
        self.assertIn("symbol", chain_energy)
        self.assertEqual(chain_energy["symbol"], "TEST")
        
        # Check that energy_distribution is present (may be empty in this test)
        self.assertIn("energy_distribution", chain_energy)
    
    def test_error_handling(self):
        """Test error handling with invalid data."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = GreekEnergyAnalyzer.analyze_chain_energy(empty_df, "TEST")
        
        # Should return a result with error information or at least not crash
        self.assertIsNotNone(result)
        
        # Test with missing columns
        invalid_df = pd.DataFrame({"symbol": ["TEST"], "strike": [100]})
        result = GreekEnergyAnalyzer.analyze_chain_energy(invalid_df, "TEST")
        
        # Should return a result without crashing
        self.assertIsNotNone(result)
    
    def test_generate_full_report(self):
        """Test generation of full text report."""
        # Create sample data for report generation
        formatted_support = "100.00 (80%), 95.00 (70%)"
        formatted_resistance = "105.00 (60%), 110.00 (50%)"
        greek_zones = {
            "gamma_zones": "95-105",
            "vanna_exposure": "Moderate positive",
            "charm_decay": "Low impact"
        }
        trade_context = {
            "price_implications": "Bullish bias with strong support",
            "hedging_behavior": "Dealers likely delta-hedging at 100"
        }
        
        # Generate report
        from analyzer_visualizations.formatters import generate_full_report
        report = generate_full_report(
            formatted_support,
            formatted_resistance,
            greek_zones,
            trade_context,
            self.spot_price
        )
        
        # Check that report is a non-empty string
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)
        
        # Check that key sections are included
        self.assertIn("Greek Energy Levels Analysis", report)
        self.assertIn("RESISTANCE LEVELS", report)
        self.assertIn("SUPPORT LEVELS", report)
        self.assertIn("GREEK CONCENTRATION ZONES", report)
        self.assertIn("TRADE CONTEXT", report)
        
        # Check that specific values are included
        self.assertIn("100.00", report)
        self.assertIn("105.00", report)
        # The string "gamma_zones" is not in the report, but "GAMMA CONCENTRATION ZONES" is
        self.assertIn("GAMMA CONCENTRATION ZONES", report)
        self.assertIn("Bullish bias", report)
        
        # Save report for inspection if needed
        output_dir = "test_results"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "greek_report.txt"), "w") as f:
            f.write(report)

if __name__ == "__main__":
    unittest.main()
