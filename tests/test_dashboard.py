"""
Test script for the trade dashboard functionality.

This script tests:
1. Dashboard initialization
2. Loading and displaying recommendations
3. Trade context handling
4. UI component functionality
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import tkinter as tk
import json
from datetime import datetime

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dashboard module
try:
    from tools.trade_dashboard import IntegratedDashboard
except ImportError as e:
    print(f"Error importing dashboard: {e}")
    sys.exit(1)

class TestDashboard(unittest.TestCase):
    """Test cases for the Trade Dashboard."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create sample recommendation data
        cls.sample_recommendation = {
            "symbol": "SPY",
            "strategy_name": "Greek Energy Flow",
            "action": "BUY",
            "timestamp": datetime.now().isoformat(),
            "risk_category": "MEDIUM",
            "rr_ratio_str": "1:3",
            "roi": 15.5,
            "entry_zone": [420.50, 422.75],
            "stop_loss_percent": 2.5,
            "profit_target_percent": 7.5,
            "days_to_hold": 14,
            "greek_analysis": {
                "delta": 0.65,
                "gamma": 0.08,
                "vanna": 0.12,
                "charm": -0.03
            },
            "market_context": {
                "regime": "Bullish Trend",
                "volatility_regime": "Normal"
            },
            "TradeContext": {
                "market_regime": {
                    "primary": "Bullish Trend",
                    "volatility": "Normal",
                    "confidence": 0.85
                },
                "volatility_regime": "Normal",
                "dominant_greek": "vanna",
                "energy_state": "Accumulation"
            }
        }
        
        # Create test output directory
        cls.test_output_dir = os.path.join(os.path.dirname(__file__), "test_output")
        os.makedirs(cls.test_output_dir, exist_ok=True)
        
        # Save sample recommendation to file
        cls.recommendation_file = os.path.join(cls.test_output_dir, "test_recommendation.json")
        with open(cls.recommendation_file, 'w') as f:
            json.dump(cls.sample_recommendation, f, indent=2)
    
    def setUp(self):
        """Set up before each test."""
        # Create a root window that won't be shown
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window
        
        # Initialize dashboard with mocked root
        self.dashboard = IntegratedDashboard(self.root, base_dir=self.test_output_dir)
    
    def tearDown(self):
        """Clean up after each test."""
        self.root.destroy()
    
    def test_dashboard_initialization(self):
        """Test that dashboard initializes correctly."""
        # Check that main components were created
        self.assertIsNotNone(self.dashboard.root)
        
        # Check that the results directory was set
        self.assertEqual(self.dashboard.results_dir, os.path.join(self.test_output_dir, "results"))
        
        # Check that UI components were created
        self.assertTrue(hasattr(self.dashboard, 'main_frame'))
        self.assertTrue(hasattr(self.dashboard, 'recommendation_listbox'))
        self.assertTrue(hasattr(self.dashboard, 'detail_frame'))
        self.assertTrue(hasattr(self.dashboard, 'notebook'))
    
    @patch('tools.trade_dashboard.IntegratedDashboard.load_all_data')
    def test_initialization_loads_data(self, mock_load_all_data):
        """Test that initialization calls load_all_data."""
        # Create a new dashboard instance
        dashboard = IntegratedDashboard(self.root, base_dir=self.test_output_dir)
        
        # Check that load_all_data was called
        mock_load_all_data.assert_called_once()
    
    def test_setup_ui(self):
        """Test that UI setup creates necessary components."""
        # Check that UI components were created
        self.assertTrue(hasattr(self.dashboard, 'main_frame'))
        self.assertTrue(hasattr(self.dashboard, 'recommendation_listbox'))
        self.assertTrue(hasattr(self.dashboard, 'detail_frame'))

if __name__ == '__main__':
    unittest.main()

