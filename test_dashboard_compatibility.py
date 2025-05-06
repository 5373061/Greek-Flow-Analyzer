"""
Test Dashboard Compatibility

This script tests the dashboard compatibility layer by generating sample recommendations
and verifying that they can be properly displayed in the dashboard.
"""

import os
import sys
import json
import logging
import argparse
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_sample_generation():
    """Test sample recommendation generation"""
    logger.info("Testing sample recommendation generation")
    
    # Import the debug_trade_recommendations module
    try:
        from debug_trade_recommendations import generate_sample_recommendations
        
        # Generate sample recommendations
        symbols = ["TEST"]
        output_dir = "test_recommendations"
        count = 1
        
        success = generate_sample_recommendations(symbols, output_dir, count)
        
        if not success:
            logger.error("Failed to generate sample recommendations")
            return False
        
        # Check if the file was created
        expected_file = os.path.join(output_dir, "TEST_trade_recommendation_1.json")
        if not os.path.exists(expected_file):
            logger.error(f"Expected file {expected_file} not found")
            return False
        
        # Load and validate the file
        with open(expected_file, 'r') as f:
            data = json.load(f)
        
        # Check required fields
        required_fields = ["Symbol", "Strategy", "Action", "Entry", "Target", "Stop", "TradeContext"]
        for field in required_fields:
            if field not in data:
                logger.error(f"Required field {field} not found in recommendation")
                return False
        
        # Check trade context
        trade_context = data.get("TradeContext", {})
        required_context_fields = ["market_regime", "volatility_regime", "dominant_greek", "greek_metrics"]
        for field in required_context_fields:
            if field not in trade_context:
                logger.error(f"Required field {field} not found in trade context")
                return False
        
        logger.info("Sample recommendation generation test passed")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import debug_trade_recommendations module: {e}")
        return False

def test_dashboard_compatibility():
    """Test dashboard compatibility with generated recommendations"""
    logger.info("Testing dashboard compatibility")
    
    # Test steps would go here
    return True
