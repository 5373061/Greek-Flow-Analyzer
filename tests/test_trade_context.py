"""
Tests for the trade_context module
"""

import sys
import os
import unittest

# Add the project root to the path to ensure imports work correctly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Try to locate the trade_context module
try:
    # Try direct import first
    from utils.trade_context import MarketRegime, GreekMetrics, TradeContext, create_default_context, merge_contexts
except ImportError:
    try:
        # Try with project name prefix
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.trade_context import MarketRegime, GreekMetrics, TradeContext, create_default_context, merge_contexts
    except ImportError:
        # Try relative import as last resort
        from ..utils.trade_context import MarketRegime, GreekMetrics, TradeContext, create_default_context, merge_contexts

# Define test cases as a class for better organization
class TestTradeContext(unittest.TestCase):
    def test_market_regime(self):
        """Test MarketRegime class"""
        regime = MarketRegime(primary="Bullish Trend", volatility="High", confidence=0.85)
        
        self.assertEqual(regime.primary, "Bullish Trend")
        self.assertEqual(regime.volatility, "High")
        self.assertEqual(regime.confidence, 0.85)
        
        # Test to_dict method
        regime_dict = regime.to_dict()
        self.assertEqual(regime_dict["primary"], "Bullish Trend")
        self.assertEqual(regime_dict["volatility"], "High")
        self.assertEqual(regime_dict["confidence"], 0.85)

    def test_greek_metrics(self):
        """Test GreekMetrics class"""
        metrics = GreekMetrics(delta=0.5, gamma=0.2, vanna=0.1, charm=0.05, vomma=0.3)
        
        self.assertEqual(metrics.delta, 0.5)
        self.assertEqual(metrics.gamma, 0.2)
        self.assertEqual(metrics.vanna, 0.1)
        self.assertEqual(metrics.charm, 0.05)
        self.assertEqual(metrics.vomma, 0.3)
        
        # Test to_dict method
        metrics_dict = metrics.to_dict()
        self.assertEqual(metrics_dict["delta"], 0.5)
        self.assertEqual(metrics_dict["vomma"], 0.3)

    def test_trade_context(self):
        """Test TradeContext class"""
        # Create a trade context
        context = TradeContext(
            market_regime=MarketRegime(primary="Vanna-Driven", confidence=0.7),
            volatility_regime="Low",
            dominant_greek="vanna",
            energy_state="Accumulation",
            entropy_score=0.3,
            support_levels=[100.0, 95.0],
            resistance_levels=[110.0, 115.0],
            greek_metrics=GreekMetrics(delta=0.6, gamma=0.3),
            anomalies=["Unusual vanna spike"],
            hold_time_days=5,
            confidence_score=0.8,
            ml_prediction="Bullish",
            ml_confidence=0.75
        )
        
        # Test basic properties
        self.assertEqual(context.market_regime.primary, "Vanna-Driven")
        self.assertEqual(context.volatility_regime, "Low")
        self.assertEqual(context.dominant_greek, "vanna")
        self.assertEqual(context.support_levels, [100.0, 95.0])
        self.assertEqual(context.greek_metrics.delta, 0.6)
        self.assertEqual(context.ml_prediction, "Bullish")
        
        # Test to_dict method
        context_dict = context.to_dict()
        self.assertEqual(context_dict["market_regime"]["primary"], "Vanna-Driven")
        self.assertEqual(context_dict["dominant_greek"], "vanna")
        self.assertEqual(context_dict["ml_prediction"], "Bullish")
        
        # Test to_json and from_json methods
        json_str = context.to_json()
        restored_context = TradeContext.from_json(json_str)
        self.assertEqual(restored_context.market_regime.primary, "Vanna-Driven")
        self.assertEqual(restored_context.dominant_greek, "vanna")
        self.assertEqual(restored_context.ml_prediction, "Bullish")
        
        # Test validation
        self.assertTrue(context.validate())

    def test_create_default_context(self):
        """Test create_default_context function"""
        context = create_default_context()
        
        self.assertEqual(context.market_regime.primary, "Unknown")
        self.assertEqual(context.volatility_regime, "Normal")
        self.assertEqual(context.dominant_greek, "")
        self.assertEqual(context.support_levels, [])
        self.assertEqual(context.hold_time_days, 0)
        self.assertIsNone(context.ml_prediction)

    def test_merge_contexts(self):
        """Test merge_contexts function"""
        # Create base context
        base = TradeContext(
            market_regime=MarketRegime(primary="Neutral"),
            volatility_regime="Normal",
            support_levels=[90.0, 85.0],
            resistance_levels=[110.0, 115.0]
        )
        
        # Create overlay context
        overlay = TradeContext(
            market_regime=MarketRegime(primary="Bullish"),
            support_levels=[95.0],
            hold_time_days=7
        )
        
        # Merge contexts
        merged = merge_contexts(base, overlay)
        
        # Test merged properties
        self.assertEqual(merged.market_regime.primary, "Bullish")
        self.assertEqual(merged.volatility_regime, "Normal")  # Kept from base
        self.assertEqual(merged.support_levels, [95.0])  # Replaced by overlay
        self.assertEqual(merged.resistance_levels, [110.0, 115.0])  # Kept from base
        self.assertEqual(merged.hold_time_days, 7)  # From overlay

if __name__ == "__main__":
    unittest.main()

