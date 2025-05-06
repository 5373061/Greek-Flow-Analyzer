"""
Implementation of the extract_market_regime_from_recommendations method for IntegratedDashboard.

This file contains the implementation of the missing method that needs to be added to the 
trade_dashboard.py file's IntegratedDashboard class.

Instructions:
1. Copy the method below
2. Open trade_dashboard.py
3. Paste the method inside the IntegratedDashboard class
4. Save the file and restart the dashboard
"""

def extract_market_regime_from_recommendations(self):
    """
    Extract market regime data from recommendations and create a market_regime.json file.
    """
    try:
        import logging
        logger = logging.getLogger(__name__)  # Use the dashboard's logger
        
        logger.info("Extracting market regime from recommendations...")
        
        if not self.recommendations:
            logger.warning("No recommendations available to extract market regime")
            return
        
        # Extract regimes from recommendations
        import os
        from datetime import datetime
        import json
        
        regimes = {}
        regime_counts = {"Bullish": 0, "Bearish": 0, "Neutral": 0, "Unknown": 0}
        
        for rec in self.recommendations:
            ticker = rec.get("Symbol", "")
            if not ticker:
                continue
                
            # Get regime from recommendation
            regime = rec.get("Regime", "Unknown")
            regimes[ticker] = regime
            
            # Count by regime type
            if "Bull" in regime:
                regime_counts["Bullish"] += 1
            elif "Bear" in regime:
                regime_counts["Bearish"] += 1
            elif "Neutral" in regime:
                regime_counts["Neutral"] += 1
            else:
                regime_counts["Unknown"] += 1
        
        # Determine overall bias
        overall_bias = "NEUTRAL"
        if regime_counts["Bullish"] > regime_counts["Bearish"]:
            overall_bias = "BULLISH"
        elif regime_counts["Bearish"] > regime_counts["Bullish"]:
            overall_bias = "BEARISH"
        
        # Create market regime summary
        market_regime = {
            "timestamp": datetime.now().isoformat(),
            "overall_bias": overall_bias,
            "bullish_count": regime_counts["Bullish"],
            "bearish_count": regime_counts["Bearish"],
            "neutral_count": regime_counts["Neutral"],
            "regimes": regimes
        }
        
        # Save market regime file
        regime_file = os.path.join(self.base_dir, "results", "market_regime.json")
        os.makedirs(os.path.join(self.base_dir, "results"), exist_ok=True)
        
        with open(regime_file, 'w') as f:
            json.dump(market_regime, f, indent=2)
        
        logger.info(f"Created market regime summary: {overall_bias} " +
                   f"(Bullish: {regime_counts['Bullish']}, " +
                   f"Bearish: {regime_counts['Bearish']}, " +
                   f"Neutral: {regime_counts['Neutral']})")
                   
        # Store market regime data for dashboard use
        self.market_regime = market_regime
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error extracting market regime from recommendations: {e}")
