"""
cross_greek_patterns.py - Module for analyzing relationships between patterns in different Greeks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from ordinal_pattern_analyzer import GreekOrdinalPatternAnalyzer

class CrossGreekPatternAnalyzer:
    """
    Analyzes relationships between ordinal patterns in different Greek metrics.
    """
    
    def __init__(self, base_analyzer: GreekOrdinalPatternAnalyzer = None):
        """
        Initialize the cross-Greek pattern analyzer.
        
        Args:
            base_analyzer: GreekOrdinalPatternAnalyzer instance to use
        """
        self.logger = logging.getLogger(__name__)
        
        if base_analyzer is None:
            self.base_analyzer = GreekOrdinalPatternAnalyzer()
        else:
            self.base_analyzer = base_analyzer
            
        # Store cross-Greek relationships
        self.cross_patterns = {}
        
    def analyze_cross_greek_patterns(self, data: pd.DataFrame, forward_period: int = 5, 
                                  moneyness_filter: str = None) -> Dict[str, Dict]:
        """
        Analyze relationships between patterns in different Greeks.
        
        Args:
            data: DataFrame with Greek data
            forward_period: How many periods forward to check for pattern relationships
            moneyness_filter: Filter by moneyness category
            
        Returns:
            Dictionary of cross-Greek pattern relationships
        """
        # Extract patterns for each Greek
        all_patterns = self.base_analyzer.extract_patterns(data, moneyness_filter)
        
        greeks = list(all_patterns.keys())
        cross_patterns = {}
        
        # For each pair of Greeks
        for i, greek1 in enumerate(greeks):
            for greek2 in greeks[i+1:]:
                pair_key = f"{greek1}_{greek2}"
                cross_patterns[pair_key] = {}
                
                patterns1 = [p[1] for p in all_patterns[greek1]]  # Extract just the patterns
                patterns2 = [p[1] for p in all_patterns[greek2]]
                
                if len(patterns1) <= forward_period or len(patterns2) <= forward_period:
                    self.logger.warning(f"Insufficient data for cross-Greek analysis of {pair_key}")
                    continue
                
                # For each pattern in the first Greek
                for idx in range(len(patterns1) - forward_period):
                    if idx >= len(patterns1) or idx + forward_period >= len(patterns2):
                        continue
                        
                    pattern1 = patterns1[idx]
                    pattern2 = patterns2[idx + forward_period]
                    
                    # Record this sequence
                    pattern_pair = (pattern1, pattern2)
                    if pattern_pair in cross_patterns[pair_key]:
                        cross_patterns[pair_key][pattern_pair] += 1
                    else:
                        cross_patterns[pair_key][pattern_pair] = 1
        
        self.cross_patterns = cross_patterns
        self.logger.info(f"Analyzed {len(cross_patterns)} cross-Greek pattern relationships")
        return cross_patterns
    
    def find_predictive_relationships(self, min_occurrences: int = 3) -> Dict[str, List]:
        """
        Find the most significant predictive relationships between Greek patterns.
        
        Args:
            min_occurrences: Minimum number of times a pattern pair must occur
            
        Returns:
            Dictionary of predictive relationships
        """
        predictive_relationships = {}
        
        for pair_key, pattern_pairs in self.cross_patterns.items():
            predictive_relationships[pair_key] = []
            
            # Split the pair key to get the individual Greeks
            greek1, greek2 = pair_key.split('_')
            
            # Find significant pattern pairs
            for pattern_pair, count in pattern_pairs.items():
                if count >= min_occurrences:
                    source_pattern, target_pattern = pattern_pair
                    
                    # Add to predictive relationships
                    predictive_relationships[pair_key].append({
                        'source_pattern': source_pattern,
                        'target_pattern': target_pattern,
                        'source_description': self.base_analyzer._describe_pattern(source_pattern),
                        'target_description': self.base_analyzer._describe_pattern(target_pattern),
                        'occurrences': count
                    })
            
            # Sort by occurrence count (descending)
            predictive_relationships[pair_key].sort(key=lambda x: x['occurrences'], reverse=True)
        
        return predictive_relationships
    
    def enhance_recommendation_with_cross_patterns(self, recommendation: Dict[str, Any], 
                                               data: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhance a trade recommendation based on cross-Greek pattern relationships.
        
        Args:
            recommendation: Original trade recommendation
            data: Recent Greek data
            
        Returns:
            Enhanced trade recommendation
        """
        enhanced_rec = recommendation.copy()
        
        # Determine relevant moneyness
        moneyness = self._determine_moneyness(recommendation)
        
        # Extract current patterns
        current_patterns = self.base_analyzer.extract_patterns(data, moneyness_filter=moneyness)
        
        # Collect the most recent pattern for each Greek
        recent_patterns = {}
        for greek, pattern_list in current_patterns.items():
            if pattern_list:
                _, recent_pattern = pattern_list[-1]
                recent_patterns[greek] = recent_pattern
        
        # Check for predictive patterns
        predictive_insights = []
        
        for greek1 in recent_patterns.keys():
            for greek2 in recent_patterns.keys():
                if greek1 != greek2:
                    pair_key = f"{greek1}_{greek2}"
                    alt_pair_key = f"{greek2}_{greek1}"
                    
                    if pair_key in self.cross_patterns:
                        pattern1 = recent_patterns[greek1]
                        
                        # Find if this pattern predicts any pattern in the other Greek
                        for (source, target), count in self.cross_patterns[pair_key].items():
                            if source == pattern1 and count >= 3:
                                predictive_insights.append({
                                    'source_greek': greek1,
                                    'target_greek': greek2,
                                    'source_pattern': self.base_analyzer._describe_pattern(source),
                                    'target_pattern': self.base_analyzer._describe_pattern(target),
                                    'confidence': min(count / 10, 0.9)  # Cap at 0.9
                                })
                    
                    elif alt_pair_key in self.cross_patterns:
                        # Check the reverse direction
                        pattern2 = recent_patterns[greek2]
                        
                        for (source, target), count in self.cross_patterns[alt_pair_key].items():
                            if source == pattern2 and count >= 3:
                                predictive_insights.append({
                                    'source_greek': greek2,
                                    'target_greek': greek1,
                                    'source_pattern': self.base_analyzer._describe_pattern(source),
                                    'target_pattern': self.base_analyzer._describe_pattern(target),
                                    'confidence': min(count / 10, 0.9)  # Cap at 0.9
                                })
        
        # Add insights to recommendation
        if predictive_insights:
            enhanced_rec['cross_greek_insights'] = predictive_insights
            
            # Adjust confidence based on insights
            if 'confidence' in enhanced_rec:
                # Calculate average insight confidence
                avg_insight_confidence = sum(i['confidence'] for i in predictive_insights) / len(predictive_insights)
                
                # Combine with original confidence (70% original, 30% insights)
                original_confidence = enhanced_rec['confidence']
                enhanced_rec['confidence'] = 0.7 * original_confidence + 0.3 * avg_insight_confidence
                
                # Flag that cross-Greek patterns were used
                enhanced_rec['cross_pattern_enhanced'] = True
        
        return enhanced_rec
    
    def _determine_moneyness(self, recommendation: Dict[str, Any]) -> str:
        """
        Determine the moneyness category from a recommendation.
        
        Args:
            recommendation: Trade recommendation
            
        Returns:
            Moneyness category (ITM, ATM, OTM)
        """
        # Default to ATM if we can't determine
        if 'current_price' not in recommendation or 'option_selection' not in recommendation:
            return 'ATM'
        
        current_price = recommendation['current_price']
        
        # Get strike price
        if 'atm_strike' in recommendation['option_selection']:
            strike = recommendation['option_selection']['atm_strike']
        else:
            return 'ATM'  # Default if no strike specified
        
        # Calculate percent difference
        pct_diff = (strike - current_price) / current_price
        
        # Determine moneyness
        if pct_diff < -0.05:  # Strike is 5% below current price
            return 'ITM'
        elif pct_diff > 0.05:  # Strike is 5% above current price
            return 'OTM'
        else:
            return 'ATM'


def main():
    """
    Example usage of the CrossGreekPatternAnalyzer.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Example data
    data = pd.DataFrame({
        'norm_delta': [0.2, 0.25, 0.3, 0.28, 0.35, 0.4, 0.38, 0.42, 0.45, 0.5],
        'norm_gamma': [0.5, 0.53, 0.48, 0.45, 0.4, 0.38, 0.35, 0.3, 0.25, 0.2],
        'norm_vanna': [0.3, 0.32, 0.35, 0.4, 0.38, 0.35, 0.32, 0.29, 0.25, 0.2],
        'norm_charm': [0.1, 0.12, 0.15, 0.18, 0.22, 0.25, 0.28, 0.3, 0.28, 0.25],
        'price': [100, 102, 105, 104, 106, 109, 107, 110, 112, 115],
        'delta': [0.5, 0.52, 0.55, 0.53, 0.57, 0.6, 0.58, 0.62, 0.65, 0.7]
    })
    
    # Initialize base analyzer
    base_analyzer = GreekOrdinalPatternAnalyzer(window_size=3)
    
    # Initialize cross-Greek analyzer
    cross_analyzer = CrossGreekPatternAnalyzer(base_analyzer)
    
    # Analyze cross-Greek patterns
    cross_patterns = cross_analyzer.analyze_cross_greek_patterns(data, forward_period=2)
    
    # Find predictive relationships
    predictive = cross_analyzer.find_predictive_relationships()
    
    # Print results
    for pair, relationships in predictive.items():
        logging.info(f"\nPredictive relationships for {pair}:")
        for rel in relationships:
            logging.info(f"  {rel['source_description']} → {rel['target_description']} (occurred {rel['occurrences']} times)")
    
    # Example recommendation
    recommendation = {
        'symbol': 'AAPL',
        'current_price': 110,
        'action': 'WAIT',
        'strategy': 'MONITOR_FOR_CLEARER_SIGNALS',
        'confidence': 0.6,
        'option_selection': {
            'atm_strike': 110,
            'otm_strike': 115
        }
    }
    
    # Enhance recommendation with cross-Greek insights
    enhanced = cross_analyzer.enhance_recommendation_with_cross_patterns(recommendation, data.iloc[-3:])
    
    # Print enhanced recommendation
    logging.info(f"\nEnhanced recommendation: {enhanced}")


if __name__ == "__main__":
    main()


