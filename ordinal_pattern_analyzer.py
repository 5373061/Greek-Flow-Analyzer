"""
ordinal_pattern_analyzer.py - Module for analyzing ordinal patterns in Greek metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import json
import os
import logging
from collections import defaultdict

class PatternMetrics:
    """
    Utility class for calculating metrics on ordinal patterns.
    """
    
    @staticmethod
    def calculate_pattern_entropy(patterns: List[Tuple]) -> float:
        """
        Calculate the entropy of a sequence of patterns.
        
        Args:
            patterns: List of ordinal patterns
            
        Returns:
            Entropy value
        """
        if not patterns:
            return 0.0
            
        # Count occurrences of each pattern
        pattern_counts = {}
        for pattern in patterns:
            if pattern in pattern_counts:
                pattern_counts[pattern] += 1
            else:
                pattern_counts[pattern] = 1
        
        # Calculate probabilities
        total_patterns = len(patterns)
        probabilities = [count / total_patterns for count in pattern_counts.values()]
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probabilities)
        return entropy
    
    @staticmethod
    def calculate_pattern_complexity(patterns: List[Tuple]) -> float:
        """
        Calculate the complexity of a sequence of patterns.
        
        Args:
            patterns: List of ordinal patterns
            
        Returns:
            Complexity value
        """
        if not patterns:
            return 0.0
            
        # Calculate entropy
        entropy = PatternMetrics.calculate_pattern_entropy(patterns)
        
        # Calculate maximum possible entropy (uniform distribution)
        unique_patterns = len(set(patterns))
        max_entropy = np.log2(unique_patterns) if unique_patterns > 0 else 0
        
        # Calculate complexity (normalized entropy)
        complexity = entropy / max_entropy if max_entropy > 0 else 0
        return complexity


class GreekOrdinalPatternAnalyzer:
    """
    Analyzes ordinal patterns in Greek metrics for options trading.
    """
    
    def __init__(self, window_size=3, min_occurrences=3, min_confidence=0.6, top_patterns=5):
        """
        Initialize the analyzer with parameters.
        
        Args:
            window_size: Size of the sliding window for pattern extraction
            min_occurrences: Minimum number of occurrences for a pattern to be considered
            min_confidence: Minimum confidence level for pattern recognition
            top_patterns: Number of top patterns to include in the library
        """
        self.window_size = window_size
        self.min_occurrences = min_occurrences
        self.min_confidence = min_confidence
        self.top_patterns = top_patterns
        
        # List of Greek metrics to analyze
        self.greeks = ['delta', 'gamma', 'theta', 'vega', 'vanna', 'charm']
        
        # Initialize pattern library
        self.pattern_library = {}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def extract_patterns(self, data: pd.DataFrame, moneyness_filter: str = None) -> Dict[str, List[Tuple]]:
        """
        Extract ordinal patterns from Greek data.
        
        Args:
            data: DataFrame with Greek data
            moneyness_filter: Filter by moneyness category
        
        Returns:
            Dictionary of patterns for each Greek
        """
        # Filter data by moneyness if specified
        if moneyness_filter:
            filtered_data = self._filter_by_moneyness(data, moneyness_filter)
            if len(filtered_data) < self.window_size:
                self.logger.warning(f"Insufficient data for pattern analysis after filtering (needed {self.window_size}, got {len(filtered_data)})")
                return {greek: [] for greek in self.greeks}
        else:
            filtered_data = data
        
        # Check if we have enough data
        if len(filtered_data) < self.window_size:
            self.logger.warning(f"Insufficient data for pattern analysis")
            return {greek: [] for greek in self.greeks}
        
        patterns = {}
        
        for greek in self.greeks:
            # Check if normalized version exists
            norm_greek = f'norm_{greek}'
            if norm_greek in filtered_data.columns and not filtered_data[norm_greek].isna().any():
                column = norm_greek
            elif greek in filtered_data.columns and not filtered_data[greek].isna().any():
                column = greek
            else:
                continue
            
            # Get values
            values = filtered_data[column].values
            
            # Extract patterns using vectorized operations
            greek_patterns = []
            for i in range(len(values) - self.window_size + 1):
                window = values[i:i+self.window_size]
                # Get ranks (argsort of argsort gives ranks)
                ranks = np.argsort(np.argsort(window))
                # Convert to tuple for hashability
                pattern = tuple(ranks)
                greek_patterns.append((i, pattern))
            
            patterns[greek] = greek_patterns
        
        return patterns
    
    def analyze_pattern_profitability(self, data: pd.DataFrame, patterns: Dict[str, List[Tuple]], 
                                   forward_periods: List[int] = [1, 3, 5]) -> Dict[str, Dict]:
        """
        Analyze the profitability of each pattern.
        
        Args:
            data: DataFrame with Greek data
            patterns: Dictionary of patterns for each Greek
            forward_periods: List of forward periods to analyze
            
        Returns:
            Dictionary with profitability analysis for each pattern
        """
        if 'price' not in data.columns:
            self.logger.warning("Price data not available for profitability analysis")
            return {}
        
        analysis = {}
        
        for greek, pattern_list in patterns.items():
            analysis[greek] = {}
            
            for idx, pattern in pattern_list:
                # Skip if we don't have enough forward data
                if idx + self.window_size + max(forward_periods) >= len(data):
                    continue
                
                # Initialize pattern analysis if not already present
                if pattern not in analysis[greek]:
                    analysis[greek][pattern] = {}
                
                # Analyze each forward period
                for period in forward_periods:
                    if period not in analysis[greek][pattern]:
                        analysis[greek][pattern][period] = {
                            'count': 0,
                            'win_rate': 0.0,
                            'avg_return': 0.0,
                            'median_return': 0.0,
                            'max_return': 0.0,
                            'min_return': 0.0,
                            'expected_value': 0.0,
                            'wins': 0,
                            'losses': 0,
                            'returns': []
                        }
                    
                    # Calculate return
                    start_price = data['price'].iloc[idx + self.window_size - 1]
                    end_price = data['price'].iloc[idx + self.window_size + period - 1]
                    
                    if start_price == 0:
                        continue
                        
                    pct_return = (end_price - start_price) / start_price
                    
                    # Update statistics
                    analysis[greek][pattern][period]['count'] += 1
                    analysis[greek][pattern][period]['returns'].append(pct_return)
                    
                    if pct_return > 0:
                        analysis[greek][pattern][period]['wins'] += 1
                    else:
                        analysis[greek][pattern][period]['losses'] += 1
            
            # Calculate aggregate statistics for each pattern and period
            for pattern in analysis[greek]:
                for period in analysis[greek][pattern]:
                    stats = analysis[greek][pattern][period]
                    returns = stats['returns']
                    
                    if returns:
                        stats['avg_return'] = sum(returns) / len(returns)
                        stats['median_return'] = np.median(returns)
                        stats['max_return'] = max(returns)
                        stats['min_return'] = min(returns)
                        stats['win_rate'] = stats['wins'] / stats['count'] if stats['count'] > 0 else 0
                        stats['expected_value'] = stats['avg_return'] * stats['win_rate']
        
        return analysis
    
    def build_pattern_library(self, analysis: Dict[str, Dict], moneyness: str, 
                           forward_period: int = 3) -> Dict[str, Dict]:
        """
        Build a pattern library for a specific moneyness category.
        
        Args:
            analysis: Pattern profitability analysis
            moneyness: Moneyness category (ITM, ATM, OTM, VOL_CRUSH)
            forward_period: Forward period to use for the library
            
        Returns:
            Pattern library for the specified moneyness
        """
        library = {}
        
        for greek, patterns in analysis.items():
            library[greek] = {}
            
            # Filter patterns by minimum occurrences
            filtered_patterns = {}
            for pattern, periods in patterns.items():
                if forward_period in periods and periods[forward_period]['count'] >= self.min_occurrences:
                    filtered_patterns[pattern] = periods[forward_period]
            
            # Sort by expected value
            sorted_patterns = sorted(
                filtered_patterns.items(),
                key=lambda x: x[1]['expected_value'],
                reverse=True
            )
            
            # Take top patterns
            for pattern, stats in sorted_patterns[:self.top_patterns]:
                library[greek][str(pattern)] = stats
        
        # Store in the pattern library
        if moneyness not in self.pattern_library:
            self.pattern_library[moneyness] = {}
        
        self.pattern_library[moneyness] = library
        return library
    
    def save_pattern_library(self, filepath: str):
        """
        Save the pattern library to a JSON file.
        
        Args:
            filepath: Path to save the library
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert tuple keys to strings for JSON serialization
        serializable_library = {}
        
        for moneyness, greeks in self.pattern_library.items():
            serializable_library[moneyness] = {}
            
            for greek, patterns in greeks.items():
                serializable_library[moneyness][greek] = {}
                
                for pattern, stats in patterns.items():
                    # Convert numpy types to Python types
                    clean_stats = {}
                    for key, value in stats.items():
                        if isinstance(value, np.float64):
                            clean_stats[key] = float(value)
                        elif isinstance(value, np.int64):
                            clean_stats[key] = int(value)
                        elif isinstance(value, list) and value and isinstance(value[0], np.float64):
                            clean_stats[key] = [float(v) for v in value]
                        else:
                            clean_stats[key] = value
                    
                    serializable_library[moneyness][greek][pattern] = clean_stats
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(serializable_library, f, indent=2)
    
    def load_pattern_library(self, filepath: str):
        """
        Load a pattern library from a JSON file.
        
        Args:
            filepath: Path to the library file
        """
        with open(filepath, 'r') as f:
            self.pattern_library = json.load(f)
    
    def recognize_current_patterns(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Recognize patterns in current market data.
        
        Args:
            data: DataFrame with current Greek data
            
        Returns:
            Dictionary of recognized patterns
        """
        recognized_patterns = {}
        
        # Check if we have enough data
        if len(data) < self.window_size:
            self.logger.warning(f"Insufficient data for pattern recognition (needed {self.window_size}, got {len(data)})")
            return recognized_patterns
        
        # Extract patterns from current data
        current_patterns = self.extract_patterns(data)
        
        # Check each moneyness category
        for moneyness, library in self.pattern_library.items():
            recognized_patterns[moneyness] = {}
            
            for greek, patterns in current_patterns.items():
                if not patterns:
                    continue
                
                # Get the most recent pattern
                _, recent_pattern = patterns[-1]
                
                # Check if this pattern is in the library
                if greek in library and str(recent_pattern) in library[greek]:
                    pattern_stats = library[greek][str(recent_pattern)]
                    
                    # Only include if confidence is high enough
                    if pattern_stats['win_rate'] >= self.min_confidence:
                        recognized_patterns[moneyness][greek] = {
                            'pattern': recent_pattern,
                            'description': self._describe_pattern(recent_pattern),
                            'win_rate': pattern_stats['win_rate'],
                            'avg_return': pattern_stats['avg_return'],
                            'expected_value': pattern_stats['expected_value'],
                            'count': pattern_stats['count']
                        }
        
        return recognized_patterns
    
    def enhance_recommendation_with_patterns(self, recommendation: Dict[str, Any], 
                                         recognized_patterns: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Enhance a trade recommendation based on recognized patterns.
        
        Args:
            recommendation: Original trade recommendation
            recognized_patterns: Dictionary of recognized patterns
            
        Returns:
            Enhanced trade recommendation
        """
        enhanced_rec = recommendation.copy()
        
        # Determine relevant moneyness
        if 'option_selection' in recommendation and 'moneyness' in recommendation['option_selection']:
            moneyness = recommendation['option_selection']['moneyness']
        else:
            # Default to ATM
            moneyness = 'ATM'
        
        # Check if we have patterns for this moneyness
        if moneyness not in recognized_patterns or not recognized_patterns[moneyness]:
            return enhanced_rec
        
        # Collect pattern insights
        pattern_insights = []
        
        for greek, pattern_info in recognized_patterns[moneyness].items():
            pattern_insights.append({
                'greek': greek,
                'pattern': pattern_info['description'],
                'win_rate': pattern_info['win_rate'],
                'avg_return': pattern_info['avg_return'],
                'expected_value': pattern_info['expected_value']
            })
        
        # Add pattern insights to recommendation
        if pattern_insights:
            enhanced_rec['pattern_insights'] = pattern_insights
            
            # Calculate average expected value
            avg_expected_value = sum(p['expected_value'] for p in pattern_insights) / len(pattern_insights)
            
            # Adjust confidence based on pattern insights
            if 'confidence' in enhanced_rec:
                # Combine with original confidence (70% original, 30% patterns)
                original_confidence = enhanced_rec['confidence']
                pattern_confidence = sum(p['win_rate'] for p in pattern_insights) / len(pattern_insights)
                enhanced_rec['confidence'] = 0.7 * original_confidence + 0.3 * pattern_confidence
            
            # Adjust action based on expected value
            if avg_expected_value > 0.02:  # Strong positive expectation
                if enhanced_rec.get('action') == 'WAIT':
                    enhanced_rec['action'] = 'BUY'
                    enhanced_rec['strategy'] = 'PATTERN_BASED_ENTRY'
            elif avg_expected_value < -0.02:  # Strong negative expectation
                if enhanced_rec.get('action') == 'BUY':
                    enhanced_rec['action'] = 'WAIT'
                    enhanced_rec['strategy'] = 'PATTERN_BASED_CAUTION'
        
        return enhanced_rec
    
    def _extract_ordinal_pattern(self, values: np.ndarray) -> Tuple:
        """
        Extract the ordinal pattern from a window of values.
        
        Args:
            values: Array of values
            
        Returns:
            Tuple representing the ordinal pattern
        """
        # Get the rank order of values
        order = np.argsort(values)
        
        # Convert to tuple for hashability
        return tuple(order)
    
    def _filter_by_moneyness(self, data: pd.DataFrame, moneyness: str) -> pd.DataFrame:
        """
        Filter data by moneyness category.
        
        Args:
            data: DataFrame with Greek data
            moneyness: Moneyness category ('ITM', 'ATM', 'OTM', or 'VOL_CRUSH')
            
        Returns:
            Filtered DataFrame
        """
        if moneyness not in ['ITM', 'ATM', 'OTM', 'VOL_CRUSH']:
            self.logger.warning(f"Invalid moneyness category: {moneyness}")
            return data
        
        # Create a copy to avoid modifying the original
        filtered = data.copy()
        
        # Apply filter based on moneyness
        if moneyness == 'ITM':
            # ITM: delta > 0.5 for calls, delta < -0.5 for puts
            if 'norm_delta' in filtered.columns:
                return filtered[filtered['norm_delta'] > 0.5]
            elif 'delta' in filtered.columns:
                return filtered[filtered['delta'] > 0.5]
        elif moneyness == 'OTM':
            # OTM: delta < 0.5 for calls, delta > -0.5 for puts
            if 'norm_delta' in filtered.columns:
                return filtered[filtered['norm_delta'] < 0.5]
            elif 'delta' in filtered.columns:
                return filtered[filtered['delta'] < 0.5]
        elif moneyness == 'ATM':
            # ATM: delta around 0.5 for calls, around -0.5 for puts
            if 'norm_delta' in filtered.columns:
                return filtered[(filtered['norm_delta'] >= 0.4) & (filtered['norm_delta'] <= 0.6)]
            elif 'delta' in filtered.columns:
                return filtered[(filtered['delta'] >= 0.4) & (filtered['delta'] <= 0.6)]
        elif moneyness == 'VOL_CRUSH':
            # VOL_CRUSH: significant decrease in vega
            if 'vega_change' in filtered.columns:
                return filtered[filtered['vega_change'] < -0.1]
            elif 'vega' in filtered.columns and 'vega_prev' in filtered.columns:
                filtered['vega_change'] = filtered['vega'] - filtered['vega_prev']
                return filtered[filtered['vega_change'] < -0.1]
        
        # Default: return all data
        return filtered
    
    def _describe_pattern(self, pattern: Tuple) -> str:
        """
        Provide a human-readable description of a pattern.
        
        Args:
            pattern: Ordinal pattern tuple
        
        Returns:
            Description string
        """
        if not pattern:
            return "Unknown"
        
        # Common patterns for window size 3
        if self.window_size == 3:
            if pattern == (0, 1, 2):
                return "Steadily increasing"
            elif pattern == (0, 2, 1):
                return "Up then down"
            elif pattern == (1, 0, 2):
                return "Down then up"
            elif pattern == (1, 2, 0):
                return "Up then down below start"
            elif pattern == (2, 0, 1):
                return "Down then up, but still below start"
            elif pattern == (2, 1, 0):
                return "Steadily decreasing"
        
        # Common patterns for window size 4
        elif self.window_size == 4:
            if pattern == (0, 1, 2, 3):
                return "Steadily increasing"
            elif pattern == (3, 2, 1, 0):
                return "Steadily decreasing"
            elif pattern == (0, 1, 3, 2):
                return "Increasing then slight pullback"
            elif pattern == (3, 2, 0, 1):
                return "Decreasing then slight bounce"
            elif pattern == (1, 0, 2, 3):
                return "Dip then strong rally"
            elif pattern == (2, 3, 1, 0):
                return "Spike then strong decline"
        
        # Generic description based on first and last values
        first_idx = pattern[0]
        last_idx = pattern[-1]
        
        if first_idx < last_idx:
            return "Generally increasing"
        elif first_idx > last_idx:
            return "Generally decreasing"
        else:
            return "Neutral/Sideways"
    
    def _calculate_pattern_confidence(self, patterns: Dict[str, Dict]) -> float:
        """
        Calculate overall confidence score from recognized patterns.
        
        Args:
            patterns: Dictionary of recognized patterns
            
        Returns:
            Confidence score between 0 and 1
        """
        if not patterns:
            return 0.0
            
        total_score = 0.0
        weights = {'delta': 0.2, 'gamma': 0.2, 'theta': 0.15, 
                  'vega': 0.15, 'charm': 0.15, 'vanna': 0.15}
        
        for greek, pattern_info in patterns.items():
            if 'stats' in pattern_info:
                stats = pattern_info['stats']
                # Weight by win rate and sample size
                weight = weights.get(greek, 0.1)
                win_rate = stats.get('win_rate', 0.5)
                count = stats.get('count', 0)
                count_factor = min(1.0, count / 20)  # Cap at 20 samples
                
                greek_score = win_rate * count_factor * weight
                total_score += greek_score
        
        # Normalize
        max_possible_score = sum(weights.values())
        normalized_score = total_score / max_possible_score if max_possible_score > 0 else 0
        
        return normalized_score

    def track_pattern_performance(self, recognized_patterns, trade_result):
        """
        Track the performance of recognized patterns to improve future predictions.
        
        Args:
            recognized_patterns: Patterns recognized for this trade
            trade_result: Actual trade result (profit/loss)
        
        Returns:
            Updated pattern statistics
        """
        if not recognized_patterns:
            return {}
        
        updated_stats = {}
        
        # Determine if trade was a win
        is_win = trade_result > 0
        
        # Update statistics for each recognized pattern
        for moneyness, greeks in recognized_patterns.items():
            if moneyness not in self.pattern_library:
                continue
            
            for greek, pattern_info in greeks.items():
                if greek not in self.pattern_library[moneyness]:
                    continue
                
                pattern = pattern_info.get('pattern')
                if pattern not in self.pattern_library[moneyness][greek]:
                    continue
                
                # Get current stats
                stats = self.pattern_library[moneyness][greek][pattern]
                
                # Update stats
                count = stats.get('count', 0) + 1
                wins = stats.get('wins', 0) + (1 if is_win else 0)
                losses = stats.get('losses', 0) + (0 if is_win else 1)
                
                # Calculate new win rate
                win_rate = wins / count if count > 0 else 0
                
                # Update returns
                returns = stats.get('returns', []) + [trade_result]
                avg_return = sum(returns) / len(returns) if returns else 0
                
                # Update stats
                self.pattern_library[moneyness][greek][pattern] = {
                    'count': count,
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'wins': wins,
                    'losses': losses,
                    'returns': returns[-20:],  # Keep last 20 returns
                    'expected_value': avg_return * win_rate
                }
                
                updated_stats[f"{moneyness}_{greek}"] = {
                    'pattern': pattern,
                    'win_rate': win_rate,
                    'count': count
                }
        
        self.logger.info(f"Updated pattern performance statistics")
        return updated_stats

    def analyze_patterns(self, symbol, greek_data):
        """
        Analyze patterns in Greek data.
        
        Args:
            symbol: Symbol being analyzed
            greek_data: DataFrame with Greek data
        
        Returns:
            Dictionary with pattern analysis results
        """
        # Extract patterns
        extracted_patterns = self.extract_patterns(greek_data)
        
        # Analyze pattern profitability
        pattern_analysis = self.analyze_pattern_profitability(greek_data, extracted_patterns)
        
        # Recognize current patterns
        current_data = greek_data.iloc[-4:].reset_index(drop=True) if len(greek_data) >= 4 else greek_data
        recognized_patterns = self.recognize_current_patterns(current_data)
        
        return {
            'extracted_patterns': extracted_patterns,
            'pattern_analysis': pattern_analysis,
            'recognized_patterns': recognized_patterns
        }

    def detect_pattern_transitions(self, data: pd.DataFrame, lookback_periods: int = 10) -> Dict[str, Any]:
        """
        Detect transitions between patterns and their significance.
        
        Args:
            data: DataFrame with Greek data
            lookback_periods: Number of periods to look back for transitions
        
        Returns:
            Dictionary with pattern transitions and significance
        """
        if len(data) < lookback_periods + self.window_size:
            self.logger.warning(f"Insufficient data for transition detection")
            return {}
        
        # Extract patterns for the lookback period
        recent_data = data.tail(lookback_periods + self.window_size - 1).reset_index(drop=True)
        patterns = self.extract_patterns(recent_data)
        
        transitions = {}
        for greek, pattern_list in patterns.items():
            if len(pattern_list) < 2:
                continue
            
            # Get sequence of patterns
            pattern_sequence = [p[1] for p in pattern_list]
            
            # Find transitions
            greek_transitions = []
            for i in range(len(pattern_sequence) - 1):
                from_pattern = pattern_sequence[i]
                to_pattern = pattern_sequence[i+1]
                
                if from_pattern != to_pattern:
                    greek_transitions.append({
                        'from': from_pattern,
                        'to': to_pattern,
                        'from_desc': self._describe_pattern(from_pattern),
                        'to_desc': self._describe_pattern(to_pattern),
                        'position': i
                    })
            
            # Calculate transition significance
            if greek_transitions:
                # Calculate entropy before and after each transition
                for t in greek_transitions:
                    pos = t['position']
                    before_entropy = PatternMetrics.calculate_pattern_entropy(pattern_sequence[max(0, pos-3):pos+1])
                    after_entropy = PatternMetrics.calculate_pattern_entropy(pattern_sequence[pos+1:min(len(pattern_sequence), pos+4)])
                    t['entropy_change'] = after_entropy - before_entropy
                    t['significance'] = abs(t['entropy_change']) / (before_entropy if before_entropy > 0 else 1)
            
            transitions[greek] = sorted(greek_transitions, key=lambda x: abs(x['significance']), reverse=True)
        
        return transitions


def main():
    """
    Example usage of the GreekOrdinalPatternAnalyzer.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Example data
    # In practice, load your actual Greek and price data here
    data = pd.DataFrame({
        'norm_delta': [0.2, 0.25, 0.3, 0.28, 0.35, 0.4, 0.38, 0.42, 0.45, 0.5],
        'norm_gamma': [0.5, 0.53, 0.48, 0.45, 0.4, 0.38, 0.35, 0.3, 0.25, 0.2],
        'norm_vanna': [0.3, 0.32, 0.35, 0.4, 0.38, 0.35, 0.32, 0.29, 0.25, 0.2],
        'norm_charm': [0.1, 0.12, 0.15, 0.18, 0.22, 0.25, 0.28, 0.3, 0.28, 0.25],
        'price': [100, 102, 105, 104, 106, 109, 107, 110, 112, 115],
        'delta': [0.5, 0.52, 0.55, 0.53, 0.57, 0.6, 0.58, 0.62, 0.65, 0.7]
    })
    
    # Initialize analyzer
    analyzer = GreekOrdinalPatternAnalyzer(window_size=4, min_occurrences=3)
    
    # Extract patterns
    patterns = analyzer.extract_patterns(data, moneyness_filter='ATM')
    
    # Analyze profitability
    analysis = analyzer.analyze_pattern_profitability(data, patterns)
    
    # Build pattern library
    library = analyzer.build_pattern_library(analysis, moneyness='ATM')
    
    # Save library
    analyzer.save_pattern_library('patterns/greek_patterns.json')
    
    # Recognize patterns in current data
    current_data = data.iloc[-4:].reset_index(drop=True)
    recognized = analyzer.recognize_current_patterns(current_data)
    
    # Example trade recommendation
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
    
    # Enhance recommendation
    enhanced = analyzer.enhance_trade_recommendation(recommendation, recognized)
    
    # Print results
    logging.info(f"Enhanced recommendation: {enhanced}")


if __name__ == "__main__":
    main()







