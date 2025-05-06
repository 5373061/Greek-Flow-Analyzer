"""
trade_signals.py - Entry and exit signal generation based on ordinal patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
import logging
from ordinal_pattern_analyzer import GreekOrdinalPatternAnalyzer
from cross_greek_patterns import CrossGreekPatternAnalyzer

class SignalType(Enum):
    ENTRY_LONG = "ENTRY_LONG"
    ENTRY_SHORT = "ENTRY_SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    HOLD = "HOLD"
    NO_SIGNAL = "NO_SIGNAL"

class TradeSignalGenerator:
    """
    Generates entry and exit signals based on ordinal patterns in Greek metrics.
    """
    
    def __init__(self, pattern_analyzer: GreekOrdinalPatternAnalyzer, 
                 cross_analyzer: Optional[CrossGreekPatternAnalyzer] = None):
        """
        Initialize the trade signal generator.
        
        Args:
            pattern_analyzer: GreekOrdinalPatternAnalyzer instance
            cross_analyzer: Optional CrossGreekPatternAnalyzer for cross-Greek signals
        """
        self.logger = logging.getLogger(__name__)
        self.pattern_analyzer = pattern_analyzer
        self.cross_analyzer = cross_analyzer
        
        # Signal configuration
        self.entry_confidence_threshold = 0.7
        self.exit_confidence_threshold = 0.6
        self.hold_exit_threshold = 0.5
        
        # Pattern significance requirements
        self.min_pattern_occurrences = 5
        self.min_win_rate = 0.6
        
    def generate_signals(self, current_data: pd.DataFrame, 
                        recognized_patterns: Dict[str, Dict],
                        current_position: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Generate trading signals based on recognized patterns.
        
        Args:
            current_data: Recent Greek metric data
            recognized_patterns: Patterns recognized in current data
            current_position: Current trading position, if any
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Generate entry signals
        if current_position is None or not self._has_active_position(current_position):
            entry_signals = self._generate_entry_signals(current_data, recognized_patterns)
            signals.extend(entry_signals)
        else:
            # Generate exit signals if we have an active position
            exit_signals = self._generate_exit_signals(
                current_data, recognized_patterns, current_position
            )
            signals.extend(exit_signals)
            
            # Check for position adjustment signals
            adjustment_signals = self._generate_adjustment_signals(
                current_data, recognized_patterns, current_position
            )
            signals.extend(adjustment_signals)
        
        # Add cross-Greek signals if available
        if self.cross_analyzer:
            cross_signals = self._generate_cross_greek_signals(
                current_data, recognized_patterns, current_position
            )
            signals.extend(cross_signals)
        
        # Sort signals by confidence and return
        signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return signals
    
    def _generate_entry_signals(self, current_data: pd.DataFrame, 
                              recognized_patterns: Dict[str, Dict]) -> List[Dict]:
        """
        Generate entry signals based on recognized patterns.
        
        Args:
            current_data: Recent Greek metric data
            recognized_patterns: Patterns recognized in current data
            
        Returns:
            List of entry signals
        """
        entry_signals = []
        
        for moneyness, greek_patterns in recognized_patterns.items():
            for greek, pattern_info in greek_patterns.items():
                stats = pattern_info.get('stats', {})
                
                if (stats.get('count', 0) >= self.min_pattern_occurrences and
                    stats.get('win_rate', 0) >= self.min_win_rate):
                    
                    # Determine signal direction based on Greek metric
                    if self._is_bullish_pattern(greek, pattern_info, current_data):
                        signal_type = SignalType.ENTRY_LONG
                    elif self._is_bearish_pattern(greek, pattern_info, current_data):
                        signal_type = SignalType.ENTRY_SHORT
                    else:
                        continue
                    
                    # Calculate signal confidence
                    confidence = self._calculate_signal_confidence(stats, pattern_info)
                    
                    if confidence >= self.entry_confidence_threshold:
                        entry_signals.append({
                            'signal_type': signal_type,
                            'confidence': confidence,
                            'target_strike': self._determine_target_strike(signal_type, moneyness, current_data),
                            'strategy': self._recommend_strategy(signal_type, moneyness, confidence),
                            'expiration': self._determine_expiration(pattern_info, current_data),
                            'take_profit': self._calculate_take_profit(signal_type, stats, current_data),
                            'stop_loss': self._calculate_stop_loss(signal_type, stats, current_data),
                            'pattern_source': {
                                'greek': greek,
                                'moneyness': moneyness,
                                'pattern': pattern_info.get('pattern'),
                                'description': pattern_info.get('description')
                            }
                        })
        
        return entry_signals
    
    def _generate_exit_signals(self, current_data: pd.DataFrame, 
                             recognized_patterns: Dict[str, Dict],
                             current_position: Dict[str, Any]) -> List[Dict]:
        """
        Generate exit signals based on recognized patterns and current position.
        
        Args:
            current_data: Recent Greek metric data
            recognized_patterns: Patterns recognized in current data
            current_position: Current trading position
            
        Returns:
            List of exit signals
        """
        exit_signals = []
        
        position_type = current_position.get('type', 'long')
        entry_moneyness = current_position.get('moneyness', 'ATM')
        
        for moneyness, greek_patterns in recognized_patterns.items():
            for greek, pattern_info in greek_patterns.items():
                stats = pattern_info.get('stats', {})
                
                # Look for exit patterns based on position type
                if self._is_exit_pattern(greek, pattern_info, position_type, current_data):
                    confidence = self._calculate_exit_confidence(
                        stats, pattern_info, current_position, current_data
                    )
                    
                    if confidence >= self.exit_confidence_threshold:
                        signal_type = (SignalType.EXIT_LONG if position_type == 'long' 
                                     else SignalType.EXIT_SHORT)
                        
                        exit_signals.append({
                            'signal_type': signal_type,
                            'confidence': confidence,
                            'exit_reason': self._determine_exit_reason(pattern_info, current_position),
                            'target_price': self._calculate_exit_price(
                                current_position, pattern_info, current_data
                            ),
                            'partial_exit': self._recommend_partial_exit(
                                confidence, current_position, stats
                            ),
                            'pattern_source': {
                                'greek': greek,
                                'moneyness': moneyness,
                                'pattern': pattern_info.get('pattern'),
                                'description': pattern_info.get('description')
                            }
                        })
        
        return exit_signals
    
    def _generate_adjustment_signals(self, current_data: pd.DataFrame,
                                   recognized_patterns: Dict[str, Dict],
                                   current_position: Dict[str, Any]) -> List[Dict]:
        """
        Generate position adjustment signals (e.g., roll options, adjust strikes).
        
        Args:
            current_data: Recent Greek metric data
            recognized_patterns: Patterns recognized in current data
            current_position: Current trading position
            
        Returns:
            List of adjustment signals
        """
        adjustment_signals = []
        
        # Check if position needs rolling (close to expiration)
        if self._needs_rolling(current_position, current_data):
            roll_signal = self._generate_roll_signal(
                current_position, recognized_patterns, current_data
            )
            if roll_signal:
                adjustment_signals.append(roll_signal)
        
        # Check if strikes need adjustment based on patterns
        strike_adjustment = self._check_strike_adjustment(
            current_position, recognized_patterns, current_data
        )
        if strike_adjustment:
            adjustment_signals.append(strike_adjustment)
        
        return adjustment_signals
    
    def _generate_cross_greek_signals(self, current_data: pd.DataFrame,
                                    recognized_patterns: Dict[str, Dict],
                                    current_position: Optional[Dict[str, Any]]) -> List[Dict]:
        """
        Generate signals based on cross-Greek pattern analysis.
        
        Args:
            current_data: Recent Greek metric data
            recognized_patterns: Patterns recognized in current data
            current_position: Current trading position, if any
            
        Returns:
            List of cross-Greek signals
        """
        cross_signals = []
        
        if not self.cross_analyzer or self.cross_analyzer.cross_patterns is None:
            return cross_signals
        
        # Analyze current cross-Greek patterns
        predictive_relationships = self.cross_analyzer.find_predictive_relationships()
        
        # Generate signals based on cross-Greek relationships
        for pair_key, relationships in predictive_relationships.items():
            for rel in relationships[:3]:  # Check top 3 relationships
                if rel['occurrences'] >= 3:  # Minimum requirement for cross signal
                    signal = self._create_cross_greek_signal(
                        rel, current_data, current_position
                    )
                    if signal:
                        cross_signals.append(signal)
        
        return cross_signals
    
    def _is_bullish_pattern(self, greek: str, pattern_info: Dict, current_data: pd.DataFrame) -> bool:
        """
        Determine if a pattern indicates bullish sentiment.
        
        Args:
            greek: Greek metric name
            pattern_info: Pattern information
            current_data: Recent Greek metric data
            
        Returns:
            True if pattern is bullish
        """
        pattern = pattern_info.get('pattern')
        stats = pattern_info.get('stats', {})
        avg_return = stats.get('avg_return', 0)
        
        # Rising delta/gamma often indicate bullish momentum
        if greek in ['delta', 'gamma'] and self._is_rising_pattern(pattern) and avg_return > 0:
            return True
        
        # Rising vanna during low volatility can be bullish
        if greek == 'vanna' and self._is_rising_pattern(pattern) and self._is_low_volatility(current_data):
            return True
        
        # Specific pattern combinations
        if self._check_bullish_pattern_combination(greek, pattern, current_data):
            return True
        
        return False
    
    def _is_bearish_pattern(self, greek: str, pattern_info: Dict, current_data: pd.DataFrame) -> bool:
        """
        Determine if a pattern indicates bearish sentiment.
        
        Args:
            greek: Greek metric name
            pattern_info: Pattern information
            current_data: Recent Greek metric data
            
        Returns:
            True if pattern is bearish
        """
        pattern = pattern_info.get('pattern')
        stats = pattern_info.get('stats', {})
        avg_return = stats.get('avg_return', 0)
        
        # Falling delta/gamma often indicate bearish momentum
        if greek in ['delta', 'gamma'] and self._is_falling_pattern(pattern) and avg_return < 0:
            return True
        
        # Strong negative theta with rising gamma (gamma scalping opportunity)
        if greek == 'theta' and avg_return < -0.02 and self._has_rising_gamma(current_data):
            return True  # For short volatility trades
        
        # Specific pattern combinations
        if self._check_bearish_pattern_combination(greek, pattern, current_data):
            return True
        
        return False
    
    def _determine_target_strike(self, signal_type: SignalType, moneyness: str, 
                               current_data: pd.DataFrame) -> float:
        """
        Determine target strike price based on signal type and moneyness.
        
        Args:
            signal_type: Type of trading signal
            moneyness: Moneyness category
            current_data: Recent Greek metric data
            
        Returns:
            Target strike price
        """
        current_price = current_data['price'].iloc[-1]
        
        if signal_type == SignalType.ENTRY_LONG:
            if moneyness == 'OTM':
                return round(current_price * 1.02 / 5) * 5  # 2% OTM calls
            elif moneyness == 'ATM':
                return round(current_price / 5) * 5  # ATM calls
            else:  # ITM
                return round(current_price * 0.98 / 5) * 5  # 2% ITM calls
        
        elif signal_type == SignalType.ENTRY_SHORT:
            if moneyness == 'OTM':
                return round(current_price * 0.98 / 5) * 5  # 2% OTM puts
            elif moneyness == 'ATM':
                return round(current_price / 5) * 5  # ATM puts
            else:  # ITM
                return round(current_price * 1.02 / 5) * 5  # 2% ITM puts
        
        return current_price
    
    def _recommend_strategy(self, signal_type: SignalType, moneyness: str, 
                          confidence: float) -> str:
        """
        Recommend an options strategy based on signal and confidence.
        
        Args:
            signal_type: Type of trading signal
            moneyness: Moneyness category
            confidence: Signal confidence level
            
        Returns:
            Recommended strategy name
        """
        if confidence >= 0.8:
            # High confidence - directional strategies
            if signal_type == SignalType.ENTRY_LONG:
                return "LONG_CALL" if moneyness == "OTM" else "BULL_CALL_SPREAD"
            else:
                return "LONG_PUT" if moneyness == "OTM" else "BEAR_PUT_SPREAD"
        
        elif confidence >= 0.7:
            # Medium-high confidence - spread strategies
            if signal_type == SignalType.ENTRY_LONG:
                return "BULL_CALL_SPREAD"
            else:
                return "BEAR_PUT_SPREAD"
        
        else:
            # Lower confidence - neutral strategies
            return "IRON_CONDOR" if moneyness == "ATM" else "BUTTERFLY"
    
    def _determine_expiration(self, pattern_info: Dict, current_data: pd.DataFrame) -> int:
        """
        Determine optimal expiration based on pattern characteristics.
        
        Args:
            pattern_info: Pattern information
            current_data: Recent Greek metric data
            
        Returns:
            Days to expiration
        """
        stats = pattern_info.get('stats', {})
        avg_return = stats.get('avg_return', 0)
        
        # Quick patterns suggest shorter expiration
        if abs(avg_return) > 0.05:  # 5% average return
            return 30  # Monthly options
        elif abs(avg_return) > 0.03:  # 3% average return
            return 45  # Quarterly options
        else:
            return 60  # Further out options for more time decay management
    
    def _calculate_take_profit(self, signal_type: SignalType, stats: Dict, 
                             current_data: pd.DataFrame) -> float:
        """
        Calculate take profit level based on pattern statistics.
        
        Args:
            signal_type: Type of trading signal
            stats: Pattern statistics
            current_data: Recent Greek metric data
            
        Returns:
            Take profit percentage
        """
        avg_return = stats.get('avg_return', 0)
        max_return = stats.get('max_return', 0)
        
        # Conservative take profit at 70% of maximum historical return
        take_profit = max_return * 0.7 if max_return > 0 else avg_return * 2
        
        # Ensure sensible bounds
        return min(max(take_profit, 0.05), 0.3)  # Between 5% and 30%
    
    def _calculate_stop_loss(self, signal_type: SignalType, stats: Dict, 
                           current_data: pd.DataFrame) -> float:
        """
        Calculate stop loss level based on pattern statistics.
        
        Args:
            signal_type: Type of trading signal
            stats: Pattern statistics
            current_data: Recent Greek metric data
            
        Returns:
            Stop loss percentage
        """
        avg_return = stats.get('avg_return', 0)
        min_return = stats.get('min_return', 0)
        
        # Conservative stop loss at 50% of maximum historical loss
        stop_loss = min_return * 0.5 if min_return < 0 else avg_return * -0.5
        
        # Ensure sensible bounds
        return max(min(stop_loss, -0.05), -0.2)  # Between -5% and -20%
    
    def _is_rising_pattern(self, pattern: Tuple[int, ...]) -> bool:
        """Check if the pattern is generally rising."""
        if not pattern:
            return False
        return sum(pattern[i+1] > pattern[i] for i in range(len(pattern)-1)) > len(pattern)/2
    
    def _is_falling_pattern(self, pattern: Tuple[int, ...]) -> bool:
        """Check if the pattern is generally falling."""
        if not pattern:
            return False
        return sum(pattern[i+1] < pattern[i] for i in range(len(pattern)-1)) > len(pattern)/2
    
    def _is_low_volatility(self, current_data: pd.DataFrame) -> bool:
        """Check if current volatility is relatively low."""
        if 'norm_vega' in current_data.columns:
            return current_data['norm_vega'].iloc[-1] < 0.3
        return False
    
    def _has_rising_gamma(self, current_data: pd.DataFrame) -> bool:
        """Check if gamma is currently rising."""
        if 'norm_gamma' in current_data.columns and len(current_data) > 1:
            return current_data['norm_gamma'].iloc[-1] > current_data['norm_gamma'].iloc[-2]
        return False
    
    def _check_bullish_pattern_combination(self, greek: str, pattern: Tuple, 
                                         current_data: pd.DataFrame) -> bool:
        """Check for specific bullish pattern combinations."""
        # Can be extended with domain-specific knowledge
        return False
    
    def _check_bearish_pattern_combination(self, greek: str, pattern: Tuple, 
                                         current_data: pd.DataFrame) -> bool:
        """Check for specific bearish pattern combinations."""
        # Can be extended with domain-specific knowledge
        return False
    
    def _calculate_signal_confidence(self, stats: Dict, pattern_info: Dict) -> float:
        """Calculate overall confidence for a signal."""
        win_rate = stats.get('win_rate', 0)
        count = stats.get('count', 0)
        avg_return = stats.get('avg_return', 0)
        
        # Base confidence on win rate
        base_confidence = win_rate
        
        # Adjust for sample size
        sample_factor = min(count / 20, 1.0)  # Max factor at 20+ samples
        
        # Adjust for return magnitude
        return_factor = min(abs(avg_return) / 0.1, 1.0)  # Max factor at 10% avg return
        
        return base_confidence * 0.5 + sample_factor * 0.25 + return_factor * 0.25
    
    def _has_active_position(self, current_position: Dict[str, Any]) -> bool:
        """Check if there's an active trading position."""
        return (current_position is not None and 
                current_position.get('status') == 'active' and
                current_position.get('expiration') > current_position.get('current_date'))
    
    def _needs_rolling(self, current_position: Dict[str, Any], current_data: pd.DataFrame) -> bool:
        """Check if position needs to be rolled."""
        if not current_position:
            return False
        
        expiration = current_position.get('expiration')
        if not expiration:
            return False
        
        # Roll at 7 days to expiration
        return (expiration - pd.Timestamp.now()).days <= 7
    
    def _generate_roll_signal(self, current_position: Dict[str, Any],
                            recognized_patterns: Dict[str, Dict],
                            current_data: pd.DataFrame) -> Optional[Dict]:
        """Generate a roll signal for an expiring position."""
        # Implementation depends on specific roll strategy
        return None
    
    def _check_strike_adjustment(self, current_position: Dict[str, Any],
                               recognized_patterns: Dict[str, Dict],
                               current_data: pd.DataFrame) -> Optional[Dict]:
        """Check if strike prices need adjustment."""
        # Implementation depends on specific adjustment criteria
        return None


def main():
    """
    Example usage of the TradeSignalGenerator.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Example data
    data = pd.DataFrame({
        'norm_delta': [0.2, 0.25, 0.3, 0.35, 0.4],
        'norm_gamma': [0.5, 0.5, 0.48, 0.45, 0.4],
        'norm_vanna': [0.3, 0.32, 0.35, 0.4, 0.38],
        'norm_theta': [-0.1, -0.12, -0.15, -0.18, -0.22],
        'price': [100, 102, 105, 107, 110]
    })
    
    # Set up analyzers
    pattern_analyzer = GreekOrdinalPatternAnalyzer(window_size=3)
    cross_analyzer = CrossGreekPatternAnalyzer(pattern_analyzer)
    
    # Build pattern library
    patterns = pattern_analyzer.extract_patterns(data)
    analysis = pattern_analyzer.analyze_pattern_profitability(data, patterns)
    pattern_analyzer.build_pattern_library(analysis, 'ATM')
    
    # Recognize current patterns
    recognized = pattern_analyzer.recognize_current_patterns(data.iloc[-3:])
    
    # Generate trade signals
    signal_generator = TradeSignalGenerator(pattern_analyzer, cross_analyzer)
    signals = signal_generator.generate_signals(data.iloc[-3:], recognized)
    
    # Print signals
    for signal in signals:
        logging.info(f"Generated Signal:")
        logging.info(f"  Type: {signal.get('signal_type')}")
        logging.info(f"  Confidence: {signal.get('confidence', 0):.2f}")
        logging.info(f"  Strategy: {signal.get('strategy')}")
        logging.info(f"  Target Strike: {signal.get('target_strike')}")
        logging.info(f"  Take Profit: {signal.get('take_profit', 0):.2%}")
        logging.info(f"  Stop Loss: {signal.get('stop_loss', 0):.2%}")
        logging.info("")


if __name__ == "__main__":
    main()