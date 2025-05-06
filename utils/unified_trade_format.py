"""
Unified Trade Format Module

This module provides functions for creating and managing standardized trade recommendations
with the new trade context structure.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from utils.trade_context import TradeContext, create_default_context

logger = logging.getLogger(__name__)

class TradeRecommendation:
    """Standardized trade recommendation class"""
    
    def __init__(self, 
                 symbol: str,
                 strategy: str,
                 direction: str,
                 entry_price: float,
                 target_price: float,
                 stop_loss: float,
                 risk_reward: float,
                 confidence: float,
                 trade_context: Optional[TradeContext] = None,
                 expiration: Optional[str] = None,
                 strike: Optional[float] = None,
                 option_type: Optional[str] = None,
                 position_size: Optional[float] = None,
                 notes: Optional[str] = None):
        """
        Initialize a trade recommendation
        
        Args:
            symbol: Ticker symbol
            strategy: Strategy type (e.g., "Greek Flow", "ML Enhanced")
            direction: Trade direction ("Long" or "Short")
            entry_price: Recommended entry price
            target_price: Price target
            stop_loss: Stop loss price
            risk_reward: Risk/reward ratio
            confidence: Confidence score (0.0-1.0)
            trade_context: TradeContext object with market context
            expiration: Option expiration date (if applicable)
            strike: Option strike price (if applicable)
            option_type: Option type ("Call" or "Put", if applicable)
            position_size: Recommended position size
            notes: Additional notes
        """
        self.symbol = symbol
        self.strategy = strategy
        self.direction = direction
        self.entry_price = entry_price
        self.target_price = target_price
        self.stop_loss = stop_loss
        self.risk_reward = risk_reward
        self.confidence = confidence
        self.trade_context = trade_context or create_default_context()
        self.expiration = expiration
        self.strike = strike
        self.option_type = option_type
        self.position_size = position_size
        self.notes = notes
        self.timestamp = datetime.now().isoformat()
        self.id = f"{symbol}_{strategy}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "id": self.id,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "risk_reward": self.risk_reward,
            "confidence": self.confidence,
            "trade_context": self.trade_context.to_dict(),
            "timestamp": self.timestamp
        }
        
        # Add optional fields if present
        if self.expiration:
            result["expiration"] = self.expiration
        if self.strike:
            result["strike"] = self.strike
        if self.option_type:
            result["option_type"] = self.option_type
        if self.position_size:
            result["position_size"] = self.position_size
        if self.notes:
            result["notes"] = self.notes
            
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeRecommendation':
        """Create TradeRecommendation from dictionary"""
        # Extract trade context
        trade_context_data = data.get("trade_context", {})
        trade_context = TradeContext.from_dict(trade_context_data)
        
        # Create recommendation
        recommendation = cls(
            symbol=data.get("symbol", ""),
            strategy=data.get("strategy", ""),
            direction=data.get("direction", ""),
            entry_price=data.get("entry_price", 0.0),
            target_price=data.get("target_price", 0.0),
            stop_loss=data.get("stop_loss", 0.0),
            risk_reward=data.get("risk_reward", 0.0),
            confidence=data.get("confidence", 0.0),
            trade_context=trade_context,
            expiration=data.get("expiration"),
            strike=data.get("strike"),
            option_type=data.get("option_type"),
            position_size=data.get("position_size"),
            notes=data.get("notes")
        )
        
        # Set timestamp and ID if present
        if "timestamp" in data:
            recommendation.timestamp = data["timestamp"]
        if "id" in data:
            recommendation.id = data["id"]
            
        return recommendation
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TradeRecommendation':
        """Create TradeRecommendation from JSON string"""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            # Return a minimal recommendation in case of error
            return cls(
                symbol="ERROR",
                strategy="Unknown",
                direction="Unknown",
                entry_price=0.0,
                target_price=0.0,
                stop_loss=0.0,
                risk_reward=0.0,
                confidence=0.0
            )
    
    def save_to_file(self, directory: str) -> str:
        """
        Save recommendation to a JSON file
        
        Args:
            directory: Directory to save the file
            
        Returns:
            Path to the saved file
        """
        os.makedirs(directory, exist_ok=True)
        filename = f"{self.symbol}_{self.strategy}_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'w') as f:
            f.write(self.to_json())
            
        return filepath
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'TradeRecommendation':
        """
        Load recommendation from a JSON file
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            TradeRecommendation object
        """
        try:
            with open(filepath, 'r') as f:
                json_str = f.read()
            return cls.from_json(json_str)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load recommendation from {filepath}: {e}")
            # Return a minimal recommendation in case of error
            return cls(
                symbol="ERROR",
                strategy="Unknown",
                direction="Unknown",
                entry_price=0.0,
                target_price=0.0,
                stop_loss=0.0,
                risk_reward=0.0,
                confidence=0.0
            )

def load_recommendations(directory: str) -> List[TradeRecommendation]:
    """
    Load all recommendations from a directory
    
    Args:
        directory: Directory containing recommendation JSON files
        
    Returns:
        List of TradeRecommendation objects
    """
    recommendations = []
    
    if not os.path.exists(directory):
        logger.warning(f"Directory {directory} does not exist")
        return recommendations
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            recommendation = TradeRecommendation.load_from_file(filepath)
            recommendations.append(recommendation)
    
    return recommendations

def convert_legacy_recommendation(legacy_data: Dict[str, Any]) -> TradeRecommendation:
    """
    Convert legacy recommendation format to new format
    
    Args:
        legacy_data: Dictionary with legacy recommendation data
        
    Returns:
        TradeRecommendation object
    """
    # Extract basic fields
    symbol = legacy_data.get("symbol", "")
    strategy = legacy_data.get("strategy", "Greek Flow")
    direction = legacy_data.get("direction", "")
    entry_price = legacy_data.get("entry_price", 0.0)
    target_price = legacy_data.get("target_price", 0.0)
    stop_loss = legacy_data.get("stop_loss", 0.0)
    risk_reward = legacy_data.get("risk_reward", 0.0)
    confidence = legacy_data.get("confidence", 0.0)
    
    # Extract option-specific fields
    expiration = legacy_data.get("expiration")
    strike = legacy_data.get("strike")
    option_type = legacy_data.get("option_type")
    
    # Extract or create trade context
    trade_context = create_default_context()
    
    # Try to extract market regime
    if "market_regime" in legacy_data:
        if isinstance(legacy_data["market_regime"], dict):
            trade_context.market_regime.primary = legacy_data["market_regime"].get("primary", "Unknown")
            trade_context.market_regime.volatility = legacy_data["market_regime"].get("volatility", "Normal")
        else:
            trade_context.market_regime.primary = str(legacy_data["market_regime"])
    
    # Try to extract other context fields
    trade_context.volatility_regime = legacy_data.get("volatility_regime", "Normal")
    trade_context.dominant_greek = legacy_data.get("dominant_greek", "")
    trade_context.energy_state = legacy_data.get("energy_state", "")
    trade_context.entropy_score = legacy_data.get("entropy_score", 0.0)
    trade_context.hold_time_days = legacy_data.get("hold_time_days", 0)
    
    # Try to extract support/resistance levels
    if "support_levels" in legacy_data:
        trade_context.support_levels = legacy_data["support_levels"]
    if "resistance_levels" in legacy_data:
        trade_context.resistance_levels = legacy_data["resistance_levels"]
    
    # Try to extract Greek metrics
    if "greek_metrics" in legacy_data and isinstance(legacy_data["greek_metrics"], dict):
        greek_data = legacy_data["greek_metrics"]
        trade_context.greek_metrics.delta = greek_data.get("delta", 0.0)
        trade_context.greek_metrics.gamma = greek_data.get("gamma", 0.0)
        trade_context.greek_metrics.vanna = greek_data.get("vanna", 0.0)
        trade_context.greek_metrics.charm = greek_data.get("charm", 0.0)
        trade_context.greek_metrics.vomma = greek_data.get("vomma", 0.0)
    
    # Try to extract anomalies
    if "anomalies" in legacy_data:
        trade_context.anomalies = legacy_data["anomalies"]
    
    # Create recommendation
    return TradeRecommendation(
        symbol=symbol,
        strategy=strategy,
        direction=direction,
        entry_price=entry_price,
        target_price=target_price,
        stop_loss=stop_loss,
        risk_reward=risk_reward,
        confidence=confidence,
        trade_context=trade_context,
        expiration=expiration,
        strike=strike,
        option_type=option_type,
        notes=legacy_data.get("notes")
    )