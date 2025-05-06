"""
Trade Context Module

This module defines the standard structure for trade context information
used throughout the system.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    """Market regime information"""
    primary: str  # e.g., "Bullish Trend", "Vanna-Driven"
    volatility: str = "Normal"  # "High", "Normal", "Low"
    confidence: float = 0.0  # Confidence score (0.0-1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class GreekMetrics:
    """Key Greek metrics"""
    delta: float = 0.0
    gamma: float = 0.0
    vanna: float = 0.0
    charm: float = 0.0
    vomma: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class TradeContext:
    """Standardized trade context structure"""
    # Market information
    market_regime: MarketRegime
    volatility_regime: str = "Normal"  # "High", "Normal", "Low"
    dominant_greek: str = ""  # e.g., "vanna", "charm"
    energy_state: str = ""  # e.g., "Accumulation", "Distribution"
    entropy_score: float = 0.0  # 0.0-1.0
    
    # Price levels
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    
    # Greek metrics
    greek_metrics: GreekMetrics = field(default_factory=GreekMetrics)
    
    # Additional information
    anomalies: List[str] = field(default_factory=list)
    hold_time_days: int = 0
    confidence_score: float = 0.0  # 0.0-1.0
    
    # ML and pattern information (optional)
    ml_prediction: Optional[str] = None
    ml_confidence: float = 0.0
    pattern_name: Optional[str] = None
    pattern_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "market_regime": self.market_regime.to_dict(),
            "volatility_regime": self.volatility_regime,
            "dominant_greek": self.dominant_greek,
            "energy_state": self.energy_state,
            "entropy_score": self.entropy_score,
            "support_levels": self.support_levels,
            "resistance_levels": self.resistance_levels,
            "greek_metrics": self.greek_metrics.to_dict(),
            "anomalies": self.anomalies,
            "hold_time_days": self.hold_time_days,
            "confidence_score": self.confidence_score
        }
        
        # Add optional fields if present
        if self.ml_prediction:
            result["ml_prediction"] = self.ml_prediction
            result["ml_confidence"] = self.ml_confidence
            
        if self.pattern_name:
            result["pattern_name"] = self.pattern_name
            result["pattern_confidence"] = self.pattern_confidence
            
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeContext':
        """Create TradeContext from dictionary"""
        # Extract and create MarketRegime
        market_regime_data = data.get("market_regime", {})
        market_regime = MarketRegime(
            primary=market_regime_data.get("primary", "Unknown"),
            volatility=market_regime_data.get("volatility", "Normal"),
            confidence=market_regime_data.get("confidence", 0.0)
        )
        
        # Extract and create GreekMetrics
        greek_metrics_data = data.get("greek_metrics", {})
        greek_metrics = GreekMetrics(
            delta=greek_metrics_data.get("delta", 0.0),
            gamma=greek_metrics_data.get("gamma", 0.0),
            vanna=greek_metrics_data.get("vanna", 0.0),
            charm=greek_metrics_data.get("charm", 0.0),
            vomma=greek_metrics_data.get("vomma", 0.0)
        )
        
        # Create TradeContext
        context = cls(
            market_regime=market_regime,
            volatility_regime=data.get("volatility_regime", "Normal"),
            dominant_greek=data.get("dominant_greek", ""),
            energy_state=data.get("energy_state", ""),
            entropy_score=data.get("entropy_score", 0.0),
            support_levels=data.get("support_levels", []),
            resistance_levels=data.get("resistance_levels", []),
            greek_metrics=greek_metrics,
            anomalies=data.get("anomalies", []),
            hold_time_days=data.get("hold_time_days", 0),
            confidence_score=data.get("confidence_score", 0.0)
        )
        
        # Add optional fields if present
        if "ml_prediction" in data:
            context.ml_prediction = data["ml_prediction"]
            context.ml_confidence = data.get("ml_confidence", 0.0)
            
        if "pattern_name" in data:
            context.pattern_name = data["pattern_name"]
            context.pattern_confidence = data.get("pattern_confidence", 0.0)
            
        return context
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TradeContext':
        """Create TradeContext from JSON string"""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            # Return a default context in case of error
            return cls(market_regime=MarketRegime(primary="Unknown"))
    
    def validate(self) -> bool:
        """Validate the trade context data"""
        # Basic validation
        if not self.market_regime.primary:
            logger.warning("Missing primary market regime")
            return False
            
        # Validate numeric ranges
        if not (0 <= self.entropy_score <= 1):
            logger.warning(f"Invalid entropy score: {self.entropy_score}")
            return False
            
        if not (0 <= self.confidence_score <= 1):
            logger.warning(f"Invalid confidence score: {self.confidence_score}")
            return False
            
        # Validate hold time
        if self.hold_time_days < 0:
            logger.warning(f"Invalid hold time: {self.hold_time_days}")
            return False
            
        return True

def create_default_context() -> TradeContext:
    """Create a default trade context with placeholder values"""
    return TradeContext(
        market_regime=MarketRegime(primary="Unknown"),
        volatility_regime="Normal",
        dominant_greek="",
        energy_state="",
        entropy_score=0.0,
        support_levels=[],
        resistance_levels=[],
        greek_metrics=GreekMetrics(),
        anomalies=[],
        hold_time_days=0,
        confidence_score=0.0
    )

def merge_contexts(base: TradeContext, overlay: TradeContext) -> TradeContext:
    """Merge two trade contexts, with overlay taking precedence"""
    base_dict = base.to_dict()
    overlay_dict = overlay.to_dict()
    
    # Deep merge the dictionaries
    for key, value in overlay_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key].update(value)
        elif value or key not in base_dict:  # Only update if value is not empty or key doesn't exist
            base_dict[key] = value
    
    return TradeContext.from_dict(base_dict)



