from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

class MarketDataCache:
    """Manages temporary storage of market data to reduce API calls"""
    
    def __init__(self, ttl_minutes: int = 15):
        self.logger = logging.getLogger(__name__)
        self.ttl = timedelta(minutes=ttl_minutes)
        self.cache: Dict[str, Dict[str, Any]] = {}
        
    def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached data if valid"""
        if symbol not in self.cache:
            return None
            
        entry = self.cache[symbol]
        if datetime.now() - entry['timestamp'] > self.ttl:
            self.logger.debug(f"Cache expired for {symbol}")
            del self.cache[symbol]
            return None
            
        return entry['data']
        
    def set(self, symbol: str, data: Dict[str, Any]) -> None:
        """Store new data in cache"""
        self.cache[symbol] = {
            'data': data,
            'timestamp': datetime.now()
        }
        self.logger.debug(f"Cached data for {symbol}")