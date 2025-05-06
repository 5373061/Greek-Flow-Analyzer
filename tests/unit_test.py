import unittest
from datetime import datetime, timedelta
import time
from pipeline.cache_manager import MarketDataCache

class TestMarketDataCache(unittest.TestCase):
    def setUp(self):
        self.cache = MarketDataCache(ttl_minutes=1)
        self.test_data = {
            'ticker': 'SPY',
            'last': 450.0,
            'volume': 1000000
        }
        
    def test_cache_set_get(self):
        """Test basic cache set and get operations"""
        self.cache.set('SPY', self.test_data)
        cached_data = self.cache.get('SPY')
        self.assertEqual(cached_data, self.test_data)
        
    def test_cache_expiration(self):
        """Test cache expiration"""
        self.cache = MarketDataCache(ttl_minutes=0.1)  # 6 seconds TTL
        self.cache.set('SPY', self.test_data)
        
        # Should be in cache
        self.assertIsNotNone(self.cache.get('SPY'))
        
        # Wait for expiration
        time.sleep(7)
        self.assertIsNone(self.cache.get('SPY'))
        
    def test_cache_override(self):
        """Test cache data override"""
        self.cache.set('SPY', self.test_data)
        new_data = {**self.test_data, 'last': 460.0}
        self.cache.set('SPY', new_data)
        
        cached_data = self.cache.get('SPY')
        self.assertEqual(cached_data['last'], 460.0)

if __name__ == '__main__':
    unittest.main()