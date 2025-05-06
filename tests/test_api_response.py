import logging
from pathlib import Path
import sys
import json
from typing import Any, Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def analyze_response_structure(data: Dict[str, Any]) -> None:
    """Analyze and print the structure of API response data"""
    logger.info("\n=== API Response Analysis ===")
    logger.info(f"Top level type: {type(data)}")
    logger.info(f"Available keys: {list(data.keys())}")
    
    for key, value in data.items():
        logger.info(f"\nField: {key}")
        logger.info(f"Type: {type(value)}")
        if isinstance(value, dict):
            logger.info(f"Nested keys: {list(value.keys())}")
        elif isinstance(value, list):
            logger.info(f"List length: {len(value)}")
            if value:
                logger.info(f"First item type: {type(value[0])}")

def inspect_api_response() -> Optional[Dict[str, Any]]:
    """Fetch and inspect the raw API response structure"""
    try:
        from api_fetcher import fetch_underlying_snapshot
        import config
        
        symbol = "SPY"
        logger.info(f"Fetching data for {symbol}...")
        
        response = fetch_underlying_snapshot(symbol, config.POLYGON_API_KEY)
        if not response:
            logger.error("Received empty response from API")
            return None
            
        # Analyze response structure
        analyze_response_structure(response)
        
        # Pretty print full response
        logger.info("\n=== Full Response Data ===")
        print(json.dumps(response, indent=2, default=str))
        
        return response
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    response = inspect_api_response()