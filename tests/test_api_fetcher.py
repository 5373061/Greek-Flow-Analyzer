import logging
from api_fetcher import fetch_options_chain_snapshot, fetch_underlying_snapshot
import config
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def validate_underlying_data(data):
    """Validate required fields in underlying data"""
    required_fields = ['ticker', 'todaysChange', 'todaysChangePerc']
    return all(field in data for field in required_fields)

def validate_options_contract(contract):
    """Validate required fields in options contract"""
    required_sections = ['day', 'details', 'greeks']
    if not all(section in contract for section in required_sections):
        return False
    
    # Validate specific fields
    if not all(field in contract['details'] for field in ['strike_price', 'expiration_date']):
        return False
    if not all(field in contract['greeks'] for field in ['delta', 'gamma', 'theta', 'vega']):
        return False
    return True

def test_api_connections():
    """Test both the options chain and underlying data fetching"""
    symbol = 'SPY'
    
    print("\n=== API Fetcher Test ===")
    
    try:
        # Test 1: Underlying Data
        print("\n1. Testing underlying snapshot fetch...")
        underlying_data = fetch_underlying_snapshot(symbol, config.POLYGON_API_KEY)
        if underlying_data and validate_underlying_data(underlying_data):
            print("✓ Underlying data fetch successful and valid")
            print(f"Data returned: {type(underlying_data)}")
            print(f"Sample data: {json.dumps(dict(list(underlying_data.items())[:3]), indent=2)}")
        else:
            print("✗ Failed to fetch underlying data or invalid data format")
        
        # Test 2: Options Chain
        print("\n2. Testing options chain snapshot...")
        options_data = fetch_options_chain_snapshot(symbol, config.POLYGON_API_KEY)
        if options_data:
            print("✓ Options chain fetch successful")
            print(f"Number of contracts: {len(options_data)}")
            if len(options_data) > 0:
                if validate_options_contract(options_data[0]):
                    print("✓ Options data format valid")
                else:
                    print("✗ Options data format invalid")
                print(f"Sample contract: {json.dumps(options_data[0], indent=2)}")
        else:
            print("✗ Failed to fetch options chain")
            
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        logging.exception("Test failed with error:")

if __name__ == '__main__':
    test_api_connections()