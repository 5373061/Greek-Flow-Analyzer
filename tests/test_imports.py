import sys
from pathlib import Path
import inspect
import logging
from typing import Tuple, Optional, Any

# Setup correct import paths
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def validate_imports() -> Tuple[bool, Optional[Any]]:
    """Validate imports and return (success, pipeline_instance)"""
    try:
        # First test api_fetcher import
        from api_fetcher import fetch_options_chain_snapshot, fetch_underlying_snapshot
        print("✓ API fetcher imports working!")
        
        # Validate API functions
        if not all(inspect.isfunction(f) for f in [fetch_options_chain_snapshot, fetch_underlying_snapshot]):
            raise TypeError("API fetcher imports are not functions")
        
        # Then test pipeline import
        from pipeline.data_pipeline import OptionsDataPipeline
        print("✓ Pipeline imports working!")
        
        # Validate pipeline class
        if not inspect.isclass(OptionsDataPipeline):
            raise TypeError("OptionsDataPipeline is not a class")
            
        # Test pipeline instantiation
        import config
        pipeline = OptionsDataPipeline(config)
        
        # Validate required methods
        required_methods = ['fetch_symbol_data', 'prepare_analysis_data']
        for method in required_methods:
            if not hasattr(pipeline, method):
                raise AttributeError(f"Pipeline missing required method: {method}")
            
        print("✓ All import validations passed!")
        return True, pipeline
        
    except Exception as e:
        logging.error(f"Validation error: {str(e)}", exc_info=True)
        print(f"✗ Error during validation: {str(e)}")
        print(f"Current path: {current_dir}")
        return False, None

def test_pipeline_functionality(pipeline) -> bool:
    """Test basic pipeline functionality"""
    try:
        symbol = "SPY"
        logging.info(f"Testing pipeline with symbol: {symbol}")
        
        # Get data from pipeline
        underlying, options = pipeline.fetch_symbol_data(symbol)
        
        # Validate core required fields
        required_fields = ['ticker', 'last', 'volume']
        missing_fields = [f for f in required_fields if f not in underlying]
        if missing_fields:
            raise ValueError(f"Missing required underlying fields: {missing_fields}")
            
        # Validate options data
        if not options or not isinstance(options, list):
            raise ValueError("Failed to fetch options data")
            
        # Test data preparation
        analysis_df = pipeline.prepare_analysis_data(symbol)
        if analysis_df is None or analysis_df.empty:
            raise ValueError("Failed to prepare analysis data")
            
        # Print success info
        print(f"✓ Pipeline successfully tested with {symbol}")
        print(f"  Underlying: {underlying['ticker']} @ ${underlying['last']}")
        print(f"  Options contracts: {len(options)}")
        return True
            
    except Exception as e:
        logging.error(f"Pipeline test error: {str(e)}")
        print(f"✗ Error testing pipeline: {str(e)}")
        return False

if __name__ == '__main__':
    success, pipeline = validate_imports()
    if success and pipeline:
        test_pipeline_functionality(pipeline)