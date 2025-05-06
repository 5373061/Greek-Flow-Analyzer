"""
Run all dashboard tests and generate a report.
"""

import os
import sys
import unittest
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"dashboard_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_dashboard_tests():
    """Run all dashboard tests and return results."""
    logger.info("Starting dashboard tests...")
    
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(__file__), pattern="test_dashboard*.py")
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Log results
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Failures: {len(result.failures)}")
    
    # Log detailed errors if any
    if result.errors:
        logger.error("Test errors:")
        for test, error in result.errors:
            logger.error(f"  {test}: {error}")
    
    if result.failures:
        logger.error("Test failures:")
        for test, failure in result.failures:
            logger.error(f"  {test}: {failure}")
    
    return result

if __name__ == "__main__":
    result = run_dashboard_tests()
    
    # Exit with appropriate code
    if result.wasSuccessful():
        logger.info("All dashboard tests passed!")
        sys.exit(0)
    else:
        logger.error("Some dashboard tests failed.")
        sys.exit(1)