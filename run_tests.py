"""
run_tests.py - Run all tests for the Greek Energy Flow II pattern analysis
"""

import unittest
import logging
import os
import sys
import re
import argparse
from datetime import datetime

def run_tests(verbose=True, pattern=None, failfast=False, filter_warnings=True):
    """
    Run all test modules and generate a report.
    
    Args:
        verbose: Whether to show verbose output
        pattern: Pattern to match test files (e.g., 'test_ordinal*')
        failfast: Stop on first failure
        filter_warnings: Filter out common warnings
        
    Returns:
        True if all tests passed, False otherwise
    """
    # Configure logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_report_{timestamp}.log")
    
    # Configure logging with a filter for common warnings
    class WarningFilter(logging.Filter):
        def filter(self, record):
            if filter_warnings:
                # Filter out common warnings
                if "Insufficient data for pattern analysis" in record.getMessage():
                    return False
                if "No patterns found for" in record.getMessage():
                    return False
                if "Cannot filter for VOL_CRUSH" in record.getMessage():
                    return False
                if "Delta column missing" in record.getMessage():
                    return False
            return True
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Add filter to console handler only if requested
    if filter_warnings:
        console_handler.addFilter(WarningFilter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Get a logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Starting tests for Greek Energy Flow II Pattern Analysis")
    
    if pattern:
        logger.info(f"Running tests matching pattern: {pattern}")
    
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    if pattern:
        test_suite = test_loader.discover('.', pattern=pattern)
    else:
        test_suite = test_loader.discover('.', pattern='test_*.py')
    
    # Run the tests
    verbosity = 2 if verbose else 1
    test_runner = unittest.TextTestRunner(verbosity=verbosity, failfast=failfast)
    result = test_runner.run(test_suite)
    
    # Log the results
    logger.info("=" * 70)
    logger.info(f"Test Results:")
    logger.info(f"  Tests Run: {result.testsRun}")
    logger.info(f"  Failures: {len(result.failures)}")
    logger.info(f"  Errors: {len(result.errors)}")
    logger.info(f"  Skipped: {len(result.skipped)}")
    
    # Log detailed failures and errors
    if result.failures:
        logger.info("\nFailures:")
        for i, (test, traceback) in enumerate(result.failures, 1):
            logger.info(f"  {i}. {test}")
            logger.info(f"     {traceback.split('Traceback')[0]}")
    
    if result.errors:
        logger.info("\nErrors:")
        for i, (test, traceback) in enumerate(result.errors, 1):
            logger.info(f"  {i}. {test}")
            logger.info(f"     {traceback.split('Traceback')[0]}")
    
    # Count warnings in the log file
    warning_count = 0
    warning_pattern = re.compile(r'\[WARNING\]')
    with open(log_file, 'r') as f:
        for line in f:
            if warning_pattern.search(line):
                warning_count += 1
    
    logger.info(f"  Warnings: {warning_count}")
    
    logger.info("=" * 70)
    logger.info(f"Test report saved to: {log_file}")
    
    return result.wasSuccessful()

def main():
    """Parse command line arguments and run tests."""
    parser = argparse.ArgumentParser(description='Run tests for Greek Energy Flow II Pattern Analysis')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose output')
    parser.add_argument('--pattern', '-p', help='Pattern to match test files (e.g., "test_ordinal*")')
    parser.add_argument('--failfast', '-f', action='store_true', help='Stop on first failure')
    parser.add_argument('--show-warnings', '-w', action='store_true', help='Show all warnings')
    
    args = parser.parse_args()
    
    success = run_tests(
        verbose=args.verbose, 
        pattern=args.pattern, 
        failfast=args.failfast,
        filter_warnings=not args.show_warnings
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()


