"""
Configuration for pytest
"""

import pytest

# Remove the coverage-related code since we're using pytest-cov
# def pytest_sessionstart(session):
#     """
#     Set up coverage before any tests are run.
#     """
#     # Start coverage collection
#     cov = coverage.Coverage(
#         source=["analysis", "greek_flow"],
#         omit=["*/test_*.py", "*/conftest.py"],
#     )
#     cov.start()
#     session.config.cov = cov

# def pytest_sessionfinish(session, exitstatus):
#     """
#     Generate coverage report after all tests are run.
#     """
#     cov = session.config.cov
#     cov.stop()
#     cov.save()
#     
#     # Print coverage report to console
#     print("\nCoverage Report:")
#     cov.report()
#     
#     # Generate HTML report
#     cov.html_report(directory="coverage_html")
#     print("HTML coverage report generated in coverage_html/")

# Add any other pytest configuration here
