"""
Comprehensive test runner for Greek Energy Flow Analysis project.
Runs all tests and provides a detailed report of passing and failing tests.
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime
import re

def run_test(test_path, verbose=True):
    """Run a specific test file or directory and return results."""
    start_time = time.time()
    
    # Command to run the test with pytest
    cmd = ["pytest", test_path]
    if verbose:
        cmd.append("-v")
    
    # Run the test
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Calculate duration
    duration = time.time() - start_time
    
    return {
        "test_path": test_path,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration": duration,
        "success": result.returncode == 0
    }

def parse_test_results(result):
    """Parse pytest output to extract individual test results."""
    lines = result["stdout"].split('\n')
    tests = []
    
    for line in lines:
        if " PASSED " in line or " FAILED " in line or " SKIPPED " in line or " ERROR " in line:
            parts = line.strip().split()
            if len(parts) >= 2:
                test_name = parts[0]
                status = "PASSED" if " PASSED " in line else "FAILED" if " FAILED " in line else "SKIPPED" if " SKIPPED " in line else "ERROR"
                tests.append({"name": test_name, "status": status})
    
    return tests

def extract_error_details(result):
    """Extract detailed error information from test output."""
    if result["success"]:
        return []
    
    errors = []
    stdout = result["stdout"]
    
    # Extract error sections
    error_sections = re.findall(r'={10,}([\s\S]*?)(?:={10,}|$)', stdout)
    
    for section in error_sections:
        if "ERROR" in section or "FAILED" in section:
            # Extract test name and error message
            test_match = re.search(r'(ERROR|FAILED) ([^\s]+)', section)
            error_match = re.search(r'E\s+(.*?)(?:\n\n|\Z)', section, re.DOTALL)
            
            if test_match and error_match:
                test_name = test_match.group(2)
                error_msg = error_match.group(1).strip()
                errors.append({
                    "test": test_name,
                    "type": test_match.group(1),
                    "message": error_msg
                })
    
    return errors

def find_test_files(include_checkpoints=False):
    """Find all test files in the project."""
    test_files = []
    
    # Look in tests directory
    if os.path.exists("tests"):
        for root, dirs, files in os.walk("tests"):
            # Skip checkpoint directories
            if ".ipynb_checkpoints" in root and not include_checkpoints:
                continue
                
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    test_files.append(os.path.join(root, file))
    
    # Look in other directories for test files
    for root, dirs, files in os.walk("."):
        # Skip certain directories
        if any(skip in root for skip in ["tests", "venv", "__pycache__", ".ipynb_checkpoints"]):
            continue
            
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                test_files.append(os.path.join(root, file))
    
    return sorted(test_files)

def run_all_tests(verbose=True, include_checkpoints=False, only_failing=False):
    """Run all tests and generate a report."""
    test_files = find_test_files(include_checkpoints)
    
    # If only_failing is True, filter to only run previously failed tests
    if only_failing and os.path.exists(".pytest_cache/v/cache/lastfailed"):
        with open(".pytest_cache/v/cache/lastfailed", "r") as f:
            lastfailed = json.load(f)
        
        # Extract file paths from the lastfailed dictionary
        failed_files = set()
        for key in lastfailed.keys():
            if "::" in key:
                file_path = key.split("::")[0]
                failed_files.add(file_path)
            else:
                failed_files.add(key)
        
        # Filter test_files to only include failed files
        test_files = [f for f in test_files if f in failed_files or f.replace("\\", "/") in failed_files]
    
    print(f"Found {len(test_files)} test files.")
    print("Running all tests...")
    
    results = []
    for test_file in test_files:
        print(f"Running {test_file}...")
        result = run_test(test_file, verbose)
        results.append(result)
    
    # Generate summary
    passing = [r for r in results if r["success"]]
    failing = [r for r in results if not r["success"]]
    
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {len(passing)} passing, {len(failing)} failing")
    print("=" * 80)
    
    if failing:
        print("\nFAILING TESTS:")
        for result in failing:
            print(f"  - {result['test_path']}")
            individual_tests = parse_test_results(result)
            for test in individual_tests:
                if test["status"] != "PASSED":
                    print(f"    * {test['name']} - {test['status']}")
            
            # Extract and display detailed error information
            errors = extract_error_details(result)
            if errors:
                print("    Detailed errors:")
                for error in errors:
                    print(f"      - {error['test']} ({error['type']}): {error['message']}")
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_report_{timestamp}.json"
    
    report = {
        "timestamp": timestamp,
        "total_tests": len(test_files),
        "passing_tests": len(passing),
        "failing_tests": len(failing),
        "detailed_results": results
    }
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to {report_file}")
    
    # Generate a more focused analysis of failing tests
    analyze_failing_tests(failing)
    
    return report

def analyze_failing_tests(failing_results):
    """Analyze failing tests to identify patterns and common issues."""
    if not failing_results:
        return
    
    print("\n" + "=" * 80)
    print("FAILING TESTS ANALYSIS")
    print("=" * 80)
    
    # Count failures by directory/module
    failure_by_module = {}
    for result in failing_results:
        path = result["test_path"]
        module = os.path.dirname(path)
        if module not in failure_by_module:
            failure_by_module[module] = []
        failure_by_module[module].append(path)
    
    print("\nFailures by module:")
    for module, files in failure_by_module.items():
        print(f"  - {module}: {len(files)} failing tests")
    
    # Look for common error patterns
    error_patterns = {}
    for result in failing_results:
        errors = extract_error_details(result)
        for error in errors:
            error_type = error.get("type", "Unknown")
            error_msg = error.get("message", "")
            
            # Extract the first line or first 50 chars of error message for grouping
            short_msg = error_msg.split('\n')[0][:50]
            
            key = f"{error_type}: {short_msg}"
            if key not in error_patterns:
                error_patterns[key] = []
            error_patterns[key].append(result["test_path"])
    
    if error_patterns:
        print("\nCommon error patterns:")
        for pattern, files in error_patterns.items():
            print(f"  - {pattern} ({len(files)} occurrences)")
    
    # Suggest next steps
    print("\nSuggested next steps:")
    if ".ipynb_checkpoints" in str(failing_results):
        print("  - Ignore failures in .ipynb_checkpoints files (these are Jupyter Notebook backups)")
    
    # Suggest focusing on specific modules with high failure rates
    high_failure_modules = [m for m, f in failure_by_module.items() if len(f) > 2 and ".ipynb_checkpoints" not in m]
    if high_failure_modules:
        print(f"  - Focus on fixing tests in these modules with high failure rates:")
        for module in high_failure_modules:
            print(f"    * {module}")
    
    # Suggest running with --lf option
    print("  - Run with --only-failing or --lf option to focus on fixing failing tests")
    print("  - Run individual test files with 'pytest -v <test_file>' for more detailed output")

def main():
    """Main function to run tests."""
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    include_checkpoints = "--include-checkpoints" in sys.argv
    only_failing = "--only-failing" in sys.argv or "--lf" in sys.argv
    
    run_all_tests(verbose, include_checkpoints, only_failing)

if __name__ == "__main__":
    main()


