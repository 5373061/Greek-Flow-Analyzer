#!/usr/bin/env python
"""
Syntax Checker Script

This script checks Python files for syntax errors without executing them.
It will scan either specified files or all Python files in the current directory.
"""

import os
import sys
import py_compile
import traceback
from pathlib import Path

def check_syntax(file_path):
    """Check a Python file for syntax errors"""
    try:
        py_compile.compile(file_path, doraise=True)
        print(f"✓ {file_path}: No syntax errors detected")
        return True
    except py_compile.PyCompileError as e:
        error_message = str(e)
        print(f"✗ {file_path}: Syntax error detected")
        
        # Extract line number and error message
        if "line" in error_message:
            try:
                # Try to get the specific error line from the traceback
                line_num = None
                for line in error_message.split('\n'):
                    if "line" in line:
                        parts = line.split("line")
                        if len(parts) > 1:
                            try:
                                line_parts = parts[1].strip().split(',')
                                if line_parts:
                                    line_num = line_parts[0].strip()
                                    break
                            except:
                                pass
                
                if line_num:
                    print(f"  Error on line {line_num}")
                    
                    # Try to display the problematic line and surrounding context
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            line_index = int(line_num) - 1
                            start = max(0, line_index - 2)
                            end = min(len(lines), line_index + 3)
                            
                            print("\nContext:")
                            for i in range(start, end):
                                prefix = "→ " if i == line_index else "  "
                                print(f"{prefix}{i+1}: {lines[i].rstrip()}")
                    except Exception as read_error:
                        print(f"  Could not read file contents: {read_error}")
            except:
                pass
                
        # Print the actual error message
        error_msg = error_message.split("SyntaxError:")[-1].strip() if "SyntaxError:" in error_message else error_message
        print(f"  Error details: {error_msg}")
        return False
    except Exception as e:
        print(f"✗ {file_path}: Error checking syntax: {e}")
        return False

def main():
    """Main function to check syntax of Python files"""
    # Get files to check from command line arguments or scan current directory
    files_to_check = []
    
    if len(sys.argv) > 1:
        # Check the specified files
        for arg in sys.argv[1:]:
            path = Path(arg)
            if path.exists():
                if path.is_file() and path.suffix == '.py':
                    files_to_check.append(path)
                elif path.is_dir():
                    files_to_check.extend(path.glob('**/*.py'))
            else:
                print(f"Warning: {arg} does not exist")
    else:
        # Check all Python files in the current directory
        current_dir = Path('.')
        files_to_check = list(current_dir.glob('*.py'))
    
    if not files_to_check:
        print("No Python files found to check.")
        return
    
    print(f"Checking {len(files_to_check)} Python file(s) for syntax errors...\n")
    
    # Check each file
    success_count = 0
    error_count = 0
    
    for file_path in files_to_check:
        if check_syntax(file_path):
            success_count += 1
        else:
            error_count += 1
    
    # Print summary
    print(f"\nSyntax check complete: {success_count} file(s) OK, {error_count} file(s) with errors")

if __name__ == "__main__":
    main()