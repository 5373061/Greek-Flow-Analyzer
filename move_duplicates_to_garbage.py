#!/usr/bin/env python3
"""
Script to find files with identical content and move duplicates to a 'garbage' folder.
"""

import os
import sys
import hashlib
import shutil
from collections import defaultdict
from datetime import datetime

# Project root directory - change this to your project root if needed
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Garbage directory
GARBAGE_DIR = os.path.join(ROOT_DIR, "garbage")

def get_file_hash(filepath):
    """Calculate the MD5 hash of a file's content."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def find_and_move_duplicates(directory=None, exclude_dirs=None, 
                            strategy="keep_shortest_path", dry_run=True):
    """Find files with identical content and move duplicates to garbage folder.
    
    Args:
        directory (str): Directory to search. Defaults to project root.
        exclude_dirs (list): Directories to exclude from search.
        strategy (str): Strategy for choosing which file to keep:
            - keep_shortest_path: Keep the file with the shortest path
            - keep_first: Keep the first file found
        dry_run (bool): If True, only print what would be done without moving files.
    
    Returns:
        tuple: (duplicates_dict, moved_count)
    """
    if directory is None:
        directory = ROOT_DIR
        
    if exclude_dirs is None:
        exclude_dirs = ['garbage']  # Only exclude the garbage directory by default
    
    # Create garbage directory if it doesn't exist
    if not dry_run and not os.path.exists(GARBAGE_DIR):
        os.makedirs(GARBAGE_DIR)
    
    # Dictionary to store file hashes
    file_hashes = defaultdict(list)
    
    # Walk through the directory
    print(f"Scanning directory: {directory}")
    total_files = 0
    
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for filename in files:
            filepath = os.path.join(root, filename)
            
            # Skip very large files (optional)
            try:
                file_size = os.path.getsize(filepath)
                if file_size > 100 * 1024 * 1024:  # Skip files larger than 100MB
                    print(f"Skipping large file: {filepath} ({file_size / (1024*1024):.2f} MB)")
                    continue
            except Exception:
                continue
            
            try:
                # Calculate file hash
                file_hash = get_file_hash(filepath)
                
                # Add file to hash dictionary
                file_hashes[file_hash].append(filepath)
                total_files += 1
                
                # Print progress every 100 files
                if total_files % 100 == 0:
                    print(f"Processed {total_files} files...")
                    
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
    
    print(f"Finished scanning. Processed {total_files} files.")
    
    # Filter out non-duplicates
    duplicates = {h: files for h, files in file_hashes.items() if len(files) > 1}
    
    # Move duplicates to garbage
    moved_count = 0
    
    for file_hash, files in duplicates.items():
        if strategy == "keep_shortest_path":
            # Keep the file with the shortest path
            files_by_length = sorted(files, key=lambda x: len(x))
            keep_file = files_by_length[0]
        else:  # keep_first
            # Keep the first file in the list
            keep_file = files[0]
        
        # Move all files except the one to keep
        for file_path in files:
            if file_path != keep_file:
                rel_path = os.path.relpath(file_path, ROOT_DIR)
                
                # Create a unique name for the file in the garbage directory
                # to avoid overwriting files with the same name
                base_name = os.path.basename(file_path)
                file_dir = os.path.dirname(file_path).replace(os.path.sep, "_")
                if file_dir:
                    unique_name = f"{file_dir}_{base_name}"
                else:
                    unique_name = base_name
                
                garbage_path = os.path.join(GARBAGE_DIR, unique_name)
                
                if dry_run:
                    print(f"Would move: {rel_path} -> garbage/{unique_name}")
                else:
                    # Ensure the garbage directory exists
                    if not os.path.exists(GARBAGE_DIR):
                        os.makedirs(GARBAGE_DIR)
                    
                    # Handle case where the file already exists in garbage
                    if os.path.exists(garbage_path):
                        name, ext = os.path.splitext(unique_name)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        unique_name = f"{name}_{timestamp}{ext}"
                        garbage_path = os.path.join(GARBAGE_DIR, unique_name)
                    
                    print(f"Moving: {rel_path} -> garbage/{unique_name}")
                    shutil.copy2(file_path, garbage_path)  # Copy with metadata
                    os.remove(file_path)  # Remove original
                
                moved_count += 1
    
    return duplicates, moved_count

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Find files with identical content and move duplicates to a 'garbage' folder.")
    parser.add_argument("--directory", type=str, default=ROOT_DIR,
                        help=f"Directory to search (default: {ROOT_DIR})")
    parser.add_argument("--exclude", nargs="+", default=["garbage"],
                        help="Directories to exclude from search (default: garbage)")
    parser.add_argument("--strategy", choices=["keep_shortest_path", "keep_first"], 
                        default="keep_shortest_path", 
                        help="Strategy for choosing which file to keep")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Only print what would be done without moving files")
    args = parser.parse_args()
    
    print(f"Searching for duplicate files in: {args.directory}")
    print(f"Excluding directories: {', '.join(args.exclude)}")
    print(f"Strategy: {args.strategy}")
    if args.dry_run:
        print("DRY RUN: No files will be moved")
    else:
        print(f"Duplicates will be moved to: {GARBAGE_DIR}")
    
    # Find and move duplicate files
    duplicates, moved_count = find_and_move_duplicates(
        directory=args.directory,
        exclude_dirs=args.exclude,
        strategy=args.strategy,
        dry_run=args.dry_run
    )
    
    # Print results
    if not duplicates:
        print("No duplicate files found.")
        sys.exit(0)
    
    print(f"\nFound {len(duplicates)} sets of duplicate files:")
    for i, (file_hash, files) in enumerate(duplicates.items(), 1):
        print(f"\nDuplicate set #{i}:")
        for file_path in files:
            rel_path = os.path.relpath(file_path, ROOT_DIR)
            if file_path == (sorted(files, key=lambda x: len(x))[0] if args.strategy == "keep_shortest_path" else files[0]):
                print(f"  - {rel_path} (KEEPING)")
            else:
                print(f"  - {rel_path} (MOVING TO GARBAGE)")
    
    if args.dry_run:
        print(f"\nWould move {moved_count} duplicate files to the garbage folder.")
        print("Run without --dry-run to actually move the files.")
    else:
        print(f"\nMoved {moved_count} duplicate files to the garbage folder.")
        print(f"You can review them in: {GARBAGE_DIR}")
        print("If you're satisfied, you can delete the garbage folder manually.")
