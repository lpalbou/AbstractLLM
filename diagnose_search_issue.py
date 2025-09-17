#!/usr/bin/env python3
"""
Diagnostic script to understand why the search result is so large.
"""

import time
import os
import re
from pathlib import Path

def diagnose_search_issue():
    """Diagnose why the search is producing such a large result."""
    
    pattern = 'Eidolon'
    path = '/Users/albou/projects/mnemosyne/memory'
    
    print("🔍 Diagnosing Search Issue")
    print("=" * 50)
    print(f"Pattern: {pattern}")
    print(f"Path: {path}")
    
    if not Path(path).exists():
        print(f"❌ Path does not exist")
        return
    
    # First, let's see what files contain the pattern
    print(f"\n📋 Finding files that contain '{pattern}':")
    print("-" * 40)
    
    regex_pattern = re.compile(pattern, re.IGNORECASE)
    matching_files = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = Path(root) / file
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        matches = list(regex_pattern.finditer(content))
                        if matches:
                            file_size = len(content)
                            line_count = content.count('\n') + 1
                            matching_files.append({
                                'path': str(file_path),
                                'size': file_size,
                                'lines': line_count,
                                'matches': len(matches)
                            })
                except:
                    continue
    
    # Sort by file size to see the largest files
    matching_files.sort(key=lambda x: x['size'], reverse=True)
    
    print(f"Found {len(matching_files)} files containing '{pattern}':")
    print()
    
    total_size = 0
    for i, file_info in enumerate(matching_files, 1):
        size_mb = file_info['size'] / (1024 * 1024)
        total_size += file_info['size']
        filename = Path(file_info['path']).name
        
        print(f"{i:2d}. {filename[:50]:50} {size_mb:8.2f} MB, {file_info['lines']:6d} lines, {file_info['matches']:3d} matches")
        
        # Show details for largest files
        if i <= 3:
            # Show some sample lines that match
            try:
                with open(file_info['path'], 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    match_lines = []
                    for line_num, line in enumerate(lines, 1):
                        if regex_pattern.search(line):
                            match_lines.append((line_num, line.strip()[:100]))
                            if len(match_lines) >= 3:  # Show max 3 sample lines
                                break
                    
                    if match_lines:
                        print(f"    Sample matches:")
                        for line_num, line_text in match_lines:
                            print(f"      Line {line_num}: {line_text}...")
                print()
            except:
                print("    (Could not read sample lines)")
                print()
    
    print(f"Total content size: {total_size / (1024 * 1024):.2f} MB")
    
    # Now let's understand why the output is so large
    print(f"\n🔍 Understanding Output Size:")
    print("-" * 40)
    
    # Calculate expected output size
    total_output_size = 0
    for file_info in matching_files:
        # Each file gets a header: "📄 filename:"
        header_size = len(f"📄 {file_info['path']}:") + 2  # +2 for newlines
        total_output_size += header_size
        
        # Each matching line gets formatted as "    Line N: content"
        try:
            with open(file_info['path'], 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for line_num, line in enumerate(lines, 1):
                    if regex_pattern.search(line):
                        line_output = f"    Line {line_num}: {line.rstrip()}"
                        total_output_size += len(line_output) + 1  # +1 for newline
        except:
            continue
    
    print(f"Expected total output size: {total_output_size / (1024 * 1024):.2f} MB")
    print(f"Expected total output size: {total_output_size:,} characters")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    print("-" * 30)
    
    largest_file = matching_files[0] if matching_files else None
    if largest_file:
        largest_size_mb = largest_file['size'] / (1024 * 1024)
        if largest_size_mb > 100:
            print(f"🚨 HUGE FILE DETECTED: {Path(largest_file['path']).name} ({largest_size_mb:.1f} MB)")
            print(f"   This file alone is causing the massive output!")
            print(f"   Consider:")
            print(f"   • Use head_limit=10 to limit matches per search")
            print(f"   • Search in specific subdirectories instead of the whole memory folder")
            print(f"   • Use output_mode='files_with_matches' to just get filenames")
            print(f"   • Use output_mode='count' to just count matches")
    
    # Show optimal search commands
    print(f"\n🚀 Optimized Search Commands:")
    print("-" * 35)
    print(f"# Just count matches:")
    print(f"search_files('{pattern}', '{path}', output_mode='count')")
    print()
    print(f"# Just get filenames:")
    print(f"search_files('{pattern}', '{path}', output_mode='files_with_matches')")
    print()
    print(f"# Limited content (much faster):")
    print(f"search_files('{pattern}', '{path}', output_mode='content', head_limit=10)")
    print()
    print(f"# Search specific file types only:")
    print(f"search_files('{pattern}', '{path}', file_pattern='*.md')")

if __name__ == "__main__":
    diagnose_search_issue()
