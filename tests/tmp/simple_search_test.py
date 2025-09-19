#!/usr/bin/env python3
"""
Simple performance test for search functionality.
Direct implementation to avoid import issues.
"""

import time
import os
import re
import glob
from pathlib import Path
from typing import Optional

def search_files(pattern: str, path: str = ".", output_mode: str = "files_with_matches", head_limit: Optional[int] = 20, file_pattern: str = "*", case_sensitive: bool = False, multiline: bool = False) -> str:
    """
    Simplified version of search_files function for testing.
    """
    try:
        search_path = Path(path)
        
        # Compile regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE | re.DOTALL
        
        try:
            regex_pattern = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern '{pattern}': {str(e)}"
        
        # Determine if path is a file or directory
        if search_path.is_file():
            files_to_search = [search_path]
        elif search_path.is_dir():
            # Find files matching pattern in directory
            if file_pattern == "*":
                # Search all files recursively
                files_to_search = []
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        file_path = Path(root) / file
                        # Skip binary files by checking if they're text files
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                f.read(1024)  # Try to read first 1KB
                            files_to_search.append(file_path)
                        except (UnicodeDecodeError, PermissionError):
                            continue  # Skip binary/inaccessible files
            else:
                # Use glob pattern
                search_pattern = str(search_path / "**" / file_pattern)
                files_to_search = [Path(f) for f in glob.glob(search_pattern, recursive=True) if Path(f).is_file()]
        else:
            return f"Error: Path '{path}' does not exist"
        
        if not files_to_search:
            return f"No files found to search in '{path}'"
        
        # Search through files
        results = []
        files_with_matches = []  # Will store (file_path, [line_numbers]) tuples
        match_counts = {}
        total_matches = 0
        
        for file_path in files_to_search:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    if multiline:
                        content = f.read()
                        matches = list(regex_pattern.finditer(content))
                        
                        if matches:
                            # Collect line numbers for files_with_matches mode
                            line_numbers = []
                            for match in matches:
                                line_num = content[:match.start()].count('\n') + 1
                                line_numbers.append(line_num)
                            
                            files_with_matches.append((str(file_path), line_numbers))
                            match_counts[str(file_path)] = len(matches)
                            total_matches += len(matches)
                            
                            if output_mode == "content":
                                results.append(f"\n📄 {file_path}:")
                                
                                # Convert content to lines for line number calculation
                                lines = content.splitlines()
                                for match in matches:
                                    # Find line number for match
                                    line_num = content[:match.start()].count('\n') + 1
                                    
                                    # Get the full line containing the match start
                                    if line_num <= len(lines):
                                        full_line = lines[line_num - 1]
                                        results.append(f"    Line {line_num}: {full_line}")
                                    
                                    # Apply head_limit for content mode
                                    if head_limit and len([r for r in results if r.startswith("    Line")]) >= head_limit:
                                        break
                    else:
                        lines = f.readlines()
                        file_matches = []
                        
                        for line_num, line in enumerate(lines, 1):
                            line_content = line.rstrip()
                            matches = list(regex_pattern.finditer(line_content))
                            
                            if matches:
                                file_matches.extend(matches)
                                if output_mode == "content":
                                    results.append(f"    Line {line_num}: {line_content}")
                        
                        if file_matches:
                            # Collect line numbers for files_with_matches mode
                            line_numbers = []
                            for line_num, line in enumerate(lines, 1):
                                line_content = line.rstrip()
                                if regex_pattern.search(line_content):
                                    line_numbers.append(line_num)
                            
                            files_with_matches.append((str(file_path), line_numbers))
                            match_counts[str(file_path)] = len(file_matches)
                            total_matches += len(file_matches)
                            
                            if output_mode == "content" and file_matches:
                                # Insert file header before the lines we just added
                                file_header_position = len(results) - len(file_matches)
                                results.insert(file_header_position, f"\n📄 {file_path}:")
                                
                                # Apply head_limit for content mode
                                if head_limit:
                                    content_lines = [r for r in results if r.startswith("    Line")]
                                    if len(content_lines) >= head_limit:
                                        break
                        
            except Exception as e:
                if output_mode == "content":
                    results.append(f"\n⚠️  Error reading {file_path}: {str(e)}")
        
        # Format output based on mode
        if output_mode == "files_with_matches":
            if head_limit:
                files_with_matches = files_with_matches[:head_limit]
            
            if files_with_matches:
                header = f"Files matching pattern '{pattern}':"
                formatted_results = [header]
                
                for file_path, line_numbers in files_with_matches:
                    # Format line numbers nicely
                    if len(line_numbers) == 1:
                        line_info = f"line {line_numbers[0]}"
                    elif len(line_numbers) <= 5:
                        line_info = f"lines {', '.join(map(str, line_numbers))}"
                    else:
                        # Show first few line numbers and total count
                        first_lines = ', '.join(map(str, line_numbers[:3]))
                        line_info = f"lines {first_lines}... ({len(line_numbers)} total)"
                    
                    formatted_results.append(f"{file_path} ({line_info})")
                
                return "\n".join(formatted_results)
            else:
                return f"No files found matching pattern '{pattern}'"
                
        elif output_mode == "count":
            if head_limit:
                count_items = list(match_counts.items())[:head_limit]
            else:
                count_items = match_counts.items()
            
            if count_items:
                header = f"Match counts for pattern '{pattern}':"
                count_results = [header]
                for file_path, count in count_items:
                    count_results.append(f"{count:3d} {file_path}")
                count_results.append(f"\nTotal: {total_matches} matches in {len(files_with_matches)} files")
                return "\n".join(count_results)
            else:
                return f"No matches found for pattern '{pattern}'"
                
        else:  # content mode
            if not results:
                return f"No matches found for pattern '{pattern}'"
            
            # Count files with matches for header
            file_count = len([r for r in results if r.startswith("\n📄")])
            header = f"Search results for pattern '{pattern}' in {file_count} files:"
            
            # Apply head_limit to final output if specified
            final_results = results
            if head_limit:
                content_lines = [r for r in results if r.startswith("    Line")]
                if len(content_lines) > head_limit:
                    # Keep file headers and trim content lines
                    trimmed_results = []
                    content_count = 0
                    for line in results:
                        if line.startswith("    Line"):
                            if content_count < head_limit:
                                trimmed_results.append(line)
                                content_count += 1
                        else:
                            trimmed_results.append(line)
                    final_results = trimmed_results
                    final_results.append(f"\n... (showing first {head_limit} matches)")
            
            return header + "\n" + "\n".join(final_results)
            
    except Exception as e:
        return f"Error performing search: {str(e)}"


def test_search_performance():
    """Test search_files function performance directly."""
    
    # Test parameters (equivalent to the LLM call)
    pattern = 'Eidolon'
    path = '/Users/albou/projects/mnemosyne/memory'
    
    print("🔍 Testing search_files performance")
    print("=" * 50)
    print(f"Pattern: {pattern}")
    print(f"Path: {path}")
    print(f"Path exists: {Path(path).exists()}")
    
    if not Path(path).exists():
        print(f"❌ Error: Path '{path}' does not exist")
        print(f"💡 Testing with current directory instead...")
        path = "."
    
    # Count files in directory first
    print(f"\n📊 Directory analysis:")
    file_count = 0
    total_size = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = Path(root) / file
            if file_path.is_file():
                try:
                    # Test if file is text
                    with open(file_path, 'r', encoding='utf-8') as f:
                        f.read(1024)
                    file_count += 1
                    total_size += file_path.stat().st_size
                except:
                    pass
    
    print(f"Text files: {file_count:,}")
    print(f"Total size: {total_size / (1024*1024):.2f} MB")
    
    # Time the search operation
    print(f"\n⏱️  Running search...")
    start_time = time.time()
    
    result = search_files(pattern, path)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"⏰ Search completed in: {duration:.3f} seconds")
    print(f"📏 Result length: {len(result):,} characters")
    
    # Show first part of result
    print(f"\n📋 Result preview (first 300 chars):")
    print("-" * 50)
    print(result[:300])
    if len(result) > 300:
        print("... (truncated)")
    
    # Performance analysis
    print(f"\n📈 Performance Analysis:")
    if file_count > 0:
        files_per_second = file_count / duration if duration > 0 else float('inf')
        mb_per_second = (total_size / (1024*1024)) / duration if duration > 0 else float('inf')
        print(f"Files processed per second: {files_per_second:.1f}")
        print(f"MB processed per second: {mb_per_second:.2f}")
    
    # Speed assessment
    if duration < 1:
        print("🚀 FAST: Tool performance is very good")
    elif duration < 5:
        print("✅ GOOD: Tool performance is acceptable")
    elif duration < 15:
        print("⚠️  SLOW: Tool may need optimization")
    else:
        print("🐌 VERY SLOW: Significant performance issue")
    
    return duration


def test_with_limits():
    """Test with different configurations."""
    
    pattern = 'Eidolon'
    path = '/Users/albou/projects/mnemosyne/memory'
    
    if not Path(path).exists():
        path = "."
        pattern = "def"  # More likely to find matches in current directory
    
    print(f"\n🧪 Testing with different configurations:")
    print("-" * 50)
    
    tests = [
        ("count", None, "Count matches only"),
        ("files_with_matches", 5, "File names only (limit 5)"),
        ("content", 5, "Content with limit 5"),
        ("content", None, "Full content (no limit)"),
    ]
    
    for output_mode, limit, description in tests:
        start_time = time.time()
        result = search_files(pattern, path, output_mode=output_mode, head_limit=limit)
        duration = time.time() - start_time
        
        print(f"{description:25}: {duration:.3f}s, {len(result):6d} chars")


if __name__ == "__main__":
    print("🚀 Simple Search Performance Test")
    print("=" * 60)
    
    # Test basic performance
    duration = test_search_performance()
    
    # Additional tests
    test_with_limits()
    
    print(f"\n🎯 Conclusion:")
    print("-" * 30)
    if duration < 2:
        print("✅ The search tool itself is FAST!")
        print("If LLM calls are slow, the bottleneck is likely:")
        print("• LLM processing large result text")
        print("• Network communication overhead")
        print("• LLM parsing/understanding the output")
        print("\n💡 Optimization suggestions:")
        print("• Use head_limit=10 for faster results")
        print("• Use output_mode='count' for quick counts")
        print("• Use output_mode='files_with_matches' for file lists")
    else:
        print("⚠️  The search tool itself needs optimization!")
        print("• Large directory or many files")
        print("• Consider using file_pattern to limit scope")
        print("• Use head_limit to reduce processing")
