#!/usr/bin/env python3
"""
Performance test script for search_files function.
Tests the equivalent of: search_files('Eidolon', '/Users/albou/projects/mnemosyne/memory')
"""

import time
import sys
import os
from pathlib import Path

# Add current directory to path to import the tool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_search_performance():
    """Test search_files function performance directly."""
    
    try:
        # Import the search function
        from abstractllm.tools.common_tools import search_files
        
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
            return
        
        # Count files in directory first
        print(f"\n📊 Directory analysis:")
        file_count = 0
        total_size = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = Path(root) / file
                if file_path.is_file():
                    file_count += 1
                    try:
                        total_size += file_path.stat().st_size
                    except:
                        pass
        
        print(f"Total files: {file_count:,}")
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
        print(f"\n📋 Result preview (first 500 chars):")
        print("-" * 50)
        print(result[:500])
        if len(result) > 500:
            print("... (truncated)")
        
        # Performance analysis
        print(f"\n📈 Performance Analysis:")
        if file_count > 0:
            files_per_second = file_count / duration
            mb_per_second = (total_size / (1024*1024)) / duration
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
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the abstractllm directory")
        return None
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return None

def test_with_limits():
    """Test with different head_limit values to see if that helps performance."""
    
    try:
        from abstractllm.tools.common_tools import search_files
        
        pattern = 'Eidolon'
        path = '/Users/albou/projects/mnemosyne/memory'
        
        if not Path(path).exists():
            print(f"❌ Path '{path}' does not exist - skipping limit tests")
            return
        
        print(f"\n🧪 Testing with different head_limit values:")
        print("-" * 50)
        
        limits = [1, 5, 10, 50, None]  # None = no limit
        
        for limit in limits:
            start_time = time.time()
            result = search_files(pattern, path, head_limit=limit)
            duration = time.time() - start_time
            
            match_count = result.count("Line ")
            limit_str = str(limit) if limit else "∞"
            
            print(f"head_limit={limit_str:>3}: {duration:.3f}s, {match_count:3d} matches, {len(result):6d} chars")
        
    except Exception as e:
        print(f"❌ Error in limit tests: {e}")

def test_output_modes():
    """Test different output modes to see performance differences."""
    
    try:
        from abstractllm.tools.common_tools import search_files
        
        pattern = 'Eidolon'
        path = '/Users/albou/projects/mnemosyne/memory'
        
        if not Path(path).exists():
            print(f"❌ Path '{path}' does not exist - skipping output mode tests")
            return
        
        print(f"\n🎯 Testing different output modes:")
        print("-" * 50)
        
        modes = ['count', 'files_with_matches', 'content']
        
        for mode in modes:
            start_time = time.time()
            result = search_files(pattern, path, output_mode=mode, head_limit=10)
            duration = time.time() - start_time
            
            print(f"{mode:>17}: {duration:.3f}s, {len(result):6d} chars")
        
    except Exception as e:
        print(f"❌ Error in output mode tests: {e}")

if __name__ == "__main__":
    print("🚀 Search Performance Test Script")
    print("=" * 60)
    
    # Test basic performance
    duration = test_search_performance()
    
    # If search worked, run additional tests
    if duration is not None:
        test_with_limits()
        test_output_modes()
        
        print(f"\n🎯 Conclusion:")
        print("-" * 30)
        if duration < 2:
            print("The tool itself is FAST. If LLM calls are slow, the bottleneck is:")
            print("- LLM processing the large result text")
            print("- Network communication with LLM")
            print("- LLM parsing/understanding the output")
        else:
            print("The tool itself is slow. Potential optimizations:")
            print("- Use head_limit to reduce output")
            print("- Use 'count' or 'files_with_matches' output modes")
            print("- Search in smaller subdirectories")
            print("- Use more specific file_pattern (e.g., '*.py')")
    else:
        print("\n❌ Could not complete performance test")
