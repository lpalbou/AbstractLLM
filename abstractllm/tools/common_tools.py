"""
Common shareable tools for AbstractLLM applications.

This module provides a collection of utility tools for file operations,
web scraping, command execution, and user interaction.
"""

import os
import json
import subprocess
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import glob
import shutil
from urllib.parse import urljoin, urlparse
import logging
import platform
import re
import time

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from abstractllm.tools.enhanced import tool
    TOOL_DECORATOR_AVAILABLE = True
except ImportError:
    TOOL_DECORATOR_AVAILABLE = False
    # Create a no-op decorator if not available
    def tool(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

# File Operations
@tool(
    description="Find and list files and directories by their names/paths using glob patterns (case-insensitive, supports multiple patterns)",
    tags=["file", "directory", "listing", "filesystem"],
    when_to_use="When you need to find files by their names, paths, or file extensions (NOT for searching file contents)",
    examples=[
        {
            "description": "List all files in current directory",
            "arguments": {
                "directory_path": ".",
                "pattern": "*"
            }
        },
        {
            "description": "Find all Python files recursively",
            "arguments": {
                "directory_path": ".",
                "pattern": "*.py",
                "recursive": True
            }
        },
        {
            "description": "Find all files with 'test' in filename (case-insensitive)",
            "arguments": {
                "directory_path": ".",
                "pattern": "*test*",
                "recursive": True
            }
        },
        {
            "description": "Find multiple file types using | separator",
            "arguments": {
                "directory_path": ".",
                "pattern": "*.py|*.js|*.md",
                "recursive": True
            }
        },
        {
            "description": "Complex multiple patterns - documentation, tests, and config files",
            "arguments": {
                "directory_path": ".",
                "pattern": "README*|*test*|config.*|*.yml",
                "recursive": True
            }
        },
        {
            "description": "List all files including hidden ones",
            "arguments": {
                "directory_path": ".",
                "pattern": "*",
                "include_hidden": True
            }
        }
    ]
)
def list_files(directory_path: str = ".", pattern: str = "*", recursive: bool = False, include_hidden: bool = False, head_limit: Optional[int] = 50) -> str:
    """
    List files and directories in a specified directory with pattern matching (case-insensitive).
    
    IMPORTANT: Use 'directory_path' parameter (not 'file_path') to specify the directory to list.
    
    Args:
        directory_path: Path to the directory to list files from (default: "." for current directory)
        pattern: Glob pattern(s) to match files. Use "|" to separate multiple patterns (default: "*")
        recursive: Whether to search recursively in subdirectories (default: False)
        include_hidden: Whether to include hidden files/directories starting with '.' (default: False)
        head_limit: Maximum number of files to return (default: 50, None for unlimited)
        
    Returns:
        Formatted string with file and directory listings or error message.
        When head_limit is applied, shows "showing X of Y files" in the header.
        
    Examples:
        list_files(directory_path="docs") - Lists files in the docs directory
        list_files(pattern="*.py") - Lists Python files (case-insensitive)
        list_files(pattern="*.py|*.js|*.md") - Lists Python, JavaScript, and Markdown files
        list_files(pattern="README*|*test*|config.*") - Lists README files, test files, and config files
        list_files(pattern="*TEST*", recursive=True) - Finds test files recursively (case-insensitive)
    """
    try:
        directory = Path(directory_path)
        
        if not directory.exists():
            return f"Error: Directory '{directory_path}' does not exist"
        
        if not directory.is_dir():
            return f"Error: '{directory_path}' is not a directory"
        
        # Split pattern by | to support multiple patterns
        patterns = [p.strip() for p in pattern.split('|')]
        
        # Get all files first, then apply case-insensitive pattern matching
        import fnmatch
        all_files = []
        
        if recursive:
            for root, dirs, dir_files in os.walk(directory):
                for f in dir_files:
                    all_files.append(Path(root) / f)
        else:
            try:
                all_files = [f for f in directory.iterdir() if f.is_file()]
                if include_hidden:
                    # Add hidden files
                    hidden_files = [f for f in directory.iterdir() if f.name.startswith('.') and f.is_file()]
                    all_files.extend(hidden_files)
            except PermissionError:
                pass
        
        # Apply case-insensitive pattern matching
        matched_files = []
        for file_path in all_files:
            filename = file_path.name
            
            # Check if file matches any pattern (case-insensitive)
            for single_pattern in patterns:
                if fnmatch.fnmatch(filename.lower(), single_pattern.lower()):
                    matched_files.append(str(file_path))
                    break
        
        files = matched_files
        
        if not files:
            return f"No files found matching pattern '{pattern}' in '{directory_path}'"
        
        # Filter out hidden files if include_hidden is False (already handled in file collection above)
        if not include_hidden:
            filtered_files = []
            for file_path in files:
                path_obj = Path(file_path)
                # Check if any part of the path (after the directory_path) starts with '.'
                relative_path = path_obj.relative_to(directory) if directory != Path('.') else path_obj
                is_hidden = any(part.startswith('.') for part in relative_path.parts)
                if not is_hidden:
                    filtered_files.append(file_path)
            files = filtered_files
        
        if not files:
            hidden_note = " (hidden files excluded)" if not include_hidden else ""
            return f"No files found matching pattern '{pattern}' in '{directory_path}'{hidden_note}"
        
        # Remove duplicates and sort files by modification time (most recent first), then alphabetically
        unique_files = set(files)
        try:
            # Sort by modification time (most recent first) for better relevance
            files = sorted(unique_files, key=lambda f: (Path(f).stat().st_mtime if Path(f).exists() else 0), reverse=True)
        except Exception:
            # Fallback to alphabetical sorting if stat fails
            files = sorted(unique_files)
        
        # Apply head_limit if specified
        total_files = len(files)
        is_truncated = False
        if head_limit is not None and head_limit > 0 and len(files) > head_limit:
            files = files[:head_limit]
            limit_note = f" (showing {head_limit} of {total_files} files)"
            is_truncated = True
        else:
            limit_note = ""
        
        hidden_note = " (hidden files excluded)" if not include_hidden else ""
        output = [f"Files in '{directory_path}' matching '{pattern}'{hidden_note}{limit_note}:"]
        
        for file_path in files:
            path_obj = Path(file_path)
            if path_obj.is_file():
                size = path_obj.stat().st_size
                size_str = f"{size:,} bytes"
                output.append(f"  📄 {path_obj.name} ({size_str})")
            elif path_obj.is_dir():
                output.append(f"  📁 {path_obj.name}/")
        
        # Add helpful hint when results are truncated
        if is_truncated:
            remaining = total_files - head_limit
            recursive_hint = ", recursive=True" if recursive else ""
            hidden_hint = ", include_hidden=True" if include_hidden else ""
            output.append(f"\n💡 {remaining} more files available. Use list_files('{directory_path}', '{pattern}'{recursive_hint}{hidden_hint}, head_limit=None) to see all.")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error listing files: {str(e)}"


@tool(
    description="Search for text patterns INSIDE files using regex (returns file paths with line numbers by default)",
    tags=["search", "content", "regex", "grep", "text"],
    when_to_use="When you need to find specific text, code patterns, or content INSIDE files (NOT for finding files by names)",
    examples=[
        {
            "description": "Find files with function definitions containing 'search'",
            "arguments": {
                "pattern": "def.*search",
                "path": ".",
                "file_pattern": "*.py"
            }
        },
        {
            "description": "Count import statements with 're' module",
            "arguments": {
                "pattern": "import.*re",
                "path": ".",
                "output_mode": "count"
            }
        },
        {
            "description": "Show content for specific patterns (limited results)",
            "arguments": {
                "pattern": "generate.*tools|create_react_cycle",
                "path": "abstractllm/session.py",
                "output_mode": "content",
                "head_limit": 5
            }
        }
    ]
)
def search_files(pattern: str, path: str = ".", output_mode: str = "files_with_matches", head_limit: Optional[int] = 20, file_pattern: str = "*", case_sensitive: bool = False, multiline: bool = False) -> str:
    """
    Enhanced search tool with regex support and flexible output modes.
    
    Similar to grep functionality, this tool can search for patterns in files
    with various output formats and options.
    
    Args:
        pattern: Regular expression pattern to search for
        path: File or directory path to search in (default: current directory)
        output_mode: Output format - "files_with_matches" (show file paths with line numbers), "content" (show matching lines), "count" (show match counts) (default: "files_with_matches")
        head_limit: Limit output to first N entries (default: 20)
        file_pattern: Glob pattern(s) for files to search. Use "|" to separate multiple patterns (default: "*" for all files)
        case_sensitive: Whether search should be case sensitive (default: False)
        multiline: Enable multiline matching where pattern can span lines (default: False)
        
    Returns:
        Search results in the specified format or error message
        
    Examples:
        search_files("generate.*react|create_react_cycle", "abstractllm/session.py")  # Returns file paths with line numbers (default)
        search_files("def.*search", ".", file_pattern="*.py")  # Search Python files only  
        search_files("import.*re", ".", file_pattern="*.py|*.js")  # Search Python and JavaScript files
        search_files("TODO|FIXME", ".", file_pattern="*.py|*.md|*.txt")  # Find TODO/FIXME in multiple file types
        search_files("import.*re", ".", "content", 10)  # Show content with 10 match limit
        search_files("pattern", ".", "count")  # Count matches per file
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
                # Support multiple patterns separated by |
                import fnmatch
                file_patterns = [p.strip() for p in file_pattern.split('|')]
                files_to_search = []
                
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        file_path = Path(root) / file
                        filename = file_path.name
                        
                        # Check if file matches any pattern (case-insensitive)
                        matches_pattern = False
                        for single_pattern in file_patterns:
                            if fnmatch.fnmatch(filename.lower(), single_pattern.lower()):
                                matches_pattern = True
                                break
                        
                        if matches_pattern:
                            # Skip binary files by checking if they're text files
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    f.read(1024)  # Try to read first 1KB
                                files_to_search.append(file_path)
                            except (UnicodeDecodeError, PermissionError):
                                continue  # Skip binary/inaccessible files
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
                                    
                                    # Get the matched text
                                    matched_text = match.group()
                                    # If multiline match, show first line only
                                    if '\n' in matched_text:
                                        matched_text = matched_text.split('\n')[0] + "..."
                                    
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
            total_files_with_matches = len(files_with_matches)
            is_truncated = False
            
            if head_limit and len(files_with_matches) > head_limit:
                files_with_matches = files_with_matches[:head_limit]
                is_truncated = True
            
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
                
                # Add helpful hint when results are truncated
                if is_truncated:
                    remaining = total_files_with_matches - head_limit
                    case_hint = "" if case_sensitive else ", case_sensitive=False"
                    multiline_hint = ", multiline=True" if multiline else ""
                    file_pattern_hint = f", file_pattern='{file_pattern}'" if file_pattern != "*" else ""
                    formatted_results.append(f"\n💡 {remaining} more files with matches available. Use search_files('{pattern}', '{path}', head_limit=None{case_hint}{multiline_hint}{file_pattern_hint}) to see all.")
                
                return "\n".join(formatted_results)
            else:
                return f"No files found matching pattern '{pattern}'"
                
        elif output_mode == "count":
            all_count_items = list(match_counts.items())
            is_count_truncated = False
            
            if head_limit and len(all_count_items) > head_limit:
                count_items = all_count_items[:head_limit]
                is_count_truncated = True
            else:
                count_items = all_count_items
            
            if count_items:
                header = f"Match counts for pattern '{pattern}':"
                count_results = [header]
                for file_path, count in count_items:
                    count_results.append(f"{count:3d} {file_path}")
                count_results.append(f"\nTotal: {total_matches} matches in {len(files_with_matches)} files")
                
                # Add helpful hint when results are truncated
                if is_count_truncated:
                    remaining = len(all_count_items) - head_limit
                    case_hint = "" if case_sensitive else ", case_sensitive=False"
                    multiline_hint = ", multiline=True" if multiline else ""
                    file_pattern_hint = f", file_pattern='{file_pattern}'" if file_pattern != "*" else ""
                    count_results.append(f"\n💡 {remaining} more files with matches available. Use search_files('{pattern}', '{path}', 'count', head_limit=None{case_hint}{multiline_hint}{file_pattern_hint}) to see all.")
                
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


@tool(
    description="Read the contents of a file with optional line range and hidden file access",
    tags=["file", "read", "content", "text"],
    when_to_use="When you need to read file contents, examine code, or extract specific line ranges from files",
    examples=[
        {
            "description": "Read entire file",
            "arguments": {
                "file_path": "README.md"
            }
        },
        {
            "description": "Read specific line range",
            "arguments": {
                "file_path": "src/main.py",
                "should_read_entire_file": False,
                "start_line_one_indexed": 10,
                "end_line_one_indexed_inclusive": 25
            }
        },
        {
            "description": "Read hidden file",
            "arguments": {
                "file_path": ".gitignore",
                "include_hidden": True
            }
        },
        {
            "description": "Read first 50 lines",
            "arguments": {
                "file_path": "large_file.txt",
                "should_read_entire_file": False,
                "end_line_one_indexed_inclusive": 50
            }
        }
    ]
)
def read_file(file_path: str, should_read_entire_file: bool = True, start_line_one_indexed: int = 1, end_line_one_indexed_inclusive: Optional[int] = None, include_hidden: bool = False) -> str:
    """
    Read the contents of a file with optional line range.

    Args:
        file_path: Path to the file to read
        should_read_entire_file: Whether to read the entire file (default: True)
        start_line_one_indexed: Starting line number (1-indexed, default: 1)
        end_line_one_indexed_inclusive: Ending line number (1-indexed, inclusive, optional)
        include_hidden: Whether to allow reading hidden files starting with '.' (default: False)

    Returns:
        File contents or error message
    """
    try:
        path = Path(file_path)

        if not path.exists():
            return f"Error: File '{file_path}' does not exist"

        if not path.is_file():
            return f"Error: '{file_path}' is not a file"

        # Check for hidden files (files starting with '.')
        if not include_hidden and path.name.startswith('.'):
            return f"Error: Access to hidden file '{file_path}' is not allowed. Use include_hidden=True to override."
        
        with open(path, 'r', encoding='utf-8') as f:
            if should_read_entire_file:
                # Read entire file
                content = f.read()
                line_count = len(content.splitlines())
                return f"File: {file_path} ({line_count} lines)\n\n{content}"
            else:
                # Read specific line range
                lines = f.readlines()
                total_lines = len(lines)
                
                # Convert to 0-indexed and validate
                start_idx = max(0, start_line_one_indexed - 1)
                end_idx = min(total_lines, end_line_one_indexed_inclusive or total_lines)
                
                if start_idx >= total_lines:
                    return f"Error: Start line {start_line_one_indexed} exceeds file length ({total_lines} lines)"
                
                selected_lines = lines[start_idx:end_idx]
                
                # Format with line numbers
                result_lines = []
                for i, line in enumerate(selected_lines, start=start_idx + 1):
#                    result_lines.append(f"{i:4d}: {line.rstrip()}")
                    result_lines.append(f"{line.rstrip()}")
               
                return "\n".join(result_lines)
                
    except UnicodeDecodeError:
        return f"Error: Cannot read '{file_path}' - file appears to be binary"
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except PermissionError:
        return f"Error: Permission denied reading file: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool(
    description="Write content to a file with robust error handling, creating directories if needed",
    tags=["file", "write", "create", "append", "content", "output"],
    when_to_use="When you need to create new files, save content, or append to existing files",
    examples=[
        {
            "description": "Write a simple text file",
            "arguments": {
                "file_path": "output.txt",
                "content": "Hello, world!"
            }
        },
        {
            "description": "Create a Python script",
            "arguments": {
                "file_path": "script.py",
                "content": "#!/usr/bin/env python3\nprint('Hello from Python!')"
            }
        },
        {
            "description": "Append to existing file",
            "arguments": {
                "file_path": "log.txt",
                "content": "\nNew log entry at 2025-01-01",
                "mode": "a"
            }
        },
        {
            "description": "Create file in nested directory",
            "arguments": {
                "file_path": "docs/api/endpoints.md",
                "content": "# API Endpoints\n\n## Authentication\n..."
            }
        },
        {
            "description": "Write JSON data",
            "arguments": {
                "file_path": "config.json",
                "content": "{\n  \"api_key\": \"test\",\n  \"debug\": true\n}"
            }
        }
    ]
)
def write_file(file_path: str, content: str = "", mode: str = "w", create_dirs: bool = True) -> str:
    """
    Write content to a file with robust error handling.

    This tool creates or overwrites a file with the specified content.
    It can optionally create parent directories if they don't exist.

    Args:
        file_path: Path to the file to write (relative or absolute)
        content: The content to write to the file (default: empty string)
        mode: Write mode - "w" to overwrite, "a" to append (default: "w")
        create_dirs: Whether to create parent directories if they don't exist (default: True)

    Returns:
        Success message with file information

    Raises:
        PermissionError: If lacking write permissions
        OSError: If there are filesystem issues
    """
    try:
        # Convert to Path object for better handling
        path = Path(file_path)

        # Create parent directories if requested and they don't exist
        if create_dirs and path.parent != path:
            path.parent.mkdir(parents=True, exist_ok=True)

        # Write the content to the file
        with open(path, mode, encoding='utf-8') as f:
            f.write(content)

        # Get file size for confirmation
        file_size = path.stat().st_size

        # Enhanced success message with emoji and formatting
        action = "appended to" if mode == "a" else "written to"
        return f"✅ Successfully {action} '{file_path}' ({file_size:,} bytes)"

    except PermissionError:
        return f"❌ Permission denied: Cannot write to '{file_path}'"
    except FileNotFoundError:
        return f"❌ Directory not found: Parent directory of '{file_path}' does not exist"
    except OSError as e:
        return f"❌ File system error: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error writing file: {str(e)}"


def update_file(file_path: str, old_text: str, new_text: str, max_replacements: int = -1) -> str:
    """
    Update a file by replacing text.
    
    Args:
        file_path: Path to the file to update
        old_text: Text to replace
        new_text: Replacement text
        max_replacements: Maximum number of replacements (-1 for unlimited)
        
    Returns:
        Success message with replacement count or error message
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            return f"Error: File '{file_path}' does not exist"
        
        # Read current content
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count potential replacements
        count = content.count(old_text)
        if count == 0:
            return f"No occurrences of '{old_text}' found in '{file_path}'"
        
        # Perform replacement
        if max_replacements == -1:
            updated_content = content.replace(old_text, new_text)
            replacements_made = count
        else:
            updated_content = content.replace(old_text, new_text, max_replacements)
            replacements_made = min(count, max_replacements)
        
        # Write back to file
        with open(path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        return f"Successfully updated '{file_path}': {replacements_made} replacement(s) made"
        
    except Exception as e:
        return f"Error updating file: {str(e)}"


@tool(
    description="Advanced file editing with line-based operations, multi-operation support, and atomic transactions",
    tags=["file", "edit", "modify", "lines", "insert", "delete", "replace", "patch"],
    when_to_use="When you need to perform precise file modifications like inserting/deleting lines, replacing specific content, or making multiple coordinated edits",
    examples=[
        {
            "description": "Insert lines at specific position",
            "arguments": {
                "file_path": "config.py",
                "operation": "insert",
                "line_number": 10,
                "content": "# New configuration option\nDEBUG = True"
            }
        },
        {
            "description": "Delete specific lines",
            "arguments": {
                "file_path": "old_code.py",
                "operation": "delete",
                "start_line": 5,
                "end_line": 8
            }
        },
        {
            "description": "Replace lines with new content",
            "arguments": {
                "file_path": "script.py",
                "operation": "replace",
                "start_line": 15,
                "end_line": 17,
                "content": "def improved_function():\n    return 'better implementation'"
            }
        },
        {
            "description": "Multiple operations in sequence",
            "arguments": {
                "file_path": "refactor.py",
                "operation": "multi",
                "operations": [
                    {"type": "replace", "start_line": 1, "end_line": 1, "content": "#!/usr/bin/env python3"},
                    {"type": "insert", "line_number": 10, "content": "import logging"},
                    {"type": "delete", "start_line": 20, "end_line": 22}
                ]
            }
        },
        {
            "description": "Preview changes without applying",
            "arguments": {
                "file_path": "test.py",
                "operation": "replace",
                "start_line": 5,
                "end_line": 7,
                "content": "new code here",
                "preview_only": True
            }
        }
    ]
)
def edit_file(
    file_path: str,
    operation: str,
    content: str = "",
    line_number: Optional[int] = None,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    operations: Optional[List[Dict[str, Any]]] = None,
    create_backup: bool = True,
    preview_only: bool = False,
    encoding: str = "utf-8"
) -> str:
    """
    Advanced file editing with multiple operation modes and safety features.
    
    This tool provides sophisticated file editing capabilities with atomic operations,
    backup creation, and preview functionality. It supports line-based operations
    for precise modifications without rewriting entire files.
    
    Args:
        file_path: Path to the file to edit
        operation: Type of operation - "insert", "delete", "replace", "multi", "transform"
        content: Content to insert or replace (for insert/replace operations)
        line_number: Specific line number for insert operations (1-indexed)
        start_line: Starting line number for range operations (1-indexed, inclusive)
        end_line: Ending line number for range operations (1-indexed, inclusive)
        operations: List of operations for "multi" mode
        create_backup: Whether to create a backup before editing (default: True)
        preview_only: Show what would be changed without applying (default: False)
        encoding: File encoding (default: "utf-8")
        
    Returns:
        Detailed results of the editing operation with change summary
        
    Operation Types:
        - "insert": Insert content at a specific line number
        - "delete": Delete line(s) from start_line to end_line
        - "replace": Replace line(s) from start_line to end_line with new content
        - "multi": Perform multiple operations in sequence
        - "transform": Apply a function to each line (advanced)
        
    Multi-operation format:
        operations = [
            {"type": "insert", "line_number": 5, "content": "new line"},
            {"type": "delete", "start_line": 10, "end_line": 12},
            {"type": "replace", "start_line": 20, "end_line": 20, "content": "replacement"}
        ]
    """
    try:
        # Validate file exists
        path = Path(file_path)
        if not path.exists():
            return f"❌ File not found: {file_path}"
        
        if not path.is_file():
            return f"❌ Path is not a file: {file_path}"
        
        # Read current content
        try:
            with open(path, 'r', encoding=encoding) as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            return f"❌ Cannot decode file with encoding '{encoding}'. File may be binary."
        except Exception as e:
            return f"❌ Error reading file: {str(e)}"
        
        original_line_count = len(lines)
        
        # Create backup if requested
        backup_path = None
        if create_backup and not preview_only:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = f"{file_path}.backup_{timestamp}"
            try:
                shutil.copy2(file_path, backup_path)
            except Exception as e:
                return f"❌ Failed to create backup: {str(e)}"
        
        # Process operations
        modified_lines = lines.copy()
        changes_made = []
        
        if operation == "insert":
            if line_number is None:
                return "❌ line_number required for insert operation"
            
            # Validate line number
            if line_number < 1 or line_number > len(modified_lines) + 1:
                return f"❌ Invalid line_number: {line_number}. File has {len(modified_lines)} lines."
            
            # Prepare content with proper line endings
            insert_content = content.rstrip('\n') + '\n' if content and not content.endswith('\n') else content
            insert_lines = insert_content.splitlines(keepends=True) if insert_content else ['']
            
            # Insert at specified position (convert to 0-indexed)
            insert_idx = line_number - 1
            modified_lines[insert_idx:insert_idx] = insert_lines
            
            changes_made.append(f"Inserted {len(insert_lines)} line(s) at line {line_number}")
            
        elif operation == "delete":
            if start_line is None:
                return "❌ start_line required for delete operation"
            
            end_line = end_line or start_line  # Default to single line
            
            # Validate line range
            if start_line < 1 or start_line > len(modified_lines):
                return f"❌ Invalid start_line: {start_line}. File has {len(modified_lines)} lines."
            if end_line < start_line or end_line > len(modified_lines):
                return f"❌ Invalid end_line: {end_line}. Must be >= start_line and <= {len(modified_lines)}."
            
            # Delete lines (convert to 0-indexed)
            start_idx = start_line - 1
            end_idx = end_line
            deleted_count = end_idx - start_idx
            
            del modified_lines[start_idx:end_idx]
            
            if deleted_count == 1:
                changes_made.append(f"Deleted line {start_line}")
            else:
                changes_made.append(f"Deleted lines {start_line}-{end_line} ({deleted_count} lines)")
                
        elif operation == "replace":
            if start_line is None:
                return "❌ start_line required for replace operation"
            
            end_line = end_line or start_line  # Default to single line
            
            # Validate line range
            if start_line < 1 or start_line > len(modified_lines):
                return f"❌ Invalid start_line: {start_line}. File has {len(modified_lines)} lines."
            if end_line < start_line or end_line > len(modified_lines):
                return f"❌ Invalid end_line: {end_line}. Must be >= start_line and <= {len(modified_lines)}."
            
            # Prepare replacement content
            replace_content = content.rstrip('\n') + '\n' if content and not content.endswith('\n') else content
            replace_lines = replace_content.splitlines(keepends=True) if replace_content else ['']
            
            # Replace lines (convert to 0-indexed)
            start_idx = start_line - 1
            end_idx = end_line
            replaced_count = end_idx - start_idx
            
            modified_lines[start_idx:end_idx] = replace_lines
            
            if replaced_count == 1:
                changes_made.append(f"Replaced line {start_line} with {len(replace_lines)} line(s)")
            else:
                changes_made.append(f"Replaced lines {start_line}-{end_line} ({replaced_count} lines) with {len(replace_lines)} line(s)")
                
        elif operation == "multi":
            if not operations:
                return "❌ operations list required for multi operation"
            
            # Sort operations by line number (descending) to avoid index shifting issues
            sorted_ops = []
            for i, op in enumerate(operations):
                op_type = op.get("type")
                if op_type in ["insert"]:
                    sort_key = op.get("line_number", 0)
                elif op_type in ["delete", "replace"]:
                    sort_key = op.get("start_line", 0)
                else:
                    return f"❌ Invalid operation type in operation {i}: {op_type}"
                sorted_ops.append((sort_key, op))
            
            # Sort in descending order to process from bottom to top
            sorted_ops.sort(key=lambda x: x[0], reverse=True)
            
            for line_num, op in sorted_ops:
                op_type = op.get("type")
                
                if op_type == "insert":
                    line_num = op.get("line_number")
                    op_content = op.get("content", "")
                    
                    if line_num < 1 or line_num > len(modified_lines) + 1:
                        return f"❌ Invalid line_number in insert operation: {line_num}"
                    
                    insert_content = op_content.rstrip('\n') + '\n' if op_content and not op_content.endswith('\n') else op_content
                    insert_lines = insert_content.splitlines(keepends=True) if insert_content else ['']
                    
                    insert_idx = line_num - 1
                    modified_lines[insert_idx:insert_idx] = insert_lines
                    changes_made.append(f"Inserted {len(insert_lines)} line(s) at line {line_num}")
                    
                elif op_type == "delete":
                    start = op.get("start_line")
                    end = op.get("end_line", start)
                    
                    if start < 1 or start > len(modified_lines) or end < start or end > len(modified_lines):
                        return f"❌ Invalid line range in delete operation: {start}-{end}"
                    
                    start_idx = start - 1
                    end_idx = end
                    deleted_count = end_idx - start_idx
                    
                    del modified_lines[start_idx:end_idx]
                    
                    if deleted_count == 1:
                        changes_made.append(f"Deleted line {start}")
                    else:
                        changes_made.append(f"Deleted lines {start}-{end} ({deleted_count} lines)")
                        
                elif op_type == "replace":
                    start = op.get("start_line")
                    end = op.get("end_line", start)
                    op_content = op.get("content", "")
                    
                    if start < 1 or start > len(modified_lines) or end < start or end > len(modified_lines):
                        return f"❌ Invalid line range in replace operation: {start}-{end}"
                    
                    replace_content = op_content.rstrip('\n') + '\n' if op_content and not op_content.endswith('\n') else op_content
                    replace_lines = replace_content.splitlines(keepends=True) if replace_content else ['']
                    
                    start_idx = start - 1
                    end_idx = end
                    replaced_count = end_idx - start_idx
                    
                    modified_lines[start_idx:end_idx] = replace_lines
                    
                    if replaced_count == 1:
                        changes_made.append(f"Replaced line {start} with {len(replace_lines)} line(s)")
                    else:
                        changes_made.append(f"Replaced lines {start}-{end} ({replaced_count} lines) with {len(replace_lines)} line(s)")
        
        else:
            return f"❌ Unknown operation: {operation}. Available: insert, delete, replace, multi"
        
        # Generate results
        final_line_count = len(modified_lines)
        line_diff = final_line_count - original_line_count
        
        results = []
        
        if preview_only:
            results.append(f"🔍 Preview Mode - Changes NOT Applied")
            results.append(f"  • File: {file_path}")
            results.append(f"  • Original lines: {original_line_count}")
            results.append(f"  • Final lines: {final_line_count}")
            if line_diff != 0:
                sign = "+" if line_diff > 0 else ""
                results.append(f"  • Line difference: {sign}{line_diff}")
            
            results.append(f"\n📝 Changes that would be made:")
            for change in changes_made:
                results.append(f"  • {change}")
            
            # Show preview of affected areas
            if operation != "multi":
                if operation == "insert" and line_number:
                    start_preview = max(1, line_number - 2)
                    end_preview = min(len(modified_lines), line_number + 3)
                    results.append(f"\n📄 Preview around line {line_number}:")
                    
                    for i in range(start_preview - 1, end_preview):
                        line_num = i + 1
                        line_content = modified_lines[i].rstrip() if i < len(modified_lines) else ""
                        marker = ">>>" if line_num == line_number else "   "
                        results.append(f"  {marker} {line_num:3d}: {line_content}")
            
            return "\n".join(results)
        
        # Apply changes to file
        try:
            with open(path, 'w', encoding=encoding) as f:
                f.writelines(modified_lines)
        except Exception as e:
            # Restore backup if write fails
            if backup_path and os.path.exists(backup_path):
                shutil.copy2(backup_path, file_path)
                return f"❌ Write failed, backup restored: {str(e)}"
            return f"❌ Write failed: {str(e)}"
        
        # Success message
        results.append(f"✅ File edited successfully: {file_path}")
        results.append(f"  • Original lines: {original_line_count}")
        results.append(f"  • Final lines: {final_line_count}")
        
        if line_diff != 0:
            sign = "+" if line_diff > 0 else ""
            results.append(f"  • Line difference: {sign}{line_diff}")
        
        if backup_path:
            results.append(f"  • Backup created: {backup_path}")
        
        results.append(f"\n📝 Changes applied:")
        for change in changes_made:
            results.append(f"  • {change}")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"❌ Error in file editing: {str(e)}"


# System Operations
def execute_command(command: str, working_directory: str = ".", timeout: int = 30) -> str:
    """
    Execute a local command safely with timeout.
    
    Args:
        command: Command to execute
        working_directory: Directory to run command in (default: current directory)
        timeout: Timeout in seconds (default: 30)
        
    Returns:
        Command output or error message
    """
    try:
        # Security check - block dangerous commands
        dangerous_patterns = ['rm -rf', 'format', 'del /f', 'shutdown', 'reboot', 'halt']
        command_lower = command.lower()
        
        for pattern in dangerous_patterns:
            if pattern in command_lower:
                return f"Error: Command blocked for security reasons: '{command}'"
        
        # Ensure working directory exists
        wd_path = Path(working_directory)
        if not wd_path.exists():
            return f"Error: Working directory '{working_directory}' does not exist"
        
        # Execute command
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_directory,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = []
        output.append(f"Command: {command}")
        output.append(f"Working directory: {working_directory}")
        output.append(f"Exit code: {result.returncode}")
        
        if result.stdout:
            output.append(f"\nSTDOUT:\n{result.stdout}")
        
        if result.stderr:
            output.append(f"\nSTDERR:\n{result.stderr}")
        
        return "\n".join(output)
        
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds: '{command}'"
    except Exception as e:
        return f"Error executing command: {str(e)}"


# Web Operations
@tool(
    description="Search the web for real-time information using DuckDuckGo (no API key required)",
    tags=["web", "search", "internet", "information", "research"],
    when_to_use="When you need current information, research topics, or verify facts that might not be in your training data",
    examples=[
        {
            "description": "Search for current programming best practices",
            "arguments": {
                "query": "python best practices 2025",
                "num_results": 5
            }
        },
        {
            "description": "Research a technology or framework",
            "arguments": {
                "query": "semantic search embedding models comparison",
                "num_results": 3
            }
        },
        {
            "description": "Get current news or events",
            "arguments": {
                "query": "AI developments 2025"
            }
        },
        {
            "description": "Find documentation or tutorials",
            "arguments": {
                "query": "LanceDB vector database tutorial",
                "num_results": 4
            }
        },
        {
            "description": "Search with strict content filtering",
            "arguments": {
                "query": "machine learning basics",
                "safe_search": "strict"
            }
        },
        {
            "description": "Get UK-specific results",
            "arguments": {
                "query": "data protection regulations",
                "region": "uk-en"
            }
        }
    ]
)
def web_search(query: str, num_results: int = 5, safe_search: str = "moderate", region: str = "us-en") -> str:
    """
    Search the internet using DuckDuckGo (no API key required).
    
    Args:
        query: Search query
        num_results: Number of results to return (default: 5)
        safe_search: Content filtering level - "strict", "moderate", or "off" (default: "moderate")
        region: Regional results preference - "us-en", "uk-en", "ca-en", "au-en", etc. (default: "us-en")
        
    Returns:
        Search results or error message
        
    Note:
        DuckDuckGo Instant Answer API does not support time range filtering.
        For time-specific searches, include date terms in your query (e.g., "python best practices 2025").
    """
    try:
        # Simple DuckDuckGo instant answer API
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1',
            'no_redirect': '1',  # Faster responses
            'safe_search': safe_search,
            'region': region
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        results.append(f"Search results for: '{query}'")
        
        # Abstract (main result)
        if data.get('Abstract'):
            results.append(f"\n📝 Summary: {data['Abstract']}")
            if data.get('AbstractURL'):
                results.append(f"Source: {data['AbstractURL']}")
        
        # Related topics
        if data.get('RelatedTopics'):
            results.append(f"\n🔗 Related Topics:")
            for i, topic in enumerate(data['RelatedTopics'][:num_results], 1):
                if isinstance(topic, dict) and 'Text' in topic:
                    text = topic['Text'][:200] + "..." if len(topic['Text']) > 200 else topic['Text']
                    results.append(f"{i}. {text}")
                    if 'FirstURL' in topic:
                        results.append(f"   URL: {topic['FirstURL']}")
        
        # Answer (if available)
        if data.get('Answer'):
            results.append(f"\n💡 Direct Answer: {data['Answer']}")
        
        if len(results) == 1:  # Only the header
            results.append("\nNo detailed results found. Try a more specific query.")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Error searching internet: {str(e)}"


def fetch_url(url: str, timeout: int = 10) -> str:
    """
    Fetch content from a URL.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds (default: 10)
        
    Returns:
        URL content or error message
    """
    try:
        # Basic URL validation
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return f"Error: Invalid URL format: '{url}'"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; AbstractLLM-Tool/1.0)'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        
        results = []
        results.append(f"URL: {url}")
        results.append(f"Status: {response.status_code}")
        results.append(f"Content-Type: {content_type}")
        results.append(f"Content-Length: {len(response.content):,} bytes")
        results.append("")
        
        # Handle different content types
        if 'text/' in content_type or 'application/json' in content_type:
            # Return FULL VERBATIM content - NO TRUNCATION
            results.append(response.text)
        else:
            results.append(f"[Binary content - {content_type}]")
        
        return "\n".join(results)
        
    except requests.exceptions.Timeout:
        return f"Error: Request timed out after {timeout} seconds"
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def fetch_and_parse_html(url: str, extract_text: bool = True, extract_links: bool = False) -> str:
    """
    Fetch and parse HTML content from a URL.
    
    Args:
        url: URL to fetch and parse
        extract_text: Whether to extract readable text (default: True)
        extract_links: Whether to extract links (default: False)
        
    Returns:
        Parsed HTML content or error message
    """
    try:
        if not BS4_AVAILABLE:
            return "Error: BeautifulSoup4 is required for HTML parsing. Install with: pip install beautifulsoup4"
        
        # Fetch the content first
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; AbstractLLM-Tool/1.0)'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        if 'text/html' not in response.headers.get('content-type', '').lower():
            return f"Error: URL does not return HTML content: {response.headers.get('content-type')}"
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        results = []
        results.append(f"Parsed HTML from: {url}")
        
        # Extract title
        title = soup.find('title')
        if title:
            results.append(f"Title: {title.get_text().strip()}")
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            results.append(f"Description: {meta_desc['content']}")
        
        results.append("")
        
        if extract_text:
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Return FULL VERBATIM text - NO TRUNCATION
            
            results.append("Extracted Text:")
            results.append(text)
        
        if extract_links:
            results.append("\nExtracted Links:")
            links = soup.find_all('a', href=True)
            for i, link in enumerate(links, 1):  # ALL links - NO TRUNCATION
                href = link['href']
                link_text = link.get_text().strip()
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    href = urljoin(url, href)
                
                results.append(f"{i}. {link_text} - {href}")
            
            # Show ALL links - NO "more links" truncation message
        
        return "\n".join(results)
        
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {str(e)}"
    except Exception as e:
        return f"Error parsing HTML: {str(e)}"


# User Interaction
def ask_user_multiple_choice(question: str, choices: List[str], allow_multiple: bool = False) -> str:
    """
    Ask the user a multiple choice question.
    
    Args:
        question: The question to ask
        choices: List of choice options
        allow_multiple: Whether to allow multiple selections (default: False)
        
    Returns:
        User's selection(s) or error message
    """
    try:
        if not choices:
            return "Error: No choices provided"
        
        if len(choices) > 26:
            return "Error: Too many choices (maximum 26 supported)"
        
        # Display question and choices
        print(f"\n❓ {question}")
        print("\nChoices:")
        
        choice_map = {}
        for i, choice in enumerate(choices):
            letter = chr(ord('a') + i)
            choice_map[letter] = choice
            print(f"  {letter}) {choice}")
        
        # Get user input
        if allow_multiple:
            prompt = f"\nSelect one or more choices (e.g., 'a', 'a,c', 'a c'): "
        else:
            prompt = f"\nSelect a choice (a-{chr(ord('a') + len(choices) - 1)}): "
        
        user_input = input(prompt).strip().lower()
        
        if not user_input:
            return "Error: No selection made"
        
        # Parse selections
        if allow_multiple:
            # Handle comma-separated or space-separated input
            selections = []
            for char in user_input.replace(',', ' ').split():
                char = char.strip()
                if len(char) == 1 and char in choice_map:
                    selections.append(char)
                else:
                    return f"Error: Invalid choice '{char}'"
            
            if not selections:
                return "Error: No valid selections found"
            
            selected_choices = [choice_map[sel] for sel in selections]
            return f"Selected: {', '.join(selected_choices)}"
        
        else:
            # Single selection
            if len(user_input) == 1 and user_input in choice_map:
                return f"Selected: {choice_map[user_input]}"
            else:
                return f"Error: Invalid choice '{user_input}'"
    
    except KeyboardInterrupt:
        return "User cancelled the selection"
    except Exception as e:
        return f"Error asking user: {str(e)}"


# System Monitoring Tools (using psutil)
def get_system_info() -> str:
    """
    Get comprehensive system information including OS, hardware, and Python environment.
    
    Returns:
        Formatted system information or error message
    """
    try:
        if not PSUTIL_AVAILABLE:
            return "Error: psutil is required for system monitoring. Install with: pip install psutil"
        
        results = []
        results.append("🖥️  System Information")
        results.append("=" * 50)
        
        # Basic system info
        uname = platform.uname()
        results.append(f"System: {uname.system}")
        results.append(f"Node Name: {uname.node}")
        results.append(f"Release: {uname.release}")
        results.append(f"Version: {uname.version}")
        results.append(f"Machine: {uname.machine}")
        results.append(f"Processor: {uname.processor}")
        
        # Python info
        import sys
        results.append(f"\nPython Version: {sys.version}")
        results.append(f"Python Executable: {sys.executable}")
        
        # CPU info
        results.append(f"\n💻 CPU Information")
        results.append(f"Physical Cores: {psutil.cpu_count(logical=False)}")
        results.append(f"Total Cores: {psutil.cpu_count(logical=True)}")
        
        # Get CPU frequency if available
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                results.append(f"Max Frequency: {cpu_freq.max:.2f} MHz")
                results.append(f"Current Frequency: {cpu_freq.current:.2f} MHz")
        except:
            pass
        
        # Memory info
        memory = psutil.virtual_memory()
        results.append(f"\n🧠 Memory Information")
        results.append(f"Total RAM: {memory.total / (1024**3):.2f} GB")
        results.append(f"Available RAM: {memory.available / (1024**3):.2f} GB")
        results.append(f"Used RAM: {memory.used / (1024**3):.2f} GB")
        results.append(f"Memory Usage: {memory.percent}%")
        
        # Disk info
        disk = psutil.disk_usage('/')
        results.append(f"\n💾 Disk Information")
        results.append(f"Total Disk Space: {disk.total / (1024**3):.2f} GB")
        results.append(f"Used Disk Space: {disk.used / (1024**3):.2f} GB")
        results.append(f"Free Disk Space: {disk.free / (1024**3):.2f} GB")
        results.append(f"Disk Usage: {(disk.used / disk.total) * 100:.1f}%")
        
        # Boot time
        import datetime
        boot_time = datetime.datetime.fromtimestamp(psutil.boot_time())
        results.append(f"\n⏰ System Boot Time: {boot_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Error getting system information: {str(e)}"


def get_performance_stats() -> str:
    """
    Get current system performance statistics including CPU, memory, and disk usage.
    
    Returns:
        Current performance metrics or error message
    """
    try:
        if not PSUTIL_AVAILABLE:
            return "Error: psutil is required for performance monitoring. Install with: pip install psutil"
        
        results = []
        results.append("📊 Current Performance Statistics")
        results.append("=" * 50)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
        results.append(f"Overall CPU Usage: {cpu_percent}%")
        results.append("Per-Core CPU Usage:")
        for i, percentage in enumerate(cpu_per_core):
            results.append(f"  Core {i}: {percentage}%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        results.append(f"\n🧠 Memory Usage:")
        results.append(f"  RAM: {memory.percent}% ({memory.used / (1024**3):.2f} GB / {memory.total / (1024**3):.2f} GB)")
        results.append(f"  Swap: {swap.percent}% ({swap.used / (1024**3):.2f} GB / {swap.total / (1024**3):.2f} GB)")
        
        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                results.append(f"\n💾 Disk I/O:")
                results.append(f"  Read: {disk_io.read_bytes / (1024**3):.2f} GB")
                results.append(f"  Write: {disk_io.write_bytes / (1024**3):.2f} GB")
                results.append(f"  Read Count: {disk_io.read_count:,}")
                results.append(f"  Write Count: {disk_io.write_count:,}")
        except:
            pass
        
        # Network I/O
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                results.append(f"\n🌐 Network I/O:")
                results.append(f"  Bytes Sent: {net_io.bytes_sent / (1024**3):.2f} GB")
                results.append(f"  Bytes Received: {net_io.bytes_recv / (1024**3):.2f} GB")
                results.append(f"  Packets Sent: {net_io.packets_sent:,}")
                results.append(f"  Packets Received: {net_io.packets_recv:,}")
        except:
            pass
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Error getting performance statistics: {str(e)}"


def get_running_processes(limit: int = 10, sort_by: str = "cpu") -> str:
    """
    Get information about currently running processes.
    
    Args:
        limit: Maximum number of processes to show (default: 10)
        sort_by: Sort by 'cpu', 'memory', or 'name' (default: 'cpu')
        
    Returns:
        Information about running processes or error message
    """
    try:
        if not PSUTIL_AVAILABLE:
            return "Error: psutil is required for process monitoring. Install with: pip install psutil"
        
        results = []
        results.append(f"🔄 Top {limit} Running Processes (sorted by {sort_by})")
        results.append("=" * 60)
        
        # Get all processes
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info']):
            try:
                pinfo = proc.info
                pinfo['memory_mb'] = pinfo['memory_info'].rss / (1024 * 1024) if pinfo['memory_info'] else 0
                processes.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        # Sort processes
        if sort_by == "cpu":
            processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
        elif sort_by == "memory":
            processes.sort(key=lambda x: x['memory_percent'] or 0, reverse=True)
        elif sort_by == "name":
            processes.sort(key=lambda x: x['name'] or '')
        else:
            return f"Error: Invalid sort_by value. Use 'cpu', 'memory', or 'name'"
        
        # Display header
        results.append(f"{'PID':<8} {'Name':<25} {'CPU%':<8} {'Memory%':<10} {'Memory (MB)':<12}")
        results.append("-" * 70)
        
        # Display processes
        for proc in processes[:limit]:
            pid = proc['pid']
            name = (proc['name'] or 'Unknown')[:24]
            cpu = f"{proc['cpu_percent'] or 0:.1f}%"
            mem_percent = f"{proc['memory_percent'] or 0:.1f}%"
            mem_mb = f"{proc['memory_mb']:.1f}"
            
            results.append(f"{pid:<8} {name:<25} {cpu:<8} {mem_percent:<10} {mem_mb:<12}")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Error getting process information: {str(e)}"


def get_network_connections(limit: int = 10) -> str:
    """
    Get information about active network connections.
    
    Args:
        limit: Maximum number of connections to show (default: 10)
        
    Returns:
        Network connection information or error message
    """
    try:
        if not PSUTIL_AVAILABLE:
            return "Error: psutil is required for network monitoring. Install with: pip install psutil"
        
        results = []
        results.append(f"🌐 Active Network Connections (top {limit})")
        results.append("=" * 80)
        
        # Get network connections
        connections = psutil.net_connections(kind='inet')
        
        if not connections:
            return "No active network connections found"
        
        # Display header
        results.append(f"{'Local Address':<25} {'Remote Address':<25} {'Status':<12} {'PID':<8}")
        results.append("-" * 80)
        
        # Display connections
        count = 0
        for conn in connections:
            if count >= limit:
                break
                
            local_addr = f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "N/A"
            remote_addr = f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "N/A"
            status = conn.status or "N/A"
            pid = str(conn.pid) if conn.pid else "N/A"
            
            results.append(f"{local_addr:<25} {remote_addr:<25} {status:<12} {pid:<8}")
            count += 1
        
        if len(connections) > limit:
            results.append(f"\n... and {len(connections) - limit} more connections")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Error getting network connections: {str(e)}"


def get_disk_partitions() -> str:
    """
    Get information about disk partitions and usage.
    
    Returns:
        Disk partition information or error message
    """
    try:
        if not PSUTIL_AVAILABLE:
            return "Error: psutil is required for disk monitoring. Install with: pip install psutil"
        
        results = []
        results.append("💾 Disk Partitions and Usage")
        results.append("=" * 60)
        
        # Get disk partitions
        partitions = psutil.disk_partitions()
        
        for partition in partitions:
            results.append(f"\nDevice: {partition.device}")
            results.append(f"Mountpoint: {partition.mountpoint}")
            results.append(f"File System: {partition.fstype}")
            
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                results.append(f"  Total Size: {usage.total / (1024**3):.2f} GB")
                results.append(f"  Used: {usage.used / (1024**3):.2f} GB")
                results.append(f"  Free: {usage.free / (1024**3):.2f} GB")
                results.append(f"  Usage: {(usage.used / usage.total) * 100:.1f}%")
            except PermissionError:
                results.append("  [Permission Denied]")
            except Exception as e:
                results.append(f"  [Error: {str(e)}]")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Error getting disk partitions: {str(e)}"


def monitor_resource_usage(duration: int = 5) -> str:
    """
    Monitor system resource usage over a specified duration.
    
    Args:
        duration: Duration in seconds to monitor (default: 5)
        
    Returns:
        Resource usage monitoring report or error message
    """
    try:
        if not PSUTIL_AVAILABLE:
            return "Error: psutil is required for resource monitoring. Install with: pip install psutil"
        
        if duration < 1 or duration > 60:
            return "Error: Duration must be between 1 and 60 seconds"
        
        results = []
        results.append(f"📈 Resource Usage Monitoring ({duration} seconds)")
        results.append("=" * 50)
        
        import time
        
        # Get initial readings
        initial_cpu = psutil.cpu_percent()
        initial_memory = psutil.virtual_memory().percent
        initial_disk_io = psutil.disk_io_counters()
        initial_net_io = psutil.net_io_counters()
        
        # Wait for the monitoring duration
        time.sleep(duration)
        
        # Get final readings
        final_cpu = psutil.cpu_percent()
        final_memory = psutil.virtual_memory().percent
        final_disk_io = psutil.disk_io_counters()
        final_net_io = psutil.net_io_counters()
        
        # Calculate averages and changes
        avg_cpu = (initial_cpu + final_cpu) / 2
        avg_memory = (initial_memory + final_memory) / 2
        
        results.append(f"Average CPU Usage: {avg_cpu:.1f}%")
        results.append(f"Average Memory Usage: {avg_memory:.1f}%")
        
        # Disk I/O changes
        if initial_disk_io and final_disk_io:
            read_bytes = final_disk_io.read_bytes - initial_disk_io.read_bytes
            write_bytes = final_disk_io.write_bytes - initial_disk_io.write_bytes
            results.append(f"\nDisk I/O during monitoring:")
            results.append(f"  Data Read: {read_bytes / (1024**2):.2f} MB")
            results.append(f"  Data Written: {write_bytes / (1024**2):.2f} MB")
        
        # Network I/O changes
        if initial_net_io and final_net_io:
            sent_bytes = final_net_io.bytes_sent - initial_net_io.bytes_sent
            recv_bytes = final_net_io.bytes_recv - initial_net_io.bytes_recv
            results.append(f"\nNetwork I/O during monitoring:")
            results.append(f"  Data Sent: {sent_bytes / (1024**2):.2f} MB")
            results.append(f"  Data Received: {recv_bytes / (1024**2):.2f} MB")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Error monitoring resource usage: {str(e)}"


# Export all tools for easy importing
__all__ = [
    'list_files',
    'search_files',
    'read_file',
    'write_file',
    'update_file',
    'edit_file',
    'execute_command',
    'web_search',
    'fetch_url',
    'fetch_and_parse_html',
    'ask_user_multiple_choice',
    'get_system_info',
    'get_performance_stats',
    'get_running_processes',
    'get_network_connections',
    'get_disk_partitions',
    'monitor_resource_usage'
] 