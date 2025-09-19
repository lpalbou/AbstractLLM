# Enhanced Search Tool Usage Examples

The new `search_files` function in `abstractllm/tools/common_tools.py` provides powerful grep-like functionality with regex support and flexible output modes.

## Function Signature

```python
def search_files(pattern: str, path: str = ".", output_mode: str = "files_with_matches", head_limit: Optional[int] = 20, file_pattern: str = "*", case_sensitive: bool = False, multiline: bool = False) -> str
```

## Parameters

- **pattern**: Regular expression pattern to search for
- **path**: File or directory path to search in (default: current directory)
- **output_mode**: Output format - "files_with_matches" (show only file paths), "content" (show matching lines), "count" (show match counts) (default: "files_with_matches")
- **head_limit**: Limit output to first N entries (default: 20)
- **file_pattern**: Glob pattern for files to search (default: "*" for all files)
- **case_sensitive**: Whether search should be case sensitive (default: False)
- **multiline**: Enable multiline matching where pattern can span lines (default: False)

## Usage Examples

### 1. Basic Usage (New Defaults)
```python
search_files("generate.*react|create_react_cycle|generate.*with.*tools", "abstractllm/session.py")
```

This searches for patterns and returns **file paths only** (fast and lightweight):
- Lines containing "generate" followed by any characters and then "react"
- OR lines containing "create_react_cycle"  
- OR lines containing "generate" followed by any characters, then "with", then any characters, then "tools"

**Output**: List of file paths containing the pattern (limited to 20 files by default)

### 2. Find Files with Search Functions (File Pattern Filter)
```python
search_files("def.*search", ".", file_pattern="*.py")
```

Returns file paths that contain function definitions with "search" in the name, searching only Python files.

### 3. Count Pattern Matches
```python
search_files("import.*re", ".", "count")
```

Returns a count of how many times "import" followed by "re" appears in each file.

### 4. Show Actual Content (Slower)
```python
search_files("Generate", "abstractllm/session.py", "content", 5)
```

Shows the actual matching lines (limited to 5 matches for performance).

### 5. Multiline Pattern Matching
```python
search_files("def.*\\n.*tools", "abstractllm/session.py", "content", multiline=True)
```

Finds patterns that span multiple lines, like function definitions followed by lines containing "tools".

### 6. Search in Specific File
```python
search_files("def search", "abstractllm/tools/common_tools.py", "content", 5)
```

Search for function definitions containing "search" in a specific file, limited to first 5 matches.

## Output Modes

### "content" mode (default)
Shows the actual matching lines with line numbers:
```
Search results for pattern 'generate.*tools' in 1 files:

📄 abstractllm/session.py:
    Line 340: implementations stored for use with generate_with_tools.
    Line 1662: def generate_with_tools(
```

### "files_with_matches" mode
Shows only file paths that contain matches:
```
Files matching pattern 'def.*search':
abstractllm/tools/common_tools.py
abstractllm/utils/commands.py
tests/integration/test_e2e.py
```

### "count" mode
Shows match counts per file:
```
Match counts for pattern 'import.*re':
  15 abstractllm/session.py
   4 abstractllm/tools/common_tools.py
   3 abstractllm/types.py

Total: 22 matches in 3 files
```

## Features

✅ **Regex Support**: Full regular expression pattern matching  
✅ **Multiple Output Modes**: Content, file lists, or counts  
✅ **File Filtering**: Use glob patterns to limit which files to search  
✅ **Result Limiting**: Use head_limit to control output size  
✅ **Multiline Matching**: Search patterns that span multiple lines  
✅ **Case Control**: Case sensitive or insensitive matching  
✅ **Binary File Skipping**: Automatically skips binary files  
✅ **Error Handling**: Graceful handling of inaccessible files  

## Integration

The search_files function is automatically exported in the `__all__` list and can be imported like:

```python
from abstractllm.tools.common_tools import search_files
```

This provides grep-like functionality directly within your AbstractLLM tools, making it easy to quickly explore codebases and find specific patterns across multiple files.
