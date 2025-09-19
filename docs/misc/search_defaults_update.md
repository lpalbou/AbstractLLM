# Search Function Defaults Update

## ✅ Changes Made

Updated `search_files` function defaults for better LLM performance:

### New Default Parameters
```python
def search_files(
    pattern: str, 
    path: str = ".", 
    output_mode: str = "files_with_matches",  # ← Changed from "content"
    head_limit: Optional[int] = 20,           # ← Changed from None
    file_pattern: str = "*", 
    case_sensitive: bool = False, 
    multiline: bool = False
) -> str:
```

### Performance Improvement

| Setting | Old Default | New Default | Impact |
|---------|-------------|-------------|---------|
| **output_mode** | `"content"` | `"files_with_matches"` | Returns file paths instead of full content |
| **head_limit** | `None` (unlimited) | `20` | Limits results to first 20 files |
| **Performance** | Can be very slow | **Fast and manageable** |
| **Output size** | Can be massive (500MB+) | **Small (few KB)** |

### Usage Examples

#### ✅ **Default Usage (Fast)**
```python
search_files("pattern", "/path/to/search")
# Returns: List of up to 20 file paths containing the pattern
```

#### 🔧 **Content Mode (When Needed)**
```python
search_files("pattern", "/path/to/search", "content", 10)
# Returns: Actual matching lines (limited to 10 matches)
```

#### 📊 **Count Mode**
```python
search_files("pattern", "/path/to/search", "count")
# Returns: Count of matches per file
```

### Why These Defaults?

1. **File Paths First**: Most LLM use cases need to **find relevant files** first
2. **Performance**: Returning file paths is **much faster** than full content
3. **Manageable Output**: 20 files = small, predictable output size
4. **Two-Step Workflow**: Find files → Then examine specific files
5. **Prevents Massive Outputs**: Avoids 500MB+ text dumps that slow LLMs

### Migration Guide

#### Old Calls That Still Work
```python
# These calls work exactly the same
search_files("pattern", "path", "content", 10)
search_files("pattern", "path", "count")
```

#### New Simplified Calls
```python
# Old way (potentially slow)
search_files("pattern", "path", "content")

# New way (fast, returns file paths)
search_files("pattern", "path")

# If you need content, be explicit with limits
search_files("pattern", "path", "content", 5)
```

### Updated Documentation

- ✅ Function signature updated
- ✅ Parameter descriptions updated  
- ✅ Examples updated to show new defaults
- ✅ Tool decorator examples updated
- ✅ Usage guides updated

## 🎯 Benefits

1. **Default Fast**: New users get fast results by default
2. **LLM Friendly**: Small outputs don't overwhelm LLMs  
3. **Practical Workflow**: Find files first, then examine content
4. **Backward Compatible**: All existing calls with explicit parameters still work
5. **Performance Optimized**: Based on real-world performance analysis

The search function is now optimized for LLM tool usage while maintaining all advanced capabilities!
