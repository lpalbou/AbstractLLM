# Tool Function Distinctions Summary

## CLI Tools Overview

The CLI now includes both `list_files` and `search` functions with **clearly distinct purposes**:

### 🗂️ `list_files` - Find Files by Names/Paths
**Purpose**: Find and list files and directories by their **filenames and paths**
**Use Case**: When you need to discover what files exist, find files by extension, or locate files by their names

```python
@tool(
    description="Find and list files and directories by their names/paths using glob patterns",
    when_to_use="When you need to find files by their names, paths, or file extensions (NOT for searching file contents)"
)
```

**Examples**:
- List all files in a directory
- Find all Python files (`*.py`)
- Locate files with "test" in their filename
- Discover directory structure
- **NEW**: Include/exclude hidden files (`.git`, `.DS_Store`, etc.) with `include_hidden` parameter

### 🔍 `search` - Search Content Inside Files  
**Purpose**: Search for **text patterns and content INSIDE files** using regex
**Use Case**: When you need to find specific code, text, or patterns within file contents

```python
@tool(
    description="Search for text patterns INSIDE files using regex (grep-like content search)",
    when_to_use="When you need to find specific text, code patterns, or content INSIDE files (NOT for finding files by names)"
)
```

**Examples**:
- Find function definitions: `def.*search`
- Locate import statements: `import.*re`
- Search for specific code patterns: `generate.*tools|create_react_cycle`
- Grep-like content search with regex support

## Key Distinctions

| Aspect | `list_files` | `search` |
|--------|-------------|----------|
| **What it searches** | File/directory names and paths | Text content inside files |
| **Pattern type** | Glob patterns (`*.py`, `*test*`) | Regex patterns (`def.*search`) |
| **Output** | File and directory listings | Matching lines with line numbers |
| **Use when** | Finding files by names/extensions | Finding content within files |
| **Similar to** | `ls`, `find` commands | `grep`, `ripgrep` commands |
| **Hidden files** | `include_hidden=False` (default) | Not applicable |

## CLI Integration

Both functions are now included in the CLI agent's default tools:

```python
'tools': [read_file, list_files, search, write_file]
```

This provides users with comprehensive file system capabilities:
1. **Discover** files with `list_files`
2. **Read** file contents with `read_file` 
3. **Search** within file contents with `search`
4. **Create** new files with `write_file`

## Enhanced Descriptions

The tool decorators now clearly emphasize:
- `list_files`: "(NOT for searching file contents)"
- `search`: "(NOT for finding files by names)"

This prevents confusion and helps users choose the right tool for their specific needs.

## Hidden Files Handling

The `list_files` function now includes an `include_hidden` parameter to control visibility of hidden files and directories:

### Default Behavior (`include_hidden=False`)
- **Excludes** files and directories starting with `.` (like `.git`, `.DS_Store`, `.venv`)
- Shows "(hidden files excluded)" note in output
- Provides cleaner, focused file listings for most use cases

### Include Hidden Files (`include_hidden=True`)
- **Includes** all files and directories, including hidden ones
- No exclusion note in output
- Useful for system administration, debugging, or comprehensive file discovery

### Examples:
```python
# Default - exclude hidden files
list_files(".", "*")
# Output: "Files in '.' matching '*' (hidden files excluded):"

# Include hidden files  
list_files(".", "*", include_hidden=True)
# Output: "Files in '.' matching '*':" (with .git, .DS_Store, etc.)
```

This feature ensures that by default, users see clean file listings without system/configuration files cluttering the output, while still providing the option to see everything when needed.
