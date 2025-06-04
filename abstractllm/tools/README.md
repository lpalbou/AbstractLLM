# AbstractLLM Common Tools

This module provides a comprehensive collection of shareable tools for AbstractLLM applications. These tools can be easily imported and used in any AbstractLLM session to provide powerful capabilities for file operations, web interactions, system commands, and user interactions.

## Installation Requirements

The basic tools work with just the standard library. For additional features:

```bash
# For web scraping and HTML parsing
pip install beautifulsoup4

# For web requests (usually included)
pip install requests
```

## Available Tools

### 📁 File Operations

#### `list_files(directory_path=".", pattern="*", recursive=False)`
List files in a directory with optional pattern matching.

```python
from abstractllm.tools.common_tools import list_files

# List all files in current directory
result = list_files()

# List Python files recursively
result = list_files(".", "*.py", recursive=True)

# List files in specific directory
result = list_files("/path/to/dir", "*.txt")
```

#### `search_files(search_term, directory_path=".", file_pattern="*.py", case_sensitive=False)`
Search for text within files in a directory.

```python
from abstractllm.tools.common_tools import search_files

# Search for "TODO" in Python files
result = search_files("TODO", ".", "*.py")

# Case-sensitive search in all text files
result = search_files("API_KEY", ".", "*.txt", case_sensitive=True)
```

#### `read_file(file_path, start_line=None, end_line=None)`
Read file contents with optional line range.

```python
from abstractllm.tools.common_tools import read_file

# Read entire file
content = read_file("README.md")

# Read specific line range
content = read_file("script.py", start_line=10, end_line=20)
```

#### `write_file(file_path, content, mode="w")`
Write content to a file.

```python
from abstractllm.tools.common_tools import write_file

# Create new file
result = write_file("output.txt", "Hello, World!")

# Append to existing file
result = write_file("log.txt", "New entry\n", mode="a")
```

#### `update_file(file_path, old_text, new_text, max_replacements=-1)`
Update a file by replacing text.

```python
from abstractllm.tools.common_tools import update_file

# Replace all occurrences
result = update_file("config.py", "localhost", "production.com")

# Replace only first occurrence
result = update_file("text.txt", "old", "new", max_replacements=1)
```

### 💻 System Operations

#### `execute_command(command, working_directory=".", timeout=30)`
Execute local commands safely with timeout and security checks.

```python
from abstractllm.tools.common_tools import execute_command

# Safe command execution
result = execute_command("ls -la")
result = execute_command("python --version")
result = execute_command("git status", working_directory="/path/to/repo")

# Note: Dangerous commands are automatically blocked
```

**Security Features:**
- Blocks dangerous commands (`rm -rf`, `format`, `shutdown`, etc.)
- Configurable timeout (default: 30 seconds)
- Working directory validation
- Captures both stdout and stderr

### 🌐 Web Operations

#### `search_internet(query, num_results=5)`
Search the internet using DuckDuckGo (no API key required).

```python
from abstractllm.tools.common_tools import search_internet

# Search for information
result = search_internet("Python best practices")
result = search_internet("machine learning tutorials", num_results=10)
```

#### `fetch_url(url, timeout=10)`
Fetch content from a URL.

```python
from abstractllm.tools.common_tools import fetch_url

# Fetch web content
result = fetch_url("https://api.github.com/repos/user/repo")
result = fetch_url("https://example.com/data.json")
```

#### `fetch_and_parse_html(url, extract_text=True, extract_links=False)`
Fetch and parse HTML content with text and link extraction.

```python
from abstractllm.tools.common_tools import fetch_and_parse_html

# Extract text content from webpage
result = fetch_and_parse_html("https://example.com")

# Extract both text and links
result = fetch_and_parse_html("https://news.ycombinator.com", 
                             extract_text=True, extract_links=True)
```

### 👤 User Interaction

#### `ask_user_multiple_choice(question, choices, allow_multiple=False)`
Ask the user interactive multiple choice questions.

```python
from abstractllm.tools.common_tools import ask_user_multiple_choice

# Single choice question
result = ask_user_multiple_choice(
    "What's your favorite programming language?",
    ["Python", "JavaScript", "Rust", "Go"]
)

# Multiple choice question
result = ask_user_multiple_choice(
    "Which features do you want to enable?",
    ["Logging", "Caching", "Authentication", "Monitoring"],
    allow_multiple=True
)
```

## Usage in AbstractLLM Sessions

### Basic Usage

```python
from abstractllm import create_llm
from abstractllm.session import Session
from abstractllm.tools.common_tools import read_file, list_files, search_internet

# Create provider
provider = create_llm("openai", model="gpt-4")

# Create session with tools
session = Session(
    provider=provider,
    tools=[read_file, list_files, search_internet]
)

# Use in conversation
response = session.generate("List all Python files and read the main one")
```

### Advanced Usage

```python
from abstractllm.tools.common_tools import *

# Import all tools
all_tools = [
    list_files, search_files, read_file, write_file, update_file,
    execute_command, search_internet, fetch_url, fetch_and_parse_html,
    ask_user_multiple_choice
]

session = Session(
    provider=provider,
    tools=all_tools,
    system_prompt="""You have access to comprehensive file operations, 
    web search, command execution, and user interaction tools. 
    Use them proactively to help users accomplish their tasks."""
)
```

## Examples

### Complete File Management Assistant

```python
#!/usr/bin/env python3
from abstractllm import create_llm
from abstractllm.session import Session
from abstractllm.tools.common_tools import (
    list_files, search_files, read_file, write_file, update_file
)

provider = create_llm("anthropic", model="claude-3-5-sonnet-20241022")

session = Session(
    provider=provider,
    tools=[list_files, search_files, read_file, write_file, update_file],
    system_prompt="""You are a file management assistant. Help users:
    - Navigate and explore directories
    - Search for content across files
    - Read, write, and modify files
    - Organize and manage file content
    
    Always use the appropriate tools to complete requests."""
)

# Example interactions:
# "Show me all Python files in this project"
# "Search for all TODO comments in the codebase"  
# "Read the configuration file and update the database URL"
```

### Web Research Assistant

```python
from abstractllm.tools.common_tools import (
    search_internet, fetch_url, fetch_and_parse_html, write_file
)

session = Session(
    provider=provider,
    tools=[search_internet, fetch_url, fetch_and_parse_html, write_file],
    system_prompt="""You are a web research assistant. Help users:
    - Search for information online
    - Fetch and analyze web content
    - Extract useful information from websites
    - Compile research into organized reports
    
    Save important findings to files when requested."""
)

# Example interactions:
# "Research the latest trends in AI and create a summary report"
# "Find the documentation for the FastAPI library and extract key concepts"
# "Search for Python deployment best practices and save to a file"
```

## Error Handling

All tools include comprehensive error handling and return descriptive error messages:

```python
# File not found
result = read_file("nonexistent.txt")
# Returns: "Error: File 'nonexistent.txt' does not exist"

# Invalid URL
result = fetch_url("not-a-url")
# Returns: "Error: Invalid URL format: 'not-a-url'"

# Blocked command
result = execute_command("rm -rf /")
# Returns: "Error: Command blocked for security reasons: 'rm -rf /'"
```

## Security Considerations

- **Command Execution**: Dangerous commands are automatically blocked
- **File Operations**: Path traversal and permission errors are handled safely
- **Web Requests**: Timeouts prevent hanging, user-agent headers are set appropriately
- **Input Validation**: All inputs are validated before processing

## Demo Script

Run the included demo to see all tools in action:

```bash
python examples/common_tools_demo.py
```

This provides an interactive demonstration of all available tools and shows how to integrate them into your own applications. 