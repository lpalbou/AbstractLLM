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

logger = logging.getLogger(__name__)

# File Operations
def list_files(directory_path: str = ".", pattern: str = "*", recursive: bool = False) -> str:
    """
    List files and directories in a specified directory with optional pattern matching.
    
    IMPORTANT: Use 'directory_path' parameter (not 'file_path') to specify the directory to list.
    
    Args:
        directory_path: Path to the directory to list files from (default: "." for current directory)
        pattern: Glob pattern to match files (default: "*" for all files and directories)
        recursive: Whether to search recursively in subdirectories (default: False)
        
    Returns:
        Formatted string with file and directory listings or error message
        
    Example:
        list_files(directory_path="docs") - Lists files in the docs directory
        list_files(directory_path=".", pattern="*.py") - Lists Python files in current directory
    """
    try:
        directory = Path(directory_path)
        
        if not directory.exists():
            return f"Error: Directory '{directory_path}' does not exist"
        
        if not directory.is_dir():
            return f"Error: '{directory_path}' is not a directory"
        
        if recursive:
            # Use ** for recursive pattern matching
            search_pattern = str(directory / "**" / pattern)
            files = glob.glob(search_pattern, recursive=True)
        else:
            search_pattern = str(directory / pattern)
            files = glob.glob(search_pattern)
        
        if not files:
            return f"No files found matching pattern '{pattern}' in '{directory_path}'"
        
        # Sort files and prepare output
        files.sort()
        output = [f"Files in '{directory_path}' matching '{pattern}':"]
        
        for file_path in files:
            path_obj = Path(file_path)
            if path_obj.is_file():
                size = path_obj.stat().st_size
                size_str = f"{size:,} bytes"
                output.append(f"  📄 {path_obj.name} ({size_str})")
            elif path_obj.is_dir():
                output.append(f"  📁 {path_obj.name}/")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error listing files: {str(e)}"


def search_files(search_term: str, directory_path: str = ".", file_pattern: str = "*.py", case_sensitive: bool = False) -> str:
    """
    Search for text within files in a directory.
    
    Args:
        search_term: Text to search for
        directory_path: Directory to search in (default: current directory)
        file_pattern: Glob pattern for files to search (default: "*.py")
        case_sensitive: Whether search should be case sensitive (default: False)
        
    Returns:
        Search results or error message
    """
    try:
        directory = Path(directory_path)
        
        if not directory.exists():
            return f"Error: Directory '{directory_path}' does not exist"
        
        # Find files matching pattern
        search_pattern = str(directory / "**" / file_pattern)
        files = glob.glob(search_pattern, recursive=True)
        
        if not files:
            return f"No files found matching pattern '{file_pattern}' in '{directory_path}'"
        
        results = []
        search_lower = search_term.lower() if not case_sensitive else search_term
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    
                matches = []
                for line_num, line in enumerate(lines, 1):
                    line_content = line.rstrip()
                    check_line = line_content.lower() if not case_sensitive else line_content
                    
                    if search_lower in check_line:
                        matches.append(f"    Line {line_num}: {line_content}")
                
                if matches:
                    results.append(f"\n📄 {file_path}:")
                    results.extend(matches)
                    
            except Exception as e:
                results.append(f"\n⚠️  Error reading {file_path}: {str(e)}")
        
        if not results:
            return f"No matches found for '{search_term}' in files matching '{file_pattern}'"
        
        # Count files with matches
        file_count = len([r for r in results if r.startswith("\n📄")])
        header = f"Search results for '{search_term}' in {file_count} files:"
        return header + "\n" + "\n".join(results)
        
    except Exception as e:
        return f"Error searching files: {str(e)}"


def read_file(file_path: str, should_read_entire_file: bool = True, start_line_one_indexed: int = 1, end_line_one_indexed_inclusive: Optional[int] = None) -> str:
    """
    Read the contents of a file with optional line range.
    
    Args:
        file_path: Path to the file to read
        should_read_entire_file: Whether to read the entire file (default: True)
        start_line_one_indexed: Starting line number (1-indexed, default: 1)
        end_line_one_indexed_inclusive: Ending line number (1-indexed, inclusive, optional)
        
    Returns:
        File contents or error message
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            return f"Error: File '{file_path}' does not exist"
        
        if not path.is_file():
            return f"Error: '{file_path}' is not a file"
        
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
                    result_lines.append(f"{i:4d}: {line.rstrip()}")
                
                return "\n".join(result_lines)
                
    except UnicodeDecodeError:
        return f"Error: Cannot read '{file_path}' - file appears to be binary"
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except PermissionError:
        return f"Error: Permission denied reading file: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(file_path: str, content: str, mode: str = "w") -> str:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        mode: Write mode - "w" to overwrite, "a" to append (default: "w")
        
    Returns:
        Success message or error message
    """
    try:
        path = Path(file_path)
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, mode, encoding='utf-8') as f:
            f.write(content)
        
        size = path.stat().st_size
        action = "appended to" if mode == "a" else "written to"
        return f"Successfully {action} '{file_path}' ({size:,} bytes)"
        
    except Exception as e:
        return f"Error writing file: {str(e)}"


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
def search_internet(query: str, num_results: int = 5) -> str:
    """
    Search the internet using DuckDuckGo (no API key required).
    
    Args:
        query: Search query
        num_results: Number of results to return (default: 5)
        
    Returns:
        Search results or error message
    """
    try:
        # Simple DuckDuckGo instant answer API
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
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
            content = response.text[:10000]  # Limit to first 10KB
            if len(response.text) > 10000:
                content += "\n\n[Content truncated - showing first 10KB]"
            results.append(content)
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
            
            # Limit text length
            if len(text) > 5000:
                text = text[:5000] + "\n\n[Text truncated - showing first 5000 characters]"
            
            results.append("Extracted Text:")
            results.append(text)
        
        if extract_links:
            results.append("\nExtracted Links:")
            links = soup.find_all('a', href=True)
            for i, link in enumerate(links[:20], 1):  # Limit to first 20 links
                href = link['href']
                link_text = link.get_text().strip()
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    href = urljoin(url, href)
                
                results.append(f"{i}. {link_text} - {href}")
            
            if len(links) > 20:
                results.append(f"... and {len(links) - 20} more links")
        
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
    'execute_command',
    'search_internet',
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