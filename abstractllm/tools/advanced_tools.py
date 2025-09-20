"""
Advanced tools for autonomous agents with SOTA capabilities.

This module provides sophisticated tools for code intelligence, system operations,
and autonomous workflows. These tools are designed for complex multi-step operations
with safety, observability, and composability as core principles.

Based on analysis of leading autonomous agent frameworks:
- Devin (Cognition Labs): Professional coding workflows
- SWE-Agent: Software engineering task automation
- AutoGPT: Autonomous goal achievement
- Code Interpreter: Safe code execution and analysis
"""

import ast
import os
import sys
import json
import subprocess
import time
import shutil
import threading
import tempfile
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import logging
import hashlib
import re

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

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
    def tool(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code or command execution."""
    success: bool
    output: str
    error: str = ""
    duration: float = 0.0
    resources_used: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeAnalysis:
    """Result of code analysis."""
    file_path: str
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    complexity: Dict[str, int] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


# 1. CODE INTELLIGENCE TOOL
@tool(
    description="Analyze and understand code structure using AST parsing and static analysis",
    tags=["code", "analysis", "ast", "refactoring", "intelligence"],
    when_to_use="When you need to understand code structure, analyze dependencies, or perform safe refactoring",
    examples=[
        {
            "description": "Analyze a Python file's structure",
            "arguments": {
                "file_path": "src/main.py",
                "analysis_type": "structure"
            }
        },
        {
            "description": "Find all function dependencies",
            "arguments": {
                "file_path": "src/utils.py",
                "analysis_type": "dependencies"
            }
        }
    ]
)
def code_intelligence(
    file_path: str,
    analysis_type: str = "structure",
    include_metrics: bool = True,
    max_complexity: int = 10
) -> str:
    """
    Analyze code using AST parsing for deep understanding.

    Args:
        file_path: Path to the code file to analyze
        analysis_type: Type of analysis - "structure", "dependencies", "complexity", "issues"
        include_metrics: Whether to include code metrics
        max_complexity: Maximum acceptable complexity

    Returns:
        Detailed code analysis results
    """
    try:
        if not os.path.exists(file_path):
            return f"❌ File not found: {file_path}"

        # Read and parse the file
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        # Parse AST
        try:
            tree = ast.parse(source_code, filename=file_path)
        except SyntaxError as e:
            return f"❌ Syntax error in {file_path}: {e}"

        analysis = CodeAnalysis(file_path=file_path)

        # Analyze AST nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                analysis.functions.append(node.name)
                if include_metrics:
                    # Calculate cyclomatic complexity (simplified)
                    complexity = _calculate_complexity(node)
                    analysis.complexity[node.name] = complexity
                    if complexity > max_complexity:
                        analysis.issues.append(f"High complexity in function '{node.name}': {complexity}")

            elif isinstance(node, ast.ClassDef):
                analysis.classes.append(node.name)

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis.imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        analysis.imports.append(f"{module}.{alias.name}")

        # Generate dependencies (simplified)
        if analysis_type in ["dependencies", "all"]:
            analysis.dependencies = list(set([
                imp.split('.')[0] for imp in analysis.imports
                if not imp.startswith('.') and imp not in ['os', 'sys', 'json']
            ]))

        # Calculate metrics
        if include_metrics:
            lines = source_code.splitlines()
            analysis.metrics = {
                "total_lines": len(lines),
                "code_lines": len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
                "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
                "function_count": len(analysis.functions),
                "class_count": len(analysis.classes),
                "import_count": len(analysis.imports)
            }

        # Format results
        result = [f"📊 Code Analysis: {file_path}"]
        result.append(f"  • Functions: {len(analysis.functions)}")
        result.append(f"  • Classes: {len(analysis.classes)}")
        result.append(f"  • Imports: {len(analysis.imports)}")

        if analysis_type in ["structure", "all"]:
            if analysis.functions:
                result.append(f"\n🔧 Functions: {', '.join(analysis.functions[:5])}")
            if analysis.classes:
                result.append(f"📦 Classes: {', '.join(analysis.classes[:5])}")

        if analysis_type in ["dependencies", "all"]:
            if analysis.dependencies:
                result.append(f"\n📚 Dependencies: {', '.join(analysis.dependencies[:10])}")

        if analysis_type in ["complexity", "all"] and analysis.complexity:
            result.append(f"\n⚡ Complexity:")
            for func, complexity in list(analysis.complexity.items())[:5]:
                status = "⚠️" if complexity > max_complexity else "✅"
                result.append(f"    {status} {func}: {complexity}")

        if analysis.issues:
            result.append(f"\n⚠️ Issues Found:")
            for issue in analysis.issues[:5]:
                result.append(f"    • {issue}")

        if include_metrics and analysis.metrics:
            result.append(f"\n📈 Metrics:")
            result.append(f"    • Total lines: {analysis.metrics['total_lines']}")
            result.append(f"    • Code lines: {analysis.metrics['code_lines']}")
            result.append(f"    • Comment ratio: {analysis.metrics['comment_lines']/analysis.metrics['total_lines']*100:.1f}%")

        return "\n".join(result)

    except Exception as e:
        return f"❌ Error analyzing code: {str(e)}"


def _calculate_complexity(node):
    """Calculate cyclomatic complexity for a function (simplified)."""
    complexity = 1  # Base complexity
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    return complexity


# 2. TEST SUITE MANAGER
@tool(
    description="Execute and analyze tests with comprehensive reporting and coverage analysis",
    tags=["testing", "pytest", "unittest", "coverage", "quality"],
    when_to_use="When you need to run tests, analyze test results, or understand test coverage",
    examples=[
        {
            "description": "Run all tests in a directory",
            "arguments": {
                "test_path": "tests/",
                "framework": "pytest"
            }
        },
        {
            "description": "Run specific test with coverage",
            "arguments": {
                "test_path": "tests/test_api.py",
                "include_coverage": True
            }
        }
    ]
)
def test_suite_manager(
    test_path: str = "tests/",
    framework: str = "auto",
    include_coverage: bool = False,
    timeout: int = 300,
    parallel: bool = False
) -> str:
    """
    Execute tests and provide comprehensive analysis.

    Args:
        test_path: Path to test files or directory
        framework: Test framework - "pytest", "unittest", "auto"
        include_coverage: Whether to include coverage analysis
        timeout: Maximum execution time in seconds
        parallel: Whether to run tests in parallel

    Returns:
        Detailed test execution results and analysis
    """
    try:
        if not os.path.exists(test_path):
            return f"❌ Test path not found: {test_path}"

        # Auto-detect framework
        if framework == "auto":
            if os.path.exists("pytest.ini") or any(f.startswith("test_") for f in os.listdir(".") if f.endswith(".py")):
                framework = "pytest"
            else:
                framework = "unittest"

        # Build command
        if framework == "pytest":
            cmd = ["python", "-m", "pytest", test_path, "-v", "--tb=short"]
            if include_coverage:
                cmd.extend(["--cov=.", "--cov-report=term-missing"])
            if parallel:
                cmd.extend(["-n", "auto"])
        else:
            cmd = ["python", "-m", "unittest", "discover", "-s", test_path, "-v"]

        # Execute tests
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            duration = time.time() - start_time

            # Parse results
            output_lines = result.stdout.splitlines() + result.stderr.splitlines()

            # Extract test statistics
            passed = failed = skipped = 0
            coverage_percent = None

            for line in output_lines:
                if "passed" in line and "failed" in line:
                    # Parse pytest summary line
                    import re
                    numbers = re.findall(r'(\d+)', line)
                    if len(numbers) >= 2:
                        passed = int(numbers[0]) if "passed" in line else 0
                        failed = int(numbers[1]) if "failed" in line else 0
                elif "TOTAL" in line and "%" in line:
                    # Parse coverage line
                    parts = line.split()
                    for part in parts:
                        if part.endswith('%'):
                            coverage_percent = part
                            break

            # Format results
            status = "✅" if result.returncode == 0 else "❌"
            results = [f"{status} Test Execution Summary"]
            results.append(f"  • Framework: {framework}")
            results.append(f"  • Duration: {duration:.2f}s")
            results.append(f"  • Exit code: {result.returncode}")

            if passed or failed or skipped:
                results.append(f"\n📊 Test Results:")
                results.append(f"  • Passed: {passed}")
                results.append(f"  • Failed: {failed}")
                results.append(f"  • Skipped: {skipped}")

                if failed > 0:
                    success_rate = (passed / (passed + failed)) * 100
                    results.append(f"  • Success Rate: {success_rate:.1f}%")

            if coverage_percent:
                results.append(f"\n📈 Coverage: {coverage_percent}")

            # Show output (truncated)
            if result.stdout:
                results.append(f"\n📄 Output (last 10 lines):")
                last_lines = result.stdout.splitlines()[-10:]
                for line in last_lines:
                    results.append(f"    {line}")

            if result.stderr and result.returncode != 0:
                results.append(f"\n❌ Errors:")
                error_lines = result.stderr.splitlines()[-5:]
                for line in error_lines:
                    results.append(f"    {line}")

            return "\n".join(results)

        except subprocess.TimeoutExpired:
            return f"❌ Test execution timed out after {timeout} seconds"

    except Exception as e:
        return f"❌ Error running tests: {str(e)}"


# 3. VERSION CONTROL MANAGER
@tool(
    description="Manage Git operations with safety checks and intelligent conflict resolution",
    tags=["git", "version-control", "repository", "merge", "diff"],
    when_to_use="When you need to perform Git operations, analyze repository state, or manage branches",
    examples=[
        {
            "description": "Check repository status",
            "arguments": {
                "operation": "status"
            }
        },
        {
            "description": "Create and switch to new branch",
            "arguments": {
                "operation": "branch",
                "branch_name": "feature/new-feature"
            }
        }
    ]
)
def version_control_manager(
    operation: str,
    target: str = "",
    branch_name: str = "",
    commit_message: str = "",
    safe_mode: bool = True
) -> str:
    """
    Manage Git operations with safety and intelligence.

    Args:
        operation: Git operation - "status", "diff", "commit", "branch", "merge", "log"
        target: Target file/directory for operation
        branch_name: Branch name for branch operations
        commit_message: Commit message for commit operation
        safe_mode: Enable safety checks and confirmations

    Returns:
        Git operation results with analysis
    """
    try:
        # Check if we're in a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return "❌ Not in a Git repository"

        if operation == "status":
            result = subprocess.run(
                ["git", "status", "--porcelain", "-b"],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return f"❌ Git status failed: {result.stderr}"

            lines = result.stdout.splitlines()
            if not lines:
                return "✅ Working directory clean"

            branch_line = lines[0] if lines[0].startswith("##") else "## unknown"
            branch = branch_line.replace("##", "").strip()

            status_lines = [line for line in lines[1:] if line.strip()]

            output = [f"📊 Repository Status"]
            output.append(f"  • Branch: {branch}")
            output.append(f"  • Changes: {len(status_lines)} files")

            if status_lines:
                output.append(f"\n📝 Modified Files:")
                for line in status_lines[:10]:  # Show first 10
                    status = line[:2]
                    filename = line[3:]
                    emoji = "🔴" if "D" in status else "🟡" if "M" in status else "🟢"
                    output.append(f"    {emoji} {filename}")

                if len(status_lines) > 10:
                    output.append(f"    ... and {len(status_lines) - 10} more files")

            return "\n".join(output)

        elif operation == "diff":
            cmd = ["git", "diff"]
            if target:
                cmd.append(target)
            cmd.extend(["--stat", "--no-color"])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                return f"❌ Git diff failed: {result.stderr}"

            if not result.stdout.strip():
                return "✅ No differences found"

            # Parse diff stats
            lines = result.stdout.splitlines()
            files_changed = [line for line in lines if "|" in line]
            summary_line = lines[-1] if lines and ("insertion" in lines[-1] or "deletion" in lines[-1]) else ""

            output = [f"📊 Diff Summary"]
            if summary_line:
                output.append(f"  • {summary_line}")
            output.append(f"  • Files changed: {len(files_changed)}")

            if files_changed:
                output.append(f"\n📝 Changed Files:")
                for line in files_changed[:10]:
                    output.append(f"    {line}")

            return "\n".join(output)

        elif operation == "commit":
            if not commit_message:
                return "❌ Commit message required"

            if safe_mode:
                # Check for unstaged changes
                status_result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True
                )

                if status_result.stdout.strip():
                    unstaged = [line for line in status_result.stdout.splitlines()
                              if line.startswith(" M") or line.startswith("??")]
                    if unstaged:
                        return f"⚠️ Warning: {len(unstaged)} unstaged changes. Stage files first with 'git add'"

            # Perform commit
            result = subprocess.run(
                ["git", "commit", "-m", commit_message],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return f"❌ Commit failed: {result.stderr}"

            return f"✅ Committed successfully: {commit_message}"

        elif operation == "branch":
            if not branch_name:
                # List branches
                result = subprocess.run(
                    ["git", "branch", "-v"],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    return f"❌ Branch listing failed: {result.stderr}"

                lines = result.stdout.splitlines()
                output = ["🌿 Branches:"]
                for line in lines:
                    emoji = "👉" if line.startswith("*") else "  "
                    output.append(f"  {emoji} {line.strip()}")

                return "\n".join(output)
            else:
                # Create and switch to branch
                result = subprocess.run(
                    ["git", "checkout", "-b", branch_name],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    return f"❌ Branch creation failed: {result.stderr}"

                return f"✅ Created and switched to branch: {branch_name}"

        elif operation == "log":
            result = subprocess.run(
                ["git", "log", "--oneline", "-10"],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return f"❌ Git log failed: {result.stderr}"

            lines = result.stdout.splitlines()
            output = ["📜 Recent Commits:"]
            for line in lines:
                output.append(f"  • {line}")

            return "\n".join(output)

        else:
            return f"❌ Unknown operation: {operation}. Available: status, diff, commit, branch, log"

    except Exception as e:
        return f"❌ Error in Git operation: {str(e)}"


# 4. SYSTEM EXECUTOR
@tool(
    description="Execute system commands and processes with advanced safety, monitoring, and resource control",
    tags=["system", "process", "execution", "monitoring", "safety"],
    when_to_use="When you need to run system commands, manage processes, or execute code with safety controls",
    examples=[
        {
            "description": "Run a Python script with monitoring",
            "arguments": {
                "command": "python script.py",
                "monitor_resources": True,
                "timeout": 60
            }
        },
        {
            "description": "Execute command in specific directory",
            "arguments": {
                "command": "npm install",
                "working_dir": "./frontend",
                "environment": {"NODE_ENV": "development"}
            }
        }
    ]
)
def system_executor(
    command: str,
    working_dir: str = ".",
    environment: Optional[Dict[str, str]] = None,
    timeout: int = 300,
    monitor_resources: bool = True,
    max_memory_mb: int = 1024,
    safe_mode: bool = True
) -> str:
    """
    Execute system commands with comprehensive monitoring and safety.

    Args:
        command: Command to execute
        working_dir: Working directory for execution
        environment: Environment variables to set
        timeout: Maximum execution time in seconds
        monitor_resources: Whether to monitor resource usage
        max_memory_mb: Maximum memory usage in MB
        safe_mode: Enable safety checks

    Returns:
        Execution results with resource monitoring and analysis
    """
    try:
        # Safety checks
        if safe_mode:
            dangerous_patterns = [
                r'rm\s+-rf\s+/',
                r'dd\s+if=',
                r'mkfs\.',
                r'fdisk',
                r'shutdown',
                r'reboot',
                r':(){ :|:& };:',  # Fork bomb
                r'wget.*\|.*sh',
                r'curl.*\|.*sh'
            ]

            for pattern in dangerous_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    return f"❌ Dangerous command blocked: {command}"

        # Setup environment
        env = os.environ.copy()
        if environment:
            env.update(environment)

        # Validate working directory
        if not os.path.exists(working_dir):
            return f"❌ Working directory not found: {working_dir}"

        # Start execution with monitoring
        start_time = time.time()
        peak_memory = 0

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=working_dir
            )

            # Monitor resources if requested
            if monitor_resources and PSUTIL_AVAILABLE:
                try:
                    psutil_process = psutil.Process(process.pid)

                    # Monitor in background thread
                    def monitor():
                        nonlocal peak_memory
                        while process.poll() is None:
                            try:
                                memory_info = psutil_process.memory_info()
                                memory_mb = memory_info.rss / 1024 / 1024
                                peak_memory = max(peak_memory, memory_mb)

                                # Check memory limit
                                if memory_mb > max_memory_mb:
                                    process.terminate()
                                    return

                                time.sleep(0.1)
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                break

                    monitor_thread = threading.Thread(target=monitor)
                    monitor_thread.daemon = True
                    monitor_thread.start()

                except Exception:
                    pass  # Continue without monitoring if psutil fails

            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                duration = time.time() - start_time

                # Format results
                status = "✅" if process.returncode == 0 else "❌"
                results = [f"{status} Command Execution"]
                results.append(f"  • Command: {command}")
                results.append(f"  • Working dir: {working_dir}")
                results.append(f"  • Duration: {duration:.2f}s")
                results.append(f"  • Exit code: {process.returncode}")

                if monitor_resources and peak_memory > 0:
                    results.append(f"  • Peak memory: {peak_memory:.1f} MB")

                # Add output (truncated)
                if stdout:
                    lines = stdout.splitlines()
                    results.append(f"\n📤 Output ({len(lines)} lines):")
                    for line in lines[-20:]:  # Last 20 lines
                        results.append(f"    {line}")

                if stderr and process.returncode != 0:
                    error_lines = stderr.splitlines()
                    results.append(f"\n❌ Errors ({len(error_lines)} lines):")
                    for line in error_lines[-10:]:  # Last 10 error lines
                        results.append(f"    {line}")

                return "\n".join(results)

            except subprocess.TimeoutExpired:
                process.kill()
                return f"❌ Command timed out after {timeout} seconds: {command}"

        except Exception as e:
            return f"❌ Failed to start command: {str(e)}"

    except Exception as e:
        return f"❌ Error in system execution: {str(e)}"


# 5. FILE OPERATIONS MANAGER
@tool(
    description="Advanced file operations with safety, backup, and batch processing capabilities",
    tags=["files", "operations", "backup", "batch", "permissions"],
    when_to_use="When you need advanced file operations like copying, moving, or managing permissions",
    examples=[
        {
            "description": "Copy files with backup",
            "arguments": {
                "operation": "copy",
                "source": "src/",
                "destination": "backup/",
                "create_backup": True
            }
        },
        {
            "description": "Set file permissions",
            "arguments": {
                "operation": "permissions",
                "target": "script.py",
                "permissions": "755"
            }
        }
    ]
)
def file_operations_manager(
    operation: str,
    source: str = "",
    destination: str = "",
    target: str = "",
    permissions: str = "",
    create_backup: bool = False,
    recursive: bool = False,
    pattern: str = "*"
) -> str:
    """
    Perform advanced file operations with safety and intelligence.

    Args:
        operation: Operation type - "copy", "move", "delete", "permissions", "backup", "sync"
        source: Source file or directory
        destination: Destination path
        target: Target file/directory for single-file operations
        permissions: Permission string (e.g., "755", "644")
        create_backup: Whether to create backup before destructive operations
        recursive: Whether to operate recursively on directories
        pattern: File pattern for filtering (e.g., "*.py")

    Returns:
        Operation results with detailed feedback
    """
    try:
        if operation == "copy":
            if not source or not destination:
                return "❌ Source and destination required for copy operation"

            if not os.path.exists(source):
                return f"❌ Source not found: {source}"

            # Create backup if requested
            backup_path = None
            if create_backup and os.path.exists(destination):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_path = f"{destination}.backup_{timestamp}"
                shutil.move(destination, backup_path)

            try:
                if os.path.isdir(source):
                    if recursive:
                        shutil.copytree(source, destination, dirs_exist_ok=True)
                        copied_count = sum(len(files) for _, _, files in os.walk(destination))
                        result = f"✅ Copied directory tree: {copied_count} files"
                    else:
                        return "❌ Use recursive=True to copy directories"
                else:
                    shutil.copy2(source, destination)
                    size = os.path.getsize(destination)
                    result = f"✅ Copied file: {source} → {destination} ({size:,} bytes)"

                if backup_path:
                    result += f"\n📦 Backup created: {backup_path}"

                return result

            except Exception as e:
                # Restore backup if copy failed
                if backup_path and os.path.exists(backup_path):
                    shutil.move(backup_path, destination)
                    return f"❌ Copy failed, backup restored: {str(e)}"
                return f"❌ Copy failed: {str(e)}"

        elif operation == "move":
            if not source or not destination:
                return "❌ Source and destination required for move operation"

            if not os.path.exists(source):
                return f"❌ Source not found: {source}"

            try:
                shutil.move(source, destination)
                return f"✅ Moved: {source} → {destination}"
            except Exception as e:
                return f"❌ Move failed: {str(e)}"

        elif operation == "permissions":
            target_path = target or source
            if not target_path:
                return "❌ Target file required for permissions operation"

            if not os.path.exists(target_path):
                return f"❌ Target not found: {target_path}"

            try:
                # Convert permission string to octal
                if permissions.isdigit() and len(permissions) == 3:
                    octal_perms = int(permissions, 8)
                    os.chmod(target_path, octal_perms)
                    return f"✅ Set permissions {permissions} on {target_path}"
                else:
                    return f"❌ Invalid permissions format: {permissions} (use format like '755')"
            except Exception as e:
                return f"❌ Permission change failed: {str(e)}"

        elif operation == "backup":
            target_path = target or source
            if not target_path:
                return "❌ Target required for backup operation"

            if not os.path.exists(target_path):
                return f"❌ Target not found: {target_path}"

            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                if os.path.isdir(target_path):
                    backup_name = f"{target_path}_backup_{timestamp}"
                    shutil.copytree(target_path, backup_name)
                    file_count = sum(len(files) for _, _, files in os.walk(backup_name))
                    return f"✅ Backup created: {backup_name} ({file_count} files)"
                else:
                    backup_name = f"{target_path}.backup_{timestamp}"
                    shutil.copy2(target_path, backup_name)
                    size = os.path.getsize(backup_name)
                    return f"✅ Backup created: {backup_name} ({size:,} bytes)"
            except Exception as e:
                return f"❌ Backup failed: {str(e)}"

        elif operation == "sync":
            if not source or not destination:
                return "❌ Source and destination required for sync operation"

            if not os.path.exists(source):
                return f"❌ Source not found: {source}"

            try:
                # Simple sync implementation
                if os.path.isdir(source):
                    for root, dirs, files in os.walk(source):
                        # Calculate relative path
                        rel_path = os.path.relpath(root, source)
                        dest_dir = os.path.join(destination, rel_path) if rel_path != '.' else destination

                        # Create directory if it doesn't exist
                        os.makedirs(dest_dir, exist_ok=True)

                        # Copy files that are newer or don't exist
                        for file in files:
                            src_file = os.path.join(root, file)
                            dest_file = os.path.join(dest_dir, file)

                            if not os.path.exists(dest_file) or os.path.getmtime(src_file) > os.path.getmtime(dest_file):
                                shutil.copy2(src_file, dest_file)

                    return f"✅ Synchronized: {source} → {destination}"
                else:
                    # Single file sync
                    if not os.path.exists(destination) or os.path.getmtime(source) > os.path.getmtime(destination):
                        shutil.copy2(source, destination)
                        return f"✅ Synchronized file: {source} → {destination}"
                    else:
                        return f"✅ File already up to date: {destination}"
            except Exception as e:
                return f"❌ Sync failed: {str(e)}"

        else:
            return f"❌ Unknown operation: {operation}. Available: copy, move, delete, permissions, backup, sync"

    except Exception as e:
        return f"❌ Error in file operation: {str(e)}"


# 6. NETWORK INTELLIGENCE
@tool(
    description="Advanced HTTP client with authentication, retries, and intelligent web operations",
    tags=["network", "http", "api", "web", "scraping", "download"],
    when_to_use="When you need to make HTTP requests, interact with APIs, or download/upload data",
    examples=[
        {
            "description": "Make authenticated API request",
            "arguments": {
                "url": "https://api.example.com/data",
                "method": "GET",
                "headers": {"Authorization": "Bearer token"}
            }
        },
        {
            "description": "Download file with progress",
            "arguments": {
                "url": "https://example.com/file.zip",
                "operation": "download",
                "local_path": "downloads/file.zip"
            }
        }
    ]
)
def network_intelligence(
    url: str,
    method: str = "GET",
    operation: str = "request",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[str] = None,
    local_path: str = "",
    timeout: int = 30,
    max_retries: int = 3,
    follow_redirects: bool = True
) -> str:
    """
    Perform intelligent network operations with robust error handling.

    Args:
        url: Target URL
        method: HTTP method (GET, POST, PUT, DELETE)
        operation: Operation type - "request", "download", "upload", "probe"
        headers: HTTP headers dictionary
        data: Request body data (JSON string or form data)
        local_path: Local file path for download/upload operations
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        follow_redirects: Whether to follow HTTP redirects

    Returns:
        Network operation results with detailed analysis
    """
    if not REQUESTS_AVAILABLE:
        return "❌ requests library not available. Install with: pip install requests"

    try:
        # Validate URL
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return f"❌ Invalid URL: {url}"

        # Setup session with retries
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'AbstractLLM-Agent/1.0 (Autonomous Agent)'
        })

        if headers:
            session.headers.update(headers)

        # Configure retries
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        if operation == "request":
            # Standard HTTP request
            start_time = time.time()

            response = session.request(
                method=method.upper(),
                url=url,
                data=data,
                timeout=timeout,
                allow_redirects=follow_redirects
            )

            duration = time.time() - start_time

            # Analyze response
            results = [f"🌐 HTTP {method.upper()} Request"]
            results.append(f"  • URL: {url}")
            results.append(f"  • Status: {response.status_code} {response.reason}")
            results.append(f"  • Duration: {duration:.2f}s")
            results.append(f"  • Content-Type: {response.headers.get('content-type', 'unknown')}")
            results.append(f"  • Content-Length: {len(response.content):,} bytes")

            if response.headers.get('content-type', '').startswith('application/json'):
                try:
                    json_data = response.json()
                    results.append(f"  • JSON Response: {len(str(json_data))} characters")
                    if isinstance(json_data, dict):
                        results.append(f"  • JSON Keys: {list(json_data.keys())[:5]}")
                except:
                    results.append(f"  • Invalid JSON response")

            # Show response preview (truncated)
            if response.text:
                preview = response.text[:500]
                results.append(f"\n📄 Response Preview:")
                results.append(f"    {preview}...")

            return "\n".join(results)

        elif operation == "download":
            if not local_path:
                return "❌ local_path required for download operation"

            # Create directory if needed
            os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)

            start_time = time.time()
            response = session.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            duration = time.time() - start_time
            speed = downloaded / duration if duration > 0 else 0

            results = [f"📥 Download Complete"]
            results.append(f"  • URL: {url}")
            results.append(f"  • Local path: {local_path}")
            results.append(f"  • Size: {downloaded:,} bytes")
            results.append(f"  • Duration: {duration:.2f}s")
            results.append(f"  • Speed: {speed/1024:.1f} KB/s")

            if total_size > 0:
                results.append(f"  • Progress: {downloaded}/{total_size} bytes")

            return "\n".join(results)

        elif operation == "probe":
            # Probe URL for information
            try:
                response = session.head(url, timeout=timeout, allow_redirects=follow_redirects)
            except:
                response = session.get(url, timeout=timeout, allow_redirects=follow_redirects, stream=True)

            results = [f"🔍 URL Probe"]
            results.append(f"  • URL: {url}")
            results.append(f"  • Status: {response.status_code}")
            results.append(f"  • Server: {response.headers.get('server', 'unknown')}")
            results.append(f"  • Content-Type: {response.headers.get('content-type', 'unknown')}")

            if 'content-length' in response.headers:
                size = int(response.headers['content-length'])
                results.append(f"  • Content-Length: {size:,} bytes")

            if 'last-modified' in response.headers:
                results.append(f"  • Last-Modified: {response.headers['last-modified']}")

            # Check for common security headers
            security_headers = ['x-frame-options', 'x-content-type-options', 'strict-transport-security']
            found_security = [h for h in security_headers if h in response.headers]
            if found_security:
                results.append(f"  • Security headers: {', '.join(found_security)}")

            return "\n".join(results)

        else:
            return f"❌ Unknown operation: {operation}. Available: request, download, upload, probe"

    except requests.exceptions.Timeout:
        return f"❌ Request timed out after {timeout} seconds"
    except requests.exceptions.ConnectionError:
        return f"❌ Connection failed to {url}"
    except requests.exceptions.RequestException as e:
        return f"❌ Request failed: {str(e)}"
    except Exception as e:
        return f"❌ Error in network operation: {str(e)}"


# 7. DATA PROCESSOR
@tool(
    description="Process and analyze structured data with validation, transformation, and format conversion",
    tags=["data", "json", "csv", "xml", "processing", "validation"],
    when_to_use="When you need to process structured data, validate formats, or transform between data formats",
    examples=[
        {
            "description": "Analyze JSON structure",
            "arguments": {
                "data_source": "data.json",
                "operation": "analyze",
                "format": "json"
            }
        },
        {
            "description": "Convert CSV to JSON",
            "arguments": {
                "data_source": "input.csv",
                "operation": "convert",
                "format": "csv",
                "output_format": "json"
            }
        }
    ]
)
def data_processor(
    data_source: str,
    operation: str = "analyze",
    format: str = "auto",
    output_format: str = "json",
    output_path: str = "",
    validation_schema: str = "",
    max_size_mb: int = 100
) -> str:
    """
    Process and analyze structured data with comprehensive capabilities.

    Args:
        data_source: Path to data file or raw data string
        operation: Operation type - "analyze", "validate", "convert", "transform"
        format: Input format - "json", "csv", "xml", "yaml", "auto"
        output_format: Output format for conversion
        output_path: Path for output file
        validation_schema: JSON schema for validation
        max_size_mb: Maximum file size in MB

    Returns:
        Data processing results with detailed analysis
    """
    try:
        # Check if data_source is a file path or raw data
        if os.path.exists(data_source):
            # Check file size
            file_size = os.path.getsize(data_source) / 1024 / 1024
            if file_size > max_size_mb:
                return f"❌ File too large: {file_size:.1f}MB (max: {max_size_mb}MB)"

            with open(data_source, 'r', encoding='utf-8') as f:
                raw_data = f.read()
            source_type = "file"
        else:
            raw_data = data_source
            source_type = "string"

        # Auto-detect format if needed
        if format == "auto":
            if raw_data.strip().startswith('{') or raw_data.strip().startswith('['):
                format = "json"
            elif ',' in raw_data and '\n' in raw_data:
                format = "csv"
            elif raw_data.strip().startswith('<'):
                format = "xml"
            else:
                format = "text"

        # Parse data based on format
        parsed_data = None
        parse_error = None

        try:
            if format == "json":
                parsed_data = json.loads(raw_data)
            elif format == "csv":
                import csv
                import io
                reader = csv.DictReader(io.StringIO(raw_data))
                parsed_data = list(reader)
            elif format == "xml":
                # Simple XML parsing (would need xml.etree.ElementTree for full support)
                return "⚠️ XML parsing requires additional implementation"
            else:
                parsed_data = {"content": raw_data, "lines": raw_data.splitlines()}
        except Exception as e:
            parse_error = str(e)

        if operation == "analyze":
            results = [f"📊 Data Analysis"]
            results.append(f"  • Source: {source_type}")
            results.append(f"  • Format: {format}")
            results.append(f"  • Size: {len(raw_data):,} characters")

            if parse_error:
                results.append(f"  • Parse status: ❌ {parse_error}")
                return "\n".join(results)

            results.append(f"  • Parse status: ✅ Success")

            # Format-specific analysis
            if format == "json":
                if isinstance(parsed_data, dict):
                    results.append(f"  • Type: Object")
                    results.append(f"  • Keys: {len(parsed_data)}")
                    results.append(f"  • Top keys: {list(parsed_data.keys())[:5]}")
                elif isinstance(parsed_data, list):
                    results.append(f"  • Type: Array")
                    results.append(f"  • Length: {len(parsed_data)}")
                    if parsed_data and isinstance(parsed_data[0], dict):
                        results.append(f"  • Item keys: {list(parsed_data[0].keys())[:5]}")

            elif format == "csv":
                if parsed_data:
                    results.append(f"  • Rows: {len(parsed_data)}")
                    results.append(f"  • Columns: {len(parsed_data[0]) if parsed_data else 0}")
                    if parsed_data:
                        results.append(f"  • Headers: {list(parsed_data[0].keys())[:5]}")

            return "\n".join(results)

        elif operation == "convert":
            if parse_error:
                return f"❌ Cannot convert: {parse_error}"

            # Convert data
            try:
                if output_format == "json":
                    output_data = json.dumps(parsed_data, indent=2)
                elif output_format == "csv":
                    if isinstance(parsed_data, list) and parsed_data:
                        import csv
                        import io
                        output = io.StringIO()
                        writer = csv.DictWriter(output, fieldnames=parsed_data[0].keys())
                        writer.writeheader()
                        writer.writerows(parsed_data)
                        output_data = output.getvalue()
                    else:
                        return f"❌ Cannot convert to CSV: data must be list of dictionaries"
                else:
                    return f"❌ Unsupported output format: {output_format}"

                # Save or return
                if output_path:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(output_data)
                    return f"✅ Converted {format} to {output_format}: {output_path}"
                else:
                    preview = output_data[:500] + "..." if len(output_data) > 500 else output_data
                    return f"✅ Conversion complete:\n{preview}"

            except Exception as e:
                return f"❌ Conversion failed: {str(e)}"

        elif operation == "validate":
            if parse_error:
                return f"❌ Validation failed: {parse_error}"

            results = [f"✅ Data validation passed"]
            results.append(f"  • Format: {format}")
            results.append(f"  • Structure: Valid")

            # Additional validation could be added here with schemas
            if validation_schema:
                results.append(f"  • Schema validation: Not implemented")

            return "\n".join(results)

        else:
            return f"❌ Unknown operation: {operation}. Available: analyze, validate, convert, transform"

    except Exception as e:
        return f"❌ Error in data processing: {str(e)}"


# 8. DEPENDENCY RESOLVER
@tool(
    description="Manage project dependencies with intelligent analysis and safe installation",
    tags=["dependencies", "packages", "pip", "npm", "requirements", "environment"],
    when_to_use="When you need to manage project dependencies, analyze package requirements, or install packages",
    examples=[
        {
            "description": "Analyze Python dependencies",
            "arguments": {
                "operation": "analyze",
                "package_manager": "pip"
            }
        },
        {
            "description": "Install package safely",
            "arguments": {
                "operation": "install",
                "package": "requests",
                "package_manager": "pip"
            }
        }
    ]
)
def dependency_resolver(
    operation: str,
    package: str = "",
    package_manager: str = "auto",
    requirements_file: str = "",
    virtual_env: bool = True,
    dry_run: bool = False,
    upgrade: bool = False
) -> str:
    """
    Manage project dependencies with intelligence and safety.

    Args:
        operation: Operation type - "analyze", "install", "update", "list", "check"
        package: Package name for install/update operations
        package_manager: Package manager - "pip", "npm", "yarn", "auto"
        requirements_file: Path to requirements file
        virtual_env: Whether to use virtual environment
        dry_run: Show what would be done without executing
        upgrade: Whether to upgrade package to latest version

    Returns:
        Dependency management results with analysis
    """
    try:
        # Auto-detect package manager
        if package_manager == "auto":
            if os.path.exists("requirements.txt") or os.path.exists("setup.py"):
                package_manager = "pip"
            elif os.path.exists("package.json"):
                package_manager = "npm"
            elif os.path.exists("yarn.lock"):
                package_manager = "yarn"
            else:
                package_manager = "pip"  # Default

        if operation == "analyze":
            results = [f"📦 Dependency Analysis"]
            results.append(f"  • Package manager: {package_manager}")

            if package_manager == "pip":
                # Check for requirements files
                req_files = ["requirements.txt", "requirements-dev.txt", "setup.py", "pyproject.toml"]
                found_files = [f for f in req_files if os.path.exists(f)]

                if found_files:
                    results.append(f"  • Requirement files: {', '.join(found_files)}")

                    # Analyze requirements.txt if it exists
                    if "requirements.txt" in found_files:
                        with open("requirements.txt", 'r') as f:
                            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                            results.append(f"  • Listed dependencies: {len(lines)}")

                            # Parse package names (simplified)
                            packages = []
                            for line in lines:
                                if '==' in line:
                                    pkg = line.split('==')[0]
                                elif '>=' in line:
                                    pkg = line.split('>=')[0]
                                else:
                                    pkg = line
                                packages.append(pkg)

                            results.append(f"  • Packages: {', '.join(packages[:5])}")
                            if len(packages) > 5:
                                results.append(f"    ... and {len(packages) - 5} more")

                # Check installed packages
                try:
                    result = subprocess.run(
                        ["pip", "list", "--format=json"],
                        capture_output=True,
                        text=True
                    )

                    if result.returncode == 0:
                        installed = json.loads(result.stdout)
                        results.append(f"  • Installed packages: {len(installed)}")

                        # Check for outdated packages
                        outdated_result = subprocess.run(
                            ["pip", "list", "--outdated", "--format=json"],
                            capture_output=True,
                            text=True
                        )

                        if outdated_result.returncode == 0:
                            outdated = json.loads(outdated_result.stdout)
                            if outdated:
                                results.append(f"  • Outdated packages: {len(outdated)}")
                                results.append(f"    Examples: {', '.join([p['name'] for p in outdated[:3]])}")

                except Exception:
                    results.append(f"  • Could not check installed packages")

            elif package_manager == "npm":
                if os.path.exists("package.json"):
                    with open("package.json", 'r') as f:
                        package_json = json.load(f)

                    deps = package_json.get("dependencies", {})
                    dev_deps = package_json.get("devDependencies", {})

                    results.append(f"  • Dependencies: {len(deps)}")
                    results.append(f"  • Dev dependencies: {len(dev_deps)}")

                    if deps:
                        results.append(f"  • Packages: {', '.join(list(deps.keys())[:5])}")

            return "\n".join(results)

        elif operation == "install":
            if not package:
                return "❌ Package name required for install operation"

            # Build install command
            if package_manager == "pip":
                cmd = ["pip", "install"]
                if upgrade:
                    cmd.append("--upgrade")
                cmd.append(package)

            elif package_manager == "npm":
                cmd = ["npm", "install", package]

            else:
                return f"❌ Unsupported package manager: {package_manager}"

            if dry_run:
                return f"🔍 Would execute: {' '.join(cmd)}"

            # Execute installation
            try:
                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                duration = time.time() - start_time

                if result.returncode == 0:
                    return f"✅ Successfully installed {package} in {duration:.1f}s"
                else:
                    return f"❌ Installation failed: {result.stderr}"

            except subprocess.TimeoutExpired:
                return f"❌ Installation timed out after 300 seconds"

        elif operation == "list":
            if package_manager == "pip":
                cmd = ["pip", "list"]
            elif package_manager == "npm":
                cmd = ["npm", "list", "--depth=0"]
            else:
                return f"❌ Unsupported package manager: {package_manager}"

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    lines = result.stdout.splitlines()
                    results = [f"📋 Installed Packages ({package_manager})"]
                    results.extend([f"  {line}" for line in lines[:20]])

                    if len(lines) > 20:
                        results.append(f"  ... and {len(lines) - 20} more packages")

                    return "\n".join(results)
                else:
                    return f"❌ Failed to list packages: {result.stderr}"

            except Exception as e:
                return f"❌ Error listing packages: {str(e)}"

        else:
            return f"❌ Unknown operation: {operation}. Available: analyze, install, update, list, check"

    except Exception as e:
        return f"❌ Error in dependency management: {str(e)}"


# Tool catalog and discovery system
def get_advanced_tools_catalog() -> Dict[str, Any]:
    """
    Get a comprehensive catalog of all advanced tools with metadata.

    Returns:
        Dictionary containing tool information for agent discovery
    """
    return {
        "code_intelligence": {
            "category": "Code Analysis",
            "complexity": "High",
            "dependencies": ["ast"],
            "description": "Deep code understanding and analysis",
            "use_cases": ["code review", "refactoring", "dependency analysis"]
        },
        "test_suite_manager": {
            "category": "Testing",
            "complexity": "Medium",
            "dependencies": ["subprocess"],
            "description": "Test execution and analysis",
            "use_cases": ["test running", "coverage analysis", "quality assurance"]
        },
        "version_control_manager": {
            "category": "Version Control",
            "complexity": "Medium",
            "dependencies": ["git"],
            "description": "Git operations and repository management",
            "use_cases": ["version control", "collaboration", "code history"]
        },
        "system_executor": {
            "category": "System Operations",
            "complexity": "High",
            "dependencies": ["subprocess", "psutil (optional)"],
            "description": "Safe system command execution",
            "use_cases": ["command execution", "process management", "system automation"]
        },
        "file_operations_manager": {
            "category": "File Operations",
            "complexity": "Medium",
            "dependencies": ["shutil", "os"],
            "description": "Advanced file and directory operations",
            "use_cases": ["file management", "backup", "synchronization"]
        },
        "network_intelligence": {
            "category": "Network Operations",
            "complexity": "Medium",
            "dependencies": ["requests"],
            "description": "Intelligent HTTP operations and web interaction",
            "use_cases": ["API calls", "web scraping", "file download"]
        },
        "data_processor": {
            "category": "Data Processing",
            "complexity": "Medium",
            "dependencies": ["json", "csv"],
            "description": "Structured data processing and transformation",
            "use_cases": ["data analysis", "format conversion", "validation"]
        },
        "dependency_resolver": {
            "category": "Package Management",
            "complexity": "Medium",
            "dependencies": ["subprocess"],
            "description": "Project dependency management",
            "use_cases": ["package installation", "dependency analysis", "environment management"]
        }
    }


# Export all tools for easy importing
__all__ = [
    'code_intelligence',
    'test_suite_manager',
    'version_control_manager',
    'system_executor',
    'file_operations_manager',
    'network_intelligence',
    'data_processor',
    'dependency_resolver',
    'get_advanced_tools_catalog'
]