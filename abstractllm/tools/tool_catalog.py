"""
Comprehensive tool catalog and discovery system for autonomous agents.

This module provides intelligent tool discovery, categorization, and recommendation
capabilities to help agents find and use the right tools for their tasks.
"""

import json
import os
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import logging

try:
    from abstractllm.tools.common_tools import (
        list_files, search_files, read_file, write_file, update_file,
        web_search, fetch_url, fetch_and_parse_html,
        ask_user_multiple_choice,
        get_system_info, get_performance_stats, get_running_processes,
        get_network_connections, get_disk_partitions, monitor_resource_usage
    )
    COMMON_TOOLS_AVAILABLE = True
except ImportError:
    COMMON_TOOLS_AVAILABLE = False

try:
    from abstractllm.tools.advanced_tools import (
        code_intelligence, test_suite_manager, version_control_manager,
        system_executor, file_operations_manager, network_intelligence,
        data_processor, dependency_resolver, get_advanced_tools_catalog
    )
    ADVANCED_TOOLS_AVAILABLE = True
except ImportError:
    ADVANCED_TOOLS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ToolInfo:
    """Information about a tool for discovery and usage."""
    name: str
    function: Callable
    category: str
    complexity: str  # "Low", "Medium", "High"
    description: str
    use_cases: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    when_to_use: Optional[str] = None


class ToolCatalog:
    """
    Comprehensive catalog for tool discovery and management.

    Provides intelligent tool recommendation based on tasks, categories,
    and capabilities. Enables autonomous agents to discover and use
    appropriate tools for their objectives.
    """

    def __init__(self):
        """Initialize the tool catalog."""
        self.tools: Dict[str, ToolInfo] = {}
        self.categories: Dict[str, List[str]] = {}
        self.use_case_index: Dict[str, List[str]] = {}
        self._build_catalog()

    def _build_catalog(self):
        """Build the comprehensive tool catalog."""

        # Common Tools (Basic Operations)
        if COMMON_TOOLS_AVAILABLE:
            common_tools = {
                "list_files": ToolInfo(
                    name="list_files",
                    function=list_files,
                    category="File Operations",
                    complexity="Low",
                    description="List files and directories with pattern matching",
                    use_cases=["explore filesystem", "find files", "directory listing"],
                    dependencies=[],
                    tags=["file", "directory", "listing", "filesystem"],
                    when_to_use="When you need to explore or find files by names/paths"
                ),
                "search_files": ToolInfo(
                    name="search_files",
                    function=search_files,
                    category="File Operations",
                    complexity="Low",
                    description="Search for text patterns inside files using regex",
                    use_cases=["find content", "grep files", "search code"],
                    dependencies=[],
                    tags=["search", "content", "regex", "grep"],
                    when_to_use="When you need to find specific text or patterns inside files"
                ),
                "read_file": ToolInfo(
                    name="read_file",
                    function=read_file,
                    category="File Operations",
                    complexity="Low",
                    description="Read file contents with optional line range",
                    use_cases=["read content", "examine files", "extract text"],
                    dependencies=[],
                    tags=["file", "read", "content"],
                    when_to_use="When you need to read file contents or examine code"
                ),
                "write_file": ToolInfo(
                    name="write_file",
                    function=write_file,
                    category="File Operations",
                    complexity="Low",
                    description="Write content to files with robust error handling",
                    use_cases=["create files", "save content", "write code"],
                    dependencies=[],
                    tags=["file", "write", "create", "save"],
                    when_to_use="When you need to create files or save content"
                ),
                "web_search": ToolInfo(
                    name="web_search",
                    function=web_search,
                    category="Network Operations",
                    complexity="Low",
                    description="Search the web using DuckDuckGo",
                    use_cases=["research", "find information", "web search"],
                    dependencies=[],
                    tags=["web", "search", "internet", "research"],
                    when_to_use="When you need current information or research topics"
                ),
                "fetch_url": ToolInfo(
                    name="fetch_url",
                    function=fetch_url,
                    category="Network Operations",
                    complexity="Low",
                    description="Fetch content from URLs",
                    use_cases=["download content", "fetch web pages", "API calls"],
                    dependencies=["requests"],
                    tags=["network", "http", "fetch", "download"],
                    when_to_use="When you need to fetch content from web URLs"
                ),
                "get_system_info": ToolInfo(
                    name="get_system_info",
                    function=get_system_info,
                    category="System Monitoring",
                    complexity="Low",
                    description="Get comprehensive system information",
                    use_cases=["system analysis", "hardware info", "diagnostics"],
                    dependencies=["psutil"],
                    tags=["system", "info", "hardware", "diagnostics"],
                    when_to_use="When you need to understand system capabilities or diagnose issues"
                )
            }

            for name, tool_info in common_tools.items():
                self.tools[name] = tool_info

        # Advanced Tools (Complex Operations)
        if ADVANCED_TOOLS_AVAILABLE:
            advanced_catalog = get_advanced_tools_catalog()
            advanced_functions = {
                "code_intelligence": code_intelligence,
                "test_suite_manager": test_suite_manager,
                "version_control_manager": version_control_manager,
                "system_executor": system_executor,
                "file_operations_manager": file_operations_manager,
                "network_intelligence": network_intelligence,
                "data_processor": data_processor,
                "dependency_resolver": dependency_resolver
            }

            for name, metadata in advanced_catalog.items():
                if name in advanced_functions:
                    self.tools[name] = ToolInfo(
                        name=name,
                        function=advanced_functions[name],
                        category=metadata["category"],
                        complexity=metadata["complexity"],
                        description=metadata["description"],
                        use_cases=metadata["use_cases"],
                        dependencies=metadata["dependencies"],
                        tags=[metadata["category"].lower().replace(" ", "_")]
                    )

        # Build indexes
        self._build_indexes()

    def _build_indexes(self):
        """Build search indexes for faster discovery."""
        # Category index
        for tool_name, tool_info in self.tools.items():
            category = tool_info.category
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(tool_name)

        # Use case index
        for tool_name, tool_info in self.tools.items():
            for use_case in tool_info.use_cases:
                if use_case not in self.use_case_index:
                    self.use_case_index[use_case] = []
                self.use_case_index[use_case].append(tool_name)

    def list_all_tools(self) -> Dict[str, Any]:
        """List all available tools with basic information."""
        return {
            name: {
                "category": tool.category,
                "complexity": tool.complexity,
                "description": tool.description,
                "use_cases": tool.use_cases
            }
            for name, tool in self.tools.items()
        }

    def get_tools_by_category(self, category: str) -> List[str]:
        """Get all tools in a specific category."""
        return self.categories.get(category, [])

    def get_categories(self) -> List[str]:
        """Get all available categories."""
        return list(self.categories.keys())

    def find_tools_by_use_case(self, use_case: str) -> List[str]:
        """Find tools that match a specific use case."""
        # Exact match
        exact_matches = self.use_case_index.get(use_case, [])

        # Fuzzy match
        fuzzy_matches = []
        use_case_lower = use_case.lower()
        for indexed_use_case, tools in self.use_case_index.items():
            if use_case_lower in indexed_use_case.lower() or indexed_use_case.lower() in use_case_lower:
                fuzzy_matches.extend(tools)

        # Combine and deduplicate
        all_matches = list(set(exact_matches + fuzzy_matches))
        return all_matches

    def search_tools(self, query: str) -> List[str]:
        """Search tools by query string (name, description, use cases)."""
        query_lower = query.lower()
        matches = []

        for tool_name, tool_info in self.tools.items():
            # Check name
            if query_lower in tool_name.lower():
                matches.append(tool_name)
                continue

            # Check description
            if query_lower in tool_info.description.lower():
                matches.append(tool_name)
                continue

            # Check use cases
            for use_case in tool_info.use_cases:
                if query_lower in use_case.lower():
                    matches.append(tool_name)
                    break

            # Check tags
            for tag in tool_info.tags:
                if query_lower in tag.lower():
                    matches.append(tool_name)
                    break

        return list(set(matches))  # Remove duplicates

    def recommend_tools(self, task_description: str, complexity_preference: str = "any") -> List[Dict[str, Any]]:
        """
        Recommend tools based on task description and complexity preference.

        Args:
            task_description: Description of what the user wants to do
            complexity_preference: "low", "medium", "high", or "any"

        Returns:
            List of recommended tools with relevance scores
        """
        recommendations = []
        task_lower = task_description.lower()

        # Keywords for different categories
        category_keywords = {
            "File Operations": ["file", "directory", "folder", "read", "write", "copy", "move", "create"],
            "Code Analysis": ["code", "function", "class", "analyze", "refactor", "ast", "syntax"],
            "Testing": ["test", "pytest", "unittest", "coverage", "verify", "check"],
            "Version Control": ["git", "commit", "branch", "merge", "repository", "version"],
            "System Operations": ["command", "process", "execute", "run", "system", "shell"],
            "Network Operations": ["http", "api", "web", "download", "request", "url", "fetch"],
            "Data Processing": ["json", "csv", "data", "parse", "convert", "analyze"],
            "Package Management": ["install", "package", "dependency", "pip", "npm", "requirement"],
            "System Monitoring": ["system", "performance", "memory", "cpu", "disk", "monitor"]
        }

        # Calculate relevance scores
        for tool_name, tool_info in self.tools.items():
            score = 0

            # Filter by complexity if specified
            if complexity_preference != "any":
                if tool_info.complexity.lower() != complexity_preference.lower():
                    continue

            # Direct name match
            if any(word in tool_name.lower() for word in task_lower.split()):
                score += 10

            # Description match
            description_words = tool_info.description.lower().split()
            task_words = task_lower.split()
            common_words = set(description_words) & set(task_words)
            score += len(common_words) * 2

            # Use case match
            for use_case in tool_info.use_cases:
                if any(word in use_case.lower() for word in task_words):
                    score += 5

            # Category keyword match
            category_words = category_keywords.get(tool_info.category, [])
            category_matches = sum(1 for word in category_words if word in task_lower)
            score += category_matches * 3

            # Tags match
            for tag in tool_info.tags:
                if any(word in tag.lower() for word in task_words):
                    score += 2

            if score > 0:
                recommendations.append({
                    "tool": tool_name,
                    "score": score,
                    "category": tool_info.category,
                    "complexity": tool_info.complexity,
                    "description": tool_info.description,
                    "reason": f"Matches {score} relevance points"
                })

        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:10]  # Top 10 recommendations

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific tool."""
        tool = self.tools.get(tool_name)
        if not tool:
            return None

        return {
            "name": tool.name,
            "category": tool.category,
            "complexity": tool.complexity,
            "description": tool.description,
            "use_cases": tool.use_cases,
            "dependencies": tool.dependencies,
            "tags": tool.tags,
            "when_to_use": tool.when_to_use,
            "available": True
        }

    def get_tool_function(self, tool_name: str) -> Optional[Callable]:
        """Get the actual function for a tool."""
        tool = self.tools.get(tool_name)
        return tool.function if tool else None

    def export_catalog(self, format: str = "json") -> str:
        """Export the tool catalog in various formats."""
        catalog_data = {
            "total_tools": len(self.tools),
            "categories": {
                category: {
                    "tools": tools,
                    "count": len(tools)
                }
                for category, tools in self.categories.items()
            },
            "tools": self.list_all_tools()
        }

        if format == "json":
            return json.dumps(catalog_data, indent=2)
        elif format == "summary":
            lines = [f"📚 Tool Catalog Summary"]
            lines.append(f"  • Total tools: {catalog_data['total_tools']}")
            lines.append(f"  • Categories: {len(catalog_data['categories'])}")
            lines.append("")

            for category, info in catalog_data['categories'].items():
                lines.append(f"🔧 {category} ({info['count']} tools)")
                for tool in info['tools'][:3]:  # Show first 3 tools per category
                    tool_info = self.tools[tool]
                    lines.append(f"    • {tool}: {tool_info.description}")
                if len(info['tools']) > 3:
                    lines.append(f"    ... and {len(info['tools']) - 3} more tools")
                lines.append("")

            return "\n".join(lines)
        else:
            return str(catalog_data)


# Global catalog instance
_global_catalog = None


def get_tool_catalog() -> ToolCatalog:
    """Get the global tool catalog instance."""
    global _global_catalog
    if _global_catalog is None:
        _global_catalog = ToolCatalog()
    return _global_catalog


def discover_tools(query: str = "") -> str:
    """
    Discover available tools based on a query or show all tools.

    Args:
        query: Search query for tool discovery

    Returns:
        Human-readable tool discovery results
    """
    catalog = get_tool_catalog()

    if not query:
        # Show overview
        return catalog.export_catalog("summary")

    # Search for tools
    matches = catalog.search_tools(query)

    if not matches:
        # Try recommendations
        recommendations = catalog.recommend_tools(query)
        if recommendations:
            results = [f"🔍 No exact matches for '{query}', but here are some recommendations:"]
            for rec in recommendations[:5]:
                results.append(f"  • {rec['tool']} ({rec['complexity']} complexity)")
                results.append(f"    {rec['description']}")
                results.append(f"    Reason: {rec['reason']}")
                results.append("")
            return "\n".join(results)
        else:
            return f"❌ No tools found matching '{query}'"

    # Show matching tools
    results = [f"🔍 Found {len(matches)} tools matching '{query}':"]
    for tool_name in matches[:10]:  # Show first 10 matches
        tool_info = catalog.get_tool_info(tool_name)
        if tool_info:
            results.append(f"  • {tool_name} ({tool_info['complexity']} complexity)")
            results.append(f"    {tool_info['description']}")
            results.append(f"    Category: {tool_info['category']}")
            results.append("")

    if len(matches) > 10:
        results.append(f"... and {len(matches) - 10} more tools")

    return "\n".join(results)


def recommend_tools_for_task(task: str, complexity: str = "any") -> str:
    """
    Get tool recommendations for a specific task.

    Args:
        task: Description of the task to accomplish
        complexity: Preferred complexity level ("low", "medium", "high", "any")

    Returns:
        Human-readable recommendations
    """
    catalog = get_tool_catalog()
    recommendations = catalog.recommend_tools(task, complexity)

    if not recommendations:
        return f"❌ No tool recommendations found for task: {task}"

    results = [f"🎯 Tool recommendations for: {task}"]
    if complexity != "any":
        results.append(f"   (Complexity preference: {complexity})")
    results.append("")

    for i, rec in enumerate(recommendations[:5], 1):
        results.append(f"{i}. {rec['tool']} (Score: {rec['score']})")
        results.append(f"   • {rec['description']}")
        results.append(f"   • Category: {rec['category']} | Complexity: {rec['complexity']}")
        results.append(f"   • {rec['reason']}")
        results.append("")

    return "\n".join(results)


# Export main functions
__all__ = [
    'ToolCatalog',
    'ToolInfo',
    'get_tool_catalog',
    'discover_tools',
    'recommend_tools_for_task'
]