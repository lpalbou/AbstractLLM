"""
Context logging module for tracking full LLM interactions.

This module provides functionality to log complete context sent to LLMs,
including system prompts, memory context, tool definitions, and responses.
It's designed to be provider-agnostic and support debugging/auditing.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os

logger = logging.getLogger(__name__)


class ContextLogger:
    """
    Manages logging of full LLM contexts and responses.

    Features:
    - Stores last context sent to LLM
    - Logs interactions with timestamps
    - Provides context replay for debugging
    - Supports multiple output formats
    """

    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize context logger.

        Args:
            log_dir: Directory for log files (defaults to ~/.abstractllm/context_logs)
        """
        if log_dir is None:
            log_dir = Path.home() / ".abstractllm" / "context_logs"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Store last context for /context command
        self.last_context: Optional[Dict[str, Any]] = None
        self.last_response: Optional[str] = None
        self.last_timestamp: Optional[datetime] = None

        # Create log file for this session
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"context_{session_id}.jsonl"

        logger.debug(f"Context logger initialized with log file: {self.log_file}")

    def log_interaction(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        memory_context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        response: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log a complete LLM interaction with timestamp.

        Args:
            prompt: User prompt
            system_prompt: System prompt if any
            messages: Full message history if using chat format
            memory_context: Memory/context injected
            tools: Tool definitions sent
            response: LLM response
            model: Model name
            provider: Provider name
            **kwargs: Additional metadata
        """
        timestamp = datetime.now()

        # Build complete context
        context = {
            "timestamp": timestamp.isoformat(),
            "timestamp_formatted": timestamp.strftime("%Y/%m/%d %H:%M:%S"),
            "model": model,
            "provider": provider,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "messages": messages,
            "memory_context": memory_context,
            "tools": self._serialize_tools(tools) if tools else None,
            "response": response,
            "metadata": kwargs
        }

        # Store as last context
        self.last_context = context
        self.last_response = response
        self.last_timestamp = timestamp

        # Write to log file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(context, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write context log: {e}")

    def get_last_context(self, format: str = "full") -> Optional[str]:
        """
        Get the last context sent to LLM.

        Args:
            format: Output format ('full', 'compact', 'debug')

        Returns:
            Formatted context string or None
        """
        if not self.last_context:
            return None

        if format == "full":
            return self._format_full_context(self.last_context)
        elif format == "compact":
            return self._format_compact_context(self.last_context)
        elif format == "debug":
            return json.dumps(self.last_context, indent=2, ensure_ascii=False)
        else:
            return str(self.last_context)

    def _format_full_context(self, context: Dict[str, Any]) -> str:
        """Format context to show EXACT verbatim content sent to LLM."""
        lines = []
        lines.append(f"╔══════════════ EXACT LLM INPUT ══════════════╗")
        lines.append(f"║ Timestamp: {context.get('timestamp_formatted', 'Unknown')}")
        lines.append(f"║ Model: {context.get('model', 'Unknown')}")
        lines.append(f"║ Provider: {context.get('provider', 'Unknown')}")
        lines.append(f"╚══════════════════════════════════════════════╝")
        lines.append("")

        # Show the raw messages array as it was sent to the LLM
        if context.get("messages"):
            lines.append("━━━ RAW MESSAGES SENT TO LLM ━━━")
            for i, msg in enumerate(context["messages"]):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                # Handle objects that might be in content
                if hasattr(content, '__dict__') and not isinstance(content, str):
                    content = str(content)
                elif not isinstance(content, str):
                    content = str(content)

                lines.append(f"Message {i+1} [{role.upper()}]:")
                lines.append(content)
                lines.append("")

        # If there's a standalone prompt (non-chat format), show it
        if context.get("prompt") and not context.get("messages"):
            lines.append("━━━ RAW PROMPT SENT TO LLM ━━━")
            lines.append(context.get("prompt", ""))
            lines.append("")

        # Tools as they were sent
        if context.get("tools"):
            lines.append("━━━ TOOLS DEFINITIONS SENT ━━━")
            for tool in context["tools"]:
                lines.append(f"Tool: {tool.get('name', 'Unknown')}")
                lines.append(f"Description: {tool.get('description', '')}")
                if tool.get('parameters'):
                    lines.append(f"Parameters: {tool.get('parameters')}")
                lines.append("")

        return "\n".join(lines)

    def _format_compact_context(self, context: Dict[str, Any]) -> str:
        """Format context for compact display."""
        lines = []
        lines.append(f"[{context.get('timestamp_formatted', '')}] {context.get('model', '')} via {context.get('provider', '')}")

        # Count components
        components = []
        if context.get("system_prompt"):
            components.append(f"system: {len(context['system_prompt'])} chars")
        if context.get("memory_context"):
            components.append(f"memory: {len(context['memory_context'])} chars")
        if context.get("messages"):
            components.append(f"history: {len(context['messages'])} messages")
        if context.get("tools"):
            components.append(f"tools: {len(context['tools'])}")

        if components:
            lines.append(f"Context: {', '.join(components)}")

        lines.append(f"Prompt: {context.get('prompt', '')[:100]}...")

        if context.get("response"):
            lines.append(f"Response: {context['response'][:100]}...")

        return "\n".join(lines)

    def _serialize_tools(self, tools: Optional[List[Any]]) -> Optional[List[Dict[str, Any]]]:
        """Serialize tool definitions for logging."""
        if not tools:
            return None

        serialized = []
        for tool in tools:
            if hasattr(tool, "__dict__"):
                # Tool object
                serialized.append({
                    "name": getattr(tool, "name", str(tool)),
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "parameters", {})
                })
            elif isinstance(tool, dict):
                # Already a dict
                serialized.append(tool)
            elif callable(tool):
                # Function
                serialized.append({
                    "name": tool.__name__,
                    "description": tool.__doc__ or "",
                    "callable": True
                })
            else:
                # Unknown
                serialized.append({"type": str(type(tool)), "value": str(tool)})

        return serialized

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about logged contexts."""
        stats = {
            "log_file": str(self.log_file),
            "log_size": self.log_file.stat().st_size if self.log_file.exists() else 0,
            "last_interaction": self.last_timestamp.isoformat() if self.last_timestamp else None
        }

        # Count interactions in log file
        if self.log_file.exists():
            with open(self.log_file, "r") as f:
                stats["total_interactions"] = sum(1 for _ in f)
        else:
            stats["total_interactions"] = 0

        return stats


# Global context logger instance
_context_logger: Optional[ContextLogger] = None


def get_context_logger() -> ContextLogger:
    """Get or create the global context logger."""
    global _context_logger
    if _context_logger is None:
        _context_logger = ContextLogger()
    return _context_logger


def log_llm_interaction(**kwargs) -> None:
    """Convenience function to log an LLM interaction."""
    logger = get_context_logger()
    logger.log_interaction(**kwargs)


def get_last_context(format: str = "full") -> Optional[str]:
    """Convenience function to get last context."""
    logger = get_context_logger()
    return logger.get_last_context(format)