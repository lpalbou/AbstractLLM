"""
Enhanced Context Tracking for AbstractLLM
========================================

This module provides comprehensive context tracking and storage for deep
observability into LLM interactions and ReAct reasoning steps.
"""

import json
import gzip
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class LLMContextTracker:
    """
    Lightweight context tracker for capturing and storing LLM contexts.

    Integrates seamlessly with existing AbstractLLM session system.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize context tracker with persistent storage."""
        self.base_dir = base_dir or Path.home() / ".abstractllm"
        self.base_dir.mkdir(exist_ok=True)

        # Generate or load session ID
        self.session_id = self._get_or_create_session_id()
        self.session_dir = self.base_dir / "sessions" / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Context storage directory
        self.contexts_dir = self.session_dir / "contexts"
        self.contexts_dir.mkdir(exist_ok=True)

        # In-memory cache for current session
        self.current_contexts: Dict[str, Dict[str, Any]] = {}

    def _get_or_create_session_id(self) -> str:
        """Get existing session ID or create new one."""
        session_file = self.base_dir / "current_session.txt"

        if session_file.exists():
            try:
                session_id = session_file.read_text().strip()
                if session_id:
                    return session_id
            except Exception:
                pass

        # Create new session ID
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        session_file.write_text(session_id)
        return session_id

    def capture_context(
        self,
        interaction_id: str,
        verbatim_context: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        step_id: Optional[str] = None,
        step_number: Optional[int] = None,
        reasoning_phase: Optional[str] = None
    ) -> str:
        """
        Capture EXACT VERBATIM LLM context from provider.

        This method stores the exact payload sent to the LLM provider,
        ensuring no formatting or reconstruction is applied.

        Args:
            interaction_id: Main interaction ID (like cycle_abc123def)
            verbatim_context: EXACT payload sent to LLM (captured by provider)
            provider: Provider name (ollama, openai, etc.)
            model: Model name
            endpoint: API endpoint used
            step_id: Step ID for ReAct cycles
            step_number: Step number in ReAct cycle
            reasoning_phase: think/act/observe phase

        Returns:
            context_id: Unique identifier for this context
        """

        # Generate context ID based on whether this is a step or main interaction
        if step_id:
            context_id = f"{interaction_id}_step_{step_number:03d}_{step_id[:8]}"
            context_type = "react_step"
        else:
            context_id = f"{interaction_id}_main"
            context_type = "interaction"

        # Create context data with EXACT verbatim payload
        context_data = {
            "context_id": context_id,
            "interaction_id": interaction_id,
            "step_id": step_id,
            "context_type": context_type,
            "timestamp": datetime.now().isoformat(),

            # EXACT VERBATIM CONTEXT (NO PROCESSING)
            "verbatim_context": verbatim_context,
            "endpoint": endpoint,

            # Request metadata
            "provider": provider,
            "model": model,
            "total_chars": len(verbatim_context) if verbatim_context else 0,

            # Step context (for ReAct cycles)
            "step_number": step_number,
            "reasoning_phase": reasoning_phase
        }

        # Store context
        self._save_context(context_id, context_data)
        self.current_contexts[context_id] = context_data

        return context_id

    def _save_context(self, context_id: str, context_data: Dict[str, Any]):
        """Save context to persistent storage with compression."""
        context_file = self.contexts_dir / f"{context_id}.json.gz"

        try:
            # Compress and save
            context_json = json.dumps(context_data, indent=2, default=str)
            with gzip.open(context_file, 'wt', encoding='utf-8') as f:
                f.write(context_json)
        except Exception as e:
            print(f"Warning: Failed to save context {context_id}: {e}")

    def _estimate_tokens(self, verbatim_context: str) -> int:
        """Rough token estimation for verbatim context size tracking."""
        if not verbatim_context:
            return 0

        # Simple word count estimation
        words = len(verbatim_context.split())
        return int(words * 1.3)  # Rough adjustment for tokenization

    def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve context data by ID."""
        # Try memory first
        if context_id in self.current_contexts:
            return self.current_contexts[context_id]

        # Try disk storage
        context_file = self.contexts_dir / f"{context_id}.json.gz"
        if not context_file.exists():
            return None

        try:
            with gzip.open(context_file, 'rt', encoding='utf-8') as f:
                context_data = json.load(f)

            # Cache in memory
            self.current_contexts[context_id] = context_data
            return context_data

        except Exception as e:
            print(f"Warning: Failed to load context {context_id}: {e}")
            return None

    def cleanup_old_contexts(self, days_old: int = 30):
        """Clean up contexts older than specified days."""
        import time
        cutoff = time.time() - (days_old * 24 * 60 * 60)

        cleaned = 0
        for context_file in self.contexts_dir.glob("*.json*"):
            try:
                if context_file.stat().st_mtime < cutoff:
                    context_file.unlink()
                    cleaned += 1
            except Exception:
                continue

        if cleaned > 0:
            print(f"Cleaned up {cleaned} old context files")


# Global context tracker instance (initialized lazily)
_global_tracker: Optional[LLMContextTracker] = None


def get_context_tracker() -> LLMContextTracker:
    """Get or create the global context tracker instance."""
    global _global_tracker

    if _global_tracker is None:
        _global_tracker = LLMContextTracker()

    return _global_tracker


def capture_llm_context(
    interaction_id: str,
    verbatim_context: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
    step_id: Optional[str] = None,
    step_number: Optional[int] = None,
    reasoning_phase: Optional[str] = None
) -> str:
    """
    Convenience function to capture EXACT VERBATIM LLM context.

    This function should be called with the exact payload captured
    by the provider's _capture_verbatim_context() method.

    Args:
        interaction_id: Main interaction ID
        verbatim_context: EXACT payload sent to LLM (NO reconstruction)
        provider: Provider name
        model: Model name
        endpoint: API endpoint
        step_id: ReAct step ID (if applicable)
        step_number: ReAct step number (if applicable)
        reasoning_phase: think/act/observe phase (if applicable)

    Returns:
        context_id: Unique identifier for stored context
    """
    tracker = get_context_tracker()
    return tracker.capture_context(
        interaction_id=interaction_id,
        verbatim_context=verbatim_context,
        provider=provider,
        model=model,
        endpoint=endpoint,
        step_id=step_id,
        step_number=step_number,
        reasoning_phase=reasoning_phase
    )


# Memory footprint management
def estimate_storage_usage() -> Dict[str, Any]:
    """Estimate current storage usage by context tracking."""
    tracker = get_context_tracker()

    total_size = 0
    file_count = 0

    for context_file in tracker.contexts_dir.glob("*.json*"):
        try:
            total_size += context_file.stat().st_size
            file_count += 1
        except Exception:
            continue

    return {
        "total_size_bytes": total_size,
        "total_size_mb": total_size / (1024 * 1024),
        "file_count": file_count,
        "average_size_kb": (total_size / 1024 / file_count) if file_count > 0 else 0,
        "storage_path": str(tracker.contexts_dir)
    }


if __name__ == "__main__":
    # Demo/test functionality
    print("Enhanced Context Tracking for AbstractLLM")
    print("=" * 50)

    storage = estimate_storage_usage()
    print(f"Storage Usage:")
    print(f"  Files: {storage['file_count']}")
    print(f"  Size: {storage['total_size_mb']:.2f} MB")
    print(f"  Path: {storage['storage_path']}")