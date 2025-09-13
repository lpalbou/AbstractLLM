"""
SOTA Scratchpad Manager for ReAct Agents with Complete Observability.

Implements comprehensive scratchpad logging with:
- COMPLETE agent reasoning traces (NO TRUNCATION)
- Event-driven phase change triggers
- Persistent serialization to session folder
- Real-time observability hooks
- Structured cycle-by-cycle breakdown

Based on 2024-2025 SOTA practices for AI agent observability.
"""

import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ReActPhase(Enum):
    """ReAct cycle phases for event triggers."""
    CYCLE_START = "cycle_start"
    THINKING = "thinking" 
    ACTING = "acting"
    OBSERVING = "observing"
    CYCLE_COMPLETE = "cycle_complete"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


@dataclass
class ScratchpadEntry:
    """Single entry in the agent's scratchpad."""
    
    timestamp: str
    cycle_id: str
    iteration: int
    phase: ReActPhase
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Specific fields for different phases
    thought_confidence: Optional[float] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None
    tool_success: Optional[bool] = None
    tool_execution_time: Optional[float] = None
    observation_success: Optional[bool] = None
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert enum to string for JSON serialization
        if isinstance(data.get('phase'), ReActPhase):
            data['phase'] = data['phase'].value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScratchpadEntry':
        """Create from dictionary."""
        # Convert phase string back to enum
        if isinstance(data.get('phase'), str):
            data['phase'] = ReActPhase(data['phase'])
        return cls(**data)


@dataclass 
class CyclePhaseEvent:
    """Event triggered when ReAct cycle changes phases."""
    
    cycle_id: str
    phase: ReActPhase
    timestamp: str
    iteration: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Phase-specific data
    tool_name: Optional[str] = None
    success: Optional[bool] = None
    execution_time: Optional[float] = None


class EventBus:
    """Event bus for real-time phase change notifications."""
    
    def __init__(self):
        self.listeners: Dict[ReActPhase, Set[Callable]] = {}
        self.global_listeners: Set[Callable] = set()
        self._lock = threading.Lock()
    
    def subscribe(self, phase: Optional[ReActPhase], callback: Callable):
        """Subscribe to phase change events."""
        with self._lock:
            if phase is None:
                # Global listener for all phases
                self.global_listeners.add(callback)
            else:
                if phase not in self.listeners:
                    self.listeners[phase] = set()
                self.listeners[phase].add(callback)
    
    def unsubscribe(self, phase: Optional[ReActPhase], callback: Callable):
        """Unsubscribe from phase change events."""
        with self._lock:
            if phase is None:
                self.global_listeners.discard(callback)
            else:
                if phase in self.listeners:
                    self.listeners[phase].discard(callback)
    
    def emit(self, event: CyclePhaseEvent):
        """Emit a phase change event to all listeners."""
        with self._lock:
            # Notify phase-specific listeners
            if event.phase in self.listeners:
                for callback in self.listeners[event.phase].copy():
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Error in event callback: {e}")
            
            # Notify global listeners
            for callback in self.global_listeners.copy():
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in global event callback: {e}")


class ScratchpadManager:
    """
    Complete scratchpad management with SOTA observability features.
    
    Features:
    - Complete trace logging with NO truncation
    - Real-time event triggers for phase changes
    - Persistent storage in session memory folder
    - Structured cycle-by-cycle breakdown
    - Observer pattern for external monitoring
    """
    
    def __init__(self, session_id: str, memory_folder: Optional[Path] = None):
        """
        Initialize scratchpad manager.
        
        Args:
            session_id: Unique session identifier
            memory_folder: Folder to store scratchpad files (optional)
        """
        self.session_id = session_id
        self.memory_folder = memory_folder or Path("./memory")
        self.memory_folder.mkdir(parents=True, exist_ok=True)
        
        # Scratchpad storage
        self.entries: List[ScratchpadEntry] = []
        self.current_cycle_id: Optional[str] = None
        self.current_iteration: int = 0
        
        # Event system
        self.event_bus = EventBus()
        
        # File paths
        self.scratchpad_file = self.memory_folder / f"scratchpad_{session_id}.json"
        self.events_file = self.memory_folder / f"events_{session_id}.json"
        
        # Load existing scratchpad if available
        self._load_scratchpad()
        
        logger.info(f"ScratchpadManager initialized for session {session_id}")
    
    def start_cycle(self, cycle_id: str, query: str) -> None:
        """Start a new ReAct cycle."""
        self.current_cycle_id = cycle_id
        self.current_iteration = 0
        
        entry = ScratchpadEntry(
            timestamp=self._get_timestamp(),
            cycle_id=cycle_id,
            iteration=0,
            phase=ReActPhase.CYCLE_START,
            content=f"Starting ReAct cycle for query: {query}",
            metadata={"query": query}
        )
        
        self._add_entry(entry)
        self._emit_event(ReActPhase.CYCLE_START, f"New cycle started: {query}")
        
        logger.info(f"Started ReAct cycle {cycle_id}")
    
    def add_thought(self, content: str, confidence: float = 1.0, metadata: Optional[Dict] = None) -> None:
        """Add a thinking phase entry."""
        if not self.current_cycle_id:
            raise ValueError("No active cycle - call start_cycle first")
        
        entry = ScratchpadEntry(
            timestamp=self._get_timestamp(),
            cycle_id=self.current_cycle_id,
            iteration=self.current_iteration,
            phase=ReActPhase.THINKING,
            content=content,
            thought_confidence=confidence,
            metadata=metadata or {}
        )
        
        self._add_entry(entry)
        self._emit_event(ReActPhase.THINKING, content, {"confidence": confidence})
    
    def add_action(self, tool_name: str, tool_args: Dict[str, Any], 
                  reasoning: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """Add an acting phase entry."""
        if not self.current_cycle_id:
            raise ValueError("No active cycle - call start_cycle first")
        
        action_content = f"Calling tool '{tool_name}' with args: {json.dumps(tool_args, indent=2)}"
        if reasoning:
            action_content = f"{reasoning}\n\nAction: {action_content}"
        
        entry = ScratchpadEntry(
            timestamp=self._get_timestamp(),
            cycle_id=self.current_cycle_id,
            iteration=self.current_iteration,
            phase=ReActPhase.ACTING,
            content=action_content,
            tool_name=tool_name,
            tool_args=tool_args,
            metadata=metadata or {}
        )
        
        action_id = f"action_{len(self.entries)}"
        entry.metadata["action_id"] = action_id
        
        self._add_entry(entry)
        self._emit_event(ReActPhase.ACTING, action_content, {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "action_id": action_id
        })
        
        return action_id
    
    def add_observation(self, action_id: str, result: Any, success: bool = True,
                       execution_time: Optional[float] = None, metadata: Optional[Dict] = None) -> None:
        """Add an observation phase entry."""
        if not self.current_cycle_id:
            raise ValueError("No active cycle - call start_cycle first")
        
        # Convert result to string but preserve full content
        result_str = str(result)
        
        observation_content = f"Tool execution {'succeeded' if success else 'failed'}:\n"
        observation_content += f"Result: {result_str}"
        
        if execution_time is not None:
            observation_content += f"\nExecution time: {execution_time:.6f}s"
        
        entry = ScratchpadEntry(
            timestamp=self._get_timestamp(),
            cycle_id=self.current_cycle_id,
            iteration=self.current_iteration,
            phase=ReActPhase.OBSERVING,
            content=observation_content,
            tool_result=result_str,
            tool_success=success,
            tool_execution_time=execution_time,
            observation_success=success,
            metadata=(metadata or {}) | {"action_id": action_id}
        )
        
        self._add_entry(entry)
        self._emit_event(ReActPhase.OBSERVING, observation_content, {
            "action_id": action_id,
            "success": success,
            "execution_time": execution_time
        })
    
    def next_iteration(self) -> None:
        """Move to next iteration within the current cycle."""
        self.current_iteration += 1
        
        entry = ScratchpadEntry(
            timestamp=self._get_timestamp(),
            cycle_id=self.current_cycle_id or "unknown",
            iteration=self.current_iteration,
            phase=ReActPhase.THINKING,
            content=f"Starting iteration {self.current_iteration}",
            metadata={"iteration_start": True}
        )
        
        self._add_entry(entry)
    
    def complete_cycle(self, final_answer: str, success: bool = True,
                      error_details: Optional[str] = None) -> None:
        """Complete the current ReAct cycle."""
        if not self.current_cycle_id:
            raise ValueError("No active cycle - call start_cycle first")
        
        # Add final answer entry
        final_entry = ScratchpadEntry(
            timestamp=self._get_timestamp(),
            cycle_id=self.current_cycle_id,
            iteration=self.current_iteration,
            phase=ReActPhase.FINAL_ANSWER,
            content=f"Final answer: {final_answer}",
            metadata={"success": success, "error_details": error_details}
        )
        
        self._add_entry(final_entry)
        self._emit_event(ReActPhase.FINAL_ANSWER, final_answer, {"success": success})
        
        # Add cycle completion entry
        completion_entry = ScratchpadEntry(
            timestamp=self._get_timestamp(),
            cycle_id=self.current_cycle_id,
            iteration=self.current_iteration,
            phase=ReActPhase.CYCLE_COMPLETE,
            content=f"ReAct cycle completed {'successfully' if success else 'with errors'}",
            metadata={"success": success, "total_iterations": self.current_iteration + 1}
        )
        
        self._add_entry(completion_entry)
        self._emit_event(ReActPhase.CYCLE_COMPLETE, "Cycle completed", {
            "success": success,
            "total_iterations": self.current_iteration + 1
        })
        
        # Reset state
        self.current_cycle_id = None
        self.current_iteration = 0
        
        # Save to disk
        self._save_scratchpad()
    
    def add_error(self, error_msg: str, error_details: Optional[str] = None) -> None:
        """Add an error entry to the scratchpad."""
        entry = ScratchpadEntry(
            timestamp=self._get_timestamp(),
            cycle_id=self.current_cycle_id or "unknown",
            iteration=self.current_iteration,
            phase=ReActPhase.ERROR,
            content=error_msg,
            error_details=error_details,
            metadata={"error": True}
        )
        
        self._add_entry(entry)
        self._emit_event(ReActPhase.ERROR, error_msg, {"error_details": error_details})
    
    def get_complete_trace(self, cycle_id: Optional[str] = None) -> str:
        """Get the complete scratchpad trace with NO truncation."""
        if cycle_id:
            entries = [e for e in self.entries if e.cycle_id == cycle_id]
        else:
            entries = self.entries
        
        if not entries:
            return "No scratchpad entries available"
        
        trace_lines = ["=== COMPLETE SCRATCHPAD TRACE ==="]
        trace_lines.append(f"Session: {self.session_id}")
        trace_lines.append(f"Total entries: {len(entries)}")
        trace_lines.append("")
        
        current_cycle = None
        current_iteration = None
        
        for entry in entries:
            # Cycle separator
            if current_cycle != entry.cycle_id:
                if current_cycle is not None:
                    trace_lines.append("")
                trace_lines.append(f"╔══ CYCLE: {entry.cycle_id} ══")
                current_cycle = entry.cycle_id
                current_iteration = None
            
            # Iteration separator
            if current_iteration != entry.iteration and entry.phase not in [ReActPhase.CYCLE_START, ReActPhase.CYCLE_COMPLETE]:
                if current_iteration is not None:
                    trace_lines.append("  ├─────────────────────")
                trace_lines.append(f"  ║ ITERATION {entry.iteration}")
                current_iteration = entry.iteration
            
            # Entry details
            phase_icon = {
                ReActPhase.CYCLE_START: "🚀",
                ReActPhase.THINKING: "💭", 
                ReActPhase.ACTING: "🔧",
                ReActPhase.OBSERVING: "👁️",
                ReActPhase.FINAL_ANSWER: "✅",
                ReActPhase.CYCLE_COMPLETE: "🏁",
                ReActPhase.ERROR: "❌"
            }.get(entry.phase, "📝")
            
            trace_lines.append(f"  ║ {entry.timestamp} {phase_icon} {entry.phase.value.upper()}")
            
            # Content (with proper indentation, NO truncation)
            content_lines = entry.content.split('\n')
            for line in content_lines:
                trace_lines.append(f"  ║   {line}")
            
            # Phase-specific metadata
            if entry.thought_confidence is not None:
                trace_lines.append(f"  ║   📊 Confidence: {entry.thought_confidence:.2f}")
            
            if entry.tool_name:
                trace_lines.append(f"  ║   🛠️ Tool: {entry.tool_name}")
                if entry.tool_args:
                    trace_lines.append(f"  ║   📋 Args: {json.dumps(entry.tool_args)}")
            
            if entry.tool_execution_time is not None:
                trace_lines.append(f"  ║   ⏱️ Execution: {entry.tool_execution_time:.6f}s")
            
            if entry.error_details:
                trace_lines.append(f"  ║   🚨 Error: {entry.error_details}")
            
            if entry.metadata:
                relevant_metadata = {k: v for k, v in entry.metadata.items() 
                                   if k not in ['action_id', 'query', 'success', 'error']}
                if relevant_metadata:
                    trace_lines.append(f"  ║   📋 Metadata: {json.dumps(relevant_metadata)}")
            
            trace_lines.append("  ║")
        
        trace_lines.append("╚══════════════════════════════════")
        trace_lines.append("")
        trace_lines.append(f"Total trace length: {len('\\n'.join(trace_lines))} characters")
        trace_lines.append("🔍 COMPLETE TRACE - NO TRUNCATION APPLIED")
        
        return '\n'.join(trace_lines)
    
    def get_cycle_summary(self, cycle_id: str) -> Dict[str, Any]:
        """Get a summary of a specific cycle."""
        cycle_entries = [e for e in self.entries if e.cycle_id == cycle_id]
        if not cycle_entries:
            return {}
        
        phases = {}
        for phase in ReActPhase:
            phases[phase.value] = len([e for e in cycle_entries if e.phase == phase])
        
        tools_used = list(set([e.tool_name for e in cycle_entries if e.tool_name]))
        errors = [e for e in cycle_entries if e.phase == ReActPhase.ERROR]
        
        start_time = min(e.timestamp for e in cycle_entries)
        end_time = max(e.timestamp for e in cycle_entries)
        
        return {
            "cycle_id": cycle_id,
            "start_time": start_time,
            "end_time": end_time,
            "total_entries": len(cycle_entries),
            "phases": phases,
            "tools_used": tools_used,
            "total_errors": len(errors),
            "iterations": max(e.iteration for e in cycle_entries) + 1,
            "success": len(errors) == 0 and any(e.phase == ReActPhase.FINAL_ANSWER for e in cycle_entries)
        }
    
    def subscribe_to_events(self, phase: Optional[ReActPhase], callback: Callable) -> None:
        """Subscribe to phase change events for real-time monitoring."""
        self.event_bus.subscribe(phase, callback)
    
    def unsubscribe_from_events(self, phase: Optional[ReActPhase], callback: Callable) -> None:
        """Unsubscribe from phase change events."""
        self.event_bus.unsubscribe(phase, callback)
    
    @contextmanager
    def monitor_phase(self, phase: ReActPhase):
        """Context manager for monitoring a specific phase."""
        events = []
        
        def capture_event(event: CyclePhaseEvent):
            if event.phase == phase:
                events.append(event)
        
        self.subscribe_to_events(phase, capture_event)
        try:
            yield events
        finally:
            self.unsubscribe_from_events(phase, capture_event)
    
    def get_scratchpad_file_path(self) -> Path:
        """Get the path to the persistent scratchpad file."""
        return self.scratchpad_file
    
    def _add_entry(self, entry: ScratchpadEntry) -> None:
        """Add entry to scratchpad and save to disk."""
        self.entries.append(entry)
        
        # Auto-save every 10 entries to prevent loss
        if len(self.entries) % 10 == 0:
            self._save_scratchpad()
    
    def _emit_event(self, phase: ReActPhase, content: str, metadata: Optional[Dict] = None) -> None:
        """Emit a phase change event."""
        event = CyclePhaseEvent(
            cycle_id=self.current_cycle_id or "unknown",
            phase=phase,
            timestamp=self._get_timestamp(),
            iteration=self.current_iteration,
            content=content,
            metadata=metadata or {}
        )
        
        self.event_bus.emit(event)
        
        # Also save event to disk for persistence
        self._save_event(event)
    
    def _save_scratchpad(self) -> None:
        """Save complete scratchpad to disk."""
        try:
            data = {
                "session_id": self.session_id,
                "created_at": self._get_timestamp(),
                "total_entries": len(self.entries),
                "entries": [entry.to_dict() for entry in self.entries]
            }
            
            with open(self.scratchpad_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save scratchpad: {e}")
    
    def _load_scratchpad(self) -> None:
        """Load existing scratchpad from disk."""
        try:
            if self.scratchpad_file.exists():
                with open(self.scratchpad_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.entries = [ScratchpadEntry.from_dict(entry_data) 
                               for entry_data in data.get('entries', [])]
                               
                logger.info(f"Loaded {len(self.entries)} scratchpad entries")
        except Exception as e:
            logger.error(f"Failed to load scratchpad: {e}")
            self.entries = []
    
    def _save_event(self, event: CyclePhaseEvent) -> None:
        """Save event to disk for persistence."""
        try:
            event_data = {
                "timestamp": event.timestamp,
                "cycle_id": event.cycle_id, 
                "phase": event.phase.value,
                "iteration": event.iteration,
                "content": event.content,
                "metadata": event.metadata
            }
            
            # Append to events file
            with open(self.events_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event_data) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to save event: {e}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()


# Global scratchpad registry for easy access
_scratchpad_registry: Dict[str, ScratchpadManager] = {}

def get_scratchpad_manager(session_id: str, memory_folder: Optional[Path] = None) -> ScratchpadManager:
    """Get or create a scratchpad manager for a session."""
    global _scratchpad_registry
    
    if session_id not in _scratchpad_registry:
        _scratchpad_registry[session_id] = ScratchpadManager(session_id, memory_folder)
    
    return _scratchpad_registry[session_id]


def cleanup_scratchpad_manager(session_id: str) -> None:
    """Clean up scratchpad manager for a session."""
    global _scratchpad_registry
    
    if session_id in _scratchpad_registry:
        # Save final state
        _scratchpad_registry[session_id]._save_scratchpad()
        del _scratchpad_registry[session_id]