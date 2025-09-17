"""
Unified Observability Store for AbstractLLM
==========================================

This module provides a centralized, efficient storage system for all AbstractLLM
observability data including contexts, facts, and ReAct scratchpads.

Key Features:
- SQLite-based storage (single file per session)
- ACID transactions for data integrity
- Compressed storage for efficiency
- Fast indexed access
- Zero external dependencies
"""

import sqlite3
import json
import gzip
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class ObservabilityStore:
    """
    Centralized storage for all AbstractLLM observability data.

    Uses SQLite for efficient, ACID-compliant storage with compression
    for large text data like contexts and scratchpads.
    """

    def __init__(self, session_id: str, base_dir: Optional[Path] = None):
        """Initialize the observability store for a session."""
        self.session_id = session_id
        self.base_dir = base_dir or Path.home() / ".abstractllm"
        self.session_dir = self.base_dir / "sessions" / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.session_dir / "observability.db"
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the SQLite database with required tables."""
        with self._get_connection() as conn:
            # Create interactions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id TEXT PRIMARY KEY,
                    context_verbatim TEXT,
                    facts_extracted TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create react_cycles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS react_cycles (
                    react_id TEXT PRIMARY KEY,
                    interaction_id TEXT NOT NULL,
                    scratchpad TEXT,
                    steps TEXT,
                    observations TEXT,
                    final_response TEXT,
                    success BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (interaction_id) REFERENCES interactions(interaction_id)
                )
            """)

            # Create indexes for fast access
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_created ON interactions(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_react_interaction ON react_cycles(interaction_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_react_created ON react_cycles(created_at)")

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def _compress_data(self, data: str) -> bytes:
        """Compress large text data for efficient storage."""
        if not data:
            return b""
        return gzip.compress(data.encode('utf-8'))

    def _decompress_data(self, compressed_data: bytes) -> str:
        """Decompress stored data."""
        if not compressed_data:
            return ""
        return gzip.decompress(compressed_data).decode('utf-8')

    # Core storage methods

    def store_context(self, interaction_id: str, verbatim_context: str, metadata: Optional[Dict] = None) -> None:
        """Store the verbatim LLM context for an interaction."""
        compressed_context = self._compress_data(verbatim_context)
        metadata_json = json.dumps(metadata or {})

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO interactions
                (interaction_id, context_verbatim, metadata)
                VALUES (?, ?, ?)
            """, (interaction_id, compressed_context, metadata_json))
            conn.commit()

        logger.debug(f"Stored context for interaction {interaction_id}")

    def store_facts(self, interaction_id: str, facts: List[Dict]) -> None:
        """Store extracted facts for an interaction."""
        facts_json = json.dumps(facts)

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO interactions
                (interaction_id, facts_extracted)
                VALUES (?, ?)
                ON CONFLICT(interaction_id) DO UPDATE SET
                facts_extracted = excluded.facts_extracted
            """, (interaction_id, facts_json))
            conn.commit()

        logger.debug(f"Stored {len(facts)} facts for interaction {interaction_id}")

    def store_react_cycle(self, react_id: str, interaction_id: str,
                         scratchpad: Dict, steps: List[Dict] = None,
                         observations: List[Dict] = None, final_response: str = None,
                         success: bool = True) -> None:
        """Store a complete ReAct cycle with scratchpad data."""
        compressed_scratchpad = self._compress_data(json.dumps(scratchpad))
        steps_json = json.dumps(steps or [])
        observations_json = json.dumps(observations or [])

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO react_cycles
                (react_id, interaction_id, scratchpad, steps, observations, final_response, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (react_id, interaction_id, compressed_scratchpad, steps_json,
                  observations_json, final_response, success))
            conn.commit()

        logger.debug(f"Stored ReAct cycle {react_id} for interaction {interaction_id}")

    # Core retrieval methods

    def get_context(self, interaction_id: str) -> Optional[str]:
        """Retrieve the verbatim LLM context for an interaction."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT context_verbatim FROM interactions
                WHERE interaction_id = ?
            """, (interaction_id,))
            row = cursor.fetchone()

            if row and row[0]:
                return self._decompress_data(row[0])
            return None

    def get_facts(self, interaction_id: str) -> List[Dict]:
        """Retrieve extracted facts for an interaction."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT facts_extracted FROM interactions
                WHERE interaction_id = ?
            """, (interaction_id,))
            row = cursor.fetchone()

            if row and row[0]:
                return json.loads(row[0])
            return []

    def get_scratchpad(self, react_id: str) -> Optional[Dict]:
        """Retrieve scratchpad data for a ReAct cycle."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT scratchpad FROM react_cycles
                WHERE react_id = ?
            """, (react_id,))
            row = cursor.fetchone()

            if row and row[0]:
                decompressed = self._decompress_data(row[0])
                return json.loads(decompressed)
            return None

    def get_interaction_metadata(self, interaction_id: str) -> Dict[str, Any]:
        """Get complete metadata for an interaction."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT interaction_id, metadata, created_at,
                       CASE WHEN context_verbatim IS NOT NULL THEN 1 ELSE 0 END as has_context,
                       CASE WHEN facts_extracted IS NOT NULL THEN 1 ELSE 0 END as has_facts
                FROM interactions
                WHERE interaction_id = ?
            """, (interaction_id,))
            row = cursor.fetchone()

            if not row:
                return {}

            metadata = json.loads(row[1] or '{}')
            return {
                'interaction_id': row[0],
                'created_at': row[2],
                'has_context': bool(row[3]),
                'has_facts': bool(row[4]),
                'metadata': metadata
            }

    def get_react_cycles_for_interaction(self, interaction_id: str) -> List[Dict]:
        """Get all ReAct cycles for an interaction."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT react_id, success, created_at, final_response
                FROM react_cycles
                WHERE interaction_id = ?
                ORDER BY created_at
            """, (interaction_id,))

            return [dict(row) for row in cursor.fetchall()]

    # Utility and maintenance methods

    def list_interactions(self, limit: int = 50) -> List[Dict]:
        """List recent interactions with basic metadata."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT interaction_id, created_at,
                       CASE WHEN context_verbatim IS NOT NULL THEN 1 ELSE 0 END as has_context,
                       CASE WHEN facts_extracted IS NOT NULL THEN 1 ELSE 0 END as has_facts
                FROM interactions
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def cleanup_old_data(self, days_old: int = 30) -> int:
        """Remove data older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)

        with self._get_connection() as conn:
            # Get IDs to be deleted for logging
            cursor = conn.execute("""
                SELECT interaction_id FROM interactions
                WHERE created_at < ?
            """, (cutoff_date,))
            old_interactions = [row[0] for row in cursor.fetchall()]

            # Delete old ReAct cycles first (foreign key constraint)
            conn.execute("""
                DELETE FROM react_cycles
                WHERE interaction_id IN (
                    SELECT interaction_id FROM interactions
                    WHERE created_at < ?
                )
            """, (cutoff_date,))

            # Delete old interactions
            cursor = conn.execute("""
                DELETE FROM interactions
                WHERE created_at < ?
            """, (cutoff_date,))

            deleted_count = cursor.rowcount
            conn.commit()

            # Vacuum to reclaim space
            conn.execute("VACUUM")

            logger.info(f"Cleaned up {deleted_count} old interactions")
            return deleted_count

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for the session."""
        with self._get_connection() as conn:
            # Get interaction counts
            cursor = conn.execute("SELECT COUNT(*) FROM interactions")
            interaction_count = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM react_cycles")
            react_count = cursor.fetchone()[0]

            # Get database size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

            return {
                'session_id': self.session_id,
                'interactions': interaction_count,
                'react_cycles': react_count,
                'database_size_mb': db_size / (1024 * 1024),
                'database_path': str(self.db_path)
            }


# Global store management
_session_stores: Dict[str, ObservabilityStore] = {}


def get_observability_store(session_id: str) -> ObservabilityStore:
    """Get or create an observability store for a session."""
    if session_id not in _session_stores:
        _session_stores[session_id] = ObservabilityStore(session_id)
    return _session_stores[session_id]


def cleanup_old_sessions(days_old: int = 30) -> int:
    """Clean up old session directories."""
    base_dir = Path.home() / ".abstractllm" / "sessions"
    if not base_dir.exists():
        return 0

    cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
    cleaned = 0

    for session_dir in base_dir.iterdir():
        if session_dir.is_dir():
            try:
                if session_dir.stat().st_mtime < cutoff_time:
                    import shutil
                    shutil.rmtree(session_dir)
                    cleaned += 1
                    # Remove from cache if present
                    if session_dir.name in _session_stores:
                        del _session_stores[session_dir.name]
            except Exception as e:
                logger.warning(f"Failed to cleanup session {session_dir.name}: {e}")

    logger.info(f"Cleaned up {cleaned} old session directories")
    return cleaned