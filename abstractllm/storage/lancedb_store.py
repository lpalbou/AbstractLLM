"""
LanceDB Observability Store for AbstractLLM
==========================================

A unified observability system that combines SQL power with vector search for RAG,
enabling time-based queries, user tracking, and semantic search over LLM interactions.

This module is designed to be modular and could potentially be extracted as a
separate library for AI observability.
"""

import lancedb
import uuid
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
import pandas as pd
import logging

from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class ObservabilityStore:
    """
    Unified observability store using LanceDB for AbstractLLM sessions.

    Provides:
    - Time-based search with exact timeframe queries
    - User management and session tracking
    - RAG capabilities with semantic search
    - SQL-powered filtering and analysis
    - Complete observability of contexts, facts, and ReAct cycles
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the observability store.

        Args:
            base_dir: Base directory for storage. Defaults to ~/.abstractllm
        """
        self.base_dir = base_dir or Path.home() / ".abstractllm"
        self.db_path = self.base_dir / "lancedb"
        self.db_path.mkdir(parents=True, exist_ok=True)

        try:
            self.db = lancedb.connect(str(self.db_path))
            self.embedder = EmbeddingManager(cache_dir=self.base_dir / "embeddings")
            self._init_tables()
            logger.info(f"ObservabilityStore initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ObservabilityStore: {e}")
            raise

    def _init_tables(self) -> None:
        """Initialize all required tables if they don't exist."""

        # Users table
        if "users" not in self.db.table_names():
            users_data = [{
                "user_id": "system",
                "username": "system",
                "created_at": datetime.now(),
                "metadata": json.dumps({})  # Store as JSON string
            }]
            self.db.create_table("users", data=users_data)
            logger.debug("Created users table")

        # Sessions table
        if "sessions" not in self.db.table_names():
            sessions_data = [{
                "session_id": str(uuid.uuid4()),
                "user_id": "system",
                "created_at": datetime.now(),
                "last_active": datetime.now(),
                "provider": "system",
                "model": "none",
                "temperature": 0.0,
                "max_tokens": 0,
                "seed": 0,
                "system_prompt": "",
                "metadata": json.dumps({})  # Store as JSON string
            }]
            self.db.create_table("sessions", data=sessions_data)
            logger.debug("Created sessions table")

        # Interactions table (will be created when first interaction is added)
        # ReAct cycles table (will be created when first cycle is added)

    def add_user(self, username: str, metadata: Optional[Dict] = None) -> str:
        """Add a new user and return the user_id.

        Args:
            username: Unique username
            metadata: Additional user metadata

        Returns:
            user_id: Generated UUID for the user
        """
        user_id = str(uuid.uuid4())
        user_data = {
            "user_id": user_id,
            "username": username,
            "created_at": datetime.now(),
            "metadata": json.dumps(metadata or {})  # Store as JSON string
        }

        try:
            table = self.db.open_table("users")
            table.add([user_data])
            logger.info(f"Added user: {username} ({user_id})")
            return user_id
        except Exception as e:
            logger.error(f"Failed to add user {username}: {e}")
            raise

    def add_session(self,
                   user_id: str,
                   provider: str,
                   model: str,
                   temperature: float = 0.7,
                   max_tokens: int = 4096,
                   seed: Optional[int] = None,
                   system_prompt: str = "",
                   metadata: Optional[Dict] = None) -> str:
        """Add a new session and return the session_id.

        Args:
            user_id: User who owns this session
            provider: LLM provider (ollama, openai, etc.)
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            seed: Random seed for reproducibility
            system_prompt: System prompt for the session
            metadata: Additional session metadata

        Returns:
            session_id: Generated UUID for the session
        """
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now(),
            "last_active": datetime.now(),
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed or 0,
            "system_prompt": system_prompt,
            "metadata": json.dumps(metadata or {})  # Store as JSON string
        }

        try:
            table = self.db.open_table("sessions")
            table.add([session_data])
            logger.info(f"Added session: {session_id} for user {user_id}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to add session: {e}")
            raise

    def add_interaction(self, interaction_data: Dict[str, Any]) -> None:
        """Add an interaction to the store."""
        normalized_data = self._normalize_interaction_data(interaction_data)

        if "interactions" not in self.db.table_names():
            self.db.create_table("interactions", data=[normalized_data])
        else:
            self.db.open_table("interactions").add([normalized_data])

        self._update_session_activity(normalized_data["session_id"])

    def _normalize_interaction_data(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize interaction data."""
        data = interaction_data.copy()

        # Normalize token_usage
        usage = data.get('token_usage', {})
        data['token_usage'] = {
            'completion_tokens': int(usage.get('completion_tokens') or 0),
            'prompt_tokens': int(usage.get('prompt_tokens') or 0),
            'total_tokens': int(usage.get('total_tokens') or 0),
            'model': str(usage.get('model', '')),
            'provider': str(usage.get('provider', ''))
        }

        # Normalize metadata
        meta = data.get('metadata', {})
        data['metadata'] = {
            'provider': str(meta.get('provider', '')),
            'model': str(meta.get('model', '')),
            'endpoint': str(meta.get('endpoint') or ''),
            'step_id': str(meta.get('step_id', '')),
            'step_number': int(meta.get('step_number') or 0),
            'reasoning_phase': str(meta.get('reasoning_phase', '')),
            'react_cycle_id': str(meta.get('react_cycle_id', '')),
            'tool_calls_count': int(meta.get('tool_calls_count') or 0),
            'facts_count': int(meta.get('facts_count') or 0),
            'scratchpad_file': str(meta.get('scratchpad_file', ''))
        }

        # Ensure facts_extracted is list
        data['facts_extracted'] = data.get('facts_extracted', [])

        return data


    def add_react_cycle(self, react_data: Dict[str, Any]) -> None:
        """Add a ReAct cycle to the store."""
        if "react_cycles" not in self.db.table_names():
            self.db.create_table("react_cycles", data=[react_data])
        else:
            self.db.open_table("react_cycles").add([react_data])

    def search_by_timeframe(self,
                           start_time: datetime,
                           end_time: datetime,
                           user_id: Optional[str] = None,
                           session_id: Optional[str] = None) -> pd.DataFrame:
        """Search interactions within a specific timeframe.

        Args:
            start_time: Start of time range
            end_time: End of time range
            user_id: Optional user filter
            session_id: Optional session filter

        Returns:
            DataFrame with matching interactions
        """
        if "interactions" not in self.db.table_names():
            return pd.DataFrame()

        try:
            table = self.db.open_table("interactions")

            # Build query with proper timestamp format
            start_iso = start_time.isoformat() if hasattr(start_time, 'isoformat') else str(start_time)
            end_iso = end_time.isoformat() if hasattr(end_time, 'isoformat') else str(end_time)
            query = f"timestamp >= to_timestamp('{start_iso}') AND timestamp <= to_timestamp('{end_iso}')"
            if user_id:
                query += f" AND user_id = '{user_id}'"
            if session_id:
                query += f" AND session_id = '{session_id}'"

            result = table.search().where(query).to_pandas()
            logger.debug(f"Timeframe search returned {len(result)} interactions")
            return result
        except Exception as e:
            logger.error(f"Failed timeframe search: {e}")
            return pd.DataFrame()

    def semantic_search(self,
                       query: str,
                       limit: int = 10,
                       filters: Optional[Dict] = None) -> List[Dict]:
        """Search interactions by semantic similarity with optional SQL filters.

        Args:
            query: Search query text
            limit: Maximum number of results
            filters: Optional filters (user_id, session_id, start_time, end_time)

        Returns:
            List of matching interactions with similarity scores
        """
        if "interactions" not in self.db.table_names():
            return []

        try:
            # Get all interactions first
            table = self.db.open_table("interactions")
            all_data = table.to_pandas()

            if len(all_data) == 0:
                return []

            # Extract clean conversation content for each interaction
            clean_contents = []
            for _, row in all_data.iterrows():
                clean_content = self._extract_clean_conversation_content(row.to_dict())
                clean_contents.append(clean_content)

            # Generate embeddings for clean content
            clean_embeddings = []
            for content in clean_contents:
                if content.strip():
                    embedding = self.embedder.embed_text(content)
                    clean_embeddings.append(embedding)
                else:
                    # Use zero vector for empty content
                    clean_embeddings.append([0.0] * len(self.embedder.embed_text("test")))

            # Generate query embedding - use content directly
            query_embedding = self.embedder.embed_text(query)

            # Calculate similarities
            import numpy as np
            similarities = []
            for i, content_embedding in enumerate(clean_embeddings):
                if len(content_embedding) > 0:
                    # Calculate cosine similarity
                    dot_product = np.dot(query_embedding, content_embedding)
                    norm_query = np.linalg.norm(query_embedding)
                    norm_content = np.linalg.norm(content_embedding)
                    similarity = dot_product / (norm_query * norm_content) if norm_query > 0 and norm_content > 0 else 0
                    similarities.append((i, similarity))
                else:
                    similarities.append((i, 0))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Convert top results back to LanceDB format and deduplicate
            results = []
            seen_interactions = set()

            for i, (row_idx, similarity) in enumerate(similarities):
                if len(results) >= limit:
                    break

                row_data = all_data.iloc[row_idx].to_dict()

                # Deduplicate by interaction_id
                interaction_id = row_data.get('interaction_id')
                if interaction_id in seen_interactions:
                    continue

                # Deduplicate by content similarity (avoid near-duplicate conversations)
                query = row_data.get('query', '')
                response = row_data.get('response', '')
                content_hash = hash(f"{query[:100]}{response[:100]}")

                if content_hash in seen_interactions:
                    continue

                seen_interactions.add(interaction_id)
                seen_interactions.add(content_hash)

                # Add distance (lower is better, so we use 1 - similarity)
                row_data['_distance'] = 1.0 - similarity
                results.append(row_data)

            logger.debug(f"Semantic search returned {len(results)} unique interactions after deduplication")
            return results

        except Exception as e:
            logger.error(f"Failed semantic search: {e}")
            return []

    def _extract_clean_conversation_content(self, interaction_data: Dict[str, Any]) -> str:
        """Extract clean conversation content (user query + assistant response) for search."""
        # Get query and response
        query = interaction_data.get('query', '')
        response = interaction_data.get('response', '')

        # Clean the query - extract just the user's actual question
        clean_query = query
        if 'User:' in query:
            # Extract from context format
            lines = query.split('\n') if isinstance(query, str) else []
            for line in reversed(lines):
                if line.strip().startswith('User:'):
                    clean_query = line.replace('User:', '').strip()
                    break
        elif 'Session:' in query:
            # Extract from session context - find the actual user input
            context = interaction_data.get('context_verbatim', '')
            if 'User:' in context:
                lines = context.split('\n')
                for line in reversed(lines):
                    if line.strip().startswith('User:'):
                        clean_query = line.replace('User:', '').strip()
                        break

        # Clean the response - remove any metadata
        clean_response = response
        if response and response != "Processing...":
            clean_response = response

        # Combine for search - CONTENT ONLY, no format tags
        clean_content = f"{clean_query} {clean_response}".strip()
        return clean_content

    def _chunk_response_content(self, response: str, max_chunk_size: int = 300) -> List[str]:
        """Chunk response into semantic segments based on SOTA 2025 practices."""
        if not response or len(response.strip()) < 50:
            return [response]

        chunks = []

        # Split by paragraphs first (double newlines or markdown headers)
        paragraphs = []
        current_paragraph = []

        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not line:  # Empty line - end of paragraph
                if current_paragraph:
                    paragraphs.append('\n'.join(current_paragraph))
                    current_paragraph = []
            elif line.startswith('#') or line.startswith('##'):  # Markdown headers - new section
                if current_paragraph:
                    paragraphs.append('\n'.join(current_paragraph))
                    current_paragraph = []
                current_paragraph.append(line)
            else:
                current_paragraph.append(line)

        # Add final paragraph
        if current_paragraph:
            paragraphs.append('\n'.join(current_paragraph))

        # Further chunk large paragraphs if needed
        for paragraph in paragraphs:
            if len(paragraph) <= max_chunk_size:
                chunks.append(paragraph)
            else:
                # Split long paragraphs by sentences
                sentences = paragraph.split('. ')
                current_chunk = ""

                for sentence in sentences:
                    test_chunk = f"{current_chunk}. {sentence}".strip('. ')

                    if len(test_chunk) <= max_chunk_size:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence

                if current_chunk:
                    chunks.append(current_chunk)

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _find_best_matching_chunks(self, query: str, chunks: List[str], max_chunks: int = 2) -> List[str]:
        """Find the best matching chunks for a query using embedding similarity."""
        if not chunks:
            return []

        if len(chunks) <= max_chunks:
            return chunks

        try:
            # Generate query embedding for chunk matching
            query_embedding = self.embedder.embed_text(query)
            chunk_similarities = []

            for i, chunk in enumerate(chunks):
                chunk_embedding = self.embedder.embed_text(chunk)

                # Calculate cosine similarity
                import numpy as np
                dot_product = np.dot(query_embedding, chunk_embedding)
                norm_query = np.linalg.norm(query_embedding)
                norm_chunk = np.linalg.norm(chunk_embedding)
                similarity = dot_product / (norm_query * norm_chunk) if norm_query > 0 and norm_chunk > 0 else 0

                chunk_similarities.append((i, similarity, chunk))

            # Sort by similarity and take top chunks
            chunk_similarities.sort(key=lambda x: x[1], reverse=True)

            # Return best matching chunks in original order
            best_indices = sorted([x[0] for x in chunk_similarities[:max_chunks]])
            return [chunks[i] for i in best_indices]

        except Exception as e:
            logger.warning(f"Failed to find best matching chunks: {e}")
            return chunks[:max_chunks]

    def search_react_cycles(self,
                           query: str,
                           limit: int = 10,
                           filters: Optional[Dict] = None) -> List[Dict]:
        """Search ReAct cycles by semantic similarity in reasoning.

        Args:
            query: Search query text
            limit: Maximum number of results
            filters: Optional filters (user_id, session_id, start_time, end_time)

        Returns:
            List of matching ReAct cycles with similarity scores
        """
        if "react_cycles" not in self.db.table_names():
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(query)
            table = self.db.open_table("react_cycles")

            # Start with vector search on scratchpad embeddings
            search = table.search(query_embedding).limit(limit)

            # Apply filters if provided
            if filters:
                if "start_time" in filters:
                    start_iso = filters['start_time'].isoformat() if hasattr(filters['start_time'], 'isoformat') else str(filters['start_time'])
                    search = search.where(f"timestamp >= to_timestamp('{start_iso}')")
                if "end_time" in filters:
                    end_iso = filters['end_time'].isoformat() if hasattr(filters['end_time'], 'isoformat') else str(filters['end_time'])
                    search = search.where(f"timestamp <= to_timestamp('{end_iso}')")

            results = search.to_list()
            logger.debug(f"ReAct search returned {len(results)} cycles")
            return results

        except Exception as e:
            logger.error(f"Failed ReAct search: {e}")
            return []

    def search_combined(self,
                       query: str,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       user_id: Optional[str] = None,
                       session_id: Optional[str] = None,
                       limit: int = 10) -> Dict[str, Any]:
        """Combined search using both semantic similarity and time/user filters.

        Args:
            query: Search query text
            start_time: Optional start time filter
            end_time: Optional end time filter
            user_id: Optional user filter
            session_id: Optional session filter
            limit: Maximum number of results per category

        Returns:
            Dictionary with interactions and react_cycles results
        """
        filters = {}
        if start_time:
            filters["start_time"] = start_time
        if end_time:
            filters["end_time"] = end_time
        if user_id:
            filters["user_id"] = user_id
        if session_id:
            filters["session_id"] = session_id

        return {
            "interactions": self.semantic_search(query, limit, filters),
            "react_cycles": self.search_react_cycles(query, limit, filters),
            "query": query,
            "filters": filters,
            "timestamp": datetime.now()
        }

    def _update_session_activity(self, session_id: Optional[str]) -> None:
        """Update last_active timestamp for a session."""
        if not session_id:
            return

        try:
            sessions_table = self.db.open_table("sessions")
            sessions_df = sessions_table.to_pandas()

            if session_id in sessions_df['session_id'].values:
                # Update last_active for the session
                sessions_df.loc[sessions_df['session_id'] == session_id, 'last_active'] = datetime.now()

                # Replace the table (LanceDB doesn't support UPDATE yet)
                self.db.drop_table("sessions")
                self.db.create_table("sessions", data=sessions_df.to_dict('records'))
        except Exception as e:
            logger.debug(f"Failed to update session activity: {e}")

    def get_sessions(self, user_id: Optional[str] = None) -> pd.DataFrame:
        """Get sessions, optionally filtered by user.

        Args:
            user_id: Optional user filter

        Returns:
            DataFrame with session data
        """
        if "sessions" not in self.db.table_names():
            return pd.DataFrame()

        try:
            table = self.db.open_table("sessions")
            if user_id:
                return table.search().where(f"user_id = '{user_id}'").to_pandas()
            else:
                return table.to_pandas()
        except Exception as e:
            logger.error(f"Failed to get sessions: {e}")
            return pd.DataFrame()

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with statistics for each table
        """
        stats = {}
        try:
            for table_name in self.db.table_names():
                table = self.db.open_table(table_name)
                df = table.to_pandas()
                stats[table_name] = {
                    "count": len(df),
                    "size_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            stats = {"error": str(e)}

        return stats

    def cleanup_old_data(self, days: int = 30) -> int:
        """Remove data older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of interactions deleted
        """
        cutoff = datetime.now() - timedelta(days=days)
        deleted_count = 0

        try:
            if "interactions" in self.db.table_names():
                table = self.db.open_table("interactions")
                df = table.to_pandas()
                original_count = len(df)

                # Filter recent data
                df_filtered = df[df['timestamp'] > cutoff]
                deleted_count = original_count - len(df_filtered)

                if deleted_count > 0:
                    # Recreate table with filtered data
                    self.db.drop_table("interactions")
                    if len(df_filtered) > 0:
                        self.db.create_table("interactions", data=df_filtered.to_dict('records'))

                    logger.info(f"Cleaned up {deleted_count} old interactions")

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

        return deleted_count

    def get_interaction_by_id(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific interaction by its exact ID."""
        if "interactions" not in self.db.table_names():
            return None

        try:
            table = self.db.open_table("interactions")

            # Try exact match first
            results = table.search().where(f"interaction_id = '{interaction_id}'").limit(1).to_list()
            if results:
                return results[0]

            # Try with cycle_ prefix if not found
            if not interaction_id.startswith('cycle_'):
                cycle_id = f'cycle_{interaction_id}'
                results = table.search().where(f"interaction_id = '{cycle_id}'").limit(1).to_list()
                if results:
                    return results[0]

            # Try without cycle_ prefix if it has one
            if interaction_id.startswith('cycle_'):
                short_id = interaction_id[6:]
                results = table.search().where(f"interaction_id = '{short_id}'").limit(1).to_list()
                if results:
                    return results[0]

            return None

        except Exception as e:
            logger.error(f"Failed to get interaction by ID: {e}")
            return None