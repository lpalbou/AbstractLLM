"""
AbstractLLM Storage Module
=========================

Unified observability and storage system with LanceDB backend for RAG capabilities.

This module provides:
- LanceDB-based observability store
- Embedding management with caching
- Time-based and semantic search
- Session and user management
"""

from .lancedb_store import ObservabilityStore
from .embeddings import EmbeddingManager

__all__ = ["ObservabilityStore", "EmbeddingManager"]