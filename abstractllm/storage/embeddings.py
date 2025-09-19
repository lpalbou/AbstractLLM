"""
Embedding Manager for AbstractLLM
=================================

Provides efficient text embedding with caching for RAG capabilities.
Uses sentence-transformers with LRU caching for performance optimization.

This module is designed to be lightweight and could be used independently
in other AI projects requiring text embeddings.
"""

import hashlib
import pickle
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union
import logging

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError as e:
    raise ImportError(
        "sentence-transformers is required for embedding functionality. "
        "Install with: pip install sentence-transformers"
    ) from e

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages text embeddings with caching for AbstractLLM observability.

    Features:
    - LRU cache for repeated text embeddings
    - Batch processing for efficiency
    - Persistent cache on disk
    - Lightweight model selection for resource efficiency
    """

    def __init__(self,
                 model_name: str = "ibm-granite/granite-embedding-30m-english",
                 cache_dir: Optional[Path] = None,
                 max_cache_size: int = 1000):
        """Initialize the embedding manager.

        Args:
            model_name: SentenceTransformer model name. Default is IBM's 2025 Granite-30M for enterprise-grade performance.
            cache_dir: Directory for persistent cache. Defaults to ~/.abstractllm/embeddings
            max_cache_size: Maximum number of embeddings to cache in memory
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or Path.home() / ".abstractllm" / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size

        # Initialize the model - try offline first to avoid network calls
        try:
            # Try to load from local cache first (no network calls)
            import os
            os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Force transformers to work offline
            os.environ['HF_HUB_OFFLINE'] = '1'  # Force HuggingFace Hub offline

            self.model = SentenceTransformer(model_name)
            logger.info(f"Initialized embedding model: {model_name} (offline mode)")
        except Exception as offline_error:
            logger.warning(f"Offline initialization failed: {offline_error}")
            try:
                # Only try online as last resort if explicitly enabled
                if os.environ.get('ABSTRACTLLM_ALLOW_DOWNLOADS') == '1':
                    # Remove offline flags and try downloading
                    os.environ.pop('TRANSFORMERS_OFFLINE', None)
                    os.environ.pop('HF_HUB_OFFLINE', None)
                    self.model = SentenceTransformer(model_name)
                    logger.info(f"Initialized embedding model: {model_name} (downloaded)")
                else:
                    logger.error(f"Embedding model {model_name} not available offline and downloads disabled")
                    raise RuntimeError(f"Embedding model {model_name} requires download. Set ABSTRACTLLM_ALLOW_DOWNLOADS=1 to enable.")
            except Exception as e:
                logger.error(f"Failed to load embedding model {model_name}: {e}")
                raise

        # Set up persistent cache
        self.cache_file = self.cache_dir / f"{model_name.replace('/', '_')}_cache.pkl"
        self._persistent_cache = self._load_persistent_cache()

    def _load_persistent_cache(self) -> dict:
        """Load persistent cache from disk."""
        try:
            if self.cache_file.exists():
                import builtins
                with builtins.open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.debug(f"Loaded {len(cache)} embeddings from persistent cache")
                return cache
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")

        return {}

    def _save_persistent_cache(self):
        """Save persistent cache to disk."""
        try:
            import builtins
            with builtins.open(self.cache_file, 'wb') as f:
                pickle.dump(self._persistent_cache, f)
            logger.debug(f"Saved {len(self._persistent_cache)} embeddings to persistent cache")
        except Exception as e:
            logger.warning(f"Failed to save persistent cache: {e}")

    def _text_hash(self, text: str) -> str:
        """Generate hash for text caching."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    @lru_cache(maxsize=1000)
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text with LRU caching.

        Args:
            text: Text to embed

        Returns:
            List of float values representing the embedding
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.model.get_sentence_embedding_dimension()

        text_hash = self._text_hash(text)

        # Check persistent cache first
        if text_hash in self._persistent_cache:
            return self._persistent_cache[text_hash]

        try:
            # Generate embedding (suppress progress bar for single text)
            embedding = self.model.encode(text, show_progress_bar=False).tolist()

            # Store in persistent cache
            self._persistent_cache[text_hash] = embedding

            # Periodically save cache (every 10 new embeddings)
            if len(self._persistent_cache) % 10 == 0:
                self._save_persistent_cache()

            logger.debug(f"Generated embedding for text (length: {len(text)})")
            return embedding

        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            # Return zero vector as fallback
            return [0.0] * self.model.get_sentence_embedding_dimension()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts efficiently using batch processing.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings, one for each input text
        """
        if not texts:
            return []

        # Separate cached and uncached texts
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                cached_embeddings[i] = [0.0] * self.model.get_sentence_embedding_dimension()
            else:
                text_hash = self._text_hash(text)
                if text_hash in self._persistent_cache:
                    cached_embeddings[i] = self._persistent_cache[text_hash]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

        # Process uncached texts in batch
        if uncached_texts:
            try:
                batch_embeddings = self.model.encode(uncached_texts, show_progress_bar=False).tolist()

                # Store new embeddings in cache
                for text, embedding, idx in zip(uncached_texts, batch_embeddings, uncached_indices):
                    text_hash = self._text_hash(text)
                    self._persistent_cache[text_hash] = embedding
                    cached_embeddings[idx] = embedding

                logger.debug(f"Generated {len(batch_embeddings)} embeddings in batch")

            except Exception as e:
                logger.error(f"Failed to embed batch: {e}")
                # Fill with zero vectors as fallback
                zero_embedding = [0.0] * self.model.get_sentence_embedding_dimension()
                for idx in uncached_indices:
                    cached_embeddings[idx] = zero_embedding

        # Save cache after batch processing
        self._save_persistent_cache()

        # Return embeddings in original order
        return [cached_embeddings[i] for i in range(len(texts))]

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model.

        Returns:
            Embedding dimension (number of features)
        """
        return self.model.get_sentence_embedding_dimension()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between 0 and 1
        """
        try:
            embedding1 = np.array(self.embed_text(text1))
            embedding2 = np.array(self.embed_text(text2))

            # Compute cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)

            if norm_product == 0:
                return 0.0

            similarity = dot_product / norm_product
            return float(similarity)

        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0

    def get_cache_stats(self) -> dict:
        """Get statistics about the embedding cache.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "persistent_cache_size": len(self._persistent_cache),
            "memory_cache_info": self.embed_text.cache_info(),
            "embedding_dimension": self.get_embedding_dimension(),
            "model_name": self.model_name,
            "cache_file": str(self.cache_file)
        }

    def clear_cache(self):
        """Clear both memory and persistent caches."""
        self.embed_text.cache_clear()
        self._persistent_cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Cleared all embedding caches")

    def __del__(self):
        """Ensure persistent cache is saved when object is destroyed."""
        try:
            self._save_persistent_cache()
        except:
            pass  # Ignore errors during cleanup