"""
Base Cognitive Abstraction

Provides common functionality for all cognitive functions including:
- LLM session management
- Error handling
- Performance monitoring
- Integration with AbstractLLM
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)


class CognitiveError(Exception):
    """Base exception for cognitive function errors"""
    pass


class BaseCognitive(ABC):
    """Base class for all cognitive abstractions"""

    def __init__(self, llm_provider: str = "ollama", model: str = "granite3.3:2b",
                 **llm_config):
        """
        Initialize base cognitive functionality

        Args:
            llm_provider: LLM provider to use (default: ollama)
            model: Model to use (default: granite3.3:2b for speed)
            **llm_config: Additional LLM configuration
        """
        self.provider = llm_provider
        self.model = model
        self.config = {
            'temperature': 0.1,  # Low for consistency
            'max_tokens': 2048,
            **llm_config
        }

        # Performance tracking
        self.execution_times = []
        self.error_count = 0
        self.total_calls = 0

        # Initialize session lazily
        self._session = None

    @property
    def session(self):
        """Lazy initialization of LLM session"""
        if self._session is None:
            from abstractllm.factory import create_session
            try:
                self._session = create_session(
                    self.provider,
                    model=self.model,
                    **self.config
                )
                logger.debug(f"Initialized cognitive session: {self.provider}/{self.model}")
            except Exception as e:
                raise CognitiveError(f"Failed to initialize LLM session: {e}")
        return self._session

    def _execute_with_monitoring(self, operation_name: str, operation_func):
        """Execute operation with performance monitoring and error handling"""
        start_time = time.time()
        self.total_calls += 1

        try:
            result = operation_func()
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)

            logger.debug(f"{operation_name} completed in {execution_time:.3f}s")
            return result

        except Exception as e:
            self.error_count += 1
            execution_time = time.time() - start_time
            logger.error(f"{operation_name} failed after {execution_time:.3f}s: {e}")
            raise CognitiveError(f"{operation_name} failed: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this cognitive function"""
        if not self.execution_times:
            return {
                "total_calls": self.total_calls,
                "error_count": self.error_count,
                "error_rate": 0.0,
                "avg_execution_time": 0.0,
                "min_execution_time": 0.0,
                "max_execution_time": 0.0
            }

        return {
            "total_calls": self.total_calls,
            "successful_calls": len(self.execution_times),
            "error_count": self.error_count,
            "error_rate": self.error_count / self.total_calls,
            "avg_execution_time": sum(self.execution_times) / len(self.execution_times),
            "min_execution_time": min(self.execution_times),
            "max_execution_time": max(self.execution_times),
            "total_execution_time": sum(self.execution_times)
        }

    def reset_performance_stats(self):
        """Reset performance tracking"""
        self.execution_times = []
        self.error_count = 0
        self.total_calls = 0

    @abstractmethod
    def _process(self, *args, **kwargs):
        """Main processing method to be implemented by subclasses"""
        pass

    def process(self, *args, **kwargs):
        """Public interface for processing with monitoring"""
        operation_name = f"{self.__class__.__name__}.process"
        return self._execute_with_monitoring(
            operation_name,
            lambda: self._process(*args, **kwargs)
        )

    def is_available(self) -> bool:
        """Check if the cognitive function is available"""
        try:
            # Test session initialization
            _ = self.session
            return True
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the underlying model"""
        return {
            "provider": self.provider,
            "model": self.model,
            "config": str(self.config)
        }


class PromptTemplate:
    """Helper class for managing optimized prompts"""

    def __init__(self, template: str, required_vars: Optional[list] = None):
        self.template = template
        self.required_vars = required_vars or []

    def format(self, **kwargs) -> str:
        """Format template with provided variables"""
        # Check required variables
        missing = [var for var in self.required_vars if var not in kwargs]
        if missing:
            raise ValueError(f"Missing required template variables: {missing}")

        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Template formatting error: {e}")

    def validate(self, **kwargs) -> bool:
        """Validate that all required variables are provided"""
        return all(var in kwargs for var in self.required_vars)


class CognitiveConfig:
    """Configuration helper for cognitive functions"""

    DEFAULT_MODELS = {
        "summarizer": "granite3.3:2b",
        "facts_extractor": "granite3.3:2b",
        "value_resonance": "granite3.3:2b"
    }

    DEFAULT_CONFIGS = {
        "summarizer": {"temperature": 0.2, "max_tokens": 1024},
        "facts_extractor": {"temperature": 0.1, "max_tokens": 2048},
        "value_resonance": {"temperature": 0.2, "max_tokens": 1024}
    }

    @classmethod
    def get_default_model(cls, cognitive_type: str) -> str:
        """Get default model for cognitive type"""
        return cls.DEFAULT_MODELS.get(cognitive_type, "granite3.3:2b")

    @classmethod
    def get_default_config(cls, cognitive_type: str) -> Dict[str, Any]:
        """Get default config for cognitive type"""
        return cls.DEFAULT_CONFIGS.get(cognitive_type, {"temperature": 0.1})


def check_cognitive_dependencies() -> Dict[str, bool]:
    """Check if cognitive function dependencies are available"""
    dependencies = {}

    # AbstractLLM factory is always available (internal module)
    from abstractllm.factory import create_session
    dependencies["abstractllm"] = True

    try:
        # Test granite3.3:2b availability
        session = create_session("ollama", model="granite3.3:2b")
        dependencies["granite3.3"] = True
    except Exception:
        dependencies["granite3.3"] = False

    return dependencies