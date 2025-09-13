"""
SOTA Retry Strategies for AbstractLLM.

Implements:
- Exponential backoff with jitter
- Circuit breaker pattern
- Smart retry with error feedback
- Validation with re-prompting
- Provider failover strategies
- Adaptive retry based on error type

Based on 2025 best practices from Tenacity, production LLM systems.
"""

import time
import random
import logging
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import re
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryableError(Enum):
    """Types of errors that can be retried."""
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK = "network"
    VALIDATION = "validation"
    PARSING = "parsing"
    TOOL_EXECUTION = "tool_execution"
    MODEL_CONFUSION = "model_confusion"
    CONTEXT_LENGTH = "context_length"
    UNKNOWN = "unknown"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_attempts: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    
    # Circuit breaker settings
    failure_threshold: int = 5  # failures before opening circuit
    recovery_timeout: float = 60.0  # seconds before trying again
    half_open_max_calls: int = 3  # calls to test in half-open state
    
    # Smart retry settings
    include_error_feedback: bool = True
    simplify_on_retry: bool = True
    reduce_temperature: bool = True
    
    # Validation retry
    validation_retries: int = 5
    re_prompt_on_validation_failure: bool = True
    
    # Provider failover
    fallback_providers: List[str] = field(default_factory=list)
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff."""
        delay = min(
            self.initial_delay * (self.exponential_base ** (attempt - 1)),
            self.max_delay
        )
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            delay *= (0.5 + random.random())
        
        return delay


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for preventing cascading failures."""
    
    config: RetryConfig
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    half_open_calls: int = 0
    
    def record_success(self):
        """Record successful call."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_calls = 0
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            logger.info("Circuit breaker closed after successful recovery")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker reopened after failure in half-open state")
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time:
                time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
                if time_since_failure >= self.config.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info("Circuit breaker entering half-open state")
                    return True
            return False
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.half_open_calls < self.config.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False
        
        return False


class RetryManager:
    """Manages retry strategies for different components."""
    
    def __init__(self, default_config: Optional[RetryConfig] = None):
        """Initialize retry manager."""
        self.config = default_config or RetryConfig()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: List[Dict[str, Any]] = []
    
    def get_circuit_breaker(self, key: str) -> CircuitBreaker:
        """Get or create circuit breaker for a key."""
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = CircuitBreaker(self.config)
        return self.circuit_breakers[key]
    
    def classify_error(self, error: Exception) -> RetryableError:
        """Classify error type for appropriate retry strategy."""
        error_str = str(error).lower()
        
        if "rate limit" in error_str or "429" in error_str:
            return RetryableError.RATE_LIMIT
        elif "timeout" in error_str or "timed out" in error_str:
            return RetryableError.TIMEOUT
        elif "network" in error_str or "connection" in error_str:
            return RetryableError.NETWORK
        elif "validation" in error_str or "schema" in error_str:
            return RetryableError.VALIDATION
        elif "json" in error_str or "parse" in error_str:
            return RetryableError.PARSING
        elif "tool" in error_str or "function" in error_str:
            return RetryableError.TOOL_EXECUTION
        elif "context" in error_str or "token" in error_str:
            return RetryableError.CONTEXT_LENGTH
        else:
            return RetryableError.UNKNOWN
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if error should be retried."""
        error_type = self.classify_error(error)
        
        # Always retry rate limits and timeouts up to max
        if error_type in [RetryableError.RATE_LIMIT, RetryableError.TIMEOUT]:
            return attempt < self.config.max_attempts
        
        # Retry network errors with backoff
        if error_type == RetryableError.NETWORK:
            return attempt < self.config.max_attempts
        
        # Retry validation/parsing with modifications
        if error_type in [RetryableError.VALIDATION, RetryableError.PARSING]:
            return attempt < self.config.validation_retries
        
        # Retry tool execution once
        if error_type == RetryableError.TOOL_EXECUTION:
            return attempt < 2
        
        # Don't retry context length errors
        if error_type == RetryableError.CONTEXT_LENGTH:
            return False
        
        # Default: retry unknown errors cautiously
        return attempt < min(2, self.config.max_attempts)
    
    def retry_with_backoff(self, 
                          func: Callable[..., T],
                          *args,
                          key: Optional[str] = None,
                          **kwargs) -> T:
        """
        Execute function with exponential backoff retry.
        
        Args:
            func: Function to execute
            key: Optional key for circuit breaker
            *args, **kwargs: Arguments for function
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        circuit_breaker = self.get_circuit_breaker(key) if key else None
        last_error = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            # Check circuit breaker
            if circuit_breaker and not circuit_breaker.can_execute():
                raise RuntimeError(f"Circuit breaker open for {key}")
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Record success
                if circuit_breaker:
                    circuit_breaker.record_success()
                
                return result
                
            except Exception as e:
                last_error = e
                error_type = self.classify_error(e)
                
                # Record error
                self.error_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "attempt": attempt,
                    "error_type": error_type.value,
                    "error": str(e),
                    "key": key
                })
                
                # Record failure in circuit breaker
                if circuit_breaker:
                    circuit_breaker.record_failure()
                
                # Check if should retry
                if not self.should_retry(e, attempt):
                    logger.error(f"Not retrying after {error_type.value} error: {e}")
                    raise
                
                # Calculate delay
                delay = self.config.get_delay(attempt)
                logger.warning(f"Attempt {attempt} failed with {error_type.value}. "
                             f"Retrying in {delay:.2f}s...")
                
                # Wait before retry
                time.sleep(delay)
        
        # All retries exhausted
        logger.error(f"All {self.config.max_attempts} attempts failed")
        raise last_error


def retry_structured_response(retry_manager: RetryManager):
    """
    Decorator for retrying structured response generation.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            config = retry_manager.config
            last_error = None
            
            # Track modifications for retry
            modified_kwargs = kwargs.copy()
            
            for attempt in range(1, config.validation_retries + 1):
                try:
                    result = func(*args, **modified_kwargs)
                    
                    # Validate if validator provided
                    if "validator" in modified_kwargs:
                        validator = modified_kwargs["validator"]
                        if not validator(result):
                            raise ValueError("Validation failed")
                    
                    return result
                    
                except (ValueError, json.JSONDecodeError) as e:
                    last_error = e
                    logger.warning(f"Structured response attempt {attempt} failed: {e}")
                    
                    if attempt >= config.validation_retries:
                        break
                    
                    # Modify for retry
                    if config.re_prompt_on_validation_failure:
                        # Add error feedback to prompt
                        if config.include_error_feedback:
                            error_feedback = f"\n\nPrevious attempt failed: {e}\nPlease correct the output."
                            if "prompt" in modified_kwargs:
                                modified_kwargs["prompt"] += error_feedback
                    
                    if config.simplify_on_retry:
                        # Simplify schema if possible
                        if "schema" in modified_kwargs and isinstance(modified_kwargs["schema"], dict):
                            # Remove optional fields
                            schema = modified_kwargs["schema"].copy()
                            if "properties" in schema:
                                required = schema.get("required", [])
                                schema["properties"] = {
                                    k: v for k, v in schema["properties"].items()
                                    if k in required
                                }
                                modified_kwargs["schema"] = schema
                    
                    if config.reduce_temperature:
                        # Lower temperature for more deterministic output
                        if "temperature" in modified_kwargs:
                            modified_kwargs["temperature"] *= 0.7
                        else:
                            modified_kwargs["temperature"] = 0.3
                    
                    # Add delay
                    delay = config.get_delay(attempt)
                    time.sleep(delay)
            
            raise last_error
        
        return wrapper
    return decorator


def retry_tool_execution(retry_manager: RetryManager):
    """
    Decorator for retrying tool execution with fallbacks.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            config = retry_manager.config
            last_error = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    # Try native tool calling first
                    result = func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    last_error = e
                    error_type = retry_manager.classify_error(e)
                    
                    logger.warning(f"Tool execution attempt {attempt} failed: {e}")
                    
                    if attempt >= config.max_attempts:
                        break
                    
                    # Modify strategy for retry
                    if error_type == RetryableError.TOOL_EXECUTION:
                        # Fallback to prompted tools
                        if "use_native_tools" in kwargs:
                            kwargs["use_native_tools"] = False
                            logger.info("Falling back to prompted tool mode")
                    
                    elif error_type == RetryableError.MODEL_CONFUSION:
                        # Simplify tool descriptions
                        if "tools" in kwargs:
                            # Keep only essential tools
                            tools = kwargs["tools"]
                            if len(tools) > 3:
                                kwargs["tools"] = tools[:3]
                                logger.info(f"Reduced tools from {len(tools)} to 3")
                    
                    # Add delay
                    delay = config.get_delay(attempt)
                    time.sleep(delay)
            
            raise last_error
        
        return wrapper
    return decorator


class AdaptiveRetryStrategy:
    """
    Adaptive retry strategy that learns from failures.
    """
    
    def __init__(self):
        """Initialize adaptive strategy."""
        self.failure_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.success_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.adaptation_rules: List[Callable] = []
    
    def record_failure(self, context: Dict[str, Any], error: Exception):
        """Record failure pattern."""
        key = self._get_pattern_key(context)
        if key not in self.failure_patterns:
            self.failure_patterns[key] = []
        
        self.failure_patterns[key].append({
            "context": context,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        })
    
    def record_success(self, context: Dict[str, Any]):
        """Record success pattern."""
        key = self._get_pattern_key(context)
        if key not in self.success_patterns:
            self.success_patterns[key] = []
        
        self.success_patterns[key].append({
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
    
    def suggest_adaptations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest adaptations based on learned patterns."""
        adaptations = {}
        key = self._get_pattern_key(context)
        
        # Check failure history
        if key in self.failure_patterns:
            failures = self.failure_patterns[key]
            
            # If high failure rate, suggest modifications
            if len(failures) > 3:
                # Reduce complexity
                adaptations["reduce_complexity"] = True
                adaptations["lower_temperature"] = 0.3
                adaptations["use_simpler_prompts"] = True
        
        # Check success patterns
        if key in self.success_patterns:
            successes = self.success_patterns[key]
            
            # Use successful configurations
            if successes:
                last_success = successes[-1]["context"]
                adaptations["use_config"] = last_success
        
        return adaptations
    
    def _get_pattern_key(self, context: Dict[str, Any]) -> str:
        """Generate pattern key from context."""
        # Create key from relevant context features
        provider = context.get("provider", "unknown")
        model = context.get("model", "unknown")
        task_type = context.get("task_type", "unknown")
        
        return f"{provider}:{model}:{task_type}"


# Global retry manager instance
default_retry_manager = RetryManager()


def with_retry(key: Optional[str] = None, 
              config: Optional[RetryConfig] = None):
    """
    Simple decorator for adding retry to any function.
    
    Usage:
        @with_retry(key="openai_api")
        def call_api():
            return api.generate(...)
    """
    retry_manager = RetryManager(config) if config else default_retry_manager
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return retry_manager.retry_with_backoff(func, *args, key=key, **kwargs)
        return wrapper
    return decorator