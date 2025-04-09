"""
Provider chains and fallback mechanisms for AbstractLLM.

This module provides utilities for creating chains of providers with fallback
mechanisms to ensure robust operation even when some providers fail.
"""

from typing import List, Dict, Any, Optional, Union, Callable, TypeVar, Generic, Tuple
import time
import logging
from enum import Enum

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.factory import create_llm
from abstractllm.exceptions import AbstractLLMError

# Configure logger
logger = logging.getLogger("abstractllm.chain")

# Type for generation results
T = TypeVar('T')


class ProviderSelectionStrategy(str, Enum):
    """Strategies for selecting providers in a chain."""
    
    SEQUENTIAL = "sequential"  # Try providers in order until successful
    FIRST_AVAILABLE = "first_available"  # Use the first provider that's available
    CAPABILITY_MATCH = "capability_match"  # Use the first provider that matches required capabilities
    ROUND_ROBIN = "round_robin"  # Rotate through providers for load balancing
    WEIGHTED = "weighted"  # Use providers according to assigned weights


class ProviderChain:
    """
    A chain of LLM providers with fallback capabilities.
    
    This class allows you to create a chain of providers and specify fallback
    behavior to ensure robust operation even when some providers fail.
    """
    
    def __init__(self, 
                providers: List[Union[str, Tuple[str, Dict[str, Any]], AbstractLLMInterface]],
                strategy: Union[str, ProviderSelectionStrategy] = ProviderSelectionStrategy.SEQUENTIAL,
                required_capabilities: Optional[List[Union[str, ModelCapability]]] = None,
                weights: Optional[List[float]] = None,
                max_retries: int = 3,
                retry_delay: float = 1.0,
                on_provider_change: Optional[Callable[[str, str, Exception], None]] = None):
        """
        Initialize a provider chain.
        
        Args:
            providers: List of providers to use in the chain. Can be provider names,
                     tuples of (provider_name, config), or provider instances.
            strategy: Strategy for selecting providers
            required_capabilities: Capabilities that providers must support to be used
            weights: Weights for providers when using the WEIGHTED strategy
            max_retries: Maximum number of retries per provider
            retry_delay: Delay between retries in seconds
            on_provider_change: Callback function when switching providers
        """
        self.providers: List[AbstractLLMInterface] = []
        self.provider_names: List[str] = []
        self.strategy = strategy if isinstance(strategy, ProviderSelectionStrategy) else ProviderSelectionStrategy(strategy)
        self.required_capabilities = required_capabilities or []
        self.weights = weights or [1.0] * len(providers)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.on_provider_change = on_provider_change
        self._current_index = 0
        
        # Initialize providers
        for provider in providers:
            if isinstance(provider, AbstractLLMInterface):
                # Provider instance
                self.providers.append(provider)
                # Try to determine provider name from instance
                name = self._get_provider_name(provider)
                self.provider_names.append(name)
            elif isinstance(provider, tuple) and len(provider) == 2:
                # Tuple of (provider_name, config)
                provider_name, config = provider
                self.providers.append(create_llm(provider_name, **config))
                self.provider_names.append(provider_name)
            else:
                # Provider name
                provider_name = provider
                self.providers.append(create_llm(provider_name))
                self.provider_names.append(provider_name)
        
        # Validate weights if provided
        if self.strategy == ProviderSelectionStrategy.WEIGHTED:
            if len(self.weights) != len(self.providers):
                raise ValueError(
                    f"Number of weights ({len(self.weights)}) must match " 
                    f"number of providers ({len(self.providers)})"
                )
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                stream: bool = False, 
                **kwargs) -> Union[str, Any]:
        """
        Generate a response using the provider chain.
        
        Args:
            prompt: The prompt to generate from
            system_prompt: Optional system prompt
            stream: Whether to stream the response
            **kwargs: Additional parameters for the provider
            
        Returns:
            Generated response or stream
            
        Raises:
            RuntimeError: If all providers fail
        """
        provider_indices = self._get_provider_indices()
        last_error = None
        
        for idx in provider_indices:
            provider = self.providers[idx]
            provider_name = self.provider_names[idx]
            
            # Check if provider meets capability requirements
            if not self._check_capabilities(provider):
                logger.info(f"Provider {provider_name} does not meet capability requirements, skipping")
                continue
            
            # Try this provider with retries
            for attempt in range(self.max_retries):
                try:
                    result = provider.generate(
                        prompt, 
                        system_prompt=system_prompt, 
                        stream=stream,
                        **kwargs
                    )
                    
                    # Switch current index for round-robin strategy
                    if self.strategy == ProviderSelectionStrategy.ROUND_ROBIN:
                        self._current_index = (idx + 1) % len(self.providers)
                    
                    return result
                
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Provider {provider_name} failed (attempt {attempt+1}/{self.max_retries}): {str(e)}"
                    )
                    
                    # Wait before retrying
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
            
            # All retries failed for this provider, try the next one
            if self.on_provider_change and last_error:
                next_idx = provider_indices[(provider_indices.index(idx) + 1) % len(provider_indices)]
                next_provider = self.provider_names[next_idx] if next_idx < len(self.provider_names) else "none"
                self.on_provider_change(provider_name, next_provider, last_error)
        
        # All providers failed
        raise RuntimeError(
            f"All providers failed to generate a response. Last error: {str(last_error)}"
        )
    
    async def generate_async(self, 
                           prompt: str, 
                           system_prompt: Optional[str] = None, 
                           stream: bool = False, 
                           **kwargs) -> Union[str, Any]:
        """
        Generate a response asynchronously using the provider chain.
        
        Args:
            prompt: The prompt to generate from
            system_prompt: Optional system prompt
            stream: Whether to stream the response
            **kwargs: Additional parameters for the provider
            
        Returns:
            Generated response or stream
            
        Raises:
            RuntimeError: If all providers fail
        """
        provider_indices = self._get_provider_indices()
        last_error = None
        
        for idx in provider_indices:
            provider = self.providers[idx]
            provider_name = self.provider_names[idx]
            
            # Check if provider meets capability requirements
            if not self._check_capabilities(provider):
                logger.info(f"Provider {provider_name} does not meet capability requirements, skipping")
                continue
            
            # Check if async is supported
            capabilities = provider.get_capabilities()
            if not capabilities.get(ModelCapability.ASYNC, False):
                logger.info(f"Provider {provider_name} does not support async generation, skipping")
                continue
            
            # Try this provider with retries
            for attempt in range(self.max_retries):
                try:
                    result = await provider.generate_async(
                        prompt, 
                        system_prompt=system_prompt, 
                        stream=stream,
                        **kwargs
                    )
                    
                    # Switch current index for round-robin strategy
                    if self.strategy == ProviderSelectionStrategy.ROUND_ROBIN:
                        self._current_index = (idx + 1) % len(self.providers)
                    
                    return result
                
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Provider {provider_name} failed (attempt {attempt+1}/{self.max_retries}): {str(e)}"
                    )
                    
                    # Wait before retrying
                    if attempt < self.max_retries - 1:
                        import asyncio
                        await asyncio.sleep(self.retry_delay)
            
            # All retries failed for this provider, try the next one
            if self.on_provider_change and last_error:
                next_idx = provider_indices[(provider_indices.index(idx) + 1) % len(provider_indices)]
                next_provider = self.provider_names[next_idx] if next_idx < len(self.provider_names) else "none"
                self.on_provider_change(provider_name, next_provider, last_error)
        
        # All providers failed
        raise RuntimeError(
            f"All providers failed to generate a response. Last error: {str(last_error)}"
        )
    
    def _get_provider_indices(self) -> List[int]:
        """
        Get provider indices based on the selected strategy.
        
        Returns:
            List of provider indices to try
        """
        if self.strategy == ProviderSelectionStrategy.SEQUENTIAL:
            return list(range(len(self.providers)))
        
        elif self.strategy == ProviderSelectionStrategy.FIRST_AVAILABLE:
            return list(range(len(self.providers)))
        
        elif self.strategy == ProviderSelectionStrategy.CAPABILITY_MATCH:
            # Return providers that match capabilities, in order
            return [i for i, provider in enumerate(self.providers) 
                   if self._check_capabilities(provider)]
        
        elif self.strategy == ProviderSelectionStrategy.ROUND_ROBIN:
            # Start from current index and wrap around
            indices = list(range(len(self.providers)))
            return indices[self._current_index:] + indices[:self._current_index]
        
        elif self.strategy == ProviderSelectionStrategy.WEIGHTED:
            # Sort by weight (descending)
            return [i for i, _ in sorted(
                enumerate(self.weights), 
                key=lambda x: x[1], 
                reverse=True
            )]
        
        # Default to sequential
        return list(range(len(self.providers)))
    
    def _check_capabilities(self, provider: AbstractLLMInterface) -> bool:
        """
        Check if a provider meets the required capabilities.
        
        Args:
            provider: Provider to check
            
        Returns:
            True if the provider meets all required capabilities, False otherwise
        """
        if not self.required_capabilities:
            return True
        
        capabilities = provider.get_capabilities()
        
        for capability in self.required_capabilities:
            # Convert string capability to enum if needed
            if isinstance(capability, str):
                try:
                    capability = ModelCapability(capability)
                except ValueError:
                    # Unknown capability, assume it's not supported
                    return False
            
            # Check if the capability is supported
            if not capabilities.get(capability, False):
                return False
        
        return True
    
    def _get_provider_name(self, provider: AbstractLLMInterface) -> str:
        """
        Get the name of a provider from its instance.
        
        Args:
            provider: Provider instance
            
        Returns:
            Provider name
        """
        class_name = provider.__class__.__name__
        
        if class_name.endswith("Provider"):
            return class_name[:-8].lower()
        
        # Fallback to checking the module path
        module = provider.__class__.__module__
        
        if "openai" in module:
            return "openai"
        elif "anthropic" in module:
            return "anthropic"
        elif "ollama" in module:
            return "ollama"
        elif "huggingface" in module:
            return "huggingface"
        
        # Generic fallback
        return "unknown"


def create_fallback_chain(providers: List[str], **kwargs) -> ProviderChain:
    """
    Create a simple fallback chain that tries providers in sequence.
    
    Args:
        providers: List of provider names to try in order
        **kwargs: Additional parameters for the provider chain
        
    Returns:
        A ProviderChain instance
    """
    return ProviderChain(
        providers=providers, 
        strategy=ProviderSelectionStrategy.SEQUENTIAL,
        **kwargs
    )


def create_capability_chain(required_capabilities: List[Union[str, ModelCapability]], 
                          preferred_providers: Optional[List[str]] = None,
                          **kwargs) -> ProviderChain:
    """
    Create a chain that selects providers based on required capabilities.
    
    Args:
        required_capabilities: List of capabilities that providers must support
        preferred_providers: Optional ordered list of preferred providers
        **kwargs: Additional parameters for the provider chain
        
    Returns:
        A ProviderChain instance
    """
    # Default providers if not specified
    providers = preferred_providers or ["openai", "anthropic", "ollama", "huggingface"]
    
    return ProviderChain(
        providers=providers,
        strategy=ProviderSelectionStrategy.CAPABILITY_MATCH,
        required_capabilities=required_capabilities,
        **kwargs
    )


def create_load_balanced_chain(providers: List[str], 
                             weights: Optional[List[float]] = None,
                             **kwargs) -> ProviderChain:
    """
    Create a chain that distributes load across providers.
    
    Args:
        providers: List of provider names
        weights: Optional weights for providers
        **kwargs: Additional parameters for the provider chain
        
    Returns:
        A ProviderChain instance
    """
    return ProviderChain(
        providers=providers,
        strategy=ProviderSelectionStrategy.ROUND_ROBIN,
        weights=weights,
        **kwargs
    ) 