"""
Configuration management utilities for AbstractLLM.
"""

import os
from typing import Dict, Any, Optional, Union, TypeVar, Generic, List, Set
from enum import Enum
import logging

from abstractllm.enums import ModelParameter


# Configure logger
logger = logging.getLogger("abstractllm.utils.config")

# Generic value type for configuration
T = TypeVar('T')

class ConfigurationManager:
    """
    Parameter management for AbstractLLM providers.
    Handles parameter storage, retrieval, and updates without provider-specific logic.
    """
    
    def __init__(self, initial_config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            initial_config: Optional initial configuration
        """
        self._config: Dict[Union[str, ModelParameter], Any] = {}
        
        if initial_config:
            self.update_config(initial_config)
    
    def get_param(self, param: Union[str, ModelParameter], default: Optional[T] = None) -> Optional[T]:
        """
        Get a parameter value from configuration, supporting both enum and string keys.
        
        Args:
            param: Parameter to retrieve (either ModelParameter enum or string)
            default: Default value if parameter is not found
            
        Returns:
            Parameter value or default
        """
        # Handle enum parameter
        if isinstance(param, ModelParameter):
            # Enum keys have precedence over string keys
            enum_value = self._config.get(param)
            if enum_value is not None:
                return enum_value
            return self._config.get(param.value, default)
        else:
            # It's a string parameter
            return self._config.get(param, default)
    
    def update_config(self, updates: Dict[Union[str, ModelParameter], Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Updates to apply
        """
        self._config.update(updates)
    
    def get_config(self) -> Dict[Union[str, ModelParameter], Any]:
        """Get a copy of the current configuration."""
        return self._config.copy()
    
    def merge_with_defaults(self, defaults: Dict[Union[str, ModelParameter], Any]) -> None:
        """
        Merge the current configuration with default values.
        Only applies defaults for parameters that are not already set.
        
        Args:
            defaults: Default values to merge
        """
        for k, v in defaults.items():
            if k not in self._config and (isinstance(k, ModelParameter) and k.value not in self._config):
                self._config[k] = v
    
