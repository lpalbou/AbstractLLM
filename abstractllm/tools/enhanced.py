"""
Enhanced tool system with Pydantic validation and advanced features.

This module provides a sophisticated @tool decorator that matches or exceeds
the capabilities of frameworks like LangChain, Instructor, and Pydantic AI.

Features:
- Automatic Pydantic model generation from type hints
- Rich docstring parsing (Google/NumPy/Sphinx styles)
- Validation with automatic retry on errors
- Timeout and confirmation support
- Streaming responses
- Deprecation warnings
- Tool choice forcing
"""

from typing import (
    Any, Dict, Optional, Union, Callable, Type, TypeVar, 
    get_type_hints, get_origin, get_args, List, AsyncGenerator
)
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum
import inspect
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Core imports
from abstractllm.tools.core import ToolDefinition as BaseToolDefinition

# Try importing optional dependencies
try:
    from pydantic import BaseModel, Field, ValidationError, create_model
    from pydantic.fields import FieldInfo
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    Field = lambda **kwargs: None
    ValidationError = Exception
    FieldInfo = object
    
    def create_model(*args, **kwargs):
        return None

try:
    import docstring_parser
    DOCSTRING_PARSER_AVAILABLE = True
except ImportError:
    DOCSTRING_PARSER_AVAILABLE = False
    docstring_parser = None

logger = logging.getLogger(__name__)


class ToolChoice(Enum):
    """Tool selection strategies for forcing specific tool usage."""
    AUTO = "auto"           # Model decides whether to use tools
    NONE = "none"          # No tools should be used
    REQUIRED = "required"   # Must use at least one tool
    SPECIFIC = "specific"   # Use specific tool(s) only


@dataclass
class ToolContext:
    """Context passed to tools that require session information."""
    session_id: str
    user_id: Optional[str] = None
    memory: Optional[Any] = None  # HierarchicalMemory if available
    provider: Optional[Any] = None  # AbstractLLMInterface
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class EnhancedToolDefinition(BaseToolDefinition):
    """
    Enhanced tool definition with validation and advanced features.
    
    Extends the base ToolDefinition with Pydantic models, retry logic,
    confirmation requirements, and other production features.
    """
    
    # Validation
    pydantic_model: Optional[Type[BaseModel]] = None
    validate_args: bool = True
    
    # Retry configuration
    retry_on_error: bool = True
    max_retries: int = 3
    retry_delay: float = 0.5  # seconds
    
    # Execution configuration
    timeout: Optional[float] = None
    requires_confirmation: bool = False
    requires_context: bool = False
    
    # Response handling
    stream_response: bool = False
    response_model: Optional[Type[BaseModel]] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    deprecated: bool = False
    deprecation_message: Optional[str] = None
    
    # Usage hints for LLMs
    examples: List[Dict[str, Any]] = field(default_factory=list)
    when_to_use: Optional[str] = None
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate arguments using Pydantic model if available.
        
        Args:
            arguments: Raw arguments to validate
            
        Returns:
            Validated and potentially coerced arguments
            
        Raises:
            ToolValidationError: If validation fails
        """
        if not self.validate_args:
            return arguments
            
        if self.pydantic_model and PYDANTIC_AVAILABLE:
            try:
                validated = self.pydantic_model(**arguments)
                return validated.model_dump() if hasattr(validated, 'model_dump') else validated.dict()
            except ValidationError as e:
                # Format errors for LLM understanding
                errors = []
                for error in e.errors():
                    loc = ".".join(str(l) for l in error["loc"])
                    errors.append(f"{loc}: {error['msg']} (got {error.get('input', 'invalid input')})")
                raise ToolValidationError(self.name, errors, arguments)
        
        # Fallback to basic validation using JSON schema
        return self._validate_with_schema(arguments)
    
    def _validate_with_schema(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Basic validation using JSON schema when Pydantic not available."""
        required = self.parameters.get("required", [])
        properties = self.parameters.get("properties", {})
        
        # Check required fields
        missing = [p for p in required if p not in arguments]
        if missing:
            raise ToolValidationError(
                self.name, 
                [f"Missing required parameter: {p}" for p in missing],
                arguments
            )
        
        # Check types (basic)
        errors = []
        for param, value in arguments.items():
            if param in properties:
                expected_type = properties[param].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    errors.append(f"{param}: Expected {expected_type}, got {type(value).__name__}")
        
        if errors:
            raise ToolValidationError(self.name, errors, arguments)
        
        return arguments
    
    def _check_type(self, value: Any, json_type: str) -> bool:
        """Check if value matches JSON schema type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        expected = type_map.get(json_type)
        if expected:
            return isinstance(value, expected)
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enhanced metadata."""
        base = super().to_dict()
        
        # Add enhanced metadata for LLMs
        if self.when_to_use:
            base["when_to_use"] = self.when_to_use
        if self.examples:
            base["examples"] = self.examples
        if self.deprecated:
            base["deprecated"] = True
            if self.deprecation_message:
                base["deprecation_message"] = self.deprecation_message
        if self.tags:
            base["tags"] = self.tags
            
        return base


class ToolValidationError(Exception):
    """
    Tool argument validation error with details for LLM retry.
    
    Provides structured error information that can be sent back
    to the LLM for automatic correction.
    """
    
    def __init__(self, tool_name: str, errors: List[str], arguments: Dict[str, Any]):
        self.tool_name = tool_name
        self.errors = errors
        self.arguments = arguments
        
        # Create human-readable message
        error_msg = f"Validation failed for tool '{tool_name}':\n"
        for error in errors:
            error_msg += f"  - {error}\n"
        error_msg += f"Arguments provided: {arguments}"
        
        super().__init__(error_msg)
    
    def to_retry_message(self) -> str:
        """Format error for LLM retry attempt."""
        return (
            f"The tool call to '{self.tool_name}' failed validation. "
            f"Errors: {'; '.join(self.errors)}. "
            f"Please correct the arguments and try again."
        )


def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parse_docstring: bool = True,
    docstring_format: Optional[str] = None,
    validate_args: bool = True,
    retry_on_error: bool = True,
    max_retries: int = 3,
    timeout: Optional[float] = None,
    requires_confirmation: bool = False,
    requires_context: bool = False,
    stream_response: bool = False,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    deprecated: bool = False,
    deprecation_message: Optional[str] = None,
    response_model: Optional[Type[BaseModel]] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
    when_to_use: Optional[str] = None,
) -> Callable:
    """
    Enhanced decorator for creating tools with rich features.
    
    This decorator provides SOTA tool creation capabilities including:
    - Automatic Pydantic model generation from type hints
    - Rich docstring parsing for parameter descriptions
    - Validation with retry support
    - Timeout and confirmation handling
    - Deprecation warnings
    - Context injection for session-aware tools
    
    Args:
        func: The function to wrap as a tool
        name: Override the tool name (defaults to function name)
        description: Override the description (defaults to docstring)
        parse_docstring: Whether to parse the docstring for metadata
        docstring_format: Format hint ('google', 'numpy', 'sphinx', or None for auto)
        validate_args: Whether to validate arguments with Pydantic
        retry_on_error: Allow retry on validation errors
        max_retries: Maximum retry attempts
        timeout: Execution timeout in seconds
        requires_confirmation: Request user confirmation before execution
        requires_context: Tool needs ToolContext injection
        stream_response: Tool returns an async generator
        tags: Tags for categorizing the tool
        version: Tool version
        deprecated: Mark tool as deprecated
        deprecation_message: Custom deprecation message
        response_model: Pydantic model to validate response
        examples: Example tool calls for LLM guidance
        when_to_use: Hint for when LLM should use this tool
        
    Returns:
        Decorated function that works as an enhanced tool
        
    Example:
        >>> @tool(
        ...     parse_docstring=True,
        ...     retry_on_error=True,
        ...     timeout=30.0,
        ...     tags=["search", "web"],
        ...     when_to_use="When user asks about current events or needs web information"
        ... )
        ... def search_web(
        ...     query: str = Field(description="Search query", min_length=1, max_length=200),
        ...     max_results: int = Field(default=10, ge=1, le=100)
        ... ) -> List[str]:
        ...     '''Search the web for information.
        ...     
        ...     Args:
        ...         query: The search query to execute
        ...         max_results: Maximum number of results to return
        ...         
        ...     Returns:
        ...         List of search result URLs
        ...     '''
        ...     return ["https://example.com/result1", "https://example.com/result2"]
    """
    
    def decorator(f: Callable) -> Callable:
        # Extract function metadata
        func_name = name or f.__name__
        sig = inspect.signature(f)
        hints = get_type_hints(f)
        
        # Initialize description and parameter descriptions
        func_description = description
        param_descriptions = {}
        
        # Parse docstring if available and requested
        if parse_docstring and f.__doc__ and DOCSTRING_PARSER_AVAILABLE:
            try:
                # Parse with specified or auto-detected format
                if docstring_format:
                    parsed = docstring_parser.parse(f.__doc__, style=docstring_format)
                else:
                    parsed = docstring_parser.parse(f.__doc__)
                
                # Extract description
                if not func_description:
                    func_description = parsed.short_description or parsed.long_description or ""
                
                # Extract parameter descriptions
                if hasattr(parsed, 'params'):
                    for param in parsed.params:
                        param_descriptions[param.arg_name] = param.description
            except Exception as e:
                logger.debug(f"Failed to parse docstring for {func_name}: {e}")
        
        # Fallback to simple docstring extraction
        if not func_description and f.__doc__:
            func_description = f.__doc__.split('\n')[0].strip()
        if not func_description:
            func_description = f"Function {func_name}"
        
        # Build Pydantic model and JSON schema
        pydantic_fields = {}
        json_properties = {}
        required_fields = []
        has_context = False
        
        for param_name, param in sig.parameters.items():
            if param_name in ["self", "cls"]:
                continue
            
            # Check for context injection
            param_type = hints.get(param_name, Any)
            if param_type == ToolContext or (hasattr(param_type, '__name__') and param_type.__name__ == 'ToolContext'):
                has_context = True
                continue  # Don't include context in tool schema
            
            # Handle Pydantic Field annotations
            if PYDANTIC_AVAILABLE and isinstance(param.default, FieldInfo):
                # Use the Field directly
                field_info = param.default
                pydantic_fields[param_name] = (param_type, field_info)
                
                # Build JSON schema property
                json_prop = {"type": _python_type_to_json_enhanced(param_type)}
                
                # Add description
                if field_info.description:
                    json_prop["description"] = field_info.description
                elif param_name in param_descriptions:
                    json_prop["description"] = param_descriptions[param_name]
                
                # Add constraints if available
                if hasattr(field_info, 'metadata'):
                    for constraint in field_info.metadata:
                        if hasattr(constraint, 'ge') and constraint.ge is not None:
                            json_prop["minimum"] = constraint.ge
                        if hasattr(constraint, 'le') and constraint.le is not None:
                            json_prop["maximum"] = constraint.le
                        if hasattr(constraint, 'min_length'):
                            json_prop["minLength"] = constraint.min_length
                        if hasattr(constraint, 'max_length'):
                            json_prop["maxLength"] = constraint.max_length
                
                json_properties[param_name] = json_prop
                
                # Check if required
                if field_info.default is None and field_info.default_factory is None:
                    required_fields.append(param_name)
            else:
                # Regular parameter
                if param.default is inspect.Parameter.empty:
                    default_value = ...
                    required_fields.append(param_name)
                else:
                    default_value = param.default
                
                # Create description
                param_desc = param_descriptions.get(param_name, f"Parameter {param_name}")
                
                if PYDANTIC_AVAILABLE:
                    pydantic_fields[param_name] = (
                        param_type,
                        Field(default=default_value, description=param_desc)
                    )
                
                # JSON schema property
                json_properties[param_name] = {
                    "type": _python_type_to_json_enhanced(param_type),
                    "description": param_desc
                }
        
        # Create Pydantic model if available
        pydantic_model = None
        if PYDANTIC_AVAILABLE and pydantic_fields:
            try:
                pydantic_model = create_model(
                    f"{func_name}_Args",
                    **pydantic_fields
                )
            except Exception as e:
                logger.debug(f"Failed to create Pydantic model for {func_name}: {e}")
        
        # Build JSON schema
        json_schema = {
            "type": "object",
            "properties": json_properties,
            "required": required_fields
        }
        
        # Create enhanced tool definition
        tool_def = EnhancedToolDefinition(
            name=func_name,
            description=func_description,
            parameters=json_schema,
            pydantic_model=pydantic_model,
            validate_args=validate_args,
            retry_on_error=retry_on_error,
            max_retries=max_retries,
            timeout=timeout,
            requires_confirmation=requires_confirmation,
            requires_context=requires_context or has_context,
            stream_response=stream_response,
            tags=tags or [],
            version=version,
            deprecated=deprecated,
            deprecation_message=deprecation_message,
            response_model=response_model,
            examples=examples or [],
            when_to_use=when_to_use
        )
        
        # Create wrapper function
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Check deprecation
            if tool_def.deprecated:
                msg = deprecation_message or f"Tool '{func_name}' is deprecated (version {version})"
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
            
            # Validate arguments if requested
            if tool_def.validate_args and kwargs:
                try:
                    kwargs = tool_def.validate_arguments(kwargs)
                except ToolValidationError as e:
                    if tool_def.retry_on_error:
                        # Return structured error for LLM retry
                        return {
                            "_error": True,
                            "_retry": True,
                            "_message": e.to_retry_message(),
                            "_errors": e.errors
                        }
                    raise
            
            # Handle timeout if specified
            if tool_def.timeout:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(f, *args, **kwargs)
                    try:
                        result = future.result(timeout=tool_def.timeout)
                    except TimeoutError:
                        return {
                            "_error": True,
                            "_retry": False,
                            "_message": f"Tool execution timed out after {tool_def.timeout} seconds"
                        }
            else:
                result = f(*args, **kwargs)
            
            # Validate response if model provided
            if response_model and PYDANTIC_AVAILABLE:
                try:
                    if isinstance(result, dict):
                        validated = response_model(**result)
                    else:
                        validated = response_model(value=result)
                    return validated.model_dump() if hasattr(validated, 'model_dump') else validated.dict()
                except ValidationError as e:
                    logger.error(f"Response validation failed for {func_name}: {e}")
                    return {
                        "_error": True,
                        "_retry": False,
                        "_message": f"Response validation failed: {e}"
                    }
            
            return result
        
        # Attach metadata to function
        wrapper.tool_definition = tool_def
        wrapper.is_tool = True
        wrapper.tool_name = func_name
        
        # Auto-register in global registry
        from abstractllm.tools.registry import get_registry
        registry = get_registry()
        registry._tools[func_name] = wrapper
        registry._definitions[func_name] = tool_def
        
        return wrapper
    
    # Handle decorator usage
    if func is None:
        # Called with arguments: @tool(name="custom")
        return decorator
    else:
        # Called without arguments: @tool
        return decorator(func)


def _python_type_to_json_enhanced(py_type: Any) -> Union[str, Dict[str, Any]]:
    """
    Enhanced Python to JSON Schema type conversion.
    
    Handles complex types including Optional, Union, List, Dict, etc.
    """
    # Handle None type
    if py_type is type(None):
        return "null"
    
    # Get origin for generic types
    origin = get_origin(py_type)
    
    # Handle Optional/Union types
    if origin is Union:
        args = get_args(py_type)
        non_none = [a for a in args if a is not type(None)]
        
        if len(non_none) == 1:
            # Optional[T] - just return T's type
            return _python_type_to_json_enhanced(non_none[0])
        else:
            # Union of multiple types
            return {
                "anyOf": [
                    {"type": _python_type_to_json_enhanced(t)} 
                    for t in non_none
                ]
            }
    
    # Handle List/Sequence types
    if origin in [list, List]:
        args = get_args(py_type)
        if args:
            return {
                "type": "array",
                "items": {"type": _python_type_to_json_enhanced(args[0])}
            }
        return "array"
    
    # Handle Dict/Mapping types
    if origin in [dict, Dict]:
        return "object"
    
    # Handle basic types
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        Any: "string",  # Default for Any type
    }
    
    # Check if it's a basic type
    json_type = type_map.get(py_type)
    if json_type:
        return json_type
    
    # Check by string name for imported types
    if hasattr(py_type, '__name__'):
        type_name = py_type.__name__
        if type_name in ['str', 'String']:
            return "string"
        elif type_name in ['int', 'Integer']:
            return "integer"
        elif type_name in ['float', 'Float', 'Number']:
            return "number"
        elif type_name in ['bool', 'Boolean']:
            return "boolean"
        elif type_name in ['list', 'List', 'Array']:
            return "array"
        elif type_name in ['dict', 'Dict', 'Object']:
            return "object"
    
    # Default to string for unknown types
    return "string"


# Utility functions for tool management

def inject_context(
    func: Callable,
    context: ToolContext
) -> Callable:
    """
    Create a version of a tool function with context pre-injected.
    
    Useful for session-specific tool instances.
    """
    @wraps(func)
    def wrapper(**kwargs):
        return func(context=context, **kwargs)
    
    # Copy tool metadata
    if hasattr(func, 'tool_definition'):
        wrapper.tool_definition = func.tool_definition
        wrapper.is_tool = True
        wrapper.tool_name = func.tool_name
    
    return wrapper


def create_tool_from_function(
    func: Callable,
    **tool_kwargs
) -> Callable:
    """
    Create a tool from an existing function without using decorator syntax.
    
    Useful for converting third-party functions to tools.
    """
    return tool(func, **tool_kwargs)


# Backward compatibility
def register(func: Optional[Callable] = None, **kwargs) -> Callable:
    """
    Legacy register decorator for backward compatibility.
    
    Deprecated: Use @tool instead for enhanced features.
    """
    warnings.warn(
        "register is deprecated, use @tool instead for enhanced features",
        DeprecationWarning,
        stacklevel=2
    )
    # Use basic tool features for compatibility
    return tool(
        func,
        parse_docstring=False,  # Keep simple for legacy
        validate_args=False,    # No validation by default
        retry_on_error=False,   # No retry for legacy
        **kwargs
    )