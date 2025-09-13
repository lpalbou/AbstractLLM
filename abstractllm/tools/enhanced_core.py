"""
Enhanced core types with SOTA features for tool support.

This module provides enhanced tool definitions with:
- Rich parameter descriptions with JSON Schema
- Tool choice forcing
- Few-shot examples
- Confidence scoring
- State management
"""

from typing import Any, Dict, List, Optional, Union, Callable, Literal
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from datetime import datetime


class ToolChoice(Enum):
    """Tool choice forcing modes."""
    AUTO = "auto"  # Let model decide
    NONE = "none"  # Disable tools
    REQUIRED = "required"  # Force tool use
    SPECIFIC = "specific"  # Force specific tool


@dataclass
class ParameterSchema:
    """Rich parameter schema with JSON Schema support."""
    
    type: str  # "string", "number", "boolean", "object", "array"
    description: Optional[str] = None
    enum: Optional[List[Any]] = None
    required: bool = True
    default: Optional[Any] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    pattern: Optional[str] = None  # Regex for strings
    items: Optional['ParameterSchema'] = None  # For arrays
    properties: Optional[Dict[str, 'ParameterSchema']] = None  # For objects
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema = {"type": self.type}
        
        if self.description:
            schema["description"] = self.description
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.pattern:
            schema["pattern"] = self.pattern
        if self.items:
            schema["items"] = self.items.to_json_schema()
        if self.properties:
            schema["properties"] = {
                k: v.to_json_schema() for k, v in self.properties.items()
            }
            
        return schema


@dataclass
class ToolExample:
    """Example of tool usage for few-shot prompting."""
    
    input_description: str  # Natural language description
    arguments: Dict[str, Any]  # Actual arguments
    expected_output: Optional[str] = None  # Expected result description
    
    def to_prompt(self) -> str:
        """Format as prompt example."""
        prompt = f"Example: {self.input_description}\n"
        prompt += f"Tool call: {json.dumps(self.arguments, indent=2)}"
        if self.expected_output:
            prompt += f"\nExpected output: {self.expected_output}"
        return prompt


@dataclass
class EnhancedToolDefinition:
    """Enhanced tool definition with rich features."""
    
    name: str
    description: str
    parameters: Dict[str, ParameterSchema]
    examples: List[ToolExample] = field(default_factory=list)
    category: Optional[str] = None  # For grouping tools
    version: str = "1.0.0"
    timeout: float = 30.0  # Execution timeout
    retry_config: Optional[Dict[str, Any]] = None
    rate_limit: Optional[int] = None  # Calls per minute
    requires_confirmation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with OpenAI/Anthropic compatible format."""
        # Build JSON Schema for parameters
        properties = {}
        required = []
        
        for param_name, param_schema in self.parameters.items():
            properties[param_name] = param_schema.to_json_schema()
            if param_schema.required:
                required.append(param_name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def to_prompt_with_examples(self) -> str:
        """Generate enhanced prompt with examples."""
        prompt = f"Tool: {self.name}\n"
        prompt += f"Description: {self.description}\n"
        
        if self.examples:
            prompt += "\nExamples:\n"
            for i, example in enumerate(self.examples, 1):
                prompt += f"\n{i}. {example.to_prompt()}\n"
                
        return prompt
    
    @classmethod
    def from_function(cls, func: Callable, 
                     examples: Optional[List[ToolExample]] = None,
                     category: Optional[str] = None) -> "EnhancedToolDefinition":
        """Create enhanced definition from function with rich annotations."""
        import inspect
        from typing import get_type_hints, get_args, get_origin
        
        name = func.__name__
        doc = inspect.getdoc(func) or ""
        
        # Parse description and parameter descriptions from docstring
        lines = doc.split('\n')
        description = lines[0] if lines else f"Function {name}"
        
        # Parse parameters from docstring (Google style)
        param_descriptions = {}
        in_args_section = False
        for line in lines:
            if line.strip().startswith("Args:"):
                in_args_section = True
                continue
            if in_args_section:
                if line.strip() and not line.startswith(" "):
                    break
                if ":" in line:
                    param_line = line.strip()
                    param_name = param_line.split(":")[0].strip()
                    param_desc = param_line.split(":", 1)[1].strip()
                    param_descriptions[param_name] = param_desc
        
        # Get parameters with type hints
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
                
            # Get type and create schema
            param_type = hints.get(param_name, Any)
            json_type = _python_type_to_json_detailed(param_type)
            
            # Build parameter schema
            param_schema = ParameterSchema(
                type=json_type["type"],
                description=param_descriptions.get(param_name, f"Parameter {param_name}"),
                required=(param.default is inspect.Parameter.empty)
            )
            
            # Add constraints from type
            if "enum" in json_type:
                param_schema.enum = json_type["enum"]
            if "pattern" in json_type:
                param_schema.pattern = json_type["pattern"]
                
            parameters[param_name] = param_schema
        
        return cls(
            name=name,
            description=description,
            parameters=parameters,
            examples=examples or [],
            category=category
        )


@dataclass
class EnhancedToolCall:
    """Enhanced tool call with confidence and metadata."""
    
    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None
    confidence: float = 1.0  # Confidence score (0-1)
    reasoning: Optional[str] = None  # Model's reasoning for this call
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = f"call_{uuid.uuid4().hex[:8]}"
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class EnhancedToolResult:
    """Enhanced tool result with detailed execution info."""
    
    tool_call_id: str
    output: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None  # Seconds
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class ToolExecutionState:
    """State management for tool execution in ReAct cycles."""
    
    conversation_id: str
    executed_tools: List[str] = field(default_factory=list)
    tool_results: List[EnhancedToolResult] = field(default_factory=list)
    facts_extracted: List[Dict[str, Any]] = field(default_factory=list)  # Knowledge graph triples
    scratchpad: str = ""  # ReAct reasoning scratchpad
    iteration_count: int = 0
    max_iterations: int = 10
    
    def add_fact(self, subject: str, predicate: str, object: Any):
        """Add a fact triple to the knowledge graph."""
        self.facts_extracted.append({
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "timestamp": datetime.now().isoformat(),
            "iteration": self.iteration_count
        })
    
    def update_scratchpad(self, thought: str, action: Optional[str] = None, 
                         observation: Optional[str] = None):
        """Update ReAct scratchpad with reasoning steps."""
        if thought:
            self.scratchpad += f"\nThought {self.iteration_count}: {thought}"
        if action:
            self.scratchpad += f"\nAction {self.iteration_count}: {action}"
        if observation:
            self.scratchpad += f"\nObservation {self.iteration_count}: {observation}"
    
    def get_context(self) -> str:
        """Get current execution context for the model."""
        context = f"Iteration: {self.iteration_count}/{self.max_iterations}\n"
        context += f"Tools executed: {', '.join(self.executed_tools)}\n"
        
        if self.scratchpad:
            context += f"\nReasoning so far:\n{self.scratchpad}\n"
            
        if self.facts_extracted:
            context += f"\nFacts discovered: {len(self.facts_extracted)}\n"
            for fact in self.facts_extracted[-3:]:  # Show last 3 facts
                context += f"- {fact['subject']} {fact['predicate']} {fact['object']}\n"
                
        return context


def _python_type_to_json_detailed(py_type: Any) -> Dict[str, Any]:
    """Convert Python type to detailed JSON Schema."""
    from typing import get_origin, get_args, Union, Literal
    import re
    
    origin = get_origin(py_type)
    
    # Handle Optional/Union types
    if origin is Union:
        args = get_args(py_type)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_detailed(non_none[0])
        else:
            # Multiple types - use oneOf
            return {
                "type": "object",
                "oneOf": [_python_type_to_json_detailed(t) for t in non_none]
            }
    
    # Handle Literal types (enums)
    if origin is Literal:
        values = get_args(py_type)
        base_type = type(values[0]).__name__
        return {
            "type": "string" if base_type == "str" else base_type,
            "enum": list(values)
        }
    
    # Handle List/Array types
    if origin in (list, List):
        args = get_args(py_type)
        item_type = args[0] if args else Any
        return {
            "type": "array",
            "items": _python_type_to_json_detailed(item_type)
        }
    
    # Handle Dict types
    if origin in (dict, Dict):
        return {"type": "object"}
    
    # Basic type mapping
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
        Any: {"type": "string"}  # Default to string for Any
    }
    
    return type_map.get(py_type, {"type": "string"})