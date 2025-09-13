"""
Unified structured response system for AbstractLLM.

Provides consistent structured output across all providers through:
- Native JSON mode (OpenAI, Anthropic)
- Prompted structured output (open source models)
- Pydantic model integration
- Retry with validation
- Schema generation and validation
"""

from typing import Any, Dict, List, Optional, Union, Type, TypeVar, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

# Type variable for Pydantic models
T = TypeVar('T')

# Try importing Pydantic
try:
    from pydantic import BaseModel, Field, ValidationError, create_model
    from pydantic.json_schema import JsonSchemaValue
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # Fallback
    Field = lambda **kwargs: None
    ValidationError = Exception


class ResponseFormat(Enum):
    """Supported response formats."""
    JSON = "json"  # Raw JSON
    JSON_SCHEMA = "json_schema"  # With schema validation
    PYDANTIC = "pydantic"  # Pydantic model
    YAML = "yaml"  # YAML format
    XML = "xml"  # XML format
    CUSTOM = "custom"  # Custom format with parser


@dataclass
class StructuredResponseConfig:
    """Configuration for structured response generation."""
    
    format: ResponseFormat = ResponseFormat.JSON
    schema: Optional[Dict[str, Any]] = None  # JSON Schema
    pydantic_model: Optional[Type[BaseModel]] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    max_retries: int = 3
    validation_fn: Optional[Callable[[Any], bool]] = None
    strip_markdown: bool = True  # Remove ```json blocks
    force_valid_json: bool = True
    temperature_override: Optional[float] = 0.0  # Lower temp for structured output
    

class StructuredResponseHandler:
    """
    Universal handler for structured responses across all providers.
    """
    
    def __init__(self, provider_name: str):
        """
        Initialize handler for specific provider.
        
        Args:
            provider_name: Name of the provider (openai, anthropic, ollama, etc.)
        """
        self.provider_name = provider_name.lower()
        self.supports_native_json = provider_name in ["openai", "anthropic"]
        self.supports_json_schema = provider_name == "openai"  # GPT-4o supports JSON schema
        
    def prepare_request(
        self,
        prompt: str,
        config: StructuredResponseConfig,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare request parameters for structured output.
        
        Args:
            prompt: User prompt
            config: Structured response configuration
            system_prompt: Optional system prompt override
            
        Returns:
            Dict with modified prompt and parameters
        """
        params = {}
        modified_prompt = prompt
        modified_system = system_prompt or ""
        
        # Temperature override for structured output
        if config.temperature_override is not None:
            params["temperature"] = config.temperature_override
        
        # Handle based on provider capabilities
        if self.supports_native_json and config.format in [ResponseFormat.JSON, ResponseFormat.JSON_SCHEMA]:
            # Native JSON mode
            if self.provider_name == "openai":
                if config.format == ResponseFormat.JSON_SCHEMA and config.schema:
                    params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": config.schema
                    }
                else:
                    params["response_format"] = {"type": "json_object"}
                    
            elif self.provider_name == "anthropic":
                # Anthropic doesn't have native JSON mode yet, use prompting
                modified_system = self._enhance_system_prompt_for_json(modified_system, config)
                
        else:
            # Prompted structured output for other providers
            modified_system = self._enhance_system_prompt_for_structure(modified_system, config)
            modified_prompt = self._enhance_prompt_with_examples(modified_prompt, config)
        
        # Add schema to prompt if provided
        if config.schema and not self.supports_json_schema:
            schema_prompt = f"\n\nOutput must conform to this JSON schema:\n```json\n{json.dumps(config.schema, indent=2)}\n```"
            modified_prompt += schema_prompt
        
        # Add Pydantic model schema if provided
        if config.pydantic_model and PYDANTIC_AVAILABLE:
            schema = config.pydantic_model.model_json_schema()
            schema_prompt = f"\n\nOutput must match this structure:\n```json\n{json.dumps(schema, indent=2)}\n```"
            modified_prompt += schema_prompt
        
        return {
            "prompt": modified_prompt,
            "system_prompt": modified_system,
            **params
        }
    
    def parse_response(
        self,
        response: Union[str, Dict[str, Any]],
        config: StructuredResponseConfig
    ) -> Any:
        """
        Parse and validate structured response.
        
        Args:
            response: Raw response from model
            config: Structured response configuration
            
        Returns:
            Parsed and validated response
            
        Raises:
            ValueError: If parsing or validation fails after retries
        """
        # Extract content string
        if isinstance(response, dict):
            content = response.get("content", str(response))
        else:
            content = str(response)
        
        # Parse based on format
        if config.format in [ResponseFormat.JSON, ResponseFormat.JSON_SCHEMA, ResponseFormat.PYDANTIC]:
            parsed = self._parse_json(content, config.strip_markdown)
            
            # Validate against schema
            if config.schema:
                self._validate_json_schema(parsed, config.schema)
            
            # Validate with Pydantic
            if config.pydantic_model and PYDANTIC_AVAILABLE:
                try:
                    return config.pydantic_model(**parsed)
                except ValidationError as e:
                    logger.error(f"Pydantic validation failed: {e}")
                    raise ValueError(f"Response doesn't match Pydantic model: {e}")
            
            # Custom validation
            if config.validation_fn:
                if not config.validation_fn(parsed):
                    raise ValueError("Custom validation failed")
            
            return parsed
            
        elif config.format == ResponseFormat.YAML:
            return self._parse_yaml(content)
            
        elif config.format == ResponseFormat.XML:
            return self._parse_xml(content)
            
        else:
            return content
    
    def generate_with_retry(
        self,
        generate_fn: Callable,
        prompt: str,
        config: StructuredResponseConfig,
        **kwargs
    ) -> Any:
        """
        Generate structured response with retry on validation failure.
        
        Args:
            generate_fn: Function to generate response
            prompt: User prompt  
            config: Structured response configuration
            **kwargs: Additional arguments for generate_fn
            
        Returns:
            Valid structured response
        """
        last_error = None
        
        for attempt in range(config.max_retries):
            try:
                # Prepare request
                request_params = self.prepare_request(prompt, config, kwargs.get("system_prompt"))
                
                # Update kwargs with structured params
                kwargs.update(request_params)
                if "system_prompt" in kwargs:
                    del kwargs["system_prompt"]  # Already in request_params
                
                # Generate response
                response = generate_fn(**kwargs)
                
                # Parse and validate
                return self.parse_response(response, config)
                
            except (ValueError, ValidationError) as e:
                last_error = e
                logger.warning(f"Structured output attempt {attempt + 1} failed: {e}")
                
                # Add error feedback to prompt for next attempt
                if attempt < config.max_retries - 1:
                    error_feedback = f"\n\nPrevious attempt failed with error: {e}\nPlease correct the output format."
                    kwargs["prompt"] = kwargs.get("prompt", prompt) + error_feedback
        
        raise ValueError(f"Failed to generate valid structured output after {config.max_retries} attempts. Last error: {last_error}")
    
    def _enhance_system_prompt_for_json(self, system_prompt: str, config: StructuredResponseConfig) -> str:
        """Enhance system prompt for JSON output."""
        json_instruction = "\n\nYou must always respond with valid JSON. Do not include any text before or after the JSON object."
        
        if config.examples:
            json_instruction += "\n\nExample outputs:\n"
            for example in config.examples[:2]:  # Show max 2 examples
                json_instruction += f"```json\n{json.dumps(example, indent=2)}\n```\n"
        
        return system_prompt + json_instruction
    
    def _enhance_system_prompt_for_structure(self, system_prompt: str, config: StructuredResponseConfig) -> str:
        """Enhance system prompt for structured output."""
        format_name = config.format.value.upper()
        
        instruction = f"\n\nYou must structure your response as valid {format_name}."
        
        if config.format == ResponseFormat.JSON:
            instruction += " Start your response with '{' and end with '}'. Do not use markdown code blocks."
        elif config.format == ResponseFormat.XML:
            instruction += " Use proper XML tags and ensure all tags are closed."
        elif config.format == ResponseFormat.YAML:
            instruction += " Use proper YAML indentation and syntax."
        
        return system_prompt + instruction
    
    def _enhance_prompt_with_examples(self, prompt: str, config: StructuredResponseConfig) -> str:
        """Add examples to the prompt."""
        if not config.examples:
            return prompt
        
        enhanced = prompt + "\n\nExamples of expected output format:\n"
        
        for i, example in enumerate(config.examples[:3], 1):
            if config.format == ResponseFormat.JSON:
                enhanced += f"\nExample {i}:\n```json\n{json.dumps(example, indent=2)}\n```"
            elif config.format == ResponseFormat.XML:
                enhanced += f"\nExample {i}:\n```xml\n{self._dict_to_xml(example)}\n```"
            elif config.format == ResponseFormat.YAML:
                enhanced += f"\nExample {i}:\n```yaml\n{self._dict_to_yaml(example)}\n```"
        
        return enhanced
    
    def _parse_json(self, content: str, strip_markdown: bool = True) -> Dict[str, Any]:
        """Parse JSON from response content."""
        # Strip markdown code blocks if present
        if strip_markdown:
            # Remove ```json and ``` blocks
            content = re.sub(r'```(?:json)?\s*\n?', '', content)
            content = re.sub(r'```\s*$', '', content)
        
        # Find JSON object in content
        content = content.strip()
        
        # Try to extract JSON if embedded in text
        json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Try to fix common issues
            if '"' not in content and "'" in content:
                # Replace single quotes with double quotes
                content = content.replace("'", '"')
                try:
                    return json.loads(content)
                except:
                    pass
            
            raise ValueError(f"Invalid JSON: {e}")
    
    def _parse_yaml(self, content: str) -> Any:
        """Parse YAML from response content."""
        try:
            import yaml
            content = re.sub(r'```(?:yaml)?\s*\n?', '', content)
            content = re.sub(r'```\s*$', '', content)
            return yaml.safe_load(content)
        except ImportError:
            raise ValueError("PyYAML not installed. Install with: pip install pyyaml")
        except Exception as e:
            raise ValueError(f"Invalid YAML: {e}")
    
    def _parse_xml(self, content: str) -> Dict[str, Any]:
        """Parse XML from response content."""
        try:
            import xml.etree.ElementTree as ET
            content = re.sub(r'```(?:xml)?\s*\n?', '', content)
            content = re.sub(r'```\s*$', '', content)
            root = ET.fromstring(content)
            return self._xml_to_dict(root)
        except Exception as e:
            raise ValueError(f"Invalid XML: {e}")
    
    def _validate_json_schema(self, data: Any, schema: Dict[str, Any]):
        """Validate data against JSON schema."""
        try:
            import jsonschema
            jsonschema.validate(data, schema)
        except ImportError:
            logger.warning("jsonschema not installed. Skipping schema validation.")
        except Exception as e:
            raise ValueError(f"Schema validation failed: {e}")
    
    def _xml_to_dict(self, element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}
        
        # Add attributes
        if element.attrib:
            result["@attributes"] = element.attrib
        
        # Add text content
        if element.text and element.text.strip():
            result["text"] = element.text.strip()
        
        # Add children
        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in result:
                # Convert to list if multiple children with same tag
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        return result if result else element.text
    
    def _dict_to_xml(self, data: Dict[str, Any], root_tag: str = "root") -> str:
        """Convert dictionary to XML string."""
        def _to_xml(tag: str, value: Any) -> str:
            if isinstance(value, dict):
                xml = f"<{tag}>"
                for k, v in value.items():
                    if k == "@attributes":
                        continue
                    xml += _to_xml(k, v)
                xml += f"</{tag}>"
                return xml
            elif isinstance(value, list):
                xml = ""
                for item in value:
                    xml += _to_xml(tag, item)
                return xml
            else:
                return f"<{tag}>{value}</{tag}>"
        
        return _to_xml(root_tag, data)
    
    def _dict_to_yaml(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to YAML string."""
        try:
            import yaml
            return yaml.dump(data, default_flow_style=False)
        except ImportError:
            # Simple fallback
            lines = []
            for key, value in data.items():
                if isinstance(value, dict):
                    lines.append(f"{key}:")
                    for k, v in value.items():
                        lines.append(f"  {k}: {v}")
                else:
                    lines.append(f"{key}: {value}")
            return "\n".join(lines)


# Convenience functions
def create_structured_handler(provider_name: str) -> StructuredResponseHandler:
    """Create a structured response handler for a provider."""
    return StructuredResponseHandler(provider_name)


def generate_pydantic_model(name: str, fields: Dict[str, tuple]) -> Type[BaseModel]:
    """
    Dynamically generate a Pydantic model.
    
    Args:
        name: Model name
        fields: Dict of field_name -> (type, description)
        
    Returns:
        Pydantic model class
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("Pydantic is required for model generation")
    
    field_definitions = {}
    for field_name, (field_type, description) in fields.items():
        field_definitions[field_name] = (field_type, Field(description=description))
    
    return create_model(name, **field_definitions)


# Example usage with providers
def example_usage():
    """Example of using structured response with different providers."""
    
    # Example 1: Simple JSON with schema
    config = StructuredResponseConfig(
        format=ResponseFormat.JSON,
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "skills": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["name", "age"]
        },
        examples=[
            {"name": "Alice", "age": 30, "skills": ["Python", "ML"]},
            {"name": "Bob", "age": 25, "skills": ["JavaScript", "React"]}
        ]
    )
    
    # Example 2: With Pydantic model
    if PYDANTIC_AVAILABLE:
        class PersonModel(BaseModel):
            name: str = Field(description="Person's name")
            age: int = Field(description="Person's age", gt=0, le=120)
            skills: List[str] = Field(description="List of skills")
            email: Optional[str] = Field(default=None, pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
        
        config_pydantic = StructuredResponseConfig(
            format=ResponseFormat.PYDANTIC,
            pydantic_model=PersonModel,
            max_retries=5,
            temperature_override=0.0
        )
    
    # Example 3: Custom validation
    def validate_person(data):
        return data.get("age", 0) >= 18  # Must be adult
    
    config_custom = StructuredResponseConfig(
        format=ResponseFormat.JSON,
        validation_fn=validate_person,
        examples=[{"name": "Charlie", "age": 21}]
    )
    
    return config, config_custom