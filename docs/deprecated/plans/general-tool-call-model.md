# AbstractLLM Internal Tool Definition Model

## 1. Overview

This document specifies the standard internal format that AbstractLLM will use to represent tool definitions. This format is adopted from the structure used by the Anthropic API for tool use. All internal components of AbstractLLM, including the `Session` class and provider adapters, will work with this representation.

## 2. Internal Format Specification

The `tools` parameter accepted by `AbstractLLMInterface.generate`, `AbstractLLMInterface.generate_async`, and `Session.send` will expect a list of Python dictionaries (`List[Dict[str, Any]]`). Each dictionary in the list represents a single tool and **must** adhere to the following structure:

```json
{
  "name": "string",         // Required: The name of the tool. Must match [a-zA-Z0-9_]+.
  "description": "string",  // Required: A description of the tool's purpose and function.
  "input_schema": {       // Required: A JSON Schema object defining the tool's input parameters.
    "type": "object",
    "properties": {
      "parameter_name_1": {
        "type": "string | number | integer | boolean | array | object",
        "description": "string" // Optional but recommended: Description of the parameter.
        // ... other JSON Schema properties like 'enum', 'items', etc. are allowed.
      },
      // ... more parameters
    },
    "required": ["parameter_name_1", ...] // Optional: List of required parameter names.
  },
  "output_schema": {      // Optional: A JSON Schema object defining the tool's return value.
    "type": "string | number | integer | boolean | array | object",
    "description": "string", // Optional: Description of the return value.
    // ... other JSON Schema properties
  }
}
```

**Key Fields:**

-   **`name` (str, Required):** The name of the function/tool. Should be descriptive and follow standard naming conventions (e.g., snake_case). Used by the model to identify which tool to call.
-   **`description` (str, Required):** A detailed natural language description of what the tool does, when it should be used, and what its parameters represent. This is crucial for the LLM to understand the tool's purpose and decide when to use it effectively.
-   **`input_schema` (dict, Required):** A valid JSON Schema object defining the structure, types, and constraints of the arguments the tool accepts. This schema **must** have `type: "object"` at the top level.
    -   **`properties` (dict):** Defines the individual parameters (arguments) the tool function accepts. Each key is the parameter name, and the value is a JSON Schema definition for that parameter (specifying `type`, `description`, `enum`, etc.).
    -   **`required` (List[str], Optional):** A list of parameter names that are mandatory for the tool call.
-   **`output_schema` (dict, Optional):** A JSON Schema object defining the expected return value from the tool. This can be used for documentation and validation of tool results.

## 3. Rationale for Adoption

This format was chosen for several reasons:

-   **Simplicity:** It offers a relatively flat and straightforward structure.
-   **Standard Compliance:** It utilizes the standard JSON Schema for defining parameters, ensuring broad compatibility and expressiveness.
-   **Convertibility:** It can be easily converted to the formats required by other major providers like OpenAI and common Ollama models.
-   **Alignment:** It aligns with the design philosophy of a major provider (Anthropic) known for its focus on robust AI interactions.
-   **Type Safety:** The addition of output schemas improves documentation and enables validation of tool results.

## 4. Example

Here's an example of a tool definition for a simple weather tool, conforming to the internal AbstractLLM standard:

```python
weather_tool = {
    "name": "get_current_weather",
    "description": "Retrieves the current weather conditions for a specified location.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state/country, e.g., 'San Francisco, CA' or 'London, UK'."
            },
            "unit": {
                "type": "string",
                "description": "The temperature unit to use.",
                "enum": ["celsius", "fahrenheit"]
            }
        },
        "required": ["location"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "temperature": {
                "type": "number",
                "description": "The current temperature in the requested unit."
            },
            "conditions": {
                "type": "string",
                "description": "Text description of the current weather conditions."
            },
            "humidity": {
                "type": "number",
                "description": "The current humidity percentage."
            }
        },
        "required": ["temperature", "conditions"]
    }
}

tools_list = [weather_tool]
```

## 5. Helper Utilities

To facilitate the creation of tool definitions in this format, AbstractLLM will provide helper functions (e.g., in `abstractllm/utils/tools.py`) to convert standard Python functions (with type hints and docstrings) into this dictionary structure.

```python
# Hypothetical usage of a helper function
from abstractllm.utils.tools import function_to_tool_definition

def get_current_weather(location: str, unit: str = 'celsius') -> dict:
    """Retrieves the current weather conditions for a specified location.

    Args:
        location (str): The city and state/country, e.g., 'San Francisco, CA'.
        unit (str): The temperature unit ('celsius' or 'fahrenheit'). Defaults to 'celsius'.

    Returns:
        dict: A dictionary containing weather information with temperature and conditions.
    """
    # ... implementation ...
    pass

# Convert the function to the internal standard dictionary format
weather_tool_dict = function_to_tool_definition(get_current_weather)

# weather_tool_dict would now conform to the specified internal format.
```

### Type Validation with Pydantic (Optional Enhancement)

While not strictly required, AbstractLLM can optionally use Pydantic for runtime validation of tool definitions and results:

```python
from pydantic import BaseModel, Field

class ToolDefinition(BaseModel):
    """Pydantic model for a tool definition."""
    name: str
    description: str
    input_schema: dict
    output_schema: dict = None
    
    class Config:
        extra = "forbid"  # Prevent extra fields
        
# For validation
def validate_tool_definition(tool_dict: dict) -> ToolDefinition:
    """Validate a tool definition dictionary against the required schema."""
    return ToolDefinition.parse_obj(tool_dict)
```

*(Note: The implementation details of `function_to_tool_definition` are part of the implementation phase)*

## 6. Provider Conversion Responsibility

Each provider implementation (`OpenAIProvider`, `AnthropicProvider`, `OllamaProvider`) will be responsible for converting tool definitions received in this internal format into the specific structure required by its target API before making the call. The `output_schema` field will typically be omitted when sending to providers, as it's primarily for internal validation and documentation. 