"""
pytest configuration file.
"""

import os
import pytest
from typing import Dict, Any, Generator, List, Callable

from abstractllm import create_llm, ModelParameter
from abstractllm.providers.openai import OpenAIProvider
from abstractllm.providers.anthropic import AnthropicProvider
from abstractllm.providers.ollama import OllamaProvider
from abstractllm.providers.huggingface import HuggingFaceProvider

# Import tool-related utilities
try:
    from abstractllm.tools import ToolDefinition, function_to_tool_definition
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    """
    Get the OpenAI API key from environment variables.
    
    Returns:
        OpenAI API key
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return api_key


@pytest.fixture(scope="session")
def anthropic_api_key() -> str:
    """
    Get the Anthropic API key from environment variables.
    
    Returns:
        Anthropic API key
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return api_key


@pytest.fixture(scope="session")
def openai_provider(openai_api_key) -> Generator[OpenAIProvider, None, None]:
    """
    Create an OpenAI provider for testing.
    
    Args:
        openai_api_key: OpenAI API key
        
    Returns:
        OpenAI provider instance
    """
    provider = create_llm("openai", **{
        ModelParameter.API_KEY: openai_api_key,
        ModelParameter.MODEL: "gpt-3.5-turbo"
    })
    yield provider


@pytest.fixture(scope="session")
def anthropic_provider(anthropic_api_key) -> Generator[AnthropicProvider, None, None]:
    """
    Create an Anthropic provider for testing.
    
    Args:
        anthropic_api_key: Anthropic API key
        
    Returns:
        Anthropic provider instance
    """
    provider = create_llm("anthropic", **{
        ModelParameter.API_KEY: anthropic_api_key,
        ModelParameter.MODEL: "claude-3-5-haiku-20241022"  # Use the latest supported model
    })
    yield provider


@pytest.fixture(scope="session")
def ollama_provider() -> Generator[OllamaProvider, None, None]:
    """
    Create an Ollama provider for testing.
    
    Returns:
        Ollama provider instance
    """
    # Skip test if Ollama is not running
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            pytest.skip("Ollama API not accessible")
            
        # Check if at least one model is available
        models = response.json().get("models", [])
        if not models:
            pytest.skip("No Ollama models available")
            
        # Use the first available model
        model_name = models[0]["name"]
    except Exception:
        pytest.skip("Ollama API not accessible or other error")
        model_name = "llama2"  # Default, won't be used if skipped
    
    provider = create_llm("ollama", **{
        ModelParameter.BASE_URL: "http://localhost:11434",
        ModelParameter.MODEL: model_name
    })
    yield provider


@pytest.fixture(scope="session")
def huggingface_provider() -> Generator[HuggingFaceProvider, None, None]:
    """
    Create a HuggingFace provider for testing.
    
    Returns:
        HuggingFace provider instance
    """
    # Use distilgpt2 model for testing as it's small and reliable
    provider = create_llm("huggingface", **{
        ModelParameter.MODEL: "distilgpt2",  # Use a small, reliable model instead of DEFAULT_MODEL
        ModelParameter.DEVICE: "cpu",        # Run on CPU to ensure it works everywhere
        ModelParameter.MAX_TOKENS: 50,       # Keep generations short for testing
        "auto_load": True,                   # Enable auto-loading
        "auto_warmup": True,                 # Enable auto-warmup
        "load_timeout": 300,                 # Longer timeout for initial load
        "generation_timeout": 30,            # Shorter timeout for generation during tests
        "trust_remote_code": True,           # Allow trusted code execution if needed
        "temperature": 0.7,                  # Set a reasonable temperature
        "top_p": 0.9                         # Set top_p for better test results
    })
    yield provider


@pytest.fixture(params=["openai_provider", "anthropic_provider", "ollama_provider", "huggingface_provider"])
def any_provider(request) -> Generator[Any, None, None]:
    """
    Parametrized fixture that returns each provider.
    This lets us run the same test against all providers.
    
    Args:
        request: pytest request object
        
    Returns:
        Provider instance
    """
    try:
        yield request.getfixturevalue(request.param)
    except pytest.skip.Exception:
        pytest.skip(f"Skipping {request.param} tests")


# Tool-related fixtures

@pytest.fixture
def calculator_function() -> Callable:
    """
    Return a calculator function for tool testing.
    
    Returns:
        Calculator function
    """
    def calculator(operation: str, a: float, b: float) -> float:
        """Perform a basic calculation.
        
        Args:
            operation: The operation to perform (add, subtract, multiply, divide)
            a: First number
            b: Second number
            
        Returns:
            The result of the calculation
        """
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    return calculator


@pytest.fixture
def weather_function() -> Callable:
    """
    Return a weather function for tool testing.
    
    Returns:
        Weather function
    """
    def get_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
        """Get the current weather for a location.
        
        Args:
            location: The city and state, e.g., "San Francisco, CA"
            unit: The unit of temperature, either "celsius" or "fahrenheit"
            
        Returns:
            A dictionary with weather information
        """
        return {
            "location": location,
            "temperature": 22.5,
            "unit": unit,
            "condition": "Sunny",
            "humidity": 65,
        }
    
    return get_weather


@pytest.fixture
def calculator_tool_definition(calculator_function) -> ToolDefinition:
    """
    Create a calculator tool definition for testing.
    
    Args:
        calculator_function: Calculator function
        
    Returns:
        ToolDefinition for calculator
    """
    if not TOOLS_AVAILABLE:
        pytest.skip("Tool support not available")
    
    return function_to_tool_definition(calculator_function)


@pytest.fixture
def weather_tool_definition(weather_function) -> ToolDefinition:
    """
    Create a weather tool definition for testing.
    
    Args:
        weather_function: Weather function
        
    Returns:
        ToolDefinition for weather
    """
    if not TOOLS_AVAILABLE:
        pytest.skip("Tool support not available")
    
    return function_to_tool_definition(weather_function)


@pytest.fixture
def tool_functions() -> Dict[str, Callable]:
    """
    Return a dictionary of tool functions for testing.
    
    Returns:
        Dictionary of tool functions
    """
    def calculator(operation: str, a: float, b: float) -> float:
        """Perform a basic calculation."""
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def get_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
        """Get the current weather for a location."""
        return {
            "location": location,
            "temperature": 22.5,
            "unit": unit,
            "condition": "Sunny",
            "humidity": 65,
        }
    
    return {
        "calculator": calculator,
        "get_weather": get_weather
    }


@pytest.fixture
def tool_definitions(tool_functions) -> List[ToolDefinition]:
    """
    Return a list of tool definitions for testing.
    
    Args:
        tool_functions: Dictionary of tool functions
        
    Returns:
        List of tool definitions
    """
    if not TOOLS_AVAILABLE:
        pytest.skip("Tool support not available")
    
    return [
        function_to_tool_definition(tool_functions["calculator"]),
        function_to_tool_definition(tool_functions["get_weather"])
    ]


# Set up environment variable handling for tests
def pytest_configure(config):
    """Configure pytest environment."""
    # Check for required environment variables
    missing_vars = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing_vars.append("OPENAI_API_KEY")
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing_vars.append("ANTHROPIC_API_KEY")
    
    if not os.environ.get("OLLAMA_HOST"):
        os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    
    if missing_vars:
        print(f"\nWarning: The following environment variables are not set: {', '.join(missing_vars)}")
        print("Some tests will be skipped. Set these variables to run all tests.")


# Skip markers for provider-specific tests
def pytest_addoption(parser):
    """Add custom command-line options to pytest."""
    parser.addoption(
        "--run-api-tests",
        action="store_true",
        default=False,
        help="Run tests that make real API calls",
    )


# Skip tests marked as "api_call" unless --run-api-tests is specified
def pytest_collection_modifyitems(config, items):
    """Modify test collection to apply skip markers."""
    if not config.getoption("--run-api-tests"):
        skip_api = pytest.mark.skip(reason="Need --run-api-tests option to run")
        for item in items:
            if "api_call" in item.keywords:
                item.add_marker(skip_api) 