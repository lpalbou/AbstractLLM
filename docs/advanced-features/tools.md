# Working with Tools in AbstractLLM

AbstractLLM provides a unified interface for using tools with various Language Model providers, making it easy to add external functionality to your AI applications. This guide explains how to define, use, and optimize tools in your AbstractLLM applications.

## What are Tools?

Tools (sometimes called "functions" or "tool calls") allow Language Models to:

1. Call external functions to retrieve information
2. Perform actions outside their knowledge cutoff
3. Interact with other systems and APIs
4. Generate structured data in specific formats

AbstractLLM's tool implementation is designed to be:

- **Provider-agnostic**: Works across multiple LLM providers with a consistent interface
- **Type-safe**: Leverages Python type hints for validation and documentation
- **Secure**: Includes built-in validation and security features
- **Flexible**: Supports both synchronous and asynchronous execution

## Basic Tool Usage

### Defining Tools as Functions

The simplest way to define tools is as regular Python functions with type hints:

```python
from abstractllm import create_llm

# Define a simple tool function with type hints
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather in a given location.
    
    Args:
        location: The city and state, e.g., "San Francisco, CA"
        unit: The unit of temperature, either "celsius" or "fahrenheit"
        
    Returns:
        A string describing the current weather
    """
    # In a real implementation, you would call a weather API here
    return f"The weather in {location} is 22°{unit[0].upper()}"

# Create an LLM instance with a model that supports tool calls
llm = create_llm("anthropic", model="claude-3-opus-20240229")

# Generate a response with the tool
response = llm.generate(
    "What's the weather like in San Francisco?",
    tools=[get_current_weather]
)

print(response)
```

### Defining Tools with Classes

For more complex tools, you can use classes:

```python
from abstractllm import create_llm, Tool
from typing import Dict, List, Union, Optional
from pydantic import BaseModel

# Define input schema using Pydantic
class SearchQuery(BaseModel):
    query: str
    max_results: Optional[int] = 5

# Define a class-based tool
class SearchTool(Tool):
    name = "search"
    description = "Search for information on the web"
    
    def execute(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search for information on the web.
        
        Args:
            query: The search query string
            max_results: Maximum number of results to return
            
        Returns:
            A list of search results
        """
        # In a real implementation, you would call a search API here
        return [
            {"title": "Example Result 1", "url": "https://example.com/1", "snippet": "This is an example search result."},
            {"title": "Example Result 2", "url": "https://example.com/2", "snippet": "Another example search result."}
        ][:max_results]

# Create an LLM instance
llm = create_llm("openai", model="gpt-4")

# Instantiate the tool
search_tool = SearchTool()

# Generate a response with the tool
response = llm.generate(
    "Find information about climate change.",
    tools=[search_tool]
)

print(response)
```

## Advanced Tool Configuration

### Tool Parameters

You can configure tools with additional parameters:

```python
from abstractllm import create_llm, Tool
import requests
from typing import Dict, Any, Optional

class StockPrice(Tool):
    name = "get_stock_price"
    description = "Get the current stock price for a given symbol"
    
    def __init__(self, api_key: str, cache_timeout: int = 300):
        self.api_key = api_key
        self.cache_timeout = cache_timeout
        self.cache = {}
    
    def execute(self, symbol: str) -> Dict[str, Any]:
        """Get the current stock price.
        
        Args:
            symbol: The stock symbol, e.g., "AAPL" for Apple
            
        Returns:
            A dictionary with stock information
        """
        symbol = symbol.upper()
        
        # Check cache first
        if symbol in self.cache:
            return self.cache[symbol]
        
        # In a real implementation, you would call a stock API here
        # response = requests.get(f"https://api.example.com/stocks/{symbol}", 
        #                        headers={"Authorization": f"Bearer {self.api_key}"})
        # data = response.json()
        
        # Simulated response
        data = {"symbol": symbol, "price": 150.25, "currency": "USD"}
        
        # Cache the result
        self.cache[symbol] = data
        return data

# Create an LLM instance
llm = create_llm("openai", model="gpt-4")

# Instantiate the tool with configuration
stock_tool = StockPrice(api_key="your_api_key", cache_timeout=600)

# Generate a response with the tool
response = llm.generate(
    "What's the current price of Apple stock?",
    tools=[stock_tool]
)

print(response)
```

### Asynchronous Tools

For non-blocking operations, you can implement asynchronous tools:

```python
from abstractllm import create_llm, AsyncTool
import aiohttp
import asyncio
from typing import Dict, Any

class AsyncWeatherTool(AsyncTool):
    name = "get_weather_async"
    description = "Get weather information asynchronously"
    
    async def execute_async(self, location: str) -> Dict[str, Any]:
        """Get weather information for a location.
        
        Args:
            location: City name or coordinates
            
        Returns:
            Weather information as a dictionary
        """
        # In a real implementation, you would use an async HTTP client
        async with aiohttp.ClientSession() as session:
            # async with session.get(f"https://api.weather.com/{location}") as response:
            #     data = await response.json()
            
            # Simulate API call delay
            await asyncio.sleep(0.5)
            data = {
                "location": location,
                "temperature": 22,
                "conditions": "Sunny",
                "humidity": 45
            }
            return data

# Create an LLM instance
llm = create_llm("openai", model="gpt-4")

# Instantiate the async tool
weather_tool = AsyncWeatherTool()

# Use the tool asynchronously
async def main():
    response = await llm.generate_async(
        "What's the weather in Tokyo?",
        tools=[weather_tool]
    )
    print(response)

# Run the async function
asyncio.run(main())
```

## Tool Validation and Security

AbstractLLM provides several levels of validation and security for tools.

### Parameter Validation

Tools automatically validate input parameters based on type hints:

```python
from abstractllm import create_llm
from typing import List, Dict, Union, Optional
from enum import Enum

class TemperatureUnit(str, Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"

def get_weather(
    location: str,
    days_forecast: int = 1,
    include_humidity: bool = False,
    unit: TemperatureUnit = TemperatureUnit.CELSIUS
) -> Dict[str, Union[str, int, List[Dict]]]:
    """Get weather forecast for a location.
    
    Args:
        location: City and country
        days_forecast: Number of days to forecast (1-10)
        include_humidity: Whether to include humidity in the response
        unit: Temperature unit (celsius or fahrenheit)
        
    Returns:
        Weather forecast data
    """
    # Parameter validation is handled automatically
    if days_forecast < 1 or days_forecast > 10:
        raise ValueError("days_forecast must be between 1 and 10")
    
    # Simplified implementation
    forecast = [
        {"day": i, "temp": 20 + i, "unit": unit, "conditions": "Partly Cloudy"}
        for i in range(days_forecast)
    ]
    
    result = {
        "location": location,
        "forecast": forecast,
    }
    
    if include_humidity:
        for day in result["forecast"]:
            day["humidity"] = 50 + day["day"]
    
    return result

# Create an LLM instance
llm = create_llm("anthropic", model="claude-3-opus-20240229")

# Generate a response with the tool
response = llm.generate(
    "What's the weather forecast for London for the next 3 days in Fahrenheit?",
    tools=[get_weather]
)

print(response)
```

### Security Decorators

For enhanced security, you can use security decorators:

```python
from abstractllm import create_llm, security

@security.validate_path
def read_file(file_path: str) -> str:
    """Read the contents of a file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        The contents of the file as a string
    """
    with open(file_path, 'r') as f:
        return f.read()

@security.rate_limit(max_calls=5, period=60)
def expensive_api_call(query: str) -> Dict[str, Any]:
    """Make an expensive API call with rate limiting.
    
    Args:
        query: The query string
        
    Returns:
        API response data
    """
    # Implementation would go here
    return {"result": f"Data for {query}"}

# Create an LLM instance
llm = create_llm("openai", model="gpt-4")

# Generate a response with the secured tools
response = llm.generate(
    "Read the contents of the README.md file.",
    tools=[read_file]
)

print(response)
```

## Working with Multiple Tools

You can provide multiple tools to an LLM:

```python
from abstractllm import create_llm

def calculator(expression: str) -> float:
    """Calculate the result of a mathematical expression.
    
    Args:
        expression: A mathematical expression as a string
        
    Returns:
        The calculated result
    """
    # SECURITY WARNING: In a real application, you should use a safer evaluation method
    return eval(expression)

def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value from one unit to another.
    
    Args:
        value: The value to convert
        from_unit: The unit to convert from
        to_unit: The unit to convert to
        
    Returns:
        The converted value
    """
    # Simplified implementation for common conversions
    conversions = {
        ("km", "miles"): lambda x: x * 0.621371,
        ("miles", "km"): lambda x: x * 1.60934,
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        return conversions[key](value)
    else:
        raise ValueError(f"Conversion from {from_unit} to {to_unit} is not supported")

# Create an LLM instance
llm = create_llm("anthropic", model="claude-3-opus-20240229")

# Generate a response with multiple tools
response = llm.generate(
    "If I drive 100 km, how many miles is that? Also, if it's 30°C outside, what's that in Fahrenheit?",
    tools=[calculator, convert_units]
)

print(response)
```

## Tool Registries

For more advanced applications, you can create a tool registry:

```python
from abstractllm import create_llm, ToolRegistry

# Create a tool registry
registry = ToolRegistry()

# Define and register tools
@registry.register
def search_database(query: str, limit: int = 10) -> List[Dict]:
    """Search the database for records matching the query.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return
        
    Returns:
        List of matching records
    """
    # Implementation would go here
    return [{"id": i, "title": f"Result {i} for {query}"} for i in range(limit)]

@registry.register
def create_record(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new record in the database.
    
    Args:
        data: Record data to create
        
    Returns:
        Created record with ID
    """
    # Implementation would go here
    return {"id": 123, **data, "created_at": "2023-04-01T12:00:00Z"}

# Create an LLM instance
llm = create_llm("openai", model="gpt-4")

# Generate a response with all tools in the registry
response = llm.generate(
    "Search for records about climate change and create a new record summarizing the findings.",
    tools=registry.tools
)

print(response)
```

## Handling Tool Results

You can customize how tool results are handled in the conversation:

```python
from abstractllm import create_llm
import json

def database_query(query: str) -> List[Dict]:
    """Query the database with SQL.
    
    Args:
        query: SQL query to execute
        
    Returns:
        List of records matching the query
    """
    # Simulated database query
    if "user" in query.lower():
        return [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"}
        ]
    return []

# Custom formatter for tool results
def format_tool_result(result):
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        # Format as a markdown table
        if not result:
            return "No results found."
        
        headers = result[0].keys()
        header_row = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        
        rows = []
        for item in result:
            row = "| " + " | ".join(str(item.get(h, "")) for h in headers) + " |"
            rows.append(row)
        
        return "\n".join([header_row, separator] + rows)
    
    # Default JSON formatting for other result types
    return f"```json\n{json.dumps(result, indent=2)}\n```"

# Create an LLM instance
llm = create_llm("anthropic", model="claude-3-opus-20240229")

# Generate a response with custom result formatting
response = llm.generate(
    "Show me all users in the database.",
    tools=[database_query],
    tool_result_formatter=format_tool_result
)

print(response)
```

## Streaming with Tools

AbstractLLM supports streaming responses with tool calls:

```python
from abstractllm import create_llm

def get_stock_data(symbol: str) -> Dict[str, Any]:
    """Get current stock data for a given symbol.
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT)
        
    Returns:
        Dictionary with stock information
    """
    # Simulated stock data
    stocks = {
        "AAPL": {"price": 175.50, "change": +1.2, "volume": 32000000},
        "MSFT": {"price": 305.75, "change": -0.5, "volume": 28000000},
        "GOOG": {"price": 138.25, "change": +0.8, "volume": 19000000},
    }
    
    symbol = symbol.upper()
    if symbol in stocks:
        return {"symbol": symbol, **stocks[symbol]}
    else:
        return {"error": f"Stock {symbol} not found"}

# Create an LLM instance
llm = create_llm("openai", model="gpt-4")

# Generate a streaming response with tools
prompt = "What's the current price of Apple and Microsoft stock?"

print("Streaming response:")
for chunk in llm.generate_streaming(
    prompt,
    tools=[get_stock_data]
):
    print(chunk, end="", flush=True)
print("\n")
```

## Provider-Specific Considerations

Different LLM providers implement tool calls in different ways. AbstractLLM normalizes these differences, but there are some provider-specific considerations.

### OpenAI

OpenAI has robust support for function calling:

```python
from abstractllm import create_llm

def get_weather(location: str, unit: str = "celsius") -> str:
    """Get current weather in a location."""
    return f"The weather in {location} is 22°{unit[0].upper()}"

# With OpenAI, you can control function calling behavior
llm = create_llm("openai", model="gpt-4")

response = llm.generate(
    "What's the weather in Paris?",
    tools=[get_weather],
    # OpenAI-specific parameter to control tool selection
    tool_choice={"type": "function", "function": {"name": "get_weather"}}
)

print(response)
```

### Anthropic

Anthropic (Claude) has recently added tool use capabilities:

```python
from abstractllm import create_llm

def get_weather(location: str, unit: str = "celsius") -> str:
    """Get current weather in a location."""
    return f"The weather in {location} is 22°{unit[0].upper()}"

# Anthropic requires Claude 3 models for tool use
llm = create_llm("anthropic", model="claude-3-opus-20240229")

response = llm.generate(
    "What's the weather in Paris?",
    tools=[get_weather]
)

print(response)
```

### Ollama

Ollama supports tool use with certain models:

```python
from abstractllm import create_llm

def get_weather(location: str, unit: str = "celsius") -> str:
    """Get current weather in a location."""
    return f"The weather in {location} is 22°{unit[0].upper()}"

# Ollama requires specific models for tool use
llm = create_llm("ollama", model="llama3")

response = llm.generate(
    "What's the weather in Paris?",
    tools=[get_weather]
)

print(response)
```

## Best Practices

### Tool Design Principles

1. **Clear Purpose**: Each tool should have a single, clear purpose.
2. **Meaningful Names**: Use descriptive names for tools and parameters.
3. **Detailed Descriptions**: Provide thorough descriptions for tools and parameters.
4. **Input Validation**: Validate inputs to prevent errors or security issues.
5. **Consistent Return Types**: Return consistent data structures from tools.
6. **Error Handling**: Return useful error messages if the tool fails.

### Security Considerations

1. **Input Validation**: Always validate inputs before processing.
2. **Rate Limiting**: Protect expensive operations with rate limits.
3. **Permissions**: Implement permission checks for sensitive operations.
4. **Environment Isolation**: Run tools in isolated environments for critical operations.
5. **Logging**: Log tool usage for auditing purposes.

### Performance Optimization

1. **Asynchronous Tools**: Use async tools for I/O-bound operations.
2. **Caching**: Cache results for frequently used tool calls.
3. **Batching**: Batch similar operations where possible.
4. **Timeout Handling**: Implement timeouts for external API calls.
5. **Resource Management**: Properly manage resources like database connections.

## Debugging Tools

When developing tools, these techniques help with debugging:

```python
from abstractllm import create_llm, ToolDebugger

# Create a debugger
debugger = ToolDebugger()

@debugger.watch
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

# Register the debugger with your LLM
llm = create_llm("openai", model="gpt-4")
llm.set_tool_debugger(debugger)

# Generate a response
response = llm.generate(
    "What is 123.45 multiplied by 67.89?",
    tools=[multiply]
)

# Get debugging information
for call in debugger.get_calls():
    print(f"Tool: {call.tool_name}")
    print(f"Inputs: {call.inputs}")
    print(f"Result: {call.result}")
    print(f"Duration: {call.duration} seconds")
    print(f"Timestamp: {call.timestamp}")
    print("---")
```

## Advanced Use Cases

### Chained Tools

You can create tools that call other tools:

```python
from abstractllm import create_llm, ToolRegistry
import json

registry = ToolRegistry()

@registry.register
def search_products(query: str, category: str = None) -> List[Dict]:
    """Search for products in the catalog."""
    # Simplified implementation
    products = [
        {"id": 1, "name": "Laptop", "price": 999, "category": "Electronics"},
        {"id": 2, "name": "Smartphone", "price": 699, "category": "Electronics"},
        {"id": 3, "name": "Desk Chair", "price": 249, "category": "Furniture"}
    ]
    
    results = products
    if category:
        results = [p for p in results if p["category"].lower() == category.lower()]
    if query:
        results = [p for p in results if query.lower() in p["name"].lower()]
    
    return results

@registry.register
def get_product_details(product_id: int) -> Dict:
    """Get detailed information about a product."""
    # Simplified implementation
    products = {
        1: {"id": 1, "name": "Laptop", "price": 999, "category": "Electronics", 
            "description": "High-performance laptop with 16GB RAM", "in_stock": True},
        2: {"id": 2, "name": "Smartphone", "price": 699, "category": "Electronics",
            "description": "Latest smartphone model with excellent camera", "in_stock": True},
        3: {"id": 3, "name": "Desk Chair", "price": 249, "category": "Furniture",
            "description": "Ergonomic desk chair with lumbar support", "in_stock": False}
    }
    
    if product_id in products:
        return products[product_id]
    else:
        raise ValueError(f"Product with ID {product_id} not found")

@registry.register
def recommend_products(product_id: int) -> List[Dict]:
    """Recommend related products based on a product ID."""
    # This tool calls other tools
    product = get_product_details(product_id)
    
    # Find products in the same category
    similar_products = search_products("", product["category"])
    
    # Filter out the original product
    recommendations = [p for p in similar_products if p["id"] != product_id]
    
    return recommendations[:2]  # Return top 2 recommendations

# Create an LLM instance
llm = create_llm("openai", model="gpt-4")

# Generate a response with chained tools
response = llm.generate(
    "Find electronics products and get recommendations for the laptop.",
    tools=registry.tools
)

print(response)
```

### Stateful Tools

For tools that need to maintain state across calls:

```python
from abstractllm import create_llm, Tool

class ShoppingCart(Tool):
    name = "shopping_cart"
    description = "Manage a shopping cart for a user"
    
    def __init__(self):
        self.carts = {}  # user_id -> cart items
    
    def execute(self, action: str, user_id: str, product_id: int = None, quantity: int = 1) -> Dict[str, Any]:
        """Manage a shopping cart.
        
        Args:
            action: One of 'add', 'remove', 'view', or 'clear'
            user_id: User identifier
            product_id: Product ID to add or remove (not needed for 'view' or 'clear')
            quantity: Quantity to add or remove (default: 1)
            
        Returns:
            Current cart state after the operation
        """
        # Initialize cart if it doesn't exist
        if user_id not in self.carts:
            self.carts[user_id] = {}
        
        cart = self.carts[user_id]
        
        if action == "add" and product_id is not None:
            cart[product_id] = cart.get(product_id, 0) + quantity
        elif action == "remove" and product_id is not None:
            if product_id in cart:
                cart[product_id] = max(0, cart[product_id] - quantity)
                if cart[product_id] == 0:
                    del cart[product_id]
        elif action == "clear":
            cart.clear()
        
        # For 'view' action, we just return the current cart state
        return {
            "user_id": user_id,
            "items": [{"product_id": pid, "quantity": qty} for pid, qty in cart.items()],
            "item_count": sum(cart.values()),
            "action_performed": action
        }

# Create an LLM instance
llm = create_llm("anthropic", model="claude-3-opus-20240229")

# Create a cart tool
cart_tool = ShoppingCart()

# Simulate a conversation with stateful cart
response1 = llm.generate(
    "Add 2 laptops (product ID 1) to my cart. My user ID is user123.",
    tools=[cart_tool]
)

print("Response 1:", response1)

response2 = llm.generate(
    "Now show me what's in my cart. My user ID is user123.",
    tools=[cart_tool]
)

print("Response 2:", response2)

response3 = llm.generate(
    "Remove 1 laptop from my cart. My user ID is user123.",
    tools=[cart_tool]
)

print("Response 3:", response3)
```

## Conclusion

Tools in AbstractLLM provide a powerful way to extend Language Models with external functionality. By following best practices for tool design, security, and performance, you can build robust AI applications that combine the reasoning capabilities of LLMs with real-world actions and data access.

For more advanced topics, refer to:
- [Security Best Practices](security.md)
- [Provider-Specific Features](provider-specific.md)
- [Custom Providers](custom-providers.md)

You can also view examples of tool implementations in the [Examples Repository](https://github.com/abstractllm/examples/tree/main/tools). 