"""
Example demonstrating the new universal tool support system.

This example shows how to use tools with any model, whether it has
native tool support or requires prompting.
"""

from abstractllm import LLMInterface
from abstractllm.tools import create_handler, register, execute_tools


# Register some example tools
@register
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # This is a mock implementation
    return f"The weather in {city} is sunny and 72°F"


@register  
def calculate(expression: str) -> float:
    """Calculate a mathematical expression."""
    # Simple eval for demo (use a proper parser in production!)
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except:
        return 0.0


def main():
    # Test with different models
    models = [
        "gpt-4",  # Native tool support
        "claude-3.5-sonnet",  # Native tool support
        "llama-3.1-8b",  # Prompted tool support
        "qwen2.5-7b",  # Prompted tool support
    ]
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Testing with {model_name}")
        print('='*60)
        
        try:
            # Create LLM interface
            llm = LLMInterface(model_name)
            
            # Create tool handler
            handler = create_handler(model_name)
            
            # Show capabilities
            caps = handler.get_capabilities()
            print(f"Tool support: {'Native' if caps['native_tools'] else 'Prompted'}")
            print(f"Tool format: {caps['tool_format']}")
            
            # Prepare request with tools
            messages = [
                {"role": "user", "content": "What's the weather in Paris and what's 25 * 4?"}
            ]
            
            request = handler.prepare_request(
                tools=[get_weather, calculate],
                messages=messages
            )
            
            print(f"Mode: {request['mode']}")
            
            # Generate response
            if request['mode'] == 'native':
                # Native tool mode
                response = llm.generate(
                    messages=request['messages'],
                    tools=request['tools']
                )
            else:
                # Prompted tool mode
                response = llm.generate(
                    messages=request['messages']
                )
            
            # Parse response for tool calls
            parsed = handler.parse_response(
                response.content if hasattr(response, 'content') else response,
                mode=request['mode']
            )
            
            print(f"\nResponse: {parsed.content[:100]}...")
            
            if parsed.has_tool_calls():
                print(f"Found {len(parsed.tool_calls)} tool calls:")
                
                # Execute tools
                results = execute_tools(parsed.tool_calls)
                
                for tc, result in zip(parsed.tool_calls, results):
                    print(f"  - {tc.name}({tc.arguments}) -> {result.output}")
                
                # Format results back
                formatted = handler.format_tool_results(results, mode=request['mode'])
                print(f"\nFormatted results: {formatted}")
            else:
                print("No tool calls detected")
                
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()