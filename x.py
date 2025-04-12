import json
import requests

def call_ollama_model(messages, functions=None):
    """
    Call the Ollama API with the given messages and optional functions.
    
    Args:
        messages (list): List of message dictionaries (role and content)
        functions (list, optional): List of function definitions
        
    Returns:
        dict: The response from the Ollama API
    """
    # Ollama API endpoint
    url = "http://localhost:11434/api/chat"
    
    # Prepare the payload
    payload = {
        "model": "qwen2.5",
        "messages": messages,
        "stream": False
    }
    
    # Add functions to the payload if provided
    if functions:
        payload["tools"] = functions
    
    # Make the API call
    response = requests.post(url, json=payload)
    
    # Return the JSON response
    return response.json()

# Define a weather function
weather_function = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use"
                }
            },
            "required": ["location"]
        }
    }
}

# Define a calculator function
calculator_function = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    }
}

# List of available functions
functions = [weather_function, calculator_function]

def main():
    # Example conversation with the model
    messages = [
        {"role": "user", "content": "What's the weather like in New York?"}
    ]

    # Call the model
    response = call_ollama_model(messages, functions)
    print("Initial Response:", json.dumps(response, indent=2))
    
    # Process the response
    if "message" in response:
        message = response["message"]
        print(f"Role: {message['role']}")
        print(f"Content: {message.get('content', 'No content')}")
        
        # Check if the model called a function
        if "tool_calls" in message:
            tool_calls = message["tool_calls"]
            print(f"Tool calls structure: {json.dumps(tool_calls, indent=2)}")
            
            for tool_call in tool_calls:
                # Debug the structure
                print(f"Tool call structure: {json.dumps(tool_call, indent=2)}")
                
                # Handle different possible formats
                function_name = None
                function_args = None
                tool_call_id = tool_call.get("id", "")
                
                # Try to extract function details based on different possible formats
                if "function" in tool_call:
                    # Format 1: {"function": {"name": "...", "arguments": "..."}}
                    function_data = tool_call["function"]
                    function_name = function_data.get("name")
                    function_args = function_data.get("arguments")
                elif "name" in tool_call and "arguments" in tool_call:
                    # Format 2: {"name": "...", "arguments": "..."}
                    function_name = tool_call.get("name")
                    function_args = tool_call.get("arguments")
                
                if function_name and function_args:
                    print(f"\nFunction called: {function_name}")
                    print(f"Arguments: {function_args}")
                    
                    try:
                        # Check if arguments is already a dict (not a string)
                        if isinstance(function_args, dict):
                            args = function_args
                        else:
                            # Parse the arguments if they're a string
                            args = json.loads(function_args)
                        
                        # Simulate function execution
                        if function_name == "get_weather":
                            location = args["location"]
                            unit = args.get("unit", "celsius")
                            
                            # This would be a real API call in a production environment
                            weather_result = {
                                "temperature": 22 if unit == "celsius" else 72,
                                "condition": "Sunny",
                                "humidity": 60,
                                "location": location
                            }
                            
                            # Create a response message with the function result
                            function_response = {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": function_name,
                                "content": json.dumps(weather_result)
                            }
                            
                            # Add the function response to messages
                            messages.append(function_response)
                            
                            # Get the final response from the model
                            final_response = call_ollama_model(messages)
                            if "message" in final_response:
                                print("\nFinal response:")
                                print(final_response["message"]["content"])
                        
                        elif function_name == "calculate":
                            expression = args["expression"]
                            
                            # This is a simple eval - in production, use a safer method
                            try:
                                result = eval(expression)
                                calc_result = {
                                    "result": result,
                                    "expression": expression
                                }
                            except Exception as e:
                                calc_result = {
                                    "error": str(e),
                                    "expression": expression
                                }
                            
                            # Create a response message with the function result
                            function_response = {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": function_name,
                                "content": json.dumps(calc_result)
                            }
                            
                            # Add the function response to messages
                            messages.append(function_response)
                            
                            # Get the final response from the model
                            final_response = call_ollama_model(messages)
                            if "message" in final_response:
                                print("\nFinal response:")
                                print(final_response["message"]["content"])
                    except json.JSONDecodeError:
                        print(f"Error: Could not parse function arguments: {function_args}")
    else:
        print("Error in response:", response)

    # Example of a direct calculation request
    print("\n" + "-"*50 + "\n")
    print("Example with calculator function:")

    calc_messages = [
        {"role": "user", "content": "Calculate 15 * 7 + 22"}
    ]

    calc_response = call_ollama_model(calc_messages, functions)
    print("Calculation Response:", json.dumps(calc_response, indent=2))
    
    # Process the calculation response using the same approach as above
    if "message" in calc_response:
        handle_function_calls(calc_response, calc_messages, functions)
    else:
        print("Error in response:", calc_response)

def handle_function_calls(response, messages, functions):
    """
    Handle function calls in the response.
    
    Args:
        response (dict): The response from the Ollama API
        messages (list): The current conversation messages
        functions (list): The list of available functions
    """
    message = response["message"]
    print(f"Role: {message['role']}")
    print(f"Content: {message.get('content', 'No content')}")
    
    # Check if the model called a function
    if "tool_calls" in message:
        tool_calls = message["tool_calls"]
        
        for tool_call in tool_calls:
            # Handle different possible formats
            function_name = None
            function_args = None
            tool_call_id = tool_call.get("id", "")
            
            # Try to extract function details based on different possible formats
            if "function" in tool_call:
                function_data = tool_call["function"]
                function_name = function_data.get("name")
                function_args = function_data.get("arguments")
            elif "name" in tool_call and "arguments" in tool_call:
                function_name = tool_call.get("name")
                function_args = tool_call.get("arguments")
            
            if function_name and function_args:
                print(f"\nFunction called: {function_name}")
                print(f"Arguments: {function_args}")
                
                try:
                    # Check if arguments is already a dict (not a string)
                    if isinstance(function_args, dict):
                        args = function_args
                    else:
                        # Parse the arguments if they're a string
                        args = json.loads(function_args)
                    
                    # Simulate function execution
                    if function_name == "get_weather":
                        simulate_weather_function(args, tool_call_id, function_name, messages)
                    elif function_name == "calculate":
                        simulate_calculator_function(args, tool_call_id, function_name, messages)
                except json.JSONDecodeError:
                    print(f"Error: Could not parse function arguments: {function_args}")

def simulate_weather_function(args, tool_call_id, function_name, messages):
    """
    Simulate the weather function execution.
    
    Args:
        args (dict): The parsed function arguments
        tool_call_id (str): The ID of the tool call
        function_name (str): The name of the function
        messages (list): The current conversation messages
    """
    location = args["location"]
    unit = args.get("unit", "celsius")
    
    # This would be a real API call in a production environment
    weather_result = {
        "temperature": 22 if unit == "celsius" else 72,
        "condition": "Sunny",
        "humidity": 60,
        "location": location
    }
    
    # Create a response message with the function result
    function_response = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": function_name,
        "content": json.dumps(weather_result)
    }
    
    # Add the function response to messages
    messages.append(function_response)
    
    # Get the final response from the model
    final_response = call_ollama_model(messages)
    if "message" in final_response:
        print("\nFinal response:")
        print(final_response["message"]["content"])

def simulate_calculator_function(args, tool_call_id, function_name, messages):
    """
    Simulate the calculator function execution.
    
    Args:
        args (dict): The parsed function arguments
        tool_call_id (str): The ID of the tool call
        function_name (str): The name of the function
        messages (list): The current conversation messages
    """
    expression = args["expression"]
    
    # This is a simple eval - in production, use a safer method
    try:
        result = eval(expression)
        calc_result = {
            "result": result,
            "expression": expression
        }
    except Exception as e:
        calc_result = {
            "error": str(e),
            "expression": expression
        }
    
    # Create a response message with the function result
    function_response = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": function_name,
        "content": json.dumps(calc_result)
    }
    
    # Add the function response to messages
    messages.append(function_response)
    
    # Get the final response from the model
    final_response = call_ollama_model(messages)
    if "message" in final_response:
        print("\nFinal response:")
        print(final_response["message"]["content"])

if __name__ == "__main__":
    main()