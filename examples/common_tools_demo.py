#!/usr/bin/env python3
"""
Demonstration of AbstractLLM common tools.

This script shows how to use the shareable tools module with AbstractLLM
for various file operations, web requests, and user interactions.
"""

from abstractllm import create_llm
from abstractllm.session import Session
from abstractllm.utils.logging import configure_logging
from abstractllm.utils.formatting import format_response_display
from abstractllm.tools.common_tools import (
    list_files, search_files, read_file, write_file, update_file,
    execute_command, search_internet, fetch_url, fetch_and_parse_html,
    ask_user_multiple_choice
)
import logging

# Configure logging
configure_logging(log_level=logging.INFO, console_output=True)

def main():
    """Demonstrate common tools usage."""
    
    print("🚀 AbstractLLM Common Tools Demo")
    print("="*50)
    
    # Initialize provider (using MLX as example)
    # Use a smaller, faster model for better demo experience
    try:
        # Try smaller models first for better responsiveness
        model_options = [
            "mlx-community/Qwen2.5-7B-Instruct-4bit",  # Much faster
            "mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX",  # Fallback
            "mlx-community/Qwen3-30B-A3B-4bit"  # Original large model as last resort
        ]
        
        provider = None
        for model in model_options:
            try:
                print(f"🔄 Trying to load model: {model}")
                provider = create_llm("mlx", 
                                     model=model,
                                     max_tokens=1024,  # Smaller for faster generation
                                     temperature=0.7)
                print(f"✅ Successfully loaded: {model}")
                break
            except Exception as model_error:
                print(f"⚠️  Failed to load {model}: {model_error}")
                continue
        
        if not provider:
            raise Exception("All MLX models failed to load")
            
    except Exception as e:
        print(f"❌ MLX not available, falling back to demo mode: {e}")
        provider = None
    
    # Create session with all the common tools
    if provider:
        session = Session(
            system_prompt="""You are a helpful assistant with access to powerful tools for file operations, web searching, command execution, and user interaction. 

When the user asks you to perform tasks:
- Use the appropriate tools to complete the request
- Provide clear explanations of what you're doing
- Show results in a user-friendly format

Available tools:
- File operations: list_files, search_files, read_file, write_file, update_file
- System: execute_command
- Web: search_internet, fetch_url, fetch_and_parse_html
- User interaction: ask_user_multiple_choice

Be proactive and use tools when they would be helpful.""",
            provider=provider,
            tools=[
                list_files, search_files, read_file, write_file, update_file,
                execute_command, search_internet, fetch_url, fetch_and_parse_html,
                ask_user_multiple_choice
            ]
        )
        
        print("✅ Session created with all common tools!")
        print("\nExample commands you can try:")
        print("- 'List all Python files in the current directory'")
        print("- 'Search for the word \"AbstractLLM\" in all Python files'")
        print("- 'Create a simple test file with some content'")
        print("- 'Search the internet for Python best practices'")
        print("- 'Ask me to choose my favorite programming language'")
        print("- 'Execute the command \"ls -la\" and show me the results'")
        print("\nType '/quit' to exit.")
        
        # Interactive loop
        while True:
            try:
                user_input = input("\n🤖 What would you like me to do? ")
                
                if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                    print("Goodbye! 👋")
                    break
                    
                if not user_input.strip():
                    continue
                
                print("\n🔄 Processing...")
                print("⏱️  Note: This may take a moment for the first request while the model loads...")
                
                try:
                    response = session.generate(
                        prompt=user_input,
                        max_tool_calls=5,  # Reduced to prevent loops
                        max_tokens=512     # Smaller for faster response
                    )
                    
                    print("\n📝 Response:")
                    format_response_display(response)
                    
                except Exception as gen_error:
                    print(f"\n❌ Generation Error: {gen_error}")
                    print("💡 Try a simpler request or restart the demo")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user (Ctrl+C). Goodbye! 👋")
                break
            except EOFError:
                print("\n\n⚠️  EOF detected. Goodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Unexpected Error: {e}")
                print("💡 Try again or type '/quit' to exit")
    
    else:
        # Demo mode - show tools individually
        print("\n🔧 Demo Mode - Testing tools individually:")
        print("-" * 40)
        
        # File operations demo
        print("\n📁 File Operations:")
        print("1. Listing files:")
        result = list_files(".", "*.py", recursive=False)
        print(result[:500] + "..." if len(result) > 500 else result)
        
        print("\n2. Creating a test file:")
        test_content = """# Test File
print("Hello from AbstractLLM common tools!")

def example_function():
    return "This is a test function"
"""
        result = write_file("test_demo.py", test_content)
        print(result)
        
        print("\n3. Reading the test file:")
        result = read_file("test_demo.py")
        print(result[:300] + "..." if len(result) > 300 else result)
        
        # Web operations demo
        print("\n🌐 Web Operations:")
        print("1. Searching the internet:")
        result = search_internet("Python programming best practices", num_results=3)
        print(result[:500] + "..." if len(result) > 500 else result)
        
        # System operations demo (safe command)
        print("\n💻 System Operations:")
        print("1. Executing a safe command:")
        result = execute_command("echo 'Hello from AbstractLLM tools!'")
        print(result)
        
        # User interaction demo
        print("\n👤 User Interaction:")
        print("1. Multiple choice question:")
        result = ask_user_multiple_choice(
            "What is your favorite programming paradigm?",
            ["Object-Oriented", "Functional", "Procedural", "Logic-based"],
            allow_multiple=False
        )
        print(f"Result: {result}")
        
        print("\n✅ Demo completed!")

if __name__ == "__main__":
    main() 