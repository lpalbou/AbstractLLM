import json
import os
import requests
import datetime
import mimetypes

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

def guess_file_description(file_path):
    """
    Try to guess a description for a file based on its name, extension, and attributes.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: A description of the file
    """
    basename = os.path.basename(file_path)
    _, extension = os.path.splitext(basename)
    extension = extension.lower()
    
    # Check if it's a directory
    if os.path.isdir(file_path):
        if basename.startswith('.'):
            return "Hidden directory for configuration or metadata"
        elif basename.lower() in ("images", "img", "pictures", "photos"):
            return "Directory containing images or visual media"
        elif basename.lower() in ("docs", "documents"):
            return "Directory containing documentation files"
        elif basename.lower() in ("src", "source"):
            return "Directory containing source code"
        elif basename.lower() in ("test", "tests"):
            return "Directory containing test files"
        elif basename.lower() in ("data", "dataset"):
            return "Directory containing data files"
        elif basename.lower() in ("build", "dist"):
            return "Directory containing build artifacts"
        elif basename.lower() in ("bin", "binaries"):
            return "Directory containing binary files"
        elif basename.lower() in ("lib", "libs", "libraries"):
            return "Directory containing library files"
        elif basename.lower() in ("node_modules", "packages", "venv", "env"):
            return "Directory containing package dependencies"
        else:
            return "Directory containing other files"
    
    # Common filenames
    if basename.lower() == "readme.md":
        return "Documentation file with information about the project"
    elif basename.lower() == "license" or basename.lower() == "license.txt":
        return "File containing licensing information"
    elif basename.lower() == "changelog.md":
        return "File tracking version changes to the project"
    elif basename.lower() == ".gitignore":
        return "Git configuration file specifying ignored files"
    elif basename.lower() == "dockerfile":
        return "Configuration file for building Docker containers"
    elif basename.lower() == "package.json":
        return "Node.js package configuration file"
    elif basename.lower() == "requirements.txt":
        return "Python package dependencies file"
    elif basename.lower() in ["makefile", "cmake"]:
        return "Build system configuration file"
    
    # File types based on extensions
    if extension in [".py", ".js", ".ts", ".java", ".c", ".cpp", ".go", ".rb", ".php"]:
        return f"Source code file ({extension[1:]} language)"
    elif extension in [".json", ".yaml", ".yml", ".toml", ".ini", ".xml"]:
        return f"Configuration or data file ({extension[1:]} format)"
    elif extension in [".md", ".txt", ".rst", ".adoc"]:
        return f"Documentation or text file ({extension[1:]} format)"
    elif extension in [".jpg", ".jpeg", ".png", ".gif", ".svg", ".bmp"]:
        return f"Image file ({extension[1:]} format)"
    elif extension in [".mp3", ".wav", ".ogg", ".flac"]:
        return f"Audio file ({extension[1:]} format)"
    elif extension in [".mp4", ".avi", ".mov", ".mkv"]:
        return f"Video file ({extension[1:]} format)"
    elif extension in [".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"]:
        return f"Document file ({extension[1:]} format)"
    elif extension in [".zip", ".tar", ".gz", ".rar", ".7z"]:
        return f"Compressed archive file ({extension[1:]} format)"
    elif extension in [".exe", ".dll", ".so", ".dylib"]:
        return f"Binary executable or library file ({extension[1:]} format)"
    elif extension in [".sql", ".db", ".sqlite"]:
        return f"Database file ({extension[1:]} format)"
    elif extension in [".html", ".htm", ".css"]:
        return f"Web file ({extension[1:]} format)"
    else:
        # Try to use mime type if available
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            main_type, sub_type = mime_type.split('/', 1)
            return f"{main_type.capitalize()} file ({sub_type} format)"
        else:
            return "File with unknown type"

def list_directory_contents(directory_path):
    """
    List the contents of a directory with descriptions.
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        dict: Information about the directory contents
    """
    # Validate directory path
    if not os.path.exists(directory_path):
        return {
            "error": f"Directory does not exist: {directory_path}",
            "exists": False
        }
    
    if not os.path.isdir(directory_path):
        return {
            "error": f"Path is not a directory: {directory_path}",
            "exists": True,
            "is_directory": False
        }
    
    try:
        # Get directory contents
        contents = os.listdir(directory_path)
        items = []
        
        for item in contents:
            item_path = os.path.join(directory_path, item)
            is_dir = os.path.isdir(item_path)
            
            # Get file stats
            stats = os.stat(item_path)
            
            # Format item information
            item_info = {
                "name": item,
                "is_directory": is_dir,
                "display_name": f"{item}{'/' if is_dir else ''}",
                "description": guess_file_description(item_path),
                "size_bytes": stats.st_size,
                "modified": datetime.datetime.fromtimestamp(stats.st_mtime).isoformat(),
            }
            
            items.append(item_info)
        
        # Sort items: directories first, then files, both alphabetically
        items.sort(key=lambda x: (not x["is_directory"], x["name"].lower()))
        
        return {
            "exists": True,
            "is_directory": True,
            "path": directory_path,
            "item_count": len(items),
            "items": items
        }
        
    except PermissionError:
        return {
            "error": f"Permission denied: {directory_path}",
            "exists": True,
            "is_directory": True
        }
    except Exception as e:
        return {
            "error": f"Error listing directory: {str(e)}",
            "exists": True,
            "is_directory": True
        }

def main():
    import sys
    
    # Get folder from command line argument if provided
    target_folder = "."
    if len(sys.argv) > 1:
        target_folder = sys.argv[1]
        print(f"Using command-line argument for folder: {target_folder}")
    
    # Define list_files function
    list_files_function = {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in a specified folder with descriptions",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "Path to the folder to list contents of"
                    }
                },
                "required": ["folder"]
            }
        }
    }
    
    # List of available functions
    functions = [list_files_function]
    
    # Start conversation with user query
    messages = [
        {"role": "user", "content": f"List all files in {target_folder}"}
    ]
    
    # Call the model
    print("Sending request to Ollama...")
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
            print(f"Tool calls found: {len(tool_calls)}")
            
            for tool_call in tool_calls:
                # Debug the structure
                print(f"Tool call structure: {json.dumps(tool_call, indent=2)}")
                
                # Handle different possible formats
                function_name = None
                function_args = None
                tool_call_id = tool_call.get("id", "tool_call_" + str(hash(str(tool_call)) % 10000))
                
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
                        
                        # Execute the list_files function
                        if function_name == "list_files":
                            folder = args.get("folder", ".")
                            print(f"Listing files in folder: {folder}")
                            
                            # Get the directory contents
                            result = list_directory_contents(folder)
                            
                            # If we got a specific folder from command line, use that instead
                            # of what the model responded with
                            if len(sys.argv) > 1:
                                target_folder = sys.argv[1]
                                print(f"Override: Using command-line folder: {target_folder}")
                                result = list_directory_contents(target_folder)
                            
                            # Create a formatted response content
                            if "error" in result:
                                formatted_response = f"Error: {result['error']}"
                            else:
                                formatted_response = []
                                for item in result["items"]:
                                    formatted_response.append(
                                        f"{item['display_name']} - {item['description']}"
                                    )
                                formatted_response = "\n".join(formatted_response)
                            
                            # Create a response message with the function result
                            function_response = {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": function_name,
                                "content": json.dumps(result)
                            }
                            
                            # Add the function response to messages
                            messages.append(function_response)
                            
                            # Get the final response from the model
                            print("\nSending function results back to model...")
                            final_response = call_ollama_model(messages)
                            if "message" in final_response:
                                print("\nFinal response from model:")
                                print(final_response["message"]["content"])
                                
                                # Also print our pre-formatted version
                                print("\nPreformatted listing:")
                                print(formatted_response)
                    except Exception as e:
                        print(f"Error executing function: {str(e)}")
    else:
        print("Error in response:", response)

if __name__ == "__main__":
    main()