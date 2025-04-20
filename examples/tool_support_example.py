#!/usr/bin/env python3
"""
Example demonstrating the tool support foundation in AbstractLLM.
"""

import math
from abstractllm.tools import (
    function_to_tool_definition,
    validate_tool_definition,
    validate_tool_arguments,
    validate_tool_result,
    create_safe_tool_wrapper,
    ToolResult
)


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate the Euclidean distance between two points.
    
    Args:
        x1: X-coordinate of the first point
        y1: Y-coordinate of the first point
        x2: X-coordinate of the second point
        y2: Y-coordinate of the second point
        
    Returns:
        The Euclidean distance between the points
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def main():
    # Convert function to tool definition
    print("Converting function to tool definition...")
    tool_def = function_to_tool_definition(calculate_distance)
    print(f"Generated tool definition: {tool_def}\n")
    
    # Validate tool definition
    print("Validating tool definition...")
    validated_tool = validate_tool_definition(tool_def.to_dict())
    print(f"Validated tool definition: {validated_tool}\n")
    
    # Create a safe wrapper
    print("Creating safe wrapper...")
    safe_calculate_distance = create_safe_tool_wrapper(calculate_distance, validated_tool)
    
    # Test with valid arguments
    print("Testing with valid arguments...")
    valid_args = {"x1": 0, "y1": 0, "x2": 3, "y2": 4}
    result = safe_calculate_distance(**valid_args)
    print(f"Result with {valid_args}: {result}\n")
    
    # Test with invalid arguments
    print("Testing with invalid arguments...")
    invalid_args = {"x1": "not a number", "y1": 0, "x2": 3, "y2": 4}
    result = safe_calculate_distance(**invalid_args)
    print(f"Result with {invalid_args}: {result}\n")
    
    # Test with missing required arguments
    print("Testing with missing required arguments...")
    missing_args = {"x1": 0, "y1": 0}
    try:
        # This will raise a TypeError since we're not going through the safe wrapper
        calculate_distance(**missing_args)
    except TypeError as e:
        print(f"Raw function with missing args error: {e}")
    
    # But the safe wrapper will handle it gracefully
    result = safe_calculate_distance(**missing_args)
    print(f"Safe wrapper result with {missing_args}: {result}\n")
    
    # Demonstrate direct validation
    print("Demonstrating direct validation...")
    try:
        validate_tool_arguments(validated_tool, invalid_args)
    except Exception as e:
        print(f"Validation error with invalid args: {e}")
    
    # Print input/output schemas
    print("\nInput Schema:")
    print(validated_tool.input_schema)
    
    print("\nOutput Schema:")
    print(validated_tool.output_schema)


if __name__ == "__main__":
    main() 