#!/usr/bin/env python3
"""
Tests for security validation in AbstractLLM tool execution.
"""

import os
import time
import unittest
from unittest.mock import patch, Mock
import tempfile

from basic_agent import (
    is_safe_path, 
    validate_tool_parameters,
    create_secure_tool_wrapper,
    sanitize_tool_output,
    read_file,
    TOOL_SECURITY_CONFIG
)

class TestSecurityValidation(unittest.TestCase):
    """Test security validation for tool execution."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory and file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.allowed_dirs = [self.temp_dir]
        
        # Update the allowed directories for testing
        self.original_allowed_dirs = TOOL_SECURITY_CONFIG["read_file"]["allowed_directories"]
        TOOL_SECURITY_CONFIG["read_file"]["allowed_directories"] = self.allowed_dirs
        
        # Create a test file in the allowed directory
        self.test_file_path = os.path.join(self.temp_dir, "test_file.txt")
        with open(self.test_file_path, "w") as f:
            f.write("This is a test file.\n" * 100)
        
        # Create a test file with sensitive information
        self.sensitive_file_path = os.path.join(self.temp_dir, "sensitive.txt")
        with open(self.sensitive_file_path, "w") as f:
            f.write("SSN: 123-45-6789\nCredit Card: 1234-5678-9012-3456\n")
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore original allowed directories
        TOOL_SECURITY_CONFIG["read_file"]["allowed_directories"] = self.original_allowed_dirs
        
        # Remove test files
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        if os.path.exists(self.sensitive_file_path):
            os.remove(self.sensitive_file_path)
        # Remove test directory
        os.rmdir(self.temp_dir)
    
    def test_is_safe_path(self):
        """Test path validation function."""
        # Test valid paths
        self.assertTrue(is_safe_path(self.test_file_path, self.allowed_dirs))
        self.assertTrue(is_safe_path(os.path.join(self.temp_dir, "subdir/file.txt"), self.allowed_dirs))
        
        # Test invalid paths (outside allowed directory)
        parent_dir = os.path.dirname(self.temp_dir)
        self.assertFalse(is_safe_path(os.path.join(parent_dir, "file.txt"), self.allowed_dirs))
        self.assertFalse(is_safe_path("/etc/passwd", self.allowed_dirs))
        self.assertFalse(is_safe_path(os.path.join(self.temp_dir, "../file.txt"), self.allowed_dirs))
    
    def test_validate_tool_parameters(self):
        """Test parameter validation."""
        # Test valid parameters
        valid_params = {"file_path": self.test_file_path, "max_lines": 10}
        is_valid, error = validate_tool_parameters("read_file", valid_params)
        self.assertTrue(is_valid, f"Validation failed with error: {error}")
        
        # Test missing required parameter
        missing_params = {"max_lines": 10}
        is_valid, error = validate_tool_parameters("read_file", missing_params)
        self.assertFalse(is_valid)
        self.assertIn("Missing required parameter", error)
        
        # Test invalid max_lines
        invalid_params = {"file_path": self.test_file_path, "max_lines": -5}
        is_valid, error = validate_tool_parameters("read_file", invalid_params)
        self.assertFalse(is_valid)
        self.assertIn("max_lines must be between", error)
        
        # Test suspicious path
        suspicious_params = {"file_path": "../../../etc/passwd", "max_lines": 10}
        is_valid, error = validate_tool_parameters("read_file", suspicious_params)
        self.assertFalse(is_valid)
        self.assertIn("suspicious patterns", error)
    
    def test_secure_tool_wrapper(self):
        """Test secure tool wrapper with timeout."""
        # Define a slow function
        def slow_function():
            time.sleep(2)
            return "Done"
        
        # Wrap it with a 1 second timeout
        wrapped_function = create_secure_tool_wrapper(slow_function, max_execution_time=1)
        
        # Execute and verify timeout
        result = wrapped_function()
        self.assertIn("timed out", result.lower())
    
    def test_sanitize_tool_output(self):
        """Test output sanitization."""
        # Test sensitive data redaction
        sensitive_data = "SSN: 123-45-6789, Credit Card: 1234-5678-9012-3456"
        sanitized = sanitize_tool_output(sensitive_data, "read_file")
        self.assertIn("***-**-****", sanitized)
        self.assertIn("****-****-****-****", sanitized)
        self.assertNotIn("123-45-6789", sanitized)
        
        # Test size limit
        large_output = "x" * (TOOL_SECURITY_CONFIG["read_file"]["max_file_size"] + 1000)
        sanitized = sanitize_tool_output(large_output, "read_file")
        self.assertLess(len(sanitized), len(large_output))
        self.assertIn("truncated", sanitized)
    
    def test_read_file_security(self):
        """Test read_file with security measures."""
        # Test reading a file in allowed directory
        content = read_file(self.test_file_path)
        self.assertIn("This is a test file", content)
        
        # Test reading a file outside allowed directory
        outside_path = "/etc/passwd"
        result = read_file(outside_path)
        self.assertIn("Error", result)
        self.assertIn("not allowed", result)
        
        # Test reading with excessive max_lines
        result = read_file(self.test_file_path, max_lines=20000)
        self.assertIn("Error", result)
        self.assertIn("max_lines must be between", result)
        
        # Test sensitive data redaction from file content
        sensitive_content = read_file(self.sensitive_file_path)
        self.assertIn("***-**-****", sensitive_content)
        self.assertIn("****-****-****-****", sensitive_content)
        self.assertNotIn("123-45-6789", sensitive_content)
        self.assertNotIn("1234-5678-9012-3456", sensitive_content)

if __name__ == "__main__":
    unittest.main() 