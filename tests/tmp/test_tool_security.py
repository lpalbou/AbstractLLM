#!/usr/bin/env python3
"""
Tests for security measures in AbstractLLM tool execution.

These tests verify that security features like path validation, parameter validation,
and execution timeouts are functioning correctly to prevent security vulnerabilities.
"""

import pytest
import os
import time
import tempfile
from unittest.mock import patch, MagicMock

from basic_agent import (
    is_safe_path, 
    validate_tool_parameters, 
    create_secure_tool_wrapper,
    sanitize_tool_output,
    read_file
)


class TestToolSecurity:
    """Test security features for tool execution."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create a temporary directory and file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.allowed_dirs = [self.temp_dir]
        
        # Create test files in the allowed directory
        self.test_file_path = os.path.join(self.temp_dir, "test_file.txt")
        with open(self.test_file_path, "w") as f:
            f.write("This is a test file.\n" * 100)
        
        # Create a test file with sensitive information
        self.sensitive_file_path = os.path.join(self.temp_dir, "sensitive.txt")
        with open(self.sensitive_file_path, "w") as f:
            f.write("SSN: 123-45-6789\nCredit Card: 1234-5678-9012-3456\n")
    
    def teardown_method(self):
        """Clean up after tests."""
        # Remove test files
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        if os.path.exists(self.sensitive_file_path):
            os.remove(self.sensitive_file_path)
        # Remove test directory
        os.rmdir(self.temp_dir)
    
    def test_safe_path_validation(self):
        """Test the is_safe_path function."""
        # Test valid paths
        assert is_safe_path(self.test_file_path, self.allowed_dirs)
        assert is_safe_path(os.path.join(self.temp_dir, "subdir/file.txt"), self.allowed_dirs)
        
        # Test invalid paths (outside allowed directory)
        parent_dir = os.path.dirname(self.temp_dir)
        assert not is_safe_path(os.path.join(parent_dir, "file.txt"), self.allowed_dirs)
        assert not is_safe_path("/etc/passwd", self.allowed_dirs)
        assert not is_safe_path(os.path.join(self.temp_dir, "../file.txt"), self.allowed_dirs)
        
        # Test directory traversal attempts
        assert not is_safe_path(f"{self.temp_dir}/../../../etc/passwd", self.allowed_dirs)
        assert not is_safe_path(f"{self.temp_dir}/subdir/../../etc/passwd", self.allowed_dirs)
        
        # Test normalized paths
        assert is_safe_path(f"{self.temp_dir}/./test.txt", self.allowed_dirs)
        assert is_safe_path(os.path.join(self.temp_dir, "subdir/../test.txt"), self.allowed_dirs)
    
    def test_validate_tool_parameters(self):
        """Test parameter validation for tools."""
        # Test valid parameters
        valid_params = {"file_path": self.test_file_path, "max_lines": 100}
        is_valid, error = validate_tool_parameters("read_file", valid_params)
        assert is_valid, f"Validation failed with error: {error}"
        
        # Test missing required parameter
        missing_params = {"max_lines": 100}
        is_valid, error = validate_tool_parameters("read_file", missing_params)
        assert not is_valid
        assert "missing required parameter" in error.lower()
        
        # Test invalid max_lines
        invalid_params = {"file_path": self.test_file_path, "max_lines": -5}
        is_valid, error = validate_tool_parameters("read_file", invalid_params)
        assert not is_valid
        assert "max_lines" in error.lower()
        
        # Test suspicious path patterns
        suspicious_params = {"file_path": "../../../etc/passwd", "max_lines": 100}
        is_valid, error = validate_tool_parameters("read_file", suspicious_params)
        assert not is_valid
        assert "suspicious patterns" in error.lower()
        
        # Test safe parameter validation
        safe_params = {"file_path": "test.txt", "max_lines": 50}
        is_valid, error = validate_tool_parameters("read_file", safe_params)
        assert is_valid, f"Validation failed with error: {error}"
    
    def test_secure_tool_wrapper_timeout(self):
        """Test that the secure tool wrapper enforces timeouts."""
        # Define a slow function
        def slow_function():
            time.sleep(3)  # Sleep for 3 seconds
            return "Done"
        
        # Wrap it with a 1-second timeout
        wrapped_function = create_secure_tool_wrapper(slow_function, max_execution_time=1)
        
        # Execute and verify timeout
        result = wrapped_function()
        assert "timeout" in result.lower() or "timed out" in result.lower()
    
    def test_secure_tool_wrapper_normal_execution(self):
        """Test that the secure tool wrapper works for normal execution."""
        # Define a normal function
        def normal_function(x, y):
            return x + y
        
        # Wrap it
        wrapped_function = create_secure_tool_wrapper(normal_function, max_execution_time=5)
        
        # Execute and verify result
        result = wrapped_function(3, 4)
        assert result == 7
    
    def test_secure_tool_wrapper_exception_handling(self):
        """Test that the secure tool wrapper handles exceptions properly."""
        # Define a function that raises an exception
        def error_function():
            raise ValueError("Test error")
        
        # Wrap it
        wrapped_function = create_secure_tool_wrapper(error_function, max_execution_time=5)
        
        # Execute and verify error is caught
        result = wrapped_function()
        assert "error" in result.lower()
        assert "test error" in result.lower()
    
    def test_sanitize_tool_output_size_limit(self):
        """Test that sanitize_tool_output enforces size limits."""
        # Create a large output
        large_output = "x" * 1_000_000  # 1MB
        
        # Sanitize it
        sanitized = sanitize_tool_output(large_output, "read_file")
        
        # Verify size limit is enforced
        assert len(sanitized) < len(large_output)
        assert "truncated" in sanitized.lower()
    
    def test_sanitize_tool_output_sensitive_data(self):
        """Test that sanitize_tool_output redacts sensitive information."""
        # Create output with sensitive data
        sensitive_data = "My SSN is 123-45-6789 and my credit card is 1234-5678-9012-3456"
        
        # Sanitize it
        sanitized = sanitize_tool_output(sensitive_data, "read_file")
        
        # Verify sensitive data is redacted
        assert "123-45-6789" not in sanitized
        assert "1234-5678-9012-3456" not in sanitized
        assert "***-**-****" in sanitized
        assert "****-****-****-****" in sanitized
    
    def test_read_file_security(self):
        """Test security features of the read_file function."""
        # Test reading a file in allowed directory with limited patch for allowed_directories
        with patch("basic_agent.TOOL_SECURITY_CONFIG", {
            "read_file": {
                "allowed_directories": self.allowed_dirs,
                "max_file_size": 10 * 1024 * 1024,
                "max_execution_time": 5,
                "max_lines": 10000
            }
        }):
            # Read a valid file
            content = read_file(self.test_file_path)
            assert "This is a test file" in content
            
            # Test reading a file outside allowed directory
            outside_path = "/etc/passwd"
            result = read_file(outside_path)
            assert "error" in result.lower()
            assert "not allowed" in result.lower()
            
            # Test reading with excessive max_lines
            result = read_file(self.test_file_path, max_lines=20000)
            assert "error" in result.lower()
            assert "max_lines" in result.lower()
            
            # Test reading a nonexistent file
            nonexistent_path = os.path.join(self.temp_dir, "nonexistent.txt")
            result = read_file(nonexistent_path)
            assert "error" in result.lower()
            assert "no such file" in result.lower() or "not found" in result.lower()
    
    def test_read_file_sensitive_data_redaction(self):
        """Test that read_file redacts sensitive information."""
        # Patch allowed directories
        with patch("basic_agent.TOOL_SECURITY_CONFIG", {
            "read_file": {
                "allowed_directories": self.allowed_dirs,
                "max_file_size": 10 * 1024 * 1024,
                "max_execution_time": 5,
                "max_lines": 10000
            }
        }):
            # Read the sensitive file
            content = read_file(self.sensitive_file_path)
            
            # Verify sensitive data is redacted
            assert "123-45-6789" not in content
            assert "1234-5678-9012-3456" not in content
            assert "***-**-****" in content
            assert "****-****-****-****" in content
    
    def test_path_traversal_defenses(self):
        """Test defenses against path traversal attacks."""
        # Patch allowed directories
        with patch("basic_agent.TOOL_SECURITY_CONFIG", {
            "read_file": {
                "allowed_directories": self.allowed_dirs,
                "max_file_size": 10 * 1024 * 1024,
                "max_execution_time": 5,
                "max_lines": 10000
            }
        }):
            # Test various path traversal attempts
            traversal_attempts = [
                os.path.join(self.temp_dir, "../../../etc/passwd"),
                os.path.join(self.temp_dir, "subdir/../../etc/passwd"),
                os.path.join(self.temp_dir, "./././../etc/passwd"),
                f"{self.temp_dir}/../etc/passwd",
                f"{self.temp_dir}/..\\../etc/passwd",  # Windows-style
                "../etc/passwd",
                "../../etc/passwd",
                "/etc/passwd",
                "~/passwords.txt",
                "$HOME/.ssh/id_rsa"
            ]
            
            for path in traversal_attempts:
                result = read_file(path)
                assert "error" in result.lower() or "not allowed" in result.lower(), f"Path traversal succeeded for: {path}"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 