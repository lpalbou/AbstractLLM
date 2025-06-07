"""
Cross-provider tool integration tests for the new tool system.

This module tests that all providers properly integrate with the new
BaseProvider tool methods and return consistent responses.
"""
import pytest
from unittest.mock import Mock, patch
from abstractllm.providers.base import BaseProvider
from abstractllm.providers.migration_status import ProviderMigrationStatus
from abstractllm.tools import ToolDefinition, ToolCallResponse
from abstractllm.types import GenerateResponse


class TestProviderToolIntegration:
    """Test that providers properly integrate with new tool system."""
    
    def test_huggingface_uses_base_methods(self):
        """Test HuggingFace provider uses base class tool methods."""
        if not ProviderMigrationStatus.HUGGINGFACE_NEW_TOOLS:
            pytest.skip("HuggingFace not yet migrated")
            
        from abstractllm.providers.huggingface import HuggingFaceProvider
        
        # Mock the model loading
        with patch.object(HuggingFaceProvider, 'load_model'):
            provider = HuggingFaceProvider()
            
            # Define a test tool
            def search(query: str) -> str:
                """Search for information."""
                return f"Results for: {query}"
            
            # Test tool context preparation
            enhanced_system, tool_defs, mode = provider._prepare_tool_context(
                [search], "You are a helpful assistant"
            )
            
            # HuggingFace should use prompted mode
            assert mode == "prompted"
            assert "Available tools:" in enhanced_system
            assert "search" in enhanced_system
    
    def test_mlx_uses_base_methods(self):
        """Test MLX provider uses base class tool methods."""
        if not ProviderMigrationStatus.MLX_NEW_TOOLS:
            pytest.skip("MLX not yet migrated")
            
        # Skip if MLX dependencies not available
        try:
            from abstractllm.providers.mlx_provider import MLXProvider
        except ImportError:
            pytest.skip("MLX dependencies not available")
            
        # Mock the model loading
        with patch.object(MLXProvider, 'load_model'):
            provider = MLXProvider()
            
            # Define a test tool
            def calculate(expression: str) -> float:
                """Calculate a mathematical expression."""
                return eval(expression)
            
            # Test tool context preparation
            enhanced_system, tool_defs, mode = provider._prepare_tool_context(
                [calculate], "You are a math assistant"
            )
            
            # MLX should use prompted mode
            assert mode == "prompted"
            assert "calculate" in enhanced_system
    
    def test_tool_response_compatibility(self):
        """Test that session compatibility layer works."""
        from abstractllm.session import Session
        
        # Create a mock response with old format
        old_response = Mock()
        old_response.tool_calls = [Mock(name="test_tool")]
        old_response.content = "Test content"
        
        # Create a mock response with new format
        new_response = ToolCallResponse(
            content="Test content",
            tool_calls=[Mock(name="test_tool")]
        )
        
        session = Session()
        
        # Test old format
        result = session._process_tool_response(old_response)
        assert result is not None
        assert hasattr(result, 'tool_calls')
        
        # Test new format
        result = session._process_tool_response(new_response)
        assert result is not None
        assert hasattr(result, 'has_tool_calls')
        assert result.has_tool_calls()
    
    def test_base_provider_extract_tool_calls(self):
        """Test BaseProvider _extract_tool_calls method."""
        
        class TestProvider(BaseProvider):
            """Test provider for unit testing."""
            
            def generate(self, *args, **kwargs):
                pass
                
            async def generate_async(self, *args, **kwargs):
                pass
        
        provider = TestProvider()
        provider.set_config(model="test-model")
        
        # Test with prompted tool response
        test_response = '''I'll search for that information.
        
        <function_call>
        {"name": "search_web", "arguments": {"query": "weather today"}}
        </function_call>'''
        
        with patch('abstractllm.tools.handler.UniversalToolHandler') as MockHandler:
            # Mock the handler
            mock_handler = Mock()
            mock_handler.supports_native = False
            mock_handler.supports_prompted = True
            mock_handler.parse_response.return_value = ToolCallResponse(
                content=test_response,
                tool_calls=[Mock(name="search_web")]
            )
            MockHandler.return_value = mock_handler
            
            # Extract tool calls
            result = provider._extract_tool_calls(test_response)
            
            assert result is not None
            assert isinstance(result, ToolCallResponse)
            assert result.has_tool_calls()
    
    def test_migration_status_tracking(self):
        """Test that migration status is properly tracked."""
        # Check current status
        assert ProviderMigrationStatus.HUGGINGFACE_NEW_TOOLS == True
        assert ProviderMigrationStatus.MLX_NEW_TOOLS == True
        
        # Test migration report
        report = ProviderMigrationStatus.migration_report()
        assert "5/5 providers migrated" in report
        assert "✅ HuggingFace: Migrated" in report
        assert "✅ MLX: Migrated" in report
        assert "✅ Ollama: Migrated" in report
        assert "✅ OpenAI: Migrated" in report
        assert "✅ Anthropic: Migrated" in report
        
        # All providers should be migrated
        assert ProviderMigrationStatus.all_migrated() == True


if __name__ == "__main__":
    # Run a quick status check
    print(ProviderMigrationStatus.migration_report())
    
    # Run tests
    pytest.main([__file__, "-v"])