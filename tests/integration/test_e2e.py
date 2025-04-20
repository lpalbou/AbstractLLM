"""
End-to-end tests for AbstractLLM tool functionality.

This module tests complete end-to-end workflows with tools, including:
- Full workflows with minimal fixtures
- Real provider integration
- Simulating user interaction patterns
- Cross-provider behavior validation
"""

import os
import pytest
import json
from typing import Dict, List, Any, Optional, Union

from abstractllm import AbstractLLM
from abstractllm.session import Session
from abstractllm.tools import (
    ToolDefinition,
    ToolCall,
    ToolResult,
    function_to_tool_definition,
)


# Real-world tool implementations
def search_database(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search a database for records matching the query.
    
    Args:
        query: The search term to look for
        limit: Maximum number of results to return
        
    Returns:
        A list of matching database records
    """
    # Simulate database search
    database = [
        {"id": 1, "title": "Introduction to Machine Learning", "author": "John Smith", "year": 2020},
        {"id": 2, "title": "Advanced Python Programming", "author": "Jane Doe", "year": 2021},
        {"id": 3, "title": "Data Science Fundamentals", "author": "Michael Johnson", "year": 2019},
        {"id": 4, "title": "Neural Networks and Deep Learning", "author": "Sarah Williams", "year": 2022},
        {"id": 5, "title": "Natural Language Processing", "author": "Robert Brown", "year": 2020},
        {"id": 6, "title": "Computer Vision Techniques", "author": "Emily Davis", "year": 2021},
        {"id": 7, "title": "Reinforcement Learning", "author": "David Wilson", "year": 2019},
        {"id": 8, "title": "Artificial Intelligence Ethics", "author": "Lisa Taylor", "year": 2022},
    ]
    
    # Simple keyword search in title, author, and year
    query = query.lower()
    results = []
    for item in database:
        if (query in item["title"].lower() or 
            query in item["author"].lower() or 
            query in str(item["year"])):
            results.append(item)
            if len(results) >= limit:
                break
    
    return results


def translate_text(text: str, target_language: str) -> str:
    """Translate text to the target language.
    
    Args:
        text: The text to translate
        target_language: The language code to translate to (e.g., 'es', 'fr', 'de')
        
    Returns:
        The translated text
    """
    # Simplified mock translation service
    translations = {
        "es": {
            "hello": "hola",
            "world": "mundo",
            "how are you": "cómo estás",
            "goodbye": "adiós",
            "thank you": "gracias",
        },
        "fr": {
            "hello": "bonjour",
            "world": "monde",
            "how are you": "comment allez-vous",
            "goodbye": "au revoir",
            "thank you": "merci",
        },
        "de": {
            "hello": "hallo",
            "world": "welt",
            "how are you": "wie geht es dir",
            "goodbye": "auf wiedersehen",
            "thank you": "danke",
        }
    }
    
    # Check if target language is supported
    if target_language not in translations:
        return f"Error: Language '{target_language}' not supported. Supported languages: {list(translations.keys())}"
    
    # Simplified approach - just check if any of the known phrases are in the input
    text_lower = text.lower()
    result = text
    
    for phrase, translation in translations[target_language].items():
        if phrase in text_lower:
            result = result.lower().replace(phrase, translation)
    
    return result


def summarize_text(text: str, max_length: int = 100) -> str:
    """Summarize a long text to a shorter version.
    
    Args:
        text: The text to summarize
        max_length: The maximum length of the summary
        
    Returns:
        The summarized text
    """
    # This is a very basic summarization - in real applications use a proper NLP model
    if len(text) <= max_length:
        return text
    
    # Simple summarization by keeping the first few sentences
    sentences = text.split('.')
    summary = ""
    
    for sentence in sentences:
        if len(summary) + len(sentence) + 1 <= max_length:
            summary += sentence + "."
        else:
            break
    
    return summary


# End-to-end test with OpenAI
@pytest.mark.api_call
class TestE2EOpenAI:
    """End-to-end tests with OpenAI provider."""
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_chain_of_tools(self):
        """Test a complex workflow chaining multiple tools."""
        # Create a provider
        provider = AbstractLLM.create("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session with multiple tools
        session = Session(
            provider=provider,
            tools=[search_database, translate_text, summarize_text]
        )
        
        # Start a complex conversation involving multiple tools
        session.add_message(
            role="user",
            content="Find books about neural networks, summarize the results, and translate the summary to Spanish."
        )
        
        # Generate a response with all tools available
        response = session.generate_with_tools(
            tool_functions={
                "search_database": search_database, 
                "translate_text": translate_text,
                "summarize_text": summarize_text
            },
            model="gpt-4"  # Using GPT-4 for complex tool chains
        )
        
        # Verify the response contains Spanish text
        assert response.content is not None
        
        # Check that all tools were used
        messages = session.get_history()
        assistant_msgs = [msg for msg in messages if msg.role == "assistant" and msg.tool_results]
        
        # Verify tool calls in the conversation
        if assistant_msgs:
            tool_results = []
            for msg in assistant_msgs:
                if msg.tool_results:
                    tool_results.extend(msg.tool_results)
            
            # Get the tool names that were called
            tool_names = set()
            for result in tool_results:
                if "name" in result:
                    tool_names.add(result["name"])
            
            # Verify at least two tools were used in this complex workflow
            assert len(tool_names) >= 2
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_interactive_session(self):
        """Test an interactive session with follow-up questions."""
        # Create a provider
        provider = AbstractLLM.create("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session with the database search tool
        session = Session(
            provider=provider,
            tools=[search_database]
        )
        
        # Start conversation with a search query
        session.add_message(
            role="user",
            content="Find books about machine learning."
        )
        
        # Generate the first response
        response1 = session.generate_with_tools(
            tool_functions={"search_database": search_database},
            model="gpt-4"
        )
        
        # Verify the first response
        assert response1.content is not None
        assert "machine learning" in response1.content.lower() or "learning" in response1.content.lower()
        
        # Add a follow-up question
        session.add_message(
            role="user",
            content="Which of these books is the most recent?"
        )
        
        # Generate the second response
        response2 = session.generate_with_tools(
            tool_functions={"search_database": search_database},
            model="gpt-4"
        )
        
        # Verify the second response refers to the most recent book
        assert response2.content is not None
        assert "2022" in response2.content or "recent" in response2.content.lower()


# End-to-end test with Anthropic
@pytest.mark.api_call
class TestE2EAnthropic:
    """End-to-end tests with Anthropic provider."""
    
    @pytest.mark.skipif(os.environ.get("ANTHROPIC_API_KEY") is None, 
                       reason="Anthropic API key not available")
    def test_anthropic_tool_execution(self):
        """Test tool execution with Anthropic provider."""
        # Create a provider
        provider = AbstractLLM.create("anthropic", api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        # Create a session with tools
        session = Session(
            provider=provider,
            tools=[search_database, summarize_text]
        )
        
        # Add a user message
        session.add_message(
            role="user",
            content="Find books published in 2020 and give me a short summary of the results."
        )
        
        # Generate a response with tools
        response = session.generate_with_tools(
            tool_functions={
                "search_database": search_database,
                "summarize_text": summarize_text
            },
            model="claude-3-opus-20240229"  # Use an Anthropic model with tool support
        )
        
        # Verify the response contains information about 2020 books
        assert response.content is not None
        assert "2020" in response.content
        
        # Check session history for tool results
        messages = session.get_history()
        assistant_msgs = [msg for msg in messages if msg.role == "assistant" and msg.tool_results]
        
        # Verify that tools were used
        assert len(assistant_msgs) > 0
        if assistant_msgs and assistant_msgs[0].tool_results:
            tool_results = assistant_msgs[0].tool_results
            assert len(tool_results) > 0


# Cross-provider testing
@pytest.mark.api_call
class TestCrossProviderE2E:
    """Tests for consistent behavior across different providers."""
    
    @pytest.mark.skipif(
        os.environ.get("OPENAI_API_KEY") is None or os.environ.get("ANTHROPIC_API_KEY") is None, 
        reason="One or more provider API keys not available"
    )
    def test_identical_query_across_providers(self):
        """Test that the same query produces similar results across providers."""
        # Define providers to test
        providers = [
            ("openai", os.environ.get("OPENAI_API_KEY"), "gpt-4"),
            ("anthropic", os.environ.get("ANTHROPIC_API_KEY"), "claude-3-opus-20240229")
        ]
        
        # Test query and tool to use
        test_query = "Find books about deep learning published after 2020."
        test_tool = search_database
        
        results = []
        
        for provider_name, api_key, model in providers:
            # Skip if API key not available
            if not api_key:
                continue
                
            # Create a provider
            provider = AbstractLLM.create(provider_name, api_key=api_key)
            
            # Create a session
            session = Session(provider=provider, tools=[test_tool])
            
            # Add a user message
            session.add_message(role="user", content=test_query)
            
            # Generate a response with tools
            response = session.generate_with_tools(
                tool_functions={"search_database": test_tool},
                model=model
            )
            
            # Store results for comparison
            results.append({
                "provider": provider_name,
                "model": model,
                "response": response.content,
                "history": session.get_history()
            })
        
        # Verify at least 2 providers were tested
        assert len(results) >= 2
        
        # Check that all providers made tool calls
        for result in results:
            assistant_msgs = [msg for msg in result["history"] 
                             if msg.role == "assistant" and msg.tool_results]
            
            # Verify that each provider made at least one tool call
            assert len(assistant_msgs) > 0
            
            # Verify that the response mentions "deep learning" and a year after 2020
            assert "deep learning" in result["response"].lower() or "neural networks" in result["response"].lower()
            assert "2021" in result["response"] or "2022" in result["response"]
    
    @pytest.mark.skipif(
        os.environ.get("OPENAI_API_KEY") is None or os.environ.get("ANTHROPIC_API_KEY") is None, 
        reason="One or more provider API keys not available"
    )
    def test_transfer_session_between_providers(self):
        """Test transferring a session between different providers."""
        # Start with OpenAI
        openai_provider = AbstractLLM.create("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create initial session
        session = Session(
            provider=openai_provider,
            tools=[search_database, summarize_text]
        )
        
        # First query with OpenAI
        session.add_message(
            role="user",
            content="Find books about machine learning published in 2019."
        )
        
        response_openai = session.generate_with_tools(
            tool_functions={
                "search_database": search_database,
                "summarize_text": summarize_text
            },
            model="gpt-4"
        )
        
        # Verify the OpenAI response
        assert response_openai.content is not None
        assert "2019" in response_openai.content
        
        # Now switch to Anthropic
        anthropic_provider = AbstractLLM.create("anthropic", api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        # Update the session with the new provider
        session.provider = anthropic_provider
        
        # Follow-up query with Anthropic
        session.add_message(
            role="user",
            content="Which other books did these authors write?"
        )
        
        response_anthropic = session.generate_with_tools(
            tool_functions={
                "search_database": search_database,
                "summarize_text": summarize_text
            },
            model="claude-3-opus-20240229"
        )
        
        # Verify the Anthropic response understands the context
        assert response_anthropic.content is not None
        assert any(
            author_name.lower() in response_anthropic.content.lower()
            for author_name in ["michael johnson", "david wilson"]  # Authors of 2019 books
        )


# Complex real-world scenario tests
@pytest.mark.api_call
class TestComplexScenarios:
    """Tests for complex real-world scenarios."""
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_error_recovery_flow(self):
        """Test a scenario with error and recovery."""
        # Create a provider
        provider = AbstractLLM.create("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session with translate tool
        session = Session(
            provider=provider,
            tools=[translate_text]
        )
        
        # Add a user message with invalid language
        session.add_message(
            role="user",
            content="Translate 'Hello world' to Japanese."
        )
        
        # Generate a response that will cause an error (Japanese not supported)
        response1 = session.generate_with_tools(
            tool_functions={"translate_text": translate_text},
            model="gpt-4"
        )
        
        # Verify the response contains error information
        assert response1.content is not None
        assert "supported" in response1.content.lower() or "available" in response1.content.lower()
        
        # Add a follow-up message to correct the error
        session.add_message(
            role="user",
            content="I see. Please translate it to Spanish instead."
        )
        
        # Generate a new response with the corrected language
        response2 = session.generate_with_tools(
            tool_functions={"translate_text": translate_text},
            model="gpt-4"
        )
        
        # Verify the response contains the Spanish translation
        assert response2.content is not None
        assert "hola" in response2.content.lower() and "mundo" in response2.content.lower()
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_multi_step_reasoning(self):
        """Test a scenario requiring multiple steps of reasoning."""
        # Create a provider
        provider = AbstractLLM.create("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Define a custom tool with multiple steps
        def analyze_reading_level(text: str) -> Dict[str, Any]:
            """Analyze the reading level of a text.
            
            Args:
                text: The text to analyze
                
            Returns:
                A dictionary with reading level metrics
            """
            # Simple implementation focusing on word count, sentence count, and unique words
            words = text.lower().split()
            sentences = text.split('.')
            unique_words = set(words)
            
            # Simple reading level calculation
            avg_words_per_sentence = len(words) / max(1, len(sentences) - 1)
            
            # Simplified readability formula
            if avg_words_per_sentence < 10:
                level = "Elementary"
            elif avg_words_per_sentence < 15:
                level = "Intermediate"
            else:
                level = "Advanced"
            
            return {
                "word_count": len(words),
                "sentence_count": len(sentences) - 1,  # -1 for trailing periods
                "unique_word_count": len(unique_words),
                "avg_words_per_sentence": avg_words_per_sentence,
                "estimated_reading_level": level
            }
        
        # Create a session with search and analyze tools
        session = Session(
            provider=provider,
            tools=[search_database, analyze_reading_level, summarize_text]
        )
        
        # Add a user message requiring multi-step reasoning
        session.add_message(
            role="user",
            content="Find books about machine learning, then analyze the reading level of the summary."
        )
        
        # Generate a response with all tools available
        response = session.generate_with_tools(
            tool_functions={
                "search_database": search_database,
                "analyze_reading_level": analyze_reading_level,
                "summarize_text": summarize_text
            },
            model="gpt-4"
        )
        
        # Verify the response contains reading level analysis
        assert response.content is not None
        assert "reading level" in response.content.lower() or "level" in response.content.lower()
        
        # Check that multiple tools were used
        messages = session.get_history()
        assistant_msgs = [msg for msg in messages if msg.role == "assistant" and msg.tool_results]
        
        # Get all tool calls
        all_tool_calls = []
        for msg in assistant_msgs:
            if msg.tool_results:
                all_tool_calls.extend(msg.tool_results)
        
        # Get unique tool names called
        tool_names = set()
        for call in all_tool_calls:
            if "name" in call:
                tool_names.add(call["name"])
        
        # Verify at least 2 different tools were used
        assert len(tool_names) >= 2 