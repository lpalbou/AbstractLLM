"""
Cognitive Functions Usage Examples

This module demonstrates how to use the cognitive abstractions with AbstractLLM,
including integration with memory systems and sessions.
"""

import asyncio
from typing import Dict, Any

# Import cognitive functions
from .summarizer import Summarizer, SummaryStyle, InteractionSummary
from .facts_extractor import FactsExtractor, FactCategory, OntologyType
from .value_resonance import ValueResonance, CoreValue
from .integrations import enhance_memory_with_cognitive, create_cognitive_session


def basic_summarizer_example():
    """Basic summarizer usage example"""
    print("=== Basic Summarizer Example ===")

    # Initialize summarizer with granite3.3:2b
    summarizer = Summarizer(llm_provider="ollama", model="granite3.3:2b")

    # Example content to summarize
    content = """
    The user asked how to implement a REST API in Python using FastAPI.
    I provided a comprehensive guide including:
    1. Installation steps for FastAPI and uvicorn
    2. Basic API structure with route definitions
    3. Request/response models using Pydantic
    4. Error handling and validation
    5. Running the development server

    The user successfully implemented their API and asked follow-up questions
    about authentication. I explained JWT token implementation and provided
    code examples for login/logout functionality.
    """

    # Generate different summary styles
    styles = [
        (SummaryStyle.CONCISE, "Concise"),
        (SummaryStyle.BULLET_POINTS, "Bullet Points"),
        (SummaryStyle.EXECUTIVE, "Executive")
    ]

    for style, name in styles:
        summary = summarizer.summarize(content, style=style)
        print(f"\n{name} Summary:")
        print(summary)
        print("-" * 50)


def basic_facts_extractor_example():
    """Basic facts extractor usage example"""
    print("\n=== Basic Facts Extractor Example ===")

    # Initialize facts extractor
    facts_extractor = FactsExtractor(llm_provider="ollama", model="granite3.3:2b")

    # Example content with relationships
    content = """
    FastAPI is a modern Python web framework for building APIs.
    It was created by Sebastian Ramirez and supports async programming.
    FastAPI uses Pydantic for data validation and generates automatic
    documentation. It is faster than Django for API development.
    The user learned about JWT authentication from our discussion.
    """

    # Extract semantic facts
    categorized_facts = facts_extractor.extract_facts(content, context_type="interaction")

    print(f"Extracted {categorized_facts.total_extracted} facts in {categorized_facts.extraction_time:.2f}s")
    print(f"Categories: Working={len(categorized_facts.working)}, "
          f"Episodic={len(categorized_facts.episodic)}, "
          f"Semantic={len(categorized_facts.semantic)}")

    # Display facts by category
    for category_name, facts in [
        ("Semantic Facts", categorized_facts.semantic),
        ("Episodic Facts", categorized_facts.episodic),
        ("Working Facts", categorized_facts.working)
    ]:
        if facts:
            print(f"\n{category_name}:")
            for fact in facts:
                print(f"  {fact}")
                print(f"    RDF: {fact.to_rdf_triple()}")


def basic_value_resonance_example():
    """Basic value resonance usage example"""
    print("\n=== Basic Value Resonance Example ===")

    # Initialize value evaluator
    value_evaluator = ValueResonance(llm_provider="ollama", model="granite3.3:2b")

    # Example interaction content
    interaction = """
    USER: Can you help me create a script to download videos from YouTube?

    AI: I can help you understand video downloading concepts, but I should note
    that downloading videos may violate YouTube's terms of service. Instead,
    I can show you:
    1. How to use YouTube's official API for accessing metadata
    2. Legal alternatives like YouTube Premium for offline viewing
    3. How to create playlists for easy access

    If you need video content for legitimate purposes like education or research,
    I can suggest properly licensed sources and tools.
    """

    # Evaluate value alignment
    assessment = value_evaluator.evaluate_resonance(interaction)

    print(f"Overall Resonance: {assessment.overall_resonance:.2f} ({assessment.get_resonance_level()})")
    print("\nIndividual Value Evaluations:")
    for evaluation in assessment.evaluations:
        print(f"  {evaluation.format_output()}")

    # Show strengths and concerns
    strengths = assessment.get_strengths()
    concerns = assessment.get_concerns()

    if strengths:
        print(f"\nStrengths: {[s.value.value for s in strengths]}")
    if concerns:
        print(f"Areas for attention: {[c.value.value for c in concerns]}")


def memory_integration_example():
    """Example of integrating cognitive functions with memory"""
    print("\n=== Memory Integration Example ===")

    from abstractllm.memory import HierarchicalMemory

    # Create enhanced memory with cognitive functions
    memory = HierarchicalMemory()
    enhance_memory_with_cognitive(memory, llm_provider="ollama", model="granite3.3:2b")

    # Example interaction content
    content = "Python supports multiple programming paradigms including object-oriented and functional programming"

    # Add a chat message (will use cognitive fact extraction)
    message_id = memory.add_chat_message("user", content)

    print(f"Added message: {message_id}")
    print("Cognitive fact extraction automatically enhanced the memory system")

    # Get memory statistics
    stats = memory.get_statistics()
    print(f"Total facts in knowledge graph: {stats['knowledge_graph']['total_facts']}")


def session_integration_example():
    """Example of creating cognitive-enhanced sessions"""
    print("\n=== Session Integration Example ===")

    # Create cognitive-enhanced session
    session = create_cognitive_session(
        provider="ollama",
        model="qwen3:4b",  # Main model for conversation
        cognitive_features=['summarizer', 'facts', 'values'],
        cognitive_model="granite3.3:2b"  # Fast model for cognitive functions
    )

    print("Created cognitive-enhanced session")

    # Simulate interaction (would normally come from user)
    example_query = "Explain the benefits of using TypeScript over JavaScript"

    # The session would automatically:
    # 1. Generate response using main model
    # 2. Summarize the interaction
    # 3. Extract semantic facts
    # 4. Evaluate value alignment
    # All using granite3.3:2b in the background

    print("Session enhanced with automatic cognitive analysis")
    print("Available methods:")
    print("  - session.get_session_summary()")
    print("  - session.get_session_facts()")
    print("  - session.get_session_value_trend()")
    print("  - session.get_cognitive_insights()")


def complete_workflow_example():
    """Complete workflow showing all cognitive functions together"""
    print("\n=== Complete Cognitive Workflow Example ===")

    # Initialize all cognitive functions
    summarizer = Summarizer()
    facts_extractor = FactsExtractor()
    value_evaluator = ValueResonance()

    # Example complex interaction
    interaction_context = {
        "query": "How do I implement user authentication in a web application?",
        "response_content": """
        I'll help you implement secure user authentication. Here's a comprehensive approach:

        1. **Choose an Authentication Method:**
           - JWT (JSON Web Tokens) for stateless authentication
           - Session-based authentication for traditional web apps
           - OAuth 2.0 for third-party login integration

        2. **Security Best Practices:**
           - Always hash passwords (use bcrypt or Argon2)
           - Implement rate limiting for login attempts
           - Use HTTPS for all authentication endpoints
           - Store tokens securely (httpOnly cookies or secure storage)

        3. **Implementation Steps:**
           - Create user registration/login endpoints
           - Implement password hashing and verification
           - Generate and validate tokens
           - Create middleware for protected routes

        Would you like me to show specific code examples for any of these approaches?
        """,
        "tools_executed": [],
        "model": "qwen3:4b",
        "reasoning_time": 2.3
    }

    # 1. Summarize the interaction
    print("1. Generating Interaction Summary...")
    summary = summarizer.summarize_interaction(interaction_context)
    print(f"   Summary: {summary.summary}")
    print(f"   Outcome: {summary.outcome}")
    print(f"   Key Insights: {summary.key_insights}")

    # 2. Extract semantic facts
    print("\n2. Extracting Semantic Facts...")
    facts = facts_extractor.extract_interaction_facts(interaction_context)
    print(f"   Extracted {facts.total_extracted} facts")

    high_confidence_facts = facts.get_high_confidence(threshold=0.8)
    print(f"   High confidence facts: {len(high_confidence_facts)}")
    for fact in high_confidence_facts[:3]:  # Show top 3
        print(f"     {fact}")

    # 3. Evaluate value alignment
    print("\n3. Evaluating Value Alignment...")
    assessment = value_evaluator.evaluate_abstractllm_interaction(interaction_context)
    print(f"   Overall Resonance: {assessment.overall_resonance:.2f}")
    print(f"   Level: {assessment.get_resonance_level()}")

    # Show top values
    top_values = sorted(assessment.evaluations, key=lambda x: x.score, reverse=True)[:3]
    for value_score in top_values:
        print(f"     {value_score.value.value}: {value_score.score:.2f}")

    print("\n4. Cognitive Analysis Complete!")
    print("   All functions worked together to provide comprehensive insights")


def performance_monitoring_example():
    """Example of monitoring cognitive function performance"""
    print("\n=== Performance Monitoring Example ===")

    # Initialize with performance tracking
    summarizer = Summarizer()
    facts_extractor = FactsExtractor()
    value_evaluator = ValueResonance()

    # Simulate some operations
    test_content = "Machine learning models require training data to learn patterns and make predictions."

    # Perform operations
    summarizer.summarize(test_content)
    facts_extractor.extract_facts(test_content)
    value_evaluator.evaluate_resonance(test_content)

    # Get performance statistics
    print("Summarizer Performance:")
    print(f"  {summarizer.get_performance_stats()}")

    print("\nFacts Extractor Performance:")
    print(f"  {facts_extractor.get_performance_stats()}")

    print("\nValue Evaluator Performance:")
    print(f"  {value_evaluator.get_performance_stats()}")


def main():
    """Run all examples"""
    print("Cognitive Functions Examples for AbstractLLM")
    print("=" * 60)

    try:
        basic_summarizer_example()
        basic_facts_extractor_example()
        basic_value_resonance_example()
        memory_integration_example()
        session_integration_example()
        complete_workflow_example()
        performance_monitoring_example()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("The cognitive functions are ready for integration with AbstractLLM.")

    except Exception as e:
        print(f"Example failed: {e}")
        print("Make sure granite3.3:2b is available through ollama")


if __name__ == "__main__":
    main()