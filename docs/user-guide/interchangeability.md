# Provider Interchangeability

One of AbstractLLM's core strengths is its ability to switch seamlessly between different LLM providers. This guide explains how to leverage provider interchangeability in your applications.

## Understanding Provider Interchangeability

Provider interchangeability means you can:
- Switch between different providers with minimal code changes
- Create fallback chains that automatically try alternative providers if one fails
- Route specific requests to the most suitable provider based on capabilities
- Test your application with different providers to find the best fit

## Basic Provider Switching

The simplest form of interchangeability is switching providers:

```python
from abstractllm import create_llm

# Initialize different providers
openai_llm = create_llm("openai", model="gpt-4")
anthropic_llm = create_llm("anthropic", model="claude-3-opus-20240229")
ollama_llm = create_llm("ollama", model="llama3")

# Same prompt to different providers
prompt = "Explain the theory of relativity simply."

# Get responses from each provider
openai_response = openai_llm.generate(prompt)
anthropic_response = anthropic_llm.generate(prompt)
ollama_response = ollama_llm.generate(prompt)

print("OpenAI:", openai_response[:100] + "...")
print("Anthropic:", anthropic_response[:100] + "...")
print("Ollama:", ollama_response[:100] + "...")
```

## Provider Switching with Sessions

You can also switch providers within an ongoing session:

```python
from abstractllm import create_llm
from abstractllm.session import Session

# Initialize providers
openai_llm = create_llm("openai", model="gpt-4")
anthropic_llm = create_llm("anthropic", model="claude-3-opus-20240229")

# Create a session starting with OpenAI
session = Session(
    system_prompt="You are a helpful assistant.",
    provider=openai_llm
)

# Get first response from OpenAI
session.add_message("user", "Tell me about quantum computing.")
openai_response = session.generate()
print("OpenAI response:", openai_response)

# Switch to Anthropic for the next response in the same conversation
anthropic_response = session.generate(
    "Explain it in simpler terms.",
    provider=anthropic_llm
)
print("Anthropic response:", anthropic_response)

# Switch back to OpenAI
openai_response_2 = session.generate(
    "Give me a practical application.",
    provider=openai_llm
)
print("OpenAI response 2:", openai_response_2)
```

## Creating Fallback Chains

You can create fallback chains to automatically try alternative providers if the primary one fails:

```python
from abstractllm import create_llm
from abstractllm.chains import FallbackChain
from abstractllm.exceptions import AbstractLLMError

# Initialize providers
primary = create_llm("openai", model="gpt-4")
fallback1 = create_llm("anthropic", model="claude-3-opus-20240229")
fallback2 = create_llm("ollama", model="llama3")

# Create a fallback chain
chain = FallbackChain(
    providers=[primary, fallback1, fallback2],
    error_types=[AbstractLLMError]  # Types of errors to catch
)

# Generate with fallback
try:
    response = chain.generate("Explain how neural networks work.")
    print("Response from provider:", chain.last_successful_provider)
    print(response)
except AbstractLLMError as e:
    print("All providers failed:", str(e))
```

## Load Balancing

You can distribute requests across multiple providers:

```python
from abstractllm import create_llm
from abstractllm.chains import LoadBalancedChain
import random

# Initialize multiple providers
providers = [
    create_llm("openai", model="gpt-4"),
    create_llm("openai", model="gpt-3.5-turbo"),
    create_llm("anthropic", model="claude-3-opus-20240229"),
    create_llm("anthropic", model="claude-3-haiku-20240307")
]

# Create a load balancing chain with a custom selection strategy
def round_robin_strategy(providers, context=None):
    # Using a simple round-robin approach
    round_robin_strategy.current = getattr(round_robin_strategy, 'current', -1) + 1
    return providers[round_robin_strategy.current % len(providers)]

# Create the load balancing chain
load_balancer = LoadBalancedChain(
    providers=providers,
    selection_strategy=round_robin_strategy
)

# Generate multiple responses
for i in range(5):
    prompt = f"Question {i+1}: What is the capital of France?"
    response = load_balancer.generate(prompt)
    print(f"Response {i+1} from {load_balancer.last_selected_provider.__class__.__name__}")
```

## Capability-Based Routing

Route requests to providers based on their capabilities:

```python
from abstractllm import create_llm, ModelCapability

# Initialize providers
openai_llm = create_llm("openai", model="gpt-4o")
anthropic_llm = create_llm("anthropic", model="claude-3-opus-20240229")
ollama_llm = create_llm("ollama", model="llama3")

def get_provider_for_task(task_type):
    """Select the appropriate provider based on task type."""
    if task_type == "vision":
        # Check which providers support vision
        providers = [openai_llm, anthropic_llm, ollama_llm]
        for provider in providers:
            if provider.get_capabilities().get(ModelCapability.VISION):
                return provider
    
    elif task_type == "long_context":
        # Find provider with largest context window
        providers = {
            openai_llm: openai_llm.get_capabilities().get(ModelCapability.MAX_TOKENS, 0),
            anthropic_llm: anthropic_llm.get_capabilities().get(ModelCapability.MAX_TOKENS, 0),
            ollama_llm: ollama_llm.get_capabilities().get(ModelCapability.MAX_TOKENS, 0)
        }
        return max(providers, key=providers.get)
    
    # Default to OpenAI for other tasks
    return openai_llm

# Use the appropriate provider for each task
vision_provider = get_provider_for_task("vision")
print(f"Using {vision_provider.__class__.__name__} for vision tasks")

long_context_provider = get_provider_for_task("long_context")
print(f"Using {long_context_provider.__class__.__name__} for long context tasks")
```

## Comparing Provider Responses

Compare responses from different providers to evaluate quality:

```python
from abstractllm import create_llm
import pandas as pd

providers = {
    "OpenAI GPT-4": create_llm("openai", model="gpt-4"),
    "OpenAI GPT-3.5": create_llm("openai", model="gpt-3.5-turbo"),
    "Anthropic Claude": create_llm("anthropic", model="claude-3-opus-20240229"),
    "Ollama Llama3": create_llm("ollama", model="llama3")
}

test_prompts = [
    "Explain the concept of recursion in programming.",
    "Summarize the main themes of 'To Kill a Mockingbird'.",
    "What are the pros and cons of electric vehicles?"
]

# Collect responses
results = []
for prompt_idx, prompt in enumerate(test_prompts):
    for provider_name, provider in providers.items():
        try:
            response = provider.generate(prompt)
            response_length = len(response)
            success = True
        except Exception as e:
            response = str(e)
            response_length = 0
            success = False
        
        results.append({
            "Prompt": f"Prompt {prompt_idx+1}",
            "Provider": provider_name,
            "Response Length": response_length,
            "Success": success
        })

# Create a comparison DataFrame
comparison_df = pd.DataFrame(results)
print(comparison_df)
```

## Best Practices for Provider Interchangeability

1. **Use Common Parameters**: Stick to parameters that work across providers when possible
   ```python
   # These parameters work across all providers
   response = llm.generate(
       prompt="Your prompt here",
       temperature=0.7,
       max_tokens=500
   )
   ```

2. **Check Capabilities Before Use**: Always check if a provider supports the capability you need
   ```python
   if provider.get_capabilities().get(ModelCapability.STREAMING):
       # Use streaming
   else:
       # Use non-streaming alternative
   ```

3. **Handle Provider-Specific Errors**: Different providers may have different error patterns
   ```python
   from abstractllm.exceptions import AbstractLLMError, QuotaExceededError, AuthenticationError
   
   try:
       response = provider.generate(prompt)
   except QuotaExceededError:
       # Handle quota exceeded
   except AuthenticationError:
       # Handle auth issues
   except AbstractLLMError as e:
       # Handle other provider errors
   ```

4. **Test Prompts Across Providers**: Some prompts may work better with specific providers
   ```python
   for provider_name, provider in providers.items():
       try:
           print(f"Testing with {provider_name}")
           response = provider.generate(prompt)
           # Evaluate response quality
       except Exception as e:
           print(f"Failed with {provider_name}: {str(e)}")
   ```

## Provider-Specific Considerations

Each provider has unique characteristics to consider when implementing interchangeability:

### OpenAI
- Strong performance on most tasks
- Generally faster response times
- Comprehensive tool calling support
- Cost scales with usage

### Anthropic (Claude)
- Excellent for complex reasoning tasks
- Supports very long contexts
- Different tool calling implementation
- May handle specific content policies differently

### Ollama
- Local deployment means no network dependency
- No data sharing with external services
- Performance depends on local hardware
- Limited to models supported by Ollama

### HuggingFace
- Widest variety of models
- Most flexible deployment options
- Variable performance based on specific model
- Different parameter names and defaults

## Conclusion

Provider interchangeability is a powerful feature of AbstractLLM that gives you flexibility, resilience, and the ability to optimize for specific use cases. By implementing proper capability checking and error handling, you can create applications that work seamlessly across multiple LLM providers. 