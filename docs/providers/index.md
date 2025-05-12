# Provider Guide

AbstractLLM supports multiple LLM providers, each with its own unique features and capabilities. This guide provides detailed information about each supported provider, including setup, configuration, and provider-specific features.

## Supported Providers

AbstractLLM currently supports these providers:

| Provider | Description | Key Features | Installation |
|----------|-------------|--------------|-------------|
| OpenAI | Cloud-based API for OpenAI models | Function calling, vision, streaming | `pip install abstractllm[openai]` |
| Anthropic | Cloud-based API for Claude models | Long contexts, multimodal, tool use | `pip install abstractllm[anthropic]` |
| Ollama | Local API for open-source models | Self-hosted, privacy-focused, custom models | `pip install abstractllm[ollama]` |
| HuggingFace | Local and cloud API for various models | Wide model selection, local deployment | `pip install abstractllm[huggingface]` |
| MLX | Local inference optimized for Apple Silicon | Hardware acceleration, local privacy, no API keys | `pip install abstractllm[mlx]` |

## Provider Selection

Choosing the right provider depends on your specific requirements:

- **OpenAI**: Best for general-purpose applications with strong tool support
- **Anthropic**: Excellent for complex reasoning tasks and long documents
- **Ollama**: Ideal for privacy-focused applications or air-gapped environments
- **HuggingFace**: Best for customization and flexibility with model selection
- **MLX**: Optimal for Apple Silicon users wanting efficient local inference

For more information about switching between providers, see the [Interchangeability Guide](../user-guide/interchangeability.md).

## Provider-Specific Guides

### [OpenAI Provider](openai.md)

The OpenAI provider supports GPT models through the OpenAI API. It provides excellent performance for a wide range of tasks and has strong support for function calling, vision capabilities, and streaming.

Key capabilities:
- GPT-4 and GPT-3.5 models
- Function calling/tool use
- Vision with GPT-4 Vision models
- Token-by-token streaming
- JSON mode

### [Anthropic Provider](anthropic.md)

The Anthropic provider supports Claude models through the Anthropic API. Claude excels at complex reasoning tasks and can handle very long contexts.

Key capabilities:
- Claude 3 and Claude 3.5 models
- Very large context windows (up to 100K tokens)
- Tool use with Claude 3
- Vision capabilities
- Strong reasoning and instruction following

### [Ollama Provider](ollama.md)

The Ollama provider supports running open-source models locally through the Ollama API. This is ideal for privacy-sensitive applications or air-gapped environments.

Key capabilities:
- Local model deployment
- No data sharing with external services
- Support for various open-source models (Llama, Mistral, etc.)
- Custom model support

### [HuggingFace Provider](huggingface.md)

The HuggingFace provider offers the most flexibility with model selection and deployment options. It supports both local models and the HuggingFace Inference API.

Key capabilities:
- Support for thousands of open-source models
- Local model deployment
- Integration with HuggingFace Hub
- Custom model fine-tuning

### [MLX Provider](mlx.md)

The MLX provider leverages Apple's MLX framework for efficient inference on Apple Silicon devices. This provider offers optimized performance for Mac users with M1/M2/M3 chips.

Key capabilities:
- Hardware acceleration on Apple Silicon
- Local inference with no data sharing
- Integration with Hugging Face models optimized for MLX
- No API keys required
- In-memory model caching for fast switching

## Provider Capabilities Matrix

The following table shows the capabilities supported by each provider:

| Capability | OpenAI | Anthropic | Ollama | HuggingFace | MLX |
|------------|--------|-----------|--------|------------|-----|
| Text Generation | ✅ | ✅ | ✅ | ✅ | ✅ |
| Streaming | ✅ | ✅ | ✅ | ✅ | ✅ |
| System Prompts | ✅ | ✅ | ✅ | ✅ | ✅ |
| Async | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tool/Function Calling | ✅ | ✅ | ⚠️ Model-dependent | ❌ | ❌ |
| Vision | ✅ | ✅ | ⚠️ Model-dependent | ⚠️ Model-dependent | ⚠️ Model-dependent |
| JSON Mode | ✅ | ✅ | ❌ | ❌ | ❌ |
| Platform | All | All | All | All | macOS/Apple Silicon only |

## API Keys and Authentication

Each provider requires different authentication:

- **OpenAI**: Requires an API key from the [OpenAI API platform](https://platform.openai.com/api-keys)
  ```python
  # Set via environment variable
  os.environ["OPENAI_API_KEY"] = "your-api-key"
  
  # Or provide directly
  llm = create_llm("openai", api_key="your-api-key")
  ```

- **Anthropic**: Requires an API key from the [Anthropic Console](https://console.anthropic.com/)
  ```python
  # Set via environment variable
  os.environ["ANTHROPIC_API_KEY"] = "your-api-key"
  
  # Or provide directly
  llm = create_llm("anthropic", api_key="your-api-key")
  ```

- **Ollama**: Typically uses a local endpoint without authentication
  ```python
  # Configure the endpoint (default is http://localhost:11434)
  llm = create_llm("ollama", api_base="http://localhost:11434")
  ```

- **HuggingFace**: Optional API key for using the Inference API
  ```python
  # Set via environment variable
  os.environ["HUGGINGFACE_API_KEY"] = "your-api-key"
  
  # Or provide directly
  llm = create_llm("huggingface", api_key="your-api-key")
  ```

- **MLX**: No API key required (runs locally)
  ```python
  # Create MLX provider with a specific model
  llm = create_llm("mlx", model="mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX")
  ```

## Common Configuration Parameters

All providers support these common parameters:

```python
llm = create_llm(
    "provider_name",
    model="model_name",        # The model to use
    temperature=0.7,           # Controls randomness (0.0 to 1.0)
    max_tokens=1000,           # Maximum number of tokens to generate
    system_prompt="Your system prompt here"  # System prompt for the model
)
```

For provider-specific parameters, see the individual provider guides.

## Cost Considerations

When selecting a provider, consider the cost implications:

- **OpenAI**: Pay-per-token pricing with different rates for different models
- **Anthropic**: Pay-per-token pricing with input/output token differentiation
- **Ollama**: Free (computation happens on your hardware)
- **HuggingFace**: Free for local models, pay-per-call for Inference API
- **MLX**: Free (computation happens on your Apple Silicon hardware)

## Availability and Reliability

Each provider has different availability characteristics:

- **OpenAI & Anthropic**: Cloud-based with high reliability but subject to rate limits and potential service disruptions
- **Ollama & HuggingFace (local)**: Fully controlled by you, not subject to external availability issues
- **HuggingFace Inference API**: Cloud-based with varying availability based on service tier
- **MLX**: Fully controlled by you, only available on Apple Silicon devices

For mission-critical applications, consider implementing provider fallbacks as described in the [Interchangeability Guide](../user-guide/interchangeability.md).

## Implementation Details

For developers interested in how each provider is implemented in AbstractLLM, see the [Provider Implementations](../architecture/providers.md) guide. 