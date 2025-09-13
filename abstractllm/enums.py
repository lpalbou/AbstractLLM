"""
Enums for AbstractLLM.
"""

from enum import Enum

class ModelParameter(str, Enum):
    """Model parameters that can be configured."""
    # Basic parameters
    TEMPERATURE = "temperature"
    MAX_TOKENS = "max_tokens"
    SYSTEM_PROMPT = "system_prompt"
    TOP_P = "top_p"
    FREQUENCY_PENALTY = "frequency_penalty"
    PRESENCE_PENALTY = "presence_penalty"
    STOP = "stop"
    MODEL = "model"  # Model identifier/name
    API_KEY = "api_key"  # API key for providers that need it
    BASE_URL = "base_url"  # Base URL for local/self-hosted models
    
    # Additional parameters
    TIMEOUT = "timeout"  # Request timeout in seconds
    RETRY_COUNT = "retry_count"  # Number of retries on failure
    LOGIT_BIAS = "logit_bias"  # Token biases for generation
    SEED = "seed"  # Random seed for reproducible generations
    TOP_K = "top_k"  # Top-k sampling parameter
    REPETITION_PENALTY = "repetition_penalty"  # Penalty for repeating tokens
    
    # Context and token management
    MAX_INPUT_TOKENS = "max_input_tokens"  # Maximum input context length
    CONTEXT_WINDOW = "context_window"  # Total context window size
    TRUNCATION_STRATEGY = "truncation_strategy"  # How to handle context overflow
    
    # Model loading parameters (for local models)
    DEVICE = "device"  # Device to load the model on (cpu, cuda, etc.)
    DEVICE_MAP = "device_map"  # Device mapping for model sharding
    LOAD_IN_8BIT = "load_in_8bit"  # Whether to load in 8-bit precision
    LOAD_IN_4BIT = "load_in_4bit"  # Whether to load in 4-bit precision
    CACHE_DIR = "cache_dir"  # Directory for model caching
    
    # Provider-specific parameters
    ORGANIZATION = "organization"  # Organization ID for OpenAI
    USER = "user"  # User ID for attribution/tracking
    PROXY = "proxy"  # Proxy URL for API requests
    REQUEST_TIMEOUT = "request_timeout"  # Timeout specifically for HTTP requests
    MAX_RETRIES = "max_retries"  # Maximum number of retry attempts
    
    # Vision support parameters
    IMAGE = "image"  # Single image input (URL, path, or base64)
    IMAGES = "images"  # Multiple image inputs (list of URLs, paths, or base64 strings)
    IMAGE_DETAIL = "image_detail"  # Detail level for image processing (e.g., 'low', 'high')
    
    # Tool support parameters
    TOOLS = "tools"  # List of tool definitions for function/tool calling
    TOOL_CHOICE = "tool_choice"  # Specifies which tool should be used

    # Security & compliance parameters
    CONTENT_FILTER = "content_filter"  # Content filtering level
    MODERATION = "moderation"  # Whether to perform moderation
    LOGGING_ENABLED = "logging_enabled"  # Whether to log requests/responses

class ModelCapability(str, Enum):
    """Capabilities that a model may support."""
    # Basic capabilities
    STREAMING = "streaming"
    MAX_TOKENS = "max_tokens"
    SYSTEM_PROMPT = "supports_system_prompt"
    ASYNC = "supports_async"
    FUNCTION_CALLING = "supports_function_calling"  # General function/tool calling capability
    TOOL_USE = "supports_tool_use"  # Specific tool use capability (may differ from functions)
    VISION = "supports_vision"
    
    # Advanced capabilities
    FINE_TUNING = "supports_fine_tuning"
    EMBEDDINGS = "supports_embeddings"
    MULTILINGUAL = "supports_multilingual"
    RAG = "supports_rag"  # Retrieval Augmented Generation
    MULTI_TURN = "supports_multi_turn"  # Multi-turn conversations
    PARALLEL_INFERENCE = "supports_parallel_inference"
    IMAGE_GENERATION = "supports_image_generation"
    AUDIO_PROCESSING = "supports_audio_processing"
    JSON_MODE = "supports_json_mode"  # Structured JSON output 

class MessageRole(str, Enum):
    """Role of a message in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"  # For tool/function call responses 