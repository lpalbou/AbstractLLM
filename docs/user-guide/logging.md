# Logging and Debugging

AbstractLLM includes a comprehensive logging system that helps you debug your applications, audit model interactions, and monitor performance. This guide explains how to configure logging and use it effectively.

## Basic Logging Configuration

AbstractLLM uses Python's standard `logging` module with some enhancements. By default, logging is set to the `WARNING` level. To enable more detailed logging:

```python
import logging
from abstractllm import create_llm

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a provider
llm = create_llm("openai", model="gpt-4")

# Now you'll see INFO level logs for your interactions
response = llm.generate("Hello, world!")
```

## Log Levels

AbstractLLM uses the following log levels:

- **DEBUG**: Detailed information, typically useful only for diagnosing problems
- **INFO**: Confirmation that things are working as expected
- **WARNING**: Indication that something unexpected happened, but the application still works
- **ERROR**: Due to a more serious problem, the application couldn't perform a function
- **CRITICAL**: A very serious error that might prevent the application from continuing

## Logging Specific Components

You can configure logging for specific components of AbstractLLM:

```python
import logging

# Configure only the API clients to DEBUG level
logging.getLogger('abstractllm.providers').setLevel(logging.DEBUG)

# Configure only the media processing to INFO level
logging.getLogger('abstractllm.media').setLevel(logging.INFO)

# Reduce noise from configuration management
logging.getLogger('abstractllm.utils.config').setLevel(logging.WARNING)
```

## Logging Request and Response Data

AbstractLLM can log requests and responses to help debug issues:

```python
import logging
from abstractllm import create_llm
from abstractllm.logging import enable_request_logging, enable_response_logging

# Enable request and response logging
enable_request_logging(level=logging.INFO)
enable_response_logging(level=logging.INFO)

# Create a provider
llm = create_llm("openai", model="gpt-4")

# Generate a response - requests and responses will be logged
response = llm.generate("Tell me about neural networks.")
```

## Advanced Logging Configuration

For more advanced logging setups:

```python
import logging
import json
from datetime import datetime
from abstractllm import create_llm

# Create a file handler
file_handler = logging.FileHandler(f'abstractllm-{datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Get the logger
logger = logging.getLogger('abstractllm')
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Now you'll get different levels of detail in the console vs. the log file
llm = create_llm("openai", model="gpt-4")
response = llm.generate("Hello, world!")
```

## Custom Log Handlers

You can create custom log handlers for specific needs:

```python
import logging
import json
from abstractllm import create_llm

class JSONFileHandler(logging.FileHandler):
    """A handler that logs messages as JSON objects."""
    
    def __init__(self, filename, mode='a', encoding=None):
        super().__init__(filename, mode, encoding)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            log_entry = {
                "timestamp": record.created,
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
                "function": record.funcName,
                "line": record.lineno
            }
            
            # Add exception info if available
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)
            
            self.stream.write(json.dumps(log_entry) + '\n')
            self.flush()
        except Exception:
            self.handleError(record)

# Add the JSON handler
json_handler = JSONFileHandler('abstractllm.json')
json_handler.setLevel(logging.INFO)
logging.getLogger('abstractllm').addHandler(json_handler)
```

## Sensitive Information Redaction

AbstractLLM's logging system automatically redacts sensitive information like API keys:

```python
import logging
from abstractllm import create_llm
from abstractllm.logging import set_sensitive_values

# Set additional values to be redacted from logs
set_sensitive_values(["your-custom-secret", "another-secret-value"])

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create a provider (the api_key will be redacted in logs)
llm = create_llm("openai", api_key="sk-123456789abcdef")
```

In the logs, you'll see:
```
DEBUG - abstractllm.providers.openai - Created OpenAI client with key sk-[REDACTED]
```

## Audit Logging

For compliance and monitoring, you might want to enable audit logging, which records all LLM interactions:

```python
from abstractllm import create_llm
from abstractllm.logging import configure_audit_logging

# Configure audit logging to a specific file
configure_audit_logging(
    filename="audit.log",
    include_prompts=True,
    include_responses=True,
    include_metadata=True
)

# Create a provider
llm = create_llm("openai", model="gpt-4")

# All interactions will be logged to audit.log
response = llm.generate("Tell me about quantum computing.")
```

## Logging with Sessions

When using sessions, you can log the entire conversation history:

```python
import logging
from abstractllm import create_llm
from abstractllm.session import Session

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('abstractllm.session').setLevel(logging.DEBUG)

# Create a session
provider = create_llm("openai", model="gpt-4")
session = Session(
    system_prompt="You are a helpful assistant.",
    provider=provider
)

# The conversation will be logged
session.add_message("user", "Hello, how are you?")
response = session.generate()
print(response)

session.add_message("user", "Tell me about neural networks.")
response = session.generate()
print(response)
```

## Debugging Tool Calls

Tool calls can be particularly complex to debug. AbstractLLM provides specialized logging for tool interactions:

```python
import logging
from abstractllm import create_llm
from abstractllm.session import Session

# Configure tool logging
logging.getLogger('abstractllm.tools').setLevel(logging.DEBUG)

# Define a tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is currently sunny and 72°F."

# Create a session with tools
provider = create_llm("openai", model="gpt-4")
session = Session(
    system_prompt="You are a helpful assistant that can check the weather.",
    provider=provider,
    tools=[get_weather]
)

# Generate a response with tool usage - detailed logs will be printed
response = session.generate_with_tools("What's the weather like in San Francisco?")
print(response)
```

## Performance Logging

For performance optimization, you can log timing information:

```python
import logging
import time
from abstractllm import create_llm

class TimingLogFilter(logging.Filter):
    """A filter that adds timing information to log records."""
    
    def __init__(self):
        super().__init__()
        self.start_times = {}
    
    def start_timer(self, name):
        self.start_times[name] = time.time()
    
    def filter(self, record):
        if hasattr(record, 'timing_name') and record.timing_name in self.start_times:
            elapsed = time.time() - self.start_times[record.timing_name]
            record.elapsed_ms = int(elapsed * 1000)
            record.msg = f"{record.msg} (took {record.elapsed_ms}ms)"
        return True

# Create and configure the filter
timing_filter = TimingLogFilter()
logger = logging.getLogger('abstractllm')
logger.addFilter(timing_filter)

# Use the timer to measure performance
timing_filter.start_timer('generation')
llm = create_llm("openai", model="gpt-4")
response = llm.generate("Explain quantum physics.")
logger.info("Generated response", extra={'timing_name': 'generation'})
```

## Common Logging Tasks

### Logging Request/Response Pairs

```python
import logging
from abstractllm import create_llm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def log_conversation(prompt, response):
    logger.info("----- Conversation Log -----")
    logger.info(f"PROMPT: {prompt}")
    logger.info(f"RESPONSE: {response}")
    logger.info("---------------------------")

llm = create_llm("openai", model="gpt-4")
prompt = "Explain the concept of machine learning."
response = llm.generate(prompt)
log_conversation(prompt, response)
```

### Logging Errors

```python
import logging
from abstractllm import create_llm
from abstractllm.exceptions import AbstractLLMError

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

llm = create_llm("openai", model="gpt-4")

try:
    # Deliberately cause an error with invalid parameter
    response = llm.generate("Hello", temperature=2.0)  # Temperature > 1.0 is invalid
except AbstractLLMError as e:
    logger.error(f"Generation failed: {str(e)}", exc_info=True)
```

### Logging API Usage

```python
import logging
from abstractllm import create_llm

class APIUsageTracker:
    def __init__(self):
        self.logger = logging.getLogger('api_usage')
        self.logger.setLevel(logging.INFO)
        
        # Create a dedicated file handler for API usage
        handler = logging.FileHandler('api_usage.log')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.request_count = 0
        self.token_count = 0
    
    def log_request(self, provider, model, prompt_tokens, completion_tokens):
        self.request_count += 1
        tokens = prompt_tokens + completion_tokens
        self.token_count += tokens
        
        self.logger.info(
            f"API Request #{self.request_count} | Provider: {provider} | " 
            f"Model: {model} | Tokens: {tokens} | "
            f"Total tokens: {self.token_count}"
        )

# Use the tracker
tracker = APIUsageTracker()
llm = create_llm("openai", model="gpt-4")
response = llm.generate("Hello, world!")

# Log the usage (in a real application, you'd get the token counts from the response metadata)
tracker.log_request("openai", "gpt-4", 5, 20)
```

## Conclusion

Effective logging is essential for debugging, monitoring, and auditing your AbstractLLM applications. By configuring the logging system to suit your needs, you can gain valuable insights into how your application is interacting with LLM providers and diagnose issues more efficiently. 