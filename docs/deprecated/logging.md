# Logging in AbstractLLM

## Overview

AbstractLLM provides a flexible logging system that can be configured to output logs to both console and files, with independent control over each output destination. The system is designed to be both developer-friendly during development and production-ready for deployment.

## Basic Configuration

The simplest way to configure logging is through the `configure_logging` function:

```python
from abstractllm import configure_logging
import logging

# Basic configuration (console only)
configure_logging(log_level=logging.INFO)

# File-only logging
configure_logging(
    log_dir="/path/to/logs",
    log_level=logging.INFO
)

# Both console and file logging
configure_logging(
    log_dir="/path/to/logs",
    log_level=logging.INFO,
    console_output=True
)
```

## Configuration Options

### Log Directory

The log directory can be set in three ways:

1. **Environment Variable**:
   ```bash
   export ABSTRACTLLM_LOG_DIR=/path/to/logs
   ```

2. **Configuration Function**:
   ```python
   configure_logging(log_dir="/path/to/logs")
   ```

3. **Default Behavior**:
   - If no log directory is specified, logs only go to console
   - If a log directory is specified, logs go to files by default

### Console Output

Console output can be controlled explicitly:

```python
# Force console output even with log directory
configure_logging(
    log_dir="/path/to/logs",
    console_output=True
)

# Force no console output
configure_logging(
    log_dir="/path/to/logs",
    console_output=False
)

# Automatic (console if no log_dir, no console if log_dir)
configure_logging(
    log_dir="/path/to/logs"  # No console by default
)
```

### Log Levels

You can set different log levels for general logs and provider-specific logs:

```python
import logging

# Set general log level
configure_logging(log_level=logging.DEBUG)

# Set different level for providers
configure_logging(
    log_level=logging.INFO,
    provider_level=logging.DEBUG
)
```

## What Gets Logged

### 1. Main Log File

When a log directory is configured, a main log file is created:
- Filename: `abstractllm_YYYYMMDD_HHMMSS.log`
- Contains: All logging events at the configured level
- Format: `timestamp - logger_name - level - message`

### 2. Request Logs

Each request is logged to a separate JSON file:
- Filename: `{provider}_request_YYYYMMDD_HHMMSS_microseconds.json`
- Contains:
  ```json
  {
    "timestamp": "ISO-8601 timestamp",
    "provider": "provider name",
    "prompt": "user prompt",
    "parameters": {
      "model": "model name",
      "temperature": 0.7,
      // ... other parameters
    }
  }
  ```

### 3. Response Logs

Each response is logged to a separate JSON file:
- Filename: `{provider}_response_YYYYMMDD_HHMMSS_microseconds.json`
- Contains:
  ```json
  {
    "timestamp": "ISO-8601 timestamp",
    "provider": "provider name",
    "response": "complete model response"
  }
  ```

### Security Considerations

The logging system automatically sanitizes sensitive information:

1. **API Keys**: Never logged
2. **Base64 Data**: Truncated in console, preserved in files
3. **Image Data**: Summarized in console, preserved in files
4. **Large Responses**: Truncated in console, preserved in files

## Using in External Programs

### Basic Integration

```python
from abstractllm import create_llm, configure_logging
import logging

# Configure logging at startup
configure_logging(
    log_dir="/var/log/myapp/abstractllm",
    log_level=logging.INFO
)

# Create and use LLM
llm = create_llm("openai")
response = llm.generate("Hello")  # Logs will be written automatically
```

### Development Setup

```python
# development.py
from abstractllm import configure_logging
import logging

# Debug everything to console
configure_logging(
    log_level=logging.DEBUG,
    console_output=True
)
```

### Production Setup

```python
# production.py
from abstractllm import configure_logging
import logging

# Info level to files, no console output
configure_logging(
    log_dir="/var/log/production/abstractllm",
    log_level=logging.INFO,
    console_output=False
)
```

### Hybrid Setup

```python
# hybrid.py
from abstractllm import configure_logging
import logging
import os

# Development vs Production configuration
if os.getenv("ENVIRONMENT") == "production":
    configure_logging(
        log_dir="/var/log/production/abstractllm",
        log_level=logging.INFO,
        console_output=False
    )
else:
    configure_logging(
        log_dir="./logs",
        log_level=logging.DEBUG,
        console_output=True
    )
```

## Best Practices

1. **Early Configuration**:
   ```python
   # Configure logging before any LLM operations
   configure_logging(...)
   ```

2. **Environment-Based Configuration**:
   ```python
   log_dir = os.getenv("ABSTRACTLLM_LOG_DIR", "./logs")
   log_level = getattr(logging, os.getenv("ABSTRACTLLM_LOG_LEVEL", "INFO"))
   configure_logging(log_dir=log_dir, log_level=log_level)
   ```

3. **Log Rotation**:
   ```python
   # Use external log rotation (e.g., logrotate)
   # /etc/logrotate.d/abstractllm
   /var/log/abstractllm/*.log {
       daily
       rotate 7
       compress
       delaycompress
       missingok
       notifempty
   }
   ```

4. **Monitoring Integration**:
   ```python
   # Send logs to both files and monitoring service
   configure_logging(
       log_dir="/var/log/abstractllm",
       console_output=True  # Monitor can read from stdout
   )
   ```

5. **Debug Mode**:
   ```python
   # Temporary debug mode
   configure_logging(
       log_dir="/var/log/abstractllm",
       log_level=logging.DEBUG,
       provider_level=logging.DEBUG,
       console_output=True
   )
   ```

## Common Patterns

### Web Application

```python
# app.py
from flask import Flask
from abstractllm import configure_logging
import logging
import os

app = Flask(__name__)

# Configure based on Flask environment
if app.debug:
    configure_logging(
        log_dir="./logs",
        log_level=logging.DEBUG,
        console_output=True
    )
else:
    configure_logging(
        log_dir="/var/log/webapp/abstractllm",
        log_level=logging.INFO,
        console_output=False
    )
```

### Script Usage

```python
# script.py
from abstractllm import configure_logging
import argparse
import logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_logging(
        log_dir=args.log_dir,
        log_level=logging.DEBUG if args.debug else logging.INFO,
        console_output=args.debug
    )
```

### Library Usage

```python
# your_library.py
from abstractllm import configure_logging
import logging

def setup_library(
    log_dir: Optional[str] = None,
    debug: bool = False
):
    """Configure your library with AbstractLLM logging."""
    configure_logging(
        log_dir=log_dir,
        log_level=logging.DEBUG if debug else logging.INFO,
        console_output=debug
    )
```

## Troubleshooting

1. **No Logs Appearing**:
   - Check if log directory exists and is writable
   - Verify log level is appropriate
   - Check console_output setting

2. **Missing Information**:
   - Set log_level to DEBUG
   - Enable console_output for immediate visibility
   - Check file permissions

3. **Performance Issues**:
   - Reduce log level in production
   - Implement log rotation
   - Use separate disk for logs

4. **Security Concerns**:
   - Review log file permissions
   - Monitor log directory size
   - Implement log retention policy 