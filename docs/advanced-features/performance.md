# Performance Optimization

This guide covers strategies to optimize AbstractLLM for performance in production environments while maintaining quality and reliability.

## Model Selection

Choosing the right model is crucial for balancing performance and quality:

```python
from abstractllm import create_llm

# For quick responses to simple queries
fast_llm = create_llm("openai", model="gpt-3.5-turbo")

# For complex reasoning tasks
powerful_llm = create_llm("openai", model="gpt-4")

# For local deployment with lower latency
local_llm = create_llm("ollama", model="mistral")

def get_appropriate_model(query, importance=1):
    """Select the appropriate model based on query complexity."""
    # Simple keyword-based complexity estimation
    complex_keywords = ["analyze", "compare", "synthesize", "evaluate", 
                      "design", "recommend", "optimize"]
    
    # Count complex keywords
    complexity = sum(1 for keyword in complex_keywords if keyword in query.lower())
    
    # Factor in user-specified importance
    adjusted_complexity = complexity * importance
    
    if adjusted_complexity > 2:
        return powerful_llm
    elif adjusted_complexity > 0:
        return fast_llm
    else:
        return local_llm
```

## Response Caching

Implement caching to avoid redundant API calls:

```python
import hashlib
import json
import redis
from abstractllm import create_llm
from functools import lru_cache

# Simple in-memory caching with LRU
@lru_cache(maxsize=100)
def cached_generate(prompt, model="gpt-3.5-turbo", temperature=0.7):
    """Cache responses for identical prompts and parameters."""
    llm = create_llm("openai", model=model)
    return llm.generate(prompt, temperature=temperature)

# Redis-based caching for distributed systems
class RedisCacheWrapper:
    def __init__(self, provider, redis_url="redis://localhost:6379/0", ttl=3600):
        """Initialize cache wrapper with Redis backend."""
        self.provider = provider
        self.redis = redis.Redis.from_url(redis_url)
        self.ttl = ttl  # Cache TTL in seconds
    
    def generate(self, prompt, **kwargs):
        """Generate with caching."""
        # Create cache key from prompt and kwargs
        cache_key = self._create_cache_key(prompt, kwargs)
        
        # Try to get from cache
        cached = self.redis.get(cache_key)
        if cached:
            return cached.decode("utf-8")
        
        # Generate and cache
        response = self.provider.generate(prompt, **kwargs)
        self.redis.setex(cache_key, self.ttl, response)
        return response
    
    def _create_cache_key(self, prompt, kwargs):
        """Create a deterministic cache key."""
        key_data = {
            "prompt": prompt,
            "params": sorted(kwargs.items())
        }
        serialized = json.dumps(key_data, sort_keys=True)
        return f"abstractllm:{hashlib.md5(serialized.encode()).hexdigest()}"

# Usage
llm = create_llm("openai", model="gpt-4")
cached_llm = RedisCacheWrapper(llm)
response = cached_llm.generate("What is the capital of France?")
```

## Asynchronous Processing

Use async APIs for concurrent requests:

```python
import asyncio
from abstractllm import create_llm

async def process_multiple_prompts(prompts, model="gpt-3.5-turbo"):
    """Process multiple prompts concurrently."""
    llm = create_llm("openai", model=model)
    
    # Create tasks for each prompt
    tasks = [llm.generate_async(prompt) for prompt in prompts]
    
    # Execute concurrently and gather results
    results = await asyncio.gather(*tasks)
    return results

# Usage
prompts = [
    "Explain quantum computing briefly.",
    "What are the main features of Python?",
    "How does blockchain work?",
    "What is machine learning?"
]

async def main():
    results = await process_multiple_prompts(prompts)
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result[:50]}...")

# Run the async function
asyncio.run(main())
```

## Batch Processing

Process requests in batches to optimize throughput:

```python
from abstractllm import create_llm
import asyncio
import time
from collections import deque

class BatchProcessor:
    """Process requests in batches for better throughput."""
    
    def __init__(self, provider, batch_size=10, max_wait_time=2.0):
        """Initialize batch processor."""
        self.provider = provider
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.queue = deque()
        self.processing = False
    
    async def add_request(self, prompt, **kwargs):
        """Add a request to the batch queue and get a future for the result."""
        future = asyncio.Future()
        self.queue.append((prompt, kwargs, future))
        
        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        """Process requests in batches."""
        self.processing = True
        
        while self.queue:
            batch_start_time = time.time()
            batch = []
            
            # Collect batch
            while self.queue and len(batch) < self.batch_size:
                batch.append(self.queue.popleft())
                
                # Check if we should wait for more requests
                if len(batch) < self.batch_size and (time.time() - batch_start_time) < self.max_wait_time:
                    # Wait a bit for more requests to accumulate
                    await asyncio.sleep(0.1)
            
            # Process batch concurrently
            batch_tasks = []
            for prompt, kwargs, future in batch:
                task = asyncio.create_task(self.provider.generate_async(prompt, **kwargs))
                batch_tasks.append((task, future))
            
            # Wait for all tasks and set results
            for task, future in batch_tasks:
                try:
                    result = await task
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
        
        self.processing = False

# Usage
async def main():
    llm = create_llm("openai", model="gpt-3.5-turbo")
    processor = BatchProcessor(llm)
    
    # Add many requests
    tasks = []
    for i in range(20):
        task = processor.add_request(f"Question {i}: What is the capital of France?")
        tasks.append(task)
    
    # Wait for all results
    results = await asyncio.gather(*tasks)
    print(f"Processed {len(results)} requests")
```

## Connection Pooling

Use connection pooling for HTTP requests:

```python
import aiohttp
from abstractllm import create_llm

async def pooled_requests():
    """Make requests with connection pooling."""
    # Create a shared session for connection pooling
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(
            limit=20,  # Max connections
            keepalive_timeout=30.0,
            limit_per_host=10  # Max connections per host
        )
    ) as session:
        # Create LLM with custom session
        llm = create_llm(
            "openai", 
            model="gpt-3.5-turbo",
            http_session=session  # Pass the shared session
        )
        
        # Process multiple requests
        tasks = []
        for i in range(10):
            task = llm.generate_async(f"Question {i}: What is the capital of France?")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
```

## Token Optimization

Optimize token usage to reduce costs and improve performance:

```python
from abstractllm import create_llm
import re

def truncate_prompt(prompt, max_length=500):
    """Truncate prompt to a maximum token length (approximate)."""
    # Simple word-based approximation (not exact but fast)
    words = prompt.split()
    if len(words) <= max_length:
        return prompt
    
    return " ".join(words[:max_length]) + "..."

def optimize_system_prompt(system_prompt):
    """Optimize system prompt for token efficiency."""
    # Remove redundant text
    optimized = re.sub(r'You are a helpful assistant that ', '', system_prompt)
    optimized = re.sub(r'You are an AI language model ', '', optimized)
    
    # Remove unnecessary qualifiers
    optimized = re.sub(r'(very|extremely|incredibly) ', '', optimized)
    
    # Consolidate instructions
    optimized = re.sub(r'Please be (concise|brief|short) and (clear|direct).', 'Be concise and clear.', optimized)
    
    return optimized.strip()

# Usage
original_system_prompt = "You are a very helpful AI assistant that provides extremely detailed and incredibly thorough responses. Please be concise and clear."
optimized_prompt = optimize_system_prompt(original_system_prompt)
print(f"Original ({len(original_system_prompt)} chars): {original_system_prompt}")
print(f"Optimized ({len(optimized_prompt)} chars): {optimized_prompt}")

llm = create_llm("openai", model="gpt-3.5-turbo", system_prompt=optimized_prompt)
```

## Load Balancing

Distribute requests across multiple providers:

```python
from abstractllm import create_llm
from abstractllm.chains import LoadBalancedChain
import random

# Create multiple providers
providers = [
    create_llm("openai", model="gpt-3.5-turbo"),
    create_llm("anthropic", model="claude-3-haiku-20240307"),
    create_llm("ollama", model="llama3")
]

# Define load balancing strategies
def round_robin_strategy(providers, context=None):
    """Simple round-robin selection."""
    round_robin_strategy.current = getattr(round_robin_strategy, 'current', -1) + 1
    return providers[round_robin_strategy.current % len(providers)]

def weighted_strategy(providers, context=None):
    """Weighted random selection."""
    weights = [0.6, 0.3, 0.1]  # 60% OpenAI, 30% Anthropic, 10% Ollama
    return random.choices(providers, weights=weights)[0]

def query_based_strategy(providers, context=None):
    """Select provider based on query complexity."""
    query = context.get('prompt', '').lower() if context else ''
    
    # Check for specific patterns
    if any(word in query for word in ['code', 'function', 'programming']):
        return providers[0]  # OpenAI for code
    elif len(query.split()) > 30:
        return providers[1]  # Anthropic for longer contexts
    else:
        return providers[2]  # Ollama for simple queries
        
# Create load balancer
load_balancer = LoadBalancedChain(
    providers=providers,
    selection_strategy=query_based_strategy
)

# Use the load balancer
response = load_balancer.generate(
    "Write a Python function to calculate Fibonacci numbers",
    context={'type': 'code_request'}
)
```

## Monitoring and Optimization

Monitor performance and optimize based on metrics:

```python
import time
import statistics
from abstractllm import create_llm
import logging

class PerformanceTracker:
    """Track and monitor LLM performance."""
    
    def __init__(self, provider):
        """Initialize with a provider."""
        self.provider = provider
        self.response_times = []
        self.token_counts = []
        self.error_count = 0
        self.request_count = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("performance_tracker")
    
    def generate(self, prompt, **kwargs):
        """Generate response with performance tracking."""
        self.request_count += 1
        start_time = time.time()
        
        try:
            # Generate response
            response = self.provider.generate(prompt, **kwargs)
            
            # Calculate metrics
            duration = time.time() - start_time
            self.response_times.append(duration)
            
            # Estimate token count (very approximate)
            approx_tokens = len(prompt.split()) + len(response.split())
            self.token_counts.append(approx_tokens)
            
            # Log performance
            self.logger.info(f"Request #{self.request_count} completed in {duration:.2f}s")
            
            # Report slow responses
            if duration > 5.0:
                self.logger.warning(f"Slow response detected: {duration:.2f}s")
            
            return response
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error in request #{self.request_count}: {str(e)}")
            raise
    
    def get_performance_stats(self):
        """Get performance statistics."""
        if not self.response_times:
            return {"status": "No data"}
        
        return {
            "request_count": self.request_count,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0,
            "avg_response_time": statistics.mean(self.response_times),
            "median_response_time": statistics.median(self.response_times),
            "p95_response_time": sorted(self.response_times)[int(len(self.response_times) * 0.95)],
            "avg_token_count": statistics.mean(self.token_counts),
        }

# Usage
llm = create_llm("openai", model="gpt-3.5-turbo")
tracked_llm = PerformanceTracker(llm)

# Generate responses
response = tracked_llm.generate("What is machine learning?")
print(response)

# Get performance stats
stats = tracked_llm.get_performance_stats()
print(stats)
```

## Next Steps

- [Multi-Modal Content](multimodal.md): Working with images and other content types efficiently
- [Custom Providers](custom-providers.md): Implementing custom providers with performance optimizations
- [Tool Calls](../user-guide/tools.md): Optimizing tool usage performance 