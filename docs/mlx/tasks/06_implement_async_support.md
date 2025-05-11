# Task 6: Implement Async Support

## Description
Implement asynchronous generation support for the MLX provider using Python's asyncio.

## Requirements
1. Implement the `generate_async()` method required by AbstractLLMInterface
2. Support both streaming and non-streaming modes in async context
3. Use asyncio's executor to run synchronous MLX code non-blockingly
4. Maintain the same functionality as the synchronous version

## Implementation Details

```python
async def generate_async(self, 
                       prompt: str, 
                       system_prompt: Optional[str] = None, 
                       files: Optional[List[Union[str, Path]]] = None,
                       stream: bool = False, 
                       tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                       **kwargs) -> Union[GenerateResponse, AsyncGenerator[GenerateResponse, None]]:
    """
    Asynchronously generate a response using the MLX model.
    
    This is currently a wrapper around the synchronous method as MLX doesn't provide
    native async support, but follows the required interface.
    """
    import asyncio
    from typing import AsyncGenerator
    
    # Use the current event loop
    loop = asyncio.get_running_loop()
    
    if stream:
        # For streaming, we need to convert the synchronous generator to an async one
        async def async_gen() -> AsyncGenerator[GenerateResponse, None]:
            # Run the sync generate in an executor to avoid blocking
            sync_gen = await loop.run_in_executor(
                None,
                lambda: self.generate(
                    prompt, system_prompt, files, stream=True, tools=tools, **kwargs
                )
            )
            
            # Yield items from the sync generator
            for item in sync_gen:
                yield item
                # Small delay to allow other tasks to run, but not too long to maintain responsiveness
                await asyncio.sleep(0.001)
        
        # Return the async generator directly
        return async_gen()
    else:
        # For non-streaming, we can just run the synchronous method in the executor
        return await loop.run_in_executor(
            None, 
            lambda: self.generate(
                prompt, system_prompt, files, stream=False, tools=tools, **kwargs
            )
        )
```

## References
- See Python asyncio documentation: https://docs.python.org/3/library/asyncio.html
- Reference the MLX Provider Implementation Guide: `docs/mlx/mlx_provider_implementation.md`
- See AbstractLLMInterface for the required async method signature

## Testing
1. Test async generation in a simple asyncio application
2. Test async streaming to ensure chunks are delivered properly
3. Test multiple concurrent generations to ensure they don't block each other
4. Compare the output of async and sync methods to ensure they produce the same results 