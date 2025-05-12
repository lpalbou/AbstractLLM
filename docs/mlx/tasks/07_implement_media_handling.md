# Task 7: Implement Basic Media Handling

## Status
**Implemented:** Yes
**Completion Date:** Current date

## Description
Implement basic media handling in the MLX provider to support text file processing and detect image inputs (although full vision support will come later).

## Requirements
1. Implement the `_process_files()` method to handle files provided to the generate method
2. Use AbstractLLM's MediaFactory to process different media types
3. Handle text files by appending their content to the prompt
4. Detect image files and check if the model supports vision

## Implementation Details

```python
def _process_files(self, prompt: str, files: List[Union[str, Path]]) -> str:
    """Process input files and append to prompt as needed."""
    from abstractllm.media.factory import MediaFactory
    from abstractllm.exceptions import UnsupportedFeatureError, MediaProcessingError
    
    processed_prompt = prompt
    has_images = False
    
    # Convert all file paths to Path objects for consistent handling
    file_paths = [Path(f) if isinstance(f, str) else f for f in files]
    
    # Verify files exist before processing
    for file_path in file_paths:
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            raise MediaProcessingError(f"File not found: {file_path}")
    
    logger.info(f"Processing {len(files)} file(s) for MLX model")
    
    # Process each file
    for file_path in file_paths:
        try:
            logger.debug(f"Processing file: {file_path}")
            media_input = MediaFactory.from_source(file_path)
            
            if media_input.media_type == "image":
                logger.debug(f"Detected image file: {file_path}")
                has_images = True
                # Actual image processing will be implemented in a future task
                # Simply flagging for now to check model compatibility
            elif media_input.media_type == "text":
                logger.debug(f"Processing text file: {file_path}")
                # Append text content to prompt with clear formatting
                processed_prompt += f"\n\n### Content from file '{file_path.name}':\n{media_input.content}\n###\n"
                logger.debug(f"Added {len(media_input.content)} chars of text content from {file_path.name}")
            else:
                logger.warning(f"Unsupported media type: {media_input.media_type} for file {file_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise MediaProcessingError(f"Failed to process file {file_path}: {str(e)}")
    
    # Check if this is a vision model if images are present
    if has_images:
        logger.debug("Images detected, checking if model supports vision")
        if not self._is_vision_model:
            logger.warning("Model does not support vision but images were provided")
            raise UnsupportedFeatureError(
                "vision",
                "This model does not support vision inputs. Try using a vision-capable model like 'mlx-community/llava-1.5-7b-mlx'",
                provider="mlx"
            )
        else:
            logger.debug("Vision-capable model confirmed for image processing")
            
    return processed_prompt
```

Then update the `generate()` method to use this for file processing:

```python
# In the generate method where files are processed:
if files:
    try:
        formatted_prompt = self._process_files(formatted_prompt, files)
    except (UnsupportedFeatureError, MediaProcessingError) as e:
        # Pass through our custom exceptions
        raise e
    except Exception as e:
        # Wrap unknown exceptions
        logger.error(f"Unexpected error processing files: {e}")
        raise MediaProcessingError(f"Failed to process input files: {str(e)}")
```

## References
- See AbstractLLM's MediaFactory implementation
- Reference the MLX Provider Implementation Guide: `docs/mlx/mlx_provider_implementation.md`
- See `abstractllm/exceptions.py` for UnsupportedFeatureError and MediaProcessingError

## Testing
1. Test processing a text file to ensure its content is properly included in the prompt
2. Test with an image file on a non-vision model to verify it raises the correct error
3. Test with various file types to ensure proper handling
4. Test with invalid files to ensure error handling works correctly 