# Media Component

## Overview
The media module provides a unified, extensible system for handling multimodal inputs (images, text, tabular data) across all LLM providers. It abstracts provider-specific formatting requirements while maintaining a clean, object-oriented architecture.

## Code Quality Assessment
**Rating: 9/10**

### Strengths
- Excellent use of design patterns (Factory, Strategy, Facade, Abstract Base Class)
- Comprehensive error handling with custom exceptions
- Efficient caching strategy for network resources
- Clear separation of concerns with single responsibility principle
- Well-documented with helpful error messages
- Defensive programming with thorough validation

### Issues
- Some code duplication in provider-specific formatting methods
- Type hints could be more specific (use Literal types)
- Optional dependencies handled ad-hoc rather than centrally
- Missing async support for network operations

## Component Mindmap
```
Media System
├── Interface (interface.py)
│   └── MediaInput (ABC)
│       ├── to_provider_format(provider, **kwargs)
│       ├── media_type property
│       └── metadata property
│
├── Factory (factory.py)
│   └── MediaFactory
│       ├── from_source(source) - Auto-detection
│       ├── from_multiple_sources(sources)
│       └── Type Detection
│           ├── File extension matching
│           ├── MIME type detection
│           └── Content inspection
│
├── Media Types
│   ├── Image (image.py)
│   │   ├── URL support
│   │   ├── File path support
│   │   ├── Base64 support
│   │   ├── PIL Image support
│   │   └── Provider Formats
│   │       ├── OpenAI: image_url objects
│   │       ├── Anthropic: base64 source
│   │       ├── Ollama: base64 strings
│   │       └── HuggingFace: file paths
│   │
│   ├── Text (text.py)
│   │   ├── Plain text files
│   │   ├── Multiple encodings
│   │   └── Line limiting
│   │
│   └── Tabular (tabular.py)
│       ├── CSV support
│       ├── TSV support
│       └── Markdown table formatting
│
└── Processor (processor.py) - Facade
    ├── process_inputs(params, provider)
    ├── Single/Multiple media handling
    └── Message structure integration
```

## Design Patterns Used
1. **Abstract Base Class**: `MediaInput` defines the interface
2. **Factory**: `MediaFactory` creates appropriate media objects
3. **Strategy**: Each media type implements provider formatting
4. **Facade**: `MediaProcessor` simplifies media handling
5. **Caching**: Memoization of downloaded/processed content

## Provider Integration
```python
# How providers use media:
params = {"prompt": "Describe this", "image": "photo.jpg"}
processed = MediaProcessor.process_inputs(params, "openai")
# Returns params with media properly formatted for OpenAI
```

## Supported Media Sources
- **Images**: URLs, file paths, base64, PIL Images, dicts
- **Text**: File paths, with encoding detection
- **Tabular**: CSV/TSV files converted to markdown

## Error Handling
- `ImageProcessingError`: Image-specific issues
- `FileProcessingError`: File access problems
- Graceful fallbacks for missing dependencies
- Detailed error messages with solutions

## Dependencies
- **Required**: None (pure Python)
- **Optional**: 
  - `PIL/Pillow`: Advanced image processing
  - `requests`: URL fetching
  - `mimetypes`: Type detection

## Recommendations
1. **Reduce duplication**: Extract common formatting patterns
2. **Add validation**: Provider-specific media requirements
3. **Implement transforms**: Auto-resize, compress for limits
4. **Add async support**: For network operations
5. **Centralize deps**: Better optional dependency management

## Technical Debt
- Provider formatting methods have similar patterns (DRY violation)
- No size/format validation before sending to providers
- Missing streaming support for large files
- No progress callbacks for long operations

## Security Considerations
- Proper User-Agent headers for HTTP requests
- Timeout settings prevent DoS
- File path validation prevents directory traversal
- No execution of untrusted content

## Performance Notes
- Lazy loading of content
- Efficient caching prevents redundant work
- Context managers for proper resource cleanup
- Could benefit from async I/O

## Future Enhancements
1. **Video support**: For providers that accept video
2. **Audio support**: For speech-to-text models  
3. **Document support**: PDF, DOCX parsing
4. **Batch processing**: Optimize multiple media items
5. **Media pipelines**: Chain transformations

## Integration Example
```python
from abstractllm.media import MediaFactory, MediaProcessor

# Create media from various sources
image = MediaFactory.from_source("https://example.com/image.jpg")
text = MediaFactory.from_source("document.txt")

# Process for specific provider
params = {
    "prompt": "Analyze these",
    "images": ["photo1.jpg", "photo2.png"],
    "file": "data.csv"
}
processed = MediaProcessor.process_inputs(params, "anthropic")
```