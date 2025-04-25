# Vision and File Handling in AbstractLLM

## Overview

This document outlines the current implementation of vision/image and file handling in AbstractLLM. The implementation follows a minimalist approach while ensuring robust cross-provider compatibility and efficient processing.

## Current Implementation

### Core Components

1. **MediaInput Interface**:
   - Abstract base class for all media types
   - Provider-agnostic format conversion
   - Metadata handling
   - Type identification

2. **ImageInput Implementation**:
   - Handles multiple input formats
   - Provider-specific formatting
   - Format caching
   - MIME type detection

3. **MediaProcessor**:
   - Message structure handling
   - Provider-specific processing
   - Input validation
   - Error management

### Data Flow

```ascii
Input Source ──────────────────────┐
  │                                │
  ▼                                │
┌──────────────────────┐           │
│     MediaFactory     │           │
│    .from_source()    │           │
└──────────┬───────────┘           │
           ▼                       │
┌──────────────────────┐           │
│     ImageInput       │           │
│    Constructor       │           │
│ ┌──────────────────┐ │           │
│ │    Properties    │ │           │
│ │ source: str/Path │ │           │
│ │ detail_level: str│ │           │
│ │ _mime_type: str  │ │           │
│ │ _cached_formats{}│ │           │
│ └──────────────────┘ │           │
└──────────┬───────────┘           │
           ▼                       │
┌──────────────────────┐           │
│   MediaProcessor     │           │
│   process_inputs()   │           │
└──────────┬───────────┘           │
           ▼                       │
┌──────────────────────┐           │
│  Provider Format     │           │
│  to_provider_format()│           │
└──────────┬───────────┘           │
           ▼                       ▼
    Provider-Specific      Original Source
    Message Structure      (if unchanged)
```

### Provider-Specific Flow (Anthropic Example)

```ascii
Input Source
    │
    ▼
┌───────────────────────────┐
│      MediaFactory         │
│      .from_source()       │
│                          │
│ STATE: Original format   │
│ (path/URL/base64/dataURL)│
└──────────┬───────────────┘
           │
           ▼
┌───────────────────────────┐
│      ImageInput           │
│      Constructor          │
│                          │
│ STATE:                   │
│ self.source: unchanged   │
│ self.detail_level: str   │
│ self._mime_type: str     │
└──────────┬───────────────┘
           │
           ▼
┌───────────────────────────┐
│   to_provider_format()    │
│   ("anthropic")          │
│                          │
│ STATE: Checks cache first│
└──────────┬───────────────┘
           │
           ▼
┌───────────────────────────┐
│  _format_for_anthropic()  │
│                          │
│ IF URL:                  │
│   {                      │
│     "type": "image",     │
│     "source": {          │
│       "type": "url",     │
│       "url": source_str  │
│     }                    │
│   }                      │
│                          │
│ ELSE:                    │
│   1. Get binary content  │
│   2. Check size (<100MB) │
│   3. Convert to base64   │
│   {                      │
│     "type": "image",     │
│     "source": {          │
│       "type": "base64",  │
│       "media_type": mime,│
│       "data": b64_data   │
│     }                    │
│   }                      │
└──────────┬───────────────┘
           │
           ▼
┌───────────────────────────┐
│    Final Message          │
│    Structure             │
│                          │
│ {                        │
│   "role": "user",        │
│   "content": [           │
│     {                    │
│       "type": "text",    │
│       "text": prompt     │
│     },                   │
│     {                    │
│       "type": "image",   │
│       "source": {...}    │
│     }                    │
│   ]                      │
│ }                        │
└──────────┬───────────────┘
           │
           ▼
      Anthropic API
```

## State Transitions

### Input States

1. **File Path**:
   - Initial: Local filesystem path
   - Final: Base64 or kept as path (HuggingFace)

2. **URL**:
   - Initial: HTTP/HTTPS URL
   - Final: Kept as URL or downloaded and converted to base64

3. **Base64**:
   - Initial: Raw base64 string
   - Final: Validated and formatted according to provider

4. **Data URL**:
   - Initial: base64 with MIME type prefix
   - Final: Extracted base64 or kept as is (OpenAI)

### Provider Format States

1. **OpenAI**:
   ```python
   {
       "type": "image_url",
       "image_url": {
           "url": str,  # URL or data URL
           "detail": str
       }
   }
   ```

2. **Anthropic**:
   ```python
   {
       "type": "image",
       "source": {
           "type": "url" | "base64",
           "url": str | None,
           "media_type": str | None,
           "data": str | None
       }
   }
   ```

3. **Ollama**:
   ```python
   str  # URL or base64 string
   ```

4. **HuggingFace**:
   ```python
   Union[str, bytes]  # Path, URL, or binary content
   ```

## Implementation Details

### Format Caching

- Provider-specific formats are cached in `_cached_formats`
- Cache key: provider name
- Cache value: formatted data structure
- No binary content caching for memory efficiency

### MIME Type Detection

1. Constructor-provided type
2. Data URL extraction
3. File extension mapping
4. URL extension analysis
5. Default to 'image/jpeg'

### Error Handling

- Custom `ImageProcessingError` with provider context
- Size validation (e.g., Anthropic's 100MB limit)
- Format validation
- Network error handling
- File access error handling

## Best Practices

1. **Input Handling**:
   - Early validation
   - Format preservation
   - Efficient conversion

2. **Provider Compatibility**:
   - Format requirements
   - Size limitations
   - Capability checking

3. **Error Management**:
   - Clear error messages
   - Provider context
   - Graceful fallbacks

4. **Performance**:
   - Format caching
   - Lazy loading
   - Memory efficiency

## Future Improvements

1. **Media Types**:
   - Document support
   - Audio handling
   - Video processing
   - Tabular data

2. **Features**:
   - Format conversion
   - Size optimization
   - Metadata extraction
   - Batch processing

3. **Provider Support**:
   - New providers
   - Enhanced capabilities
   - Optimized formats 