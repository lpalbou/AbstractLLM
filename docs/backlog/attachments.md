# Attachment Management Improvements

## Current State

### What We Have Now

AbstractLLM currently supports comprehensive attachment handling through the `MediaFactory` system:

**Supported Media Types:**
- **Images**: JPEG, PNG, GIF, WebP, BMP (via PIL)
- **Text**: Plain text, Markdown, JSON files
- **Tabular**: CSV, TSV files with automatic parsing

**Provider Support:**
- ✅ **OpenAI**: Full support (images, text, tabular)
- ✅ **Anthropic**: Full support (images, text, tabular) 
- ✅ **Ollama**: Full support (images, text, tabular)
- ✅ **HuggingFace**: Full support (images, text, tabular)
- ✅ **MLX**: Full support (images, text, tabular) - *recently added*

**Current Architecture:**
```python
# Session-level attachment handling
session = Session(provider="anthropic")
response = session.generate(
    prompt="Analyze this data",
    files=["data.csv", "chart.png", "report.md"]
)
```

### Current Limitations

#### 1. **Memory Management Issues**
- **Problem**: Once a file is read and processed, its entire content becomes part of the chat history
- **Impact**: Large files (especially images/tables) consume significant context window space
- **Example**: A 2MB CSV file becomes part of every subsequent message in the conversation
- **Result**: Inefficient memory usage and potential context window overflow

#### 2. **Static Attachment Model**
- **Problem**: No way to dynamically add/remove attachments during a conversation
- **Impact**: Cannot selectively reference files based on conversation flow
- **Example**: Cannot "forget" a large dataset after analysis is complete to make room for new files

#### 3. **No Reference-Based System**
- **Problem**: Files are always fully embedded in context
- **Impact**: Cannot maintain a "library" of attachments that can be referenced on-demand
- **Example**: Cannot keep a collection of documents that the LLM can query when needed

#### 4. **Session vs Message Level Confusion**
- **Problem**: Unclear whether attachments belong to the session or individual messages
- **Impact**: Inconsistent behavior across different use cases
- **Current**: Files are passed per `generate()` call but become part of session history

## Proposed Improvements

### 1. **Dynamic Attachment Registry**

```python
# Proposed API
session = Session(provider="anthropic")

# Register attachments without immediately loading into context
session.attachments.register("sales_data", "data/sales_2024.csv")
session.attachments.register("company_logo", "assets/logo.png") 
session.attachments.register("manual", "docs/user_manual.md")

# Query available attachments
print(session.attachments.list())  # ["sales_data", "company_logo", "manual"]

# Reference attachments on-demand
response = session.generate(
    prompt="Analyze Q1 sales trends",
    use_attachments=["sales_data"]  # Only load this file into context
)

# Later, switch context
response = session.generate(
    prompt="Update the logo in this design",
    use_attachments=["company_logo"]  # Different file, sales_data not in context
)
```

### 2. **Smart Memory Management**

```python
# Proposed memory management strategies
session.memory_config = {
    "max_attachment_tokens": 2000,  # Limit attachment content per message
    "attachment_strategy": "summarize",  # "full" | "summarize" | "reference"
    "auto_compress": True,  # Automatically compress large files
    "retention_policy": "last_used"  # Keep recently used attachments in memory
}

# Automatic summarization for large files
response = session.generate(
    prompt="What are the key findings?",
    files=["large_report.pdf"],  # Auto-summarized if > max_attachment_tokens
    attachment_strategy="summarize"
)
```

### 3. **Hierarchical Attachment Scopes**

```python
# Session-level: Available throughout conversation
session.attachments.register_persistent("company_context", "company_info.md")

# Message-level: Only for specific interactions  
response = session.generate(
    prompt="Analyze this specific dataset",
    files=["temp_data.csv"],  # Not persisted in session
    attachment_scope="message"  # vs "session" 
)

# Conversation-level: Available for a sub-conversation
with session.conversation_scope() as conv:
    conv.attachments.register("analysis_data", "analysis.xlsx")
    # Multiple messages can reference analysis_data
    # Automatically cleaned up when scope exits
```

### 4. **Attachment Preprocessing and Caching**

```python
# Proposed preprocessing pipeline
session.attachments.register(
    "large_dataset", 
    "data.csv",
    preprocess={
        "summarize": True,  # Create summary for quick reference
        "index": True,      # Create searchable index
        "chunk_size": 1000, # Break into manageable chunks
        "cache": True       # Cache processed versions
    }
)

# Smart querying
response = session.generate(
    prompt="Find records where revenue > 100k",
    attachments={"large_dataset": {"query": "revenue > 100000"}}
)
```

### 5. **Current Practices Integration**

Based on research of current practices:

#### **RAG-Style Attachment Handling**
```python
# Vector-based attachment retrieval
session.attachments.register_with_embedding("knowledge_base", "docs/")
response = session.generate(
    prompt="How do I configure SSL?",
    attachment_retrieval="semantic",  # Find relevant sections automatically
    max_retrieved_chunks=3
)
```

#### **Tool-Based File Access**
```python
# Files as tools rather than context
@session.tool
def read_file_section(filename: str, section: str) -> str:
    """Read specific section of a registered file."""
    return session.attachments.get_section(filename, section)

@session.tool  
def search_attachments(query: str) -> str:
    """Search across all registered attachments."""
    return session.attachments.search(query)

# LLM can now dynamically access files as needed
response = session.generate(
    prompt="Compare Q1 and Q2 sales",
    tools=[read_file_section, search_attachments]
)
```

## Implementation Roadmap

### Phase 1: Foundation (v1.0)
- [ ] Create `AttachmentRegistry` class
- [ ] Implement session-level attachment management
- [ ] Add attachment reference system (vs full embedding)
- [ ] Basic memory management (token limits)

### Phase 2: Smart Processing (v1.1)  
- [ ] Automatic summarization for large files
- [ ] Chunking and indexing system
- [ ] Preprocessing pipeline
- [ ] Caching layer

### Phase 3: Advanced Features (v1.2)
- [ ] Vector-based attachment retrieval
- [ ] Tool-based file access
- [ ] Hierarchical scopes (session/conversation/message)
- [ ] Advanced memory policies

### Phase 4: Optimization (v1.3)
- [ ] Performance optimization
- [ ] Memory usage analytics
- [ ] Automatic attachment cleanup
- [ ] Provider-specific optimizations

## Technical Considerations

### Memory Management Strategies

1. **Token-Based Limits**: Set maximum tokens per attachment per message
2. **LRU Cache**: Keep recently accessed attachments in memory
3. **Compression**: Automatic compression of large text files
4. **Summarization**: AI-powered summarization of large documents
5. **Chunking**: Break large files into retrievable chunks

### Provider Compatibility

Different providers handle attachments differently:
- **OpenAI**: Native multimodal support, good for images + text
- **Anthropic**: Excellent document processing, good for large texts
- **MLX**: Local processing, good for privacy-sensitive files
- **Ollama**: Variable support based on model

### Storage Considerations

- **In-Memory**: Fast access, limited by RAM
- **Disk Cache**: Persistent, slower access
- **Vector Store**: Semantic search, requires embedding model
- **Database**: Structured storage, good for metadata

## Benefits

### For Users
- **Reduced Context Pollution**: Only relevant attachments in each message
- **Better Performance**: Faster generation with smaller contexts
- **Flexible File Management**: Add/remove files dynamically
- **Cost Efficiency**: Fewer tokens used per request

### For Developers  
- **Clear Architecture**: Separation of concerns between attachments and conversation
- **Extensible System**: Easy to add new attachment types and processing strategies
- **Better Testing**: Isolated attachment handling logic
- **Provider Agnostic**: Consistent API across all providers

## Migration Strategy

### Backward Compatibility
Current `files=[]` parameter will continue to work:
```python
# Current API (still supported)
response = session.generate(prompt="...", files=["data.csv"])

# New API (recommended)
session.attachments.register("data", "data.csv")
response = session.generate(prompt="...", use_attachments=["data"])
```

### Gradual Migration
1. **Phase 1**: Implement new system alongside existing
2. **Phase 2**: Add deprecation warnings to old API
3. **Phase 3**: Migrate examples and documentation  
4. **Phase 4**: Remove old API in major version bump

This approach ensures existing code continues working while providing a path to better attachment management. 