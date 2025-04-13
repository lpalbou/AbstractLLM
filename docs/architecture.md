# AbstractLLM Architecture

## Overview

AbstractLLM is designed with modularity and extensibility in mind, built around three core concepts:
1. Provider abstraction
2. Pipeline-based model handling
3. Unified media processing

## System Architecture

```mermaid
graph TD
    A[User Interface] --> B[Provider Layer]
    B --> C[OpenAI Provider]
    B --> D[Anthropic Provider]
    B --> E[HuggingFace Provider]
    B --> F[Ollama Provider]
    
    E --> G[Pipeline Factory]
    G --> H[Text Pipeline]
    G --> I[Vision Pipeline]
    G --> J[QA Pipeline]
    G --> K[Classification Pipeline]
    G --> L[Document QA Pipeline]
    G --> M[Visual QA Pipeline]
    G --> N[TTS Pipeline]
    
    O[Media System] --> P[Media Factory]
    P --> Q[Image Input]
    P --> R[Text Input]
    P --> S[Tabular Input]
    
    Q --> T[Provider Format]
    R --> T
    S --> T
    T --> U[Pipeline Input]
```

## Core Components

### 1. Provider Layer
The provider layer implements the AbstractLLM interface for different LLM providers:

```mermaid
graph TD
    A[AbstractLLM Interface] --> B[Provider Implementation]
    B --> C[Configuration]
    B --> D[Media Handling]
    B --> E[Pipeline Selection]
    B --> F[Resource Management]
```

### 2. Pipeline System
The pipeline system manages different model architectures:

```mermaid
graph LR
    A[Pipeline Factory] --> B[Model Detection]
    B --> C[Architecture Selection]
    C --> D[Pipeline Creation]
    D --> E[Resource Management]
```

Currently implemented pipelines:
- Text Generation (CAUSAL_LM)
- Text-to-Text (SEQ2SEQ_LM)
- Image-to-Text (VISION2SEQ)
- Question Answering (QUESTION_ANSWERING)
- Text Classification (TEXT_CLASSIFICATION)
- Document QA (DOCUMENT_QA)
- Visual QA (VISUAL_QA)
- Text-to-Speech (TEXT_TO_SPEECH)

Planned pipelines:
- Token Classification (TOKEN_CLASSIFICATION)
- Speech-to-Text (SPEECH_TO_TEXT)

### 3. Media System
The media system handles various input types:

```mermaid
graph TD
    A[Media Factory] --> B[Media Detection]
    B --> C[Format Conversion]
    C --> D[Provider Format]
    D --> E[Pipeline Input]
```

## Component Interactions

### Provider to Pipeline Flow
```mermaid
sequenceDiagram
    participant U as User
    participant P as Provider
    participant F as Factory
    participant M as Model Pipeline
    
    U->>P: generate(prompt, files)
    P->>F: create_pipeline()
    F->>M: load_model()
    M-->>P: ready
    P->>M: process(inputs)
    M-->>P: result
    P-->>U: response
```

### Media to Pipeline Flow
```mermaid
sequenceDiagram
    participant U as User
    participant M as Media Factory
    participant P as Pipeline
    participant F as Provider Format
    
    U->>M: process_files()
    M->>F: convert_format()
    F->>P: provide_input()
    P->>P: process()
    P-->>U: result
```

## Resource Management

### Model Loading
```mermaid
graph TD
    A[Request Model] --> B[Check Cache]
    B --> C{Cached?}
    C -->|Yes| D[Load from Cache]
    C -->|No| E[Load New Model]
    E --> F[Cache Model]
    F --> G[Return Model]
    D --> G
```

### Memory Management
```mermaid
graph TD
    A[Resource Request] --> B[Check Available]
    B --> C{Sufficient?}
    C -->|Yes| D[Allocate]
    C -->|No| E[Cleanup Old]
    E --> F[Try Again]
    F --> B
```

## Error Handling

```mermaid
graph TD
    A[Error Detection] --> B[Error Classification]
    B --> C[Error Handling]
    C --> D[Resource Cleanup]
    D --> E[User Feedback]
```

## Extension Points

1. **New Providers**:
   ```python
   class NewProvider(AbstractLLMInterface):
       def __init__(self, config: Optional[Dict[str, Any]] = None):
           super().__init__(config)
           
       def generate(self, prompt: str, **kwargs) -> str:
           # Implementation
           pass
   ```

2. **New Pipelines**:
   ```python
   class NewPipeline(BasePipeline):
       def load(self, model_name: str, config: ModelConfig) -> None:
           # Load model and components
           pass
           
       def process(self, inputs: List[MediaInput], **kwargs) -> Any:
           # Process inputs
           pass
   ```

3. **New Media Types**:
   ```python
   class NewMediaInput(MediaInput):
       def to_provider_format(self, provider: str) -> Any:
           # Convert to provider format
           pass
           
       @property
       def media_type(self) -> str:
           return "new_type"
   ```

## Cross-References
- [Data Flow Documentation](data_flow.md)
- [Media System Documentation](../abstractllm/media/README.md)
- [Pipeline Implementation Guide](../abstractllm/providers/huggingface/README.md) 