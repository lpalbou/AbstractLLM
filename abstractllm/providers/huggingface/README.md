# HuggingFace Provider

The HuggingFace provider implements AbstractLLM's interface using HuggingFace's transformers library. It supports a wide range of model architectures and tasks through a modular pipeline system.

## Architecture

The provider is built around three core components:

1. **Provider Interface** (`HuggingFaceProvider`):
   - Implements AbstractLLM's interface
   - Handles configuration and resource management
   - Routes requests to appropriate pipelines

2. **Pipeline Factory** (`PipelineFactory`):
   - Creates and configures task-specific pipelines
   - Detects model architectures and capabilities
   - Maps model types to implementations

3. **Model Pipelines**:
   - Task-specific implementations (text generation, QA, etc.)
   - Handle model loading and inference
   - Process inputs and outputs

## Supported Pipelines

Currently implemented:
- Text Generation (`TextGenerationPipeline`)
- Text-to-Text (`Text2TextPipeline`)
- Image-to-Text (`ImageToTextPipeline`)
- Question Answering (`QuestionAnsweringPipeline`)
- Text Classification (`TextClassificationPipeline`)
- Document QA (`DocumentQuestionAnsweringPipeline`)

Planned for future implementation:
- Token Classification
- Visual Question Answering
- Text-to-Speech
- Speech-to-Text

## Model Architectures

Each pipeline supports specific model architectures:

| Pipeline Type | Architectures | Examples |
|--------------|---------------|-----------|
| Text Generation | CAUSAL_LM | GPT-2, Llama |
| Text-to-Text | SEQ2SEQ_LM | T5, BART |
| Image-to-Text | VISION2SEQ | BLIP, LLaVA |
| Question Answering | QUESTION_ANSWERING | BERT, RoBERTa |
| Text Classification | TEXT_CLASSIFICATION | DistilBERT |
| Document QA | DOCUMENT_QA | LayoutLM, Donut |

## Usage

Basic usage:
```python
from abstractllm import create_llm

# Create provider instance
llm = create_llm("huggingface", model="microsoft/phi-2")

# Generate text
response = llm.generate("Tell me a story about a robot.")

# Process files
response = llm.generate(
    "What's in this image?",
    files=["image.jpg"]
)
```

Advanced configuration:
```python
llm = create_llm("huggingface", {
    "model": "microsoft/phi-2",
    "temperature": 0.7,
    "max_tokens": 2048,
    "device_map": "auto",
    "quantization": "4bit",  # Enable 4-bit quantization
    "use_flash_attention": True  # Enable Flash Attention 2
})
```

## Media Handling

The provider supports various media types through AbstractLLM's media handling system:

- **Images**: Supported by vision-capable models (BLIP, LLaVA)
- **Documents**: PDF and DOCX processing with layout analysis
- **Text**: Plain text, markdown, and structured formats
- **Tabular**: CSV and TSV data

## Model Loading

Models are loaded with optimizations based on:
- Available hardware (CUDA, MPS, CPU)
- Memory constraints
- Model architecture
- Quantization settings

## Error Handling

The provider implements comprehensive error handling:
- Model loading errors
- Input validation
- Resource management
- Device compatibility

## Contributing

To add a new pipeline:

1. Create pipeline class inheriting from `BasePipeline`
2. Add architecture to `ModelArchitecture` enum
3. Register in `PipelineFactory._PIPELINE_MAPPING`
4. Add tests and documentation

Example:
```python
from .model_types import BasePipeline, ModelArchitecture

class NewPipeline(BasePipeline):
    """Implementation of a new pipeline type."""
    
    def load(self, model_name: str, config: ModelConfig) -> None:
        # Implementation
        pass
        
    def process(self, inputs: List[MediaInput], **kwargs) -> Any:
        # Implementation
        pass

# Register in factory
_PIPELINE_MAPPING = {
    "new-type": (NewPipeline, ModelArchitecture.NEW_TYPE)
}
```

## Future Improvements

Planned enhancements:
- More model architectures
- Better memory optimization
- Enhanced quantization support
- Improved streaming capabilities
- Additional media types

## Text Classification Pipeline

The Text Classification pipeline supports various classification tasks with both single-label and multi-label capabilities. Here's a detailed guide on using each type:

### 1. Sentiment Analysis

Sentiment analysis classifies text based on emotional tone or sentiment.

```python
from abstractllm import create_llm
from abstractllm.media.text import TextInput

# Create LLM with sentiment model
llm = create_llm("huggingface", model="distilbert-base-uncased-finetuned-sst-2-english")

# Analyze sentiment
result = llm.generate("This movie was absolutely fantastic!")
# Returns: {"labels": ["positive"], "scores": [0.97]}

# Get all sentiment scores
result = llm.generate("The service was okay.", return_all_scores=True)
# Returns:
# {
#   "labels": ["neutral"],
#   "scores": [0.75],
#   "all_scores": {
#     "negative": 0.15,
#     "neutral": 0.75,
#     "positive": 0.10
#   }
# }
```

Recommended Models:
- `distilbert-base-uncased-finetuned-sst-2-english` (binary sentiment)
- `nlptown/bert-base-multilingual-uncased-sentiment` (5-class sentiment)
- `cardiffnlp/twitter-roberta-base-sentiment` (social media sentiment)

### 2. Topic Classification

Topic classification categorizes text into predefined topics or domains.

```python
from abstractllm import create_llm
from abstractllm.media.text import TextInput

# Create LLM with topic classifier
llm = create_llm("huggingface", model="facebook/bart-large-mnli")

# Classify topic
text = "SpaceX successfully launched its latest mission to the International Space Station."
result = llm.generate(text)
# Returns: {"labels": ["science"], "scores": [0.92]}

# Multi-topic classification
text = "The new AI technology is revolutionizing healthcare systems worldwide."
result = llm.generate(text, return_all_scores=True)
# Returns:
# {
#   "labels": ["technology", "healthcare"],
#   "scores": [0.88, 0.85],
#   "all_scores": {
#     "technology": 0.88,
#     "healthcare": 0.85,
#     "science": 0.65,
#     ...
#   }
# }
```

Recommended Models:
- `facebook/bart-large-mnli` (zero-shot classification)
- `classla/bert-base-croatian-cased-news` (news topics)
- `yiyanghkust/finbert-tone` (financial topics)

### 3. Intent Detection

Intent detection identifies the purpose or intention behind text input.

```python
from abstractllm import create_llm
from abstractllm.media.text import TextInput

# Create LLM with intent classifier
llm = create_llm("huggingface", model="joeddav/bart-large-mnli-yahoo-answers")

# Detect intent
text = "What's the weather like in Paris today?"
result = llm.generate(text)
# Returns: {"labels": ["weather_query"], "scores": [0.89]}

# Get all possible intents
text = "Can you book a table for two at Le Cheval Blanc?"
result = llm.generate(text, return_all_scores=True)
# Returns:
# {
#   "labels": ["reservation_request"],
#   "scores": [0.93],
#   "all_scores": {
#     "reservation_request": 0.93,
#     "information_query": 0.45,
#     "booking_confirmation": 0.22,
#     ...
#   }
# }
```

Recommended Models:
- `joeddav/bart-large-mnli-yahoo-answers` (general intent)
- `cross-encoder/nli-distilroberta-base` (natural language inference)
- `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (multilingual intent)

### 4. Multi-label Classification

Multi-label classification allows assigning multiple labels to a single input.

```python
from abstractllm import create_llm
from abstractllm.media.text import TextInput

# Create LLM with multi-label classifier
llm = create_llm("huggingface", model="joeddav/xlm-roberta-large-xnli")

# Multi-label classification
text = "The new electric car features advanced AI for autonomous driving."
result = llm.generate(text)
# Returns:
# {
#   "labels": ["automotive", "technology", "artificial_intelligence"],
#   "scores": [0.91, 0.88, 0.85]
# }

# Adjust threshold for label inclusion
result = llm.generate(text, threshold=0.7)  # Only include labels with confidence > 0.7
```

Recommended Models:
- `joeddav/xlm-roberta-large-xnli` (multilingual multi-label)
- `microsoft/deberta-v3-large` (high-performance multi-label)
- `bert-base-multilingual-uncased` (general multi-label)

### Advanced Features

1. **Confidence Thresholds**
   ```python
   # Adjust confidence threshold for multi-label classification
   result = llm.generate(text, threshold=0.8)  # Higher threshold for more confident predictions
   ```

2. **Detailed Scores**
   ```python
   # Get detailed scores for all possible labels
   result = llm.generate(text, return_all_scores=True)
   ```

3. **Model Quantization**
   ```python
   # Use 4-bit quantization for memory efficiency
   llm = create_llm("huggingface", 
                    model="microsoft/deberta-v3-large",
                    quantization="4bit")
   ```

4. **Device Control**
   ```python
   # Force CPU usage
   llm = create_llm("huggingface",
                    model="bert-base-uncased",
                    device_map="cpu")
   ```

### Testing

The classification pipeline includes comprehensive tests:

```python
# Test single-label classification
def test_sentiment():
    llm = create_llm("huggingface", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = llm.generate("Great product!")
    assert "positive" in result["labels"]

# Test multi-label classification
def test_topics():
    llm = create_llm("huggingface", model="facebook/bart-large-mnli")
    result = llm.generate("AI in healthcare", return_all_scores=True)
    assert len(result["labels"]) >= 2  # Should detect both AI and healthcare
```

Run tests with:
```bash
pytest tests/providers/huggingface/test_classification_pipeline.py
```

## Best Practices

1. **Model Selection**
   - Choose task-specific models when possible
   - Consider model size vs. performance tradeoffs
   - Check model card for specific usage requirements

2. **Input Processing**
   - Clean and preprocess text inputs
   - Consider text length limitations
   - Handle multilingual inputs appropriately

3. **Performance Optimization**
   - Use quantization for large models
   - Enable Flash Attention 2 when available
   - Consider batch processing for multiple inputs

4. **Error Handling**
   - Always check for invalid inputs
   - Handle model loading failures gracefully
   - Validate confidence scores before using results

## Pipeline Implementation Guide

To implement a new pipeline type:

1. **Update ModelArchitecture**
   ```python
   class ModelArchitecture(Enum):
       NEW_TYPE = auto()  # Add new type
   ```

2. **Create Pipeline Class**
   ```python
   class NewPipeline(BasePipeline):
       def load(self, model_name: str, config: ModelConfig) -> None:
           # Load model and components
           pass

       def process(self, inputs: List[MediaInput], **kwargs) -> Any:
           # Process inputs and generate output
           pass

       @property
       def capabilities(self) -> ModelCapabilities:
           # Define capabilities
           pass
   ```

3. **Update Factory**
   ```python
   _PIPELINE_MAPPING = {
       "new-type": (NewPipeline, ModelArchitecture.NEW_TYPE)
   }
   ```

4. **Add Media Types** (if needed)
   - Create new media type class in `abstractllm/media/`
   - Update `MediaFactory` to handle new type
   - Implement provider-specific formatting

### Implementation Best Practices

1. **Resource Management**
   - Always implement proper cleanup
   - Use context managers where appropriate
   - Handle GPU memory carefully

2. **Error Handling**
   - Use specific exception types
   - Provide clear error messages
   - Clean up resources on failure

3. **Input Validation**
   - Check capabilities before processing
   - Validate all inputs early
   - Provide helpful error messages

4. **Model Loading**
   - Support different quantization options
   - Handle device mapping properly
   - Support model revision/variants

5. **Processing**
   - Handle streaming properly
   - Support system prompts where applicable
   - Maintain consistent output format

## Testing

Each pipeline should have:

1. **Unit Tests**
   ```python
   def test_pipeline_load():
       # Test model loading
       pass

   def test_pipeline_process():
       # Test processing
       pass

   def test_pipeline_cleanup():
       # Test resource cleanup
       pass
   ```

2. **Integration Tests**
   ```python
   def test_pipeline_with_provider():
       # Test full pipeline flow
       pass
   ```

3. **Quick Test Example**
   ```python
   from abstractllm import create_llm

   llm = create_llm("huggingface", model="model-name")
   result = llm.generate("prompt")
   ```

## Current Status

Currently implemented:
- Text Generation (CAUSAL_LM)
  - GPT-style models
  - GGUF models
- Image-to-Text (VISION2SEQ)
  - BLIP models
  - LLaVA models
- Text2Text Generation (SEQ2SEQ_LM)
  - T5 models (with language detection)
  - BART models (with optimizations)
  - mBART models (with language code handling)

### Text2Text Pipeline Features

1. **Language Detection and Handling**
   - Automatic source language detection
   - Model-specific language code mapping
   - Support for multilingual models

2. **Quality Metrics**
   - Translation: BLEU scores
   - Summarization: ROUGE scores
   - Semantic similarity metrics
   - Length ratio analysis

3. **Model-Specific Optimizations**
   - T5: Model parallelism and memory efficiency
   - BART: Optimized beam search
   - mBART: Language code handling
   - Resource cleanup and memory management

## Planned Pipeline Implementations

1. **Text Classification**
   - Sentiment analysis
   - Topic classification
   - Intent detection
   ```python
   from transformers import AutoModelForSequenceClassification
   ```

2. **Token Classification**
   - Named Entity Recognition (NER)
   - Part-of-Speech (POS) tagging
   - Chunking
   ```python
   from transformers import AutoModelForTokenClassification
   ```

3. **Question Answering**
   - Extractive QA
   - Generative QA
   ```python
   from transformers import AutoModelForQuestionAnswering
   ```

4. **Visual Question Answering**
   - Image-based QA
   - Scene understanding
   ```python
   from transformers import ViltForQuestionAnswering
   ```

5. **Document Question Answering**
   - PDF/Document understanding
   - Table question answering
   ```python
   from transformers import LayoutLMForQuestionAnswering
   ```

6. **Text-to-Speech**
   - Speech synthesis
   - Voice generation
   ```python
   from transformers import SpeechT5ForTextToSpeech
   ```

7. **Automatic Speech Recognition**
   - Speech-to-text
   - Audio transcription
   ```python
   from transformers import WhisperForConditionalGeneration
   ```

## Optional Dependencies

The Text2Text pipeline has optional dependencies for enhanced functionality:

```bash
# Language detection
pip install langdetect

# Translation metrics
pip install sacrebleu

# Summarization metrics
pip install rouge_score

# Semantic similarity
pip install sentence-transformers
```

## Adding New Pipeline Types

1. **Assess Requirements**
   - Required model types
   - Input/output formats
   - Special handling needs

2. **Design**
   - Pipeline interface
   - Resource requirements
   - Error handling strategy

3. **Implementation**
   - Create pipeline class
   - Update factory
   - Add tests
   - Update documentation

4. **Testing**
   - Unit tests
   - Integration tests
   - Resource cleanup tests
   - Error handling tests

## Common Patterns

1. **Model Loading**
   ```python
   def load(self, model_name: str, config: ModelConfig) -> None:
       try:
           # Load components
           self._load_components(model_name, config)
           self._is_loaded = True
       except Exception as e:
           self.cleanup()
           raise ModelLoadingError(f"Failed to load: {e}")
   ```

2. **Input Processing**
   ```python
   def process(self, inputs: List[MediaInput], **kwargs) -> Any:
       if not self._is_loaded:
           raise RuntimeError("Model not loaded")
       try:
           # Process inputs
           return self._generate(inputs, **kwargs)
       except Exception as e:
           raise GenerationError(f"Failed: {e}")
   ```

3. **Resource Cleanup**
   ```python
   def cleanup(self) -> None:
       if self._model is not None:
           try:
               self._cleanup_resources()
           except Exception as e:
               logger.warning(f"Cleanup error: {e}")
       super().cleanup()
   ```

### Visual Question Answering Pipeline

The Visual Question Answering (VQA) pipeline enables models to answer questions about images. It supports both open-ended questions and multiple-choice formats.

#### Capabilities

1. **General Visual QA**
   - Answer questions about image content
   - Describe objects and their attributes
   - Understand spatial relationships
   - Count objects in scenes

2. **Multiple Choice QA**
   - Select from predefined answer options
   - Provide confidence scores for choices
   - Handle binary yes/no questions

3. **Scene Understanding**
   - Interpret complex visual scenes
   - Understand object interactions
   - Recognize activities and actions
   - Analyze image attributes

#### Recommended Models

1. **ViLT Models** (Best for Multiple Choice)
   ```python
   # Best overall performance
   "dandelin/vilt-b32-finetuned-vqa"
   
   # Faster but slightly less accurate
   "dandelin/vilt-b16-finetuned-vqa"
   ```

2. **BLIP Models** (Best for Open-ended Questions)
   ```python
   # Best quality responses
   "Salesforce/blip-vqa-capfilt-large"
   
   # Good balance of speed and quality
   "Salesforce/blip-vqa-base"
   ```

#### Quick Start Example

```python
from abstractllm import create_llm
from abstractllm.media.image import ImageInput
from abstractllm.media.text import TextInput
from PIL import Image

# Create LLM with VQA model
llm = create_llm(
    "huggingface",
    model="dandelin/vilt-b32-finetuned-vqa",
    device_map="auto"  # Use GPU if available
)

# Load image and create inputs
image = Image.open("path/to/image.jpg")
image_input = ImageInput(image)
question = TextInput("What color is the car in the image?")

# Get answer
result = llm.generate([image_input, question])
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
```

#### Multiple Choice Example

```python
from abstractllm import create_llm
from abstractllm.media.image import ImageInput
from abstractllm.media.text import TextInput

# Create LLM
llm = create_llm("huggingface", model="dandelin/vilt-b32-finetuned-vqa")

# Prepare inputs
image_input = ImageInput("path/to/image.jpg")
question = TextInput("What animal is in the image?")

# Define possible answers
answer_candidates = ["cat", "dog", "bird", "fish"]

# Get answer with candidates
result = llm.generate(
    [image_input, question],
    answer_candidates=answer_candidates
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
```

#### Advanced Usage

1. **Custom Processing Parameters**
   ```python
   result = llm.generate(
       [image_input, question],
       max_answer_length=100,  # For longer answers
       num_beams=5,           # More thorough search
       return_logits=True     # Get raw model outputs
   )
   ```

2. **Batch Processing**
   ```python
   # Process multiple question-image pairs
   questions = [
       TextInput("What color is the car?"),
       TextInput("How many people are there?")
   ]
   
   for question in questions:
       result = llm.generate([image_input, question])
       print(f"Q: {question.content}")
       print(f"A: {result['answer']}\n")
   ```

3. **Error Handling**
   ```python
   try:
       result = llm.generate([image_input, question])
   except InvalidInputError as e:
       print(f"Input error: {e}")
   except GenerationError as e:
       print(f"Generation failed: {e}")
   ```

#### Performance Tips

1. **Model Selection**
   - Use ViLT for multiple-choice questions
   - Use BLIP for open-ended questions
   - Consider model size vs. speed tradeoffs

2. **Input Optimization**
   - Resize large images (1024x1024 is usually sufficient)
   - Keep questions clear and concise
   - Provide context when needed

3. **Resource Usage**
   - Enable quantization for large models
   ```python
   llm = create_llm(
       "huggingface",
       model="Salesforce/blip-vqa-capfilt-large",
       quantization="8bit"  # or "4bit" for more savings
   )
   ```

4. **Device Management**
   - Use GPU when available
   - Force CPU if memory limited
   ```python
   llm = create_llm(
       "huggingface",
       model="dandelin/vilt-b32-finetuned-vqa",
       device_map="cpu"  # Force CPU usage
   )
   ```

#### Common Issues and Solutions

1. **Out of Memory**
   - Use smaller models
   - Enable quantization
   - Reduce batch size
   - Process on CPU

2. **Slow Processing**
   - Use smaller models
   - Enable Flash Attention 2 (when supported)
   - Optimize image sizes
   - Use GPU acceleration

3. **Poor Answers**
   - Try different models
   - Rephrase questions
   - Provide clearer images
   - Use multiple-choice when possible

#### Testing

The VQA pipeline includes comprehensive tests:

```python
# Test basic functionality
def test_vqa():
    llm = create_llm("huggingface", model="dandelin/vilt-b32-finetuned-vqa")
    image = ImageInput("test_image.jpg")
    question = TextInput("What color is dominant?")
    result = llm.generate([image, question])
    assert isinstance(result["answer"], str)
    assert isinstance(result["confidence"], float)

# Test multiple choice
def test_multiple_choice():
    llm = create_llm("huggingface", model="dandelin/vilt-b32-finetuned-vqa")
    result = llm.generate(
        [image, question],
        answer_candidates=["red", "blue", "green"]
    )
    assert result["answer"] in ["red", "blue", "green"]
```

Run tests with:
```bash
pytest tests/providers/huggingface/test_vqa_pipeline.py
```

#### Future Improvements

1. **Planned Features**
   - Streaming answer generation
   - Multi-image support
   - Better confidence estimation
   - More model architectures

2. **Optimization Goals**
   - Faster processing
   - Lower memory usage
   - Better answer quality
   - More robust error handling
``` 