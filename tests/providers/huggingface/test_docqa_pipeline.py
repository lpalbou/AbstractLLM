"""Tests for the Document Question Answering pipeline."""

import pytest
from unittest.mock import Mock, patch
import torch
from PIL import Image
import numpy as np

from abstractllm.providers.huggingface.docqa_pipeline import DocumentQuestionAnsweringPipeline
from abstractllm.providers.huggingface.model_types import ModelConfig, ModelArchitecture
from abstractllm.exceptions import ModelLoadError, InvalidInputError
from abstractllm.media.text import TextInput
from abstractllm.media.image import ImageInput

@pytest.fixture
def docqa_pipeline():
    """Create a Document QA pipeline instance for testing."""
    return DocumentQuestionAnsweringPipeline()

@pytest.fixture
def mock_layoutlm_model():
    """Create a mock LayoutLM model."""
    mock = Mock()
    mock.config.max_position_embeddings = 512
    mock.config.has_table_encoder = True
    mock.device = torch.device("cpu")
    return mock

@pytest.fixture
def mock_donut_model():
    """Create a mock Donut model."""
    mock = Mock()
    mock.config.max_position_embeddings = 512
    mock.device = torch.device("cpu")
    return mock

@pytest.fixture
def mock_document():
    """Create a mock document input."""
    # Create a small test image with text
    image = Image.new('RGB', (200, 100), color='white')
    return ImageInput(image)

@pytest.fixture
def mock_ocr_results():
    """Create mock OCR results."""
    return {
        "text": ["Hello", "world", "!"],
        "left": [10, 50, 90],
        "top": [10, 10, 10],
        "width": [30, 30, 10],
        "height": [20, 20, 20]
    }

def test_init(docqa_pipeline):
    """Test pipeline initialization."""
    assert docqa_pipeline.model is None
    assert docqa_pipeline.processor is None
    assert docqa_pipeline.tokenizer is None
    assert docqa_pipeline.feature_extractor is None
    assert docqa_pipeline.model_config is None
    assert not docqa_pipeline._is_loaded
    assert docqa_pipeline._model_type == ""
    assert docqa_pipeline._max_seq_length == 512
    assert not docqa_pipeline._supports_tables

@patch("transformers.LayoutLMv3ForQuestionAnswering.from_pretrained")
@patch("transformers.LayoutLMv3TokenizerFast.from_pretrained")
def test_load_layoutlm_model(mock_tokenizer, mock_model, docqa_pipeline):
    """Test loading a LayoutLM model."""
    config = ModelConfig(
        architecture=ModelArchitecture.DOCUMENT_QA,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Mock model
    mock_model_instance = Mock()
    mock_model_instance.config.max_position_embeddings = 512
    mock_model_instance.config.has_table_encoder = True
    mock_model_instance.device = torch.device("cpu")
    mock_model.return_value = mock_model_instance
    
    docqa_pipeline.load("microsoft/layoutlmv3-base", config)
    
    mock_model.assert_called_once()
    mock_tokenizer.assert_called_once()
    assert docqa_pipeline.model is not None
    assert docqa_pipeline.processor is not None
    assert docqa_pipeline._is_loaded
    assert docqa_pipeline._model_type == "layoutlm"
    assert docqa_pipeline._supports_tables

@patch("transformers.AutoModelForDocumentQuestionAnswering.from_pretrained")
@patch("transformers.AutoProcessor.from_pretrained")
def test_load_donut_model(mock_processor, mock_model, docqa_pipeline):
    """Test loading a Donut model."""
    config = ModelConfig(
        architecture=ModelArchitecture.DOCUMENT_QA,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Mock model
    mock_model_instance = Mock()
    mock_model_instance.config.max_position_embeddings = 512
    mock_model_instance.device = torch.device("cpu")
    mock_model.return_value = mock_model_instance
    
    docqa_pipeline.load("naver-clova-ix/donut-base", config)
    
    mock_model.assert_called_once()
    mock_processor.assert_called_once()
    assert docqa_pipeline.model is not None
    assert docqa_pipeline.processor is not None
    assert docqa_pipeline._is_loaded
    assert docqa_pipeline._model_type == "donut"
    assert not docqa_pipeline._supports_tables

@patch("transformers.LayoutLMv3ForQuestionAnswering.from_pretrained")
@patch("transformers.LayoutLMv3TokenizerFast.from_pretrained")
@patch("pytesseract.image_to_data")
def test_process_layoutlm(mock_ocr, mock_tokenizer, mock_model, docqa_pipeline, mock_document, mock_ocr_results):
    """Test document processing with LayoutLM."""
    config = ModelConfig(
        architecture=ModelArchitecture.DOCUMENT_QA,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Mock OCR
    mock_ocr.return_value = mock_ocr_results
    
    # Mock model outputs
    mock_outputs = Mock()
    mock_outputs.start_logits = torch.tensor([[0.1, 0.9, 0.1]])  # Second word most likely
    mock_outputs.end_logits = torch.tensor([[0.1, 0.1, 0.9]])   # Third word most likely
    mock_model_instance = Mock()
    mock_model_instance.config.max_position_embeddings = 512
    mock_model_instance.config.has_table_encoder = True
    mock_model_instance.device = torch.device("cpu")
    mock_model_instance.__call__ = Mock(return_value=mock_outputs)
    mock_model.return_value = mock_model_instance
    
    # Mock tokenizer
    mock_tokenizer.return_value = Mock(
        __call__=lambda *args, **kwargs: {"input_ids": torch.tensor([[1, 2, 3]])},
        decode=lambda *args, **kwargs: "world!"
    )
    
    docqa_pipeline.load("microsoft/layoutlmv3-base", config)
    
    # Create inputs
    question = TextInput("What is written in the document?")
    
    result = docqa_pipeline.process([mock_document, question])
    
    assert isinstance(result, dict)
    assert "answer" in result
    assert "confidence" in result
    assert "context" in result
    assert "boxes" in result
    assert result["answer"] == "world!"
    assert result["confidence"] > 0.0
    assert len(result["boxes"]) > 0

@patch("transformers.AutoModelForDocumentQuestionAnswering.from_pretrained")
@patch("transformers.AutoProcessor.from_pretrained")
@patch("pytesseract.image_to_data")
def test_process_donut(mock_ocr, mock_processor, mock_model, docqa_pipeline, mock_document, mock_ocr_results):
    """Test document processing with Donut."""
    config = ModelConfig(
        architecture=ModelArchitecture.DOCUMENT_QA,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Mock OCR
    mock_ocr.return_value = mock_ocr_results
    
    # Mock model outputs
    mock_outputs = torch.tensor([[1, 2, 3]])  # Generated token IDs
    mock_model_instance = Mock()
    mock_model_instance.config.max_position_embeddings = 512
    mock_model_instance.device = torch.device("cpu")
    mock_model_instance.generate = Mock(return_value=mock_outputs)
    mock_model.return_value = mock_model_instance
    
    # Mock processor
    mock_processor.return_value = Mock(
        __call__=lambda *args, **kwargs: {"input_ids": torch.tensor([[1, 2, 3]])},
        decode=lambda *args, **kwargs: "Hello world!"
    )
    
    docqa_pipeline.load("naver-clova-ix/donut-base", config)
    
    # Create inputs
    question = TextInput("What is written in the document?")
    
    result = docqa_pipeline.process([mock_document, question])
    
    assert isinstance(result, dict)
    assert "answer" in result
    assert "confidence" in result
    assert "context" in result
    assert "boxes" in result
    assert result["answer"] == "Hello world!"
    assert result["confidence"] == 1.0  # Donut always returns 1.0
    assert len(result["boxes"]) == 0  # Donut doesn't provide boxes

def test_process_without_loading(docqa_pipeline, mock_document):
    """Test processing without loading model first."""
    question = TextInput("What is written in the document?")
    
    with pytest.raises(RuntimeError, match="Model not loaded"):
        docqa_pipeline.process([mock_document, question])

def test_process_invalid_inputs(docqa_pipeline):
    """Test processing with invalid inputs."""
    # Test with no inputs
    with pytest.raises(InvalidInputError):
        docqa_pipeline.process([])
    
    # Test with only document
    with pytest.raises(InvalidInputError):
        docqa_pipeline.process([mock_document])
    
    # Test with only question
    question = TextInput("What is written in the document?")
    with pytest.raises(InvalidInputError):
        docqa_pipeline.process([question])
    
    # Test with multiple documents
    with pytest.raises(InvalidInputError):
        docqa_pipeline.process([mock_document, mock_document, question])

def test_process_with_logits(docqa_pipeline, mock_document, mock_ocr_results):
    """Test processing with return_logits=True."""
    config = ModelConfig(
        architecture=ModelArchitecture.DOCUMENT_QA,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Set up mock model
    mock_outputs = Mock()
    mock_outputs.start_logits = torch.tensor([[0.1, 0.9, 0.1]])
    mock_outputs.end_logits = torch.tensor([[0.1, 0.1, 0.9]])
    mock_model = Mock()
    mock_model.config.max_position_embeddings = 512
    mock_model.config.has_table_encoder = True
    mock_model.device = torch.device("cpu")
    mock_model.__call__ = Mock(return_value=mock_outputs)
    
    docqa_pipeline.model = mock_model
    docqa_pipeline._is_loaded = True
    docqa_pipeline._model_type = "layoutlm"
    
    # Mock processor and OCR
    mock_processor = Mock()
    mock_processor.__call__ = Mock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})
    docqa_pipeline.processor = mock_processor
    
    # Create inputs
    question = TextInput("What is written in the document?")
    
    with patch("pytesseract.image_to_data", return_value=mock_ocr_results):
        result = docqa_pipeline.process([mock_document, question], return_logits=True)
    
    assert isinstance(result, dict)
    assert "answer" in result
    assert "confidence" in result
    assert "context" in result
    assert "boxes" in result
    assert "logits" in result
    assert "start_logits" in result["logits"]
    assert "end_logits" in result["logits"]
    assert isinstance(result["logits"]["start_logits"], np.ndarray)
    assert isinstance(result["logits"]["end_logits"], np.ndarray) 