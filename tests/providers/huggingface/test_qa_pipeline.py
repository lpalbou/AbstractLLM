"""Tests for the Question Answering pipeline."""

import pytest
from unittest.mock import Mock, patch
import torch

from abstractllm.providers.huggingface.qa_pipeline import QuestionAnsweringPipeline
from abstractllm.providers.huggingface.model_types import ModelConfig, ModelArchitecture
from abstractllm.exceptions import ModelLoadError, InvalidInputError
from abstractllm.media.text import TextInput

@pytest.fixture
def qa_pipeline():
    """Create a QA pipeline instance for testing."""
    return QuestionAnsweringPipeline()

@pytest.fixture
def mock_extractive_model():
    """Create a mock extractive QA model."""
    mock = Mock()
    mock.__class__.__name__ = "BertForQuestionAnswering"
    mock.config.model_type = "bert"
    mock.device = torch.device("cpu")
    return mock

@pytest.fixture
def mock_generative_model():
    """Create a mock generative QA model."""
    mock = Mock()
    mock.__class__.__name__ = "T5ForConditionalGeneration"
    mock.config.model_type = "t5"
    mock.device = torch.device("cpu")
    return mock

def test_init(qa_pipeline):
    """Test pipeline initialization."""
    assert qa_pipeline.model is None
    assert qa_pipeline.tokenizer is None
    assert qa_pipeline.model_config is None
    assert not qa_pipeline._is_loaded

def test_is_extractive_model(qa_pipeline, mock_extractive_model, mock_generative_model):
    """Test model type detection."""
    assert qa_pipeline._is_extractive_model(mock_extractive_model) is True
    assert qa_pipeline._is_extractive_model(mock_generative_model) is False

@patch("transformers.AutoModelForQuestionAnswering.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_load_extractive_model(mock_tokenizer, mock_model, qa_pipeline):
    """Test loading an extractive QA model."""
    config = ModelConfig(
        architecture=ModelArchitecture.QUESTION_ANSWERING,
        device_map="cpu",
        trust_remote_code=True
    )
    
    mock_model_instance = Mock(spec=["to", "__class__", "device"])
    mock_model_instance.__class__.__name__ = "BertForQuestionAnswering"
    mock_model_instance.config.model_type = "bert"
    mock_model_instance.device = torch.device("cpu")
    mock_model.return_value = mock_model_instance
    
    qa_pipeline.load("bert-qa-model", config)
    
    mock_model.assert_called_once()
    mock_tokenizer.assert_called_once()
    assert qa_pipeline.model is not None
    assert qa_pipeline.tokenizer is not None
    assert qa_pipeline._is_loaded

@patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_load_generative_model(mock_tokenizer, mock_model, qa_pipeline):
    """Test loading a generative QA model."""
    config = ModelConfig(
        architecture=ModelArchitecture.QUESTION_ANSWERING,
        device_map="cpu",
        trust_remote_code=True
    )
    
    mock_model_instance = Mock(spec=["to", "__class__", "device"])
    mock_model_instance.__class__.__name__ = "T5ForConditionalGeneration"
    mock_model_instance.config.model_type = "t5"
    mock_model_instance.device = torch.device("cpu")
    mock_model.return_value = mock_model_instance
    
    qa_pipeline.load("t5-qa-model", config)
    
    mock_model.assert_called_once()
    mock_tokenizer.assert_called_once()
    assert qa_pipeline.model is not None
    assert qa_pipeline.tokenizer is not None
    assert qa_pipeline._is_loaded

@patch("transformers.AutoModelForQuestionAnswering.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_process_extractive(mock_tokenizer, mock_model, qa_pipeline):
    """Test extractive answer generation."""
    config = ModelConfig(
        architecture=ModelArchitecture.QUESTION_ANSWERING,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Mock model and tokenizer
    mock_model_instance = Mock(spec=["to", "__class__", "device", "__call__"])
    mock_model_instance.__class__.__name__ = "BertForQuestionAnswering"
    mock_model_instance.config.model_type = "bert"
    mock_model_instance.device = torch.device("cpu")
    
    # Mock model outputs
    mock_outputs = Mock()
    mock_outputs.start_logits = torch.tensor([[0, 0, 1, 0, 0]])  # Position 2
    mock_outputs.end_logits = torch.tensor([[0, 0, 0, 1, 0]])    # Position 3
    mock_model_instance.__call__.return_value = mock_outputs
    mock_model.return_value = mock_model_instance
    
    # Mock tokenizer
    mock_tokenizer.return_value = Mock(
        __call__=lambda *args, **kwargs: {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])},
        convert_ids_to_tokens=lambda x: ["[CLS]", "what", "Paris", "is", "[SEP]"],
        convert_tokens_to_ids=lambda x: [3],  # ID for "Paris"
        decode=lambda x, **kwargs: "Paris"
    )
    
    qa_pipeline.load("bert-qa-model", config)
    
    # Create inputs
    question = TextInput("What is the capital of France?")
    context = TextInput("Paris is the capital of France.")
    
    result = qa_pipeline.process([question, context])
    
    assert isinstance(result, str)
    assert result == "Paris"

@patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_process_generative(mock_tokenizer, mock_model, qa_pipeline):
    """Test generative answer generation."""
    config = ModelConfig(
        architecture=ModelArchitecture.QUESTION_ANSWERING,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Mock model and tokenizer
    mock_model_instance = Mock(spec=["to", "__class__", "device", "generate"])
    mock_model_instance.__class__.__name__ = "T5ForConditionalGeneration"
    mock_model_instance.config.model_type = "t5"
    mock_model_instance.device = torch.device("cpu")
    mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3]])
    mock_model.return_value = mock_model_instance
    
    # Mock tokenizer
    mock_tokenizer.return_value = Mock(
        __call__=lambda *args, **kwargs: {"input_ids": torch.tensor([[1, 2, 3]])},
        batch_decode=lambda x, **kwargs: ["Paris is the capital of France"]
    )
    
    qa_pipeline.load("t5-qa-model", config)
    
    # Create inputs
    question = TextInput("What is the capital of France?")
    context = TextInput("Paris is the capital of France.")
    
    result = qa_pipeline.process([question, context])
    
    assert isinstance(result, str)
    assert result == "Paris is the capital of France"

def test_process_without_loading(qa_pipeline):
    """Test processing without loading model first."""
    question = TextInput("What is the capital of France?")
    context = TextInput("Paris is the capital of France.")
    
    with pytest.raises(RuntimeError, match="Model not loaded"):
        qa_pipeline.process([question, context])

def test_process_invalid_inputs(qa_pipeline):
    """Test processing with invalid inputs."""
    # Test with no inputs
    with pytest.raises(InvalidInputError):
        qa_pipeline.process([])
    
    # Test with only one input
    question = TextInput("What is the capital of France?")
    with pytest.raises(InvalidInputError):
        qa_pipeline.process([question])
    
    # Test with too many inputs
    context = TextInput("Paris is the capital of France.")
    extra = TextInput("Extra text")
    with pytest.raises(InvalidInputError):
        qa_pipeline.process([question, context, extra]) 