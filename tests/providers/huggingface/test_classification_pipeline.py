"""Tests for the Text Classification pipeline."""

import pytest
from unittest.mock import Mock, patch
import torch

from abstractllm.providers.huggingface.classification_pipeline import TextClassificationPipeline
from abstractllm.providers.huggingface.model_types import ModelConfig, ModelArchitecture
from abstractllm.exceptions import ModelLoadError, InvalidInputError
from abstractllm.media.text import TextInput

@pytest.fixture
def classification_pipeline():
    """Create a classification pipeline instance for testing."""
    return TextClassificationPipeline()

@pytest.fixture
def mock_single_label_model():
    """Create a mock single-label classification model."""
    mock = Mock()
    mock.config.problem_type = "single_label_classification"
    mock.config.id2label = {0: "negative", 1: "positive"}
    mock.device = torch.device("cpu")
    return mock

@pytest.fixture
def mock_multi_label_model():
    """Create a mock multi-label classification model."""
    mock = Mock()
    mock.config.problem_type = "multi_label_classification"
    mock.config.id2label = {0: "sports", 1: "politics", 2: "technology"}
    mock.device = torch.device("cpu")
    return mock

def test_init(classification_pipeline):
    """Test pipeline initialization."""
    assert classification_pipeline.model is None
    assert classification_pipeline.tokenizer is None
    assert classification_pipeline.model_config is None
    assert not classification_pipeline._is_loaded
    assert not classification_pipeline._is_multilabel
    assert classification_pipeline._label_names == []

@patch("transformers.AutoModelForSequenceClassification.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoConfig.from_pretrained")
def test_load_single_label_model(mock_config, mock_tokenizer, mock_model, classification_pipeline):
    """Test loading a single-label classification model."""
    config = ModelConfig(
        architecture=ModelArchitecture.TEXT_CLASSIFICATION,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Mock config
    mock_config_instance = Mock()
    mock_config_instance.problem_type = "single_label_classification"
    mock_config_instance.id2label = {0: "negative", 1: "positive"}
    mock_config.return_value = mock_config_instance
    
    # Mock model
    mock_model_instance = Mock()
    mock_model_instance.config = mock_config_instance
    mock_model_instance.device = torch.device("cpu")
    mock_model.return_value = mock_model_instance
    
    classification_pipeline.load("sentiment-model", config)
    
    mock_model.assert_called_once()
    mock_tokenizer.assert_called_once()
    assert classification_pipeline.model is not None
    assert classification_pipeline.tokenizer is not None
    assert classification_pipeline._is_loaded
    assert not classification_pipeline._is_multilabel
    assert classification_pipeline._label_names == ["negative", "positive"]

@patch("transformers.AutoModelForSequenceClassification.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoConfig.from_pretrained")
def test_load_multi_label_model(mock_config, mock_tokenizer, mock_model, classification_pipeline):
    """Test loading a multi-label classification model."""
    config = ModelConfig(
        architecture=ModelArchitecture.TEXT_CLASSIFICATION,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Mock config
    mock_config_instance = Mock()
    mock_config_instance.problem_type = "multi_label_classification"
    mock_config_instance.id2label = {0: "sports", 1: "politics", 2: "technology"}
    mock_config.return_value = mock_config_instance
    
    # Mock model
    mock_model_instance = Mock()
    mock_model_instance.config = mock_config_instance
    mock_model_instance.device = torch.device("cpu")
    mock_model.return_value = mock_model_instance
    
    classification_pipeline.load("topic-model", config)
    
    mock_model.assert_called_once()
    mock_tokenizer.assert_called_once()
    assert classification_pipeline.model is not None
    assert classification_pipeline.tokenizer is not None
    assert classification_pipeline._is_loaded
    assert classification_pipeline._is_multilabel
    assert classification_pipeline._label_names == ["sports", "politics", "technology"]

@patch("transformers.AutoModelForSequenceClassification.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoConfig.from_pretrained")
def test_process_single_label(mock_config, mock_tokenizer, mock_model, classification_pipeline):
    """Test single-label classification."""
    config = ModelConfig(
        architecture=ModelArchitecture.TEXT_CLASSIFICATION,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Mock config
    mock_config_instance = Mock()
    mock_config_instance.problem_type = "single_label_classification"
    mock_config_instance.id2label = {0: "negative", 1: "positive"}
    mock_config.return_value = mock_config_instance
    
    # Mock model outputs
    mock_outputs = Mock()
    mock_outputs.logits = torch.tensor([[0.1, 0.9]])  # Positive sentiment
    mock_model_instance = Mock()
    mock_model_instance.config = mock_config_instance
    mock_model_instance.device = torch.device("cpu")
    mock_model_instance.__call__ = Mock(return_value=mock_outputs)
    mock_model.return_value = mock_model_instance
    
    # Mock tokenizer
    mock_tokenizer.return_value = Mock(
        __call__=lambda *args, **kwargs: {"input_ids": torch.tensor([[1, 2, 3]])}
    )
    
    classification_pipeline.load("sentiment-model", config)
    
    # Create input
    text_input = TextInput("This is a great movie!")
    
    result = classification_pipeline.process([text_input])
    
    assert isinstance(result, dict)
    assert "labels" in result
    assert "scores" in result
    assert result["labels"] == ["positive"]
    assert len(result["scores"]) == 1
    assert result["scores"][0] > 0.5  # Should be confident in positive class

@patch("transformers.AutoModelForSequenceClassification.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoConfig.from_pretrained")
def test_process_multi_label(mock_config, mock_tokenizer, mock_model, classification_pipeline):
    """Test multi-label classification."""
    config = ModelConfig(
        architecture=ModelArchitecture.TEXT_CLASSIFICATION,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Mock config
    mock_config_instance = Mock()
    mock_config_instance.problem_type = "multi_label_classification"
    mock_config_instance.id2label = {0: "sports", 1: "politics", 2: "technology"}
    mock_config.return_value = mock_config_instance
    
    # Mock model outputs
    mock_outputs = Mock()
    mock_outputs.logits = torch.tensor([[2.0, -1.0, 1.5]])  # Sports and Technology
    mock_model_instance = Mock()
    mock_model_instance.config = mock_config_instance
    mock_model_instance.device = torch.device("cpu")
    mock_model_instance.__call__ = Mock(return_value=mock_outputs)
    mock_model.return_value = mock_model_instance
    
    # Mock tokenizer
    mock_tokenizer.return_value = Mock(
        __call__=lambda *args, **kwargs: {"input_ids": torch.tensor([[1, 2, 3]])}
    )
    
    classification_pipeline.load("topic-model", config)
    
    # Create input
    text_input = TextInput("AI is transforming sports analytics.")
    
    result = classification_pipeline.process([text_input])
    
    assert isinstance(result, dict)
    assert "labels" in result
    assert "scores" in result
    assert set(result["labels"]) == {"sports", "technology"}
    assert len(result["scores"]) == 2
    assert all(score > 0.5 for score in result["scores"])

def test_process_without_loading(classification_pipeline):
    """Test processing without loading model first."""
    text_input = TextInput("This is a test.")
    
    with pytest.raises(RuntimeError, match="Model not loaded"):
        classification_pipeline.process([text_input])

def test_process_invalid_inputs(classification_pipeline):
    """Test processing with invalid inputs."""
    # Test with no inputs
    with pytest.raises(InvalidInputError):
        classification_pipeline.process([])

def test_process_with_all_scores(classification_pipeline, mock_single_label_model):
    """Test processing with return_all_scores=True."""
    config = ModelConfig(
        architecture=ModelArchitecture.TEXT_CLASSIFICATION,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Set up mock model
    classification_pipeline.model = mock_single_label_model
    classification_pipeline._is_loaded = True
    classification_pipeline._label_names = ["negative", "positive"]
    
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.__call__ = Mock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})
    classification_pipeline.tokenizer = mock_tokenizer
    
    # Mock model outputs
    mock_outputs = Mock()
    mock_outputs.logits = torch.tensor([[0.1, 0.9]])
    mock_single_label_model.__call__ = Mock(return_value=mock_outputs)
    
    # Create input
    text_input = TextInput("This is a test.")
    
    result = classification_pipeline.process([text_input], return_all_scores=True)
    
    assert isinstance(result, dict)
    assert "labels" in result
    assert "scores" in result
    assert "all_scores" in result
    assert isinstance(result["all_scores"], dict)
    assert set(result["all_scores"].keys()) == {"negative", "positive"} 