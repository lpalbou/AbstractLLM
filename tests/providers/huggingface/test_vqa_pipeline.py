"""Tests for the Visual Question Answering pipeline."""

import pytest
from unittest.mock import Mock, patch
import torch
from PIL import Image
import numpy as np

from abstractllm.providers.huggingface.vqa_pipeline import VisualQuestionAnsweringPipeline
from abstractllm.providers.huggingface.model_types import ModelConfig, ModelArchitecture
from abstractllm.exceptions import ModelLoadError, InvalidInputError
from abstractllm.media.text import TextInput
from abstractllm.media.image import ImageInput

@pytest.fixture
def vqa_pipeline():
    """Create a VQA pipeline instance for testing."""
    return VisualQuestionAnsweringPipeline()

@pytest.fixture
def mock_vilt_model():
    """Create a mock ViLT model."""
    mock = Mock()
    mock.config.max_position_embeddings = 512
    mock.config.num_choices = 2  # Supports multiple choice
    mock.device = torch.device("cpu")
    return mock

@pytest.fixture
def mock_blip_model():
    """Create a mock BLIP model."""
    mock = Mock()
    mock.config.max_position_embeddings = 512
    mock.device = torch.device("cpu")
    return mock

@pytest.fixture
def mock_image():
    """Create a mock image input."""
    # Create a small test image
    image = Image.new('RGB', (100, 100), color='red')
    return ImageInput(image)

def test_init(vqa_pipeline):
    """Test pipeline initialization."""
    assert vqa_pipeline.model is None
    assert vqa_pipeline.processor is None
    assert vqa_pipeline.model_config is None
    assert not vqa_pipeline._is_loaded
    assert vqa_pipeline._model_type == ""
    assert vqa_pipeline._max_question_length == 512
    assert not vqa_pipeline._supports_multiple_choice

@patch("transformers.ViltForQuestionAnswering.from_pretrained")
@patch("transformers.ViltProcessor.from_pretrained")
def test_load_vilt_model(mock_processor, mock_model, vqa_pipeline):
    """Test loading a ViLT model."""
    config = ModelConfig(
        architecture=ModelArchitecture.VISUAL_QA,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Mock model
    mock_model_instance = Mock()
    mock_model_instance.config.max_position_embeddings = 512
    mock_model_instance.config.num_choices = 2
    mock_model_instance.device = torch.device("cpu")
    mock_model.return_value = mock_model_instance
    
    vqa_pipeline.load("dandelin/vilt-b32-finetuned-vqa", config)
    
    mock_model.assert_called_once()
    mock_processor.assert_called_once()
    assert vqa_pipeline.model is not None
    assert vqa_pipeline.processor is not None
    assert vqa_pipeline._is_loaded
    assert vqa_pipeline._model_type == "vilt"
    assert vqa_pipeline._supports_multiple_choice

@patch("transformers.BlipForQuestionAnswering.from_pretrained")
@patch("transformers.BlipProcessor.from_pretrained")
def test_load_blip_model(mock_processor, mock_model, vqa_pipeline):
    """Test loading a BLIP model."""
    config = ModelConfig(
        architecture=ModelArchitecture.VISUAL_QA,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Mock model
    mock_model_instance = Mock()
    mock_model_instance.config.max_position_embeddings = 512
    mock_model_instance.device = torch.device("cpu")
    mock_model.return_value = mock_model_instance
    
    vqa_pipeline.load("Salesforce/blip-vqa-base", config)
    
    mock_model.assert_called_once()
    mock_processor.assert_called_once()
    assert vqa_pipeline.model is not None
    assert vqa_pipeline.processor is not None
    assert vqa_pipeline._is_loaded
    assert vqa_pipeline._model_type == "blip"
    assert not vqa_pipeline._supports_multiple_choice

@patch("transformers.ViltForQuestionAnswering.from_pretrained")
@patch("transformers.ViltProcessor.from_pretrained")
def test_process_vilt_multiple_choice(mock_processor, mock_model, vqa_pipeline, mock_image):
    """Test multiple choice VQA with ViLT."""
    config = ModelConfig(
        architecture=ModelArchitecture.VISUAL_QA,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Mock model outputs
    mock_outputs = Mock()
    mock_outputs.logits = torch.tensor([[0.1, 0.9]])  # Second answer more likely
    mock_model_instance = Mock()
    mock_model_instance.config.max_position_embeddings = 512
    mock_model_instance.config.num_choices = 2
    mock_model_instance.device = torch.device("cpu")
    mock_model_instance.__call__ = Mock(return_value=mock_outputs)
    mock_model.return_value = mock_model_instance
    
    # Mock processor
    mock_processor.return_value = Mock(
        __call__=lambda *args, **kwargs: {"input_ids": torch.tensor([[1, 2, 3]])}
    )
    
    vqa_pipeline.load("dandelin/vilt-b32-finetuned-vqa", config)
    
    # Create inputs
    question = TextInput("What color is the object?")
    answer_candidates = ["blue", "red"]
    
    result = vqa_pipeline.process(
        [mock_image, question],
        answer_candidates=answer_candidates
    )
    
    assert isinstance(result, dict)
    assert "answer" in result
    assert "confidence" in result
    assert result["answer"] == "red"  # Second answer had higher probability
    assert result["confidence"] > 0.8  # High confidence

@patch("transformers.BlipForQuestionAnswering.from_pretrained")
@patch("transformers.BlipProcessor.from_pretrained")
def test_process_blip_open_ended(mock_processor, mock_model, vqa_pipeline, mock_image):
    """Test open-ended VQA with BLIP."""
    config = ModelConfig(
        architecture=ModelArchitecture.VISUAL_QA,
        device_map="cpu",
        trust_remote_code=True
    )
    
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
        decode=lambda *args, **kwargs: "The object is red"
    )
    
    vqa_pipeline.load("Salesforce/blip-vqa-base", config)
    
    # Create inputs
    question = TextInput("What color is the object?")
    
    result = vqa_pipeline.process([mock_image, question])
    
    assert isinstance(result, dict)
    assert "answer" in result
    assert "confidence" in result
    assert result["answer"] == "The object is red"
    assert result["confidence"] == 1.0  # BLIP always returns 1.0

def test_process_without_loading(vqa_pipeline, mock_image):
    """Test processing without loading model first."""
    question = TextInput("What color is the object?")
    
    with pytest.raises(RuntimeError, match="Model not loaded"):
        vqa_pipeline.process([mock_image, question])

def test_process_invalid_inputs(vqa_pipeline):
    """Test processing with invalid inputs."""
    # Test with no inputs
    with pytest.raises(InvalidInputError):
        vqa_pipeline.process([])
    
    # Test with only image
    with pytest.raises(InvalidInputError):
        vqa_pipeline.process([mock_image])
    
    # Test with only question
    question = TextInput("What color is the object?")
    with pytest.raises(InvalidInputError):
        vqa_pipeline.process([question])
    
    # Test with multiple images
    with pytest.raises(InvalidInputError):
        vqa_pipeline.process([mock_image, mock_image, question])

def test_process_with_logits(vqa_pipeline, mock_image):
    """Test processing with return_logits=True."""
    config = ModelConfig(
        architecture=ModelArchitecture.VISUAL_QA,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Set up mock model
    mock_outputs = Mock()
    mock_outputs.logits = torch.tensor([[0.1, 0.9]])
    mock_model = Mock()
    mock_model.config.max_position_embeddings = 512
    mock_model.config.num_choices = 2
    mock_model.device = torch.device("cpu")
    mock_model.__call__ = Mock(return_value=mock_outputs)
    
    vqa_pipeline.model = mock_model
    vqa_pipeline._is_loaded = True
    vqa_pipeline._model_type = "vilt"
    
    # Mock processor
    mock_processor = Mock()
    mock_processor.__call__ = Mock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})
    vqa_pipeline.processor = mock_processor
    
    # Create inputs
    question = TextInput("What color is the object?")
    
    result = vqa_pipeline.process([mock_image, question], return_logits=True)
    
    assert isinstance(result, dict)
    assert "answer" in result
    assert "confidence" in result
    assert "logits" in result
    assert isinstance(result["logits"], np.ndarray) 