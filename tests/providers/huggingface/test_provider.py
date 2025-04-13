"""Tests for the HuggingFace provider implementation."""

import pytest
from unittest.mock import Mock, patch
import torch
from pathlib import Path

from abstractllm.providers.huggingface.provider import HuggingFaceProvider
from abstractllm.providers.huggingface.model_types import (
    ModelArchitecture,
    ModelCapability,
    ModelCapabilities
)
from abstractllm.providers.huggingface.config import ModelConfig, DeviceType
from abstractllm.exceptions import (
    ModelLoadingError,
    UnsupportedFeatureError,
    ResourceError,
    GenerationError
)
from abstractllm.media.text import TextInput
from abstractllm.media.image import ImageInput
from abstractllm.enums import ModelParameter

@pytest.fixture
def provider():
    """Create a HuggingFace provider instance for testing."""
    return HuggingFaceProvider()

@pytest.fixture
def mock_hf_api():
    """Create a mock HuggingFace API."""
    mock = Mock()
    mock.model_info.return_value = Mock(
        pipeline_tag="text-generation"
    )
    return mock

@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline."""
    mock = Mock()
    mock.capabilities = Mock(
        get_all_capabilities=lambda: {
            "text_generation": ModelCapability(
                name="text_generation",
                confidence=1.0,
                requires_finetuning=False
            )
        }
    )
    return mock

def test_init(provider):
    """Test provider initialization."""
    assert provider._pipeline is None
    assert provider._model_config is None
    assert provider._hf_api is not None
    
    # Check default config
    config = provider.config_manager.get_config()
    assert config[ModelParameter.MODEL] == "microsoft/phi-2"
    assert config[ModelParameter.TEMPERATURE] == 0.7
    assert config[ModelParameter.MAX_TOKENS] == 2048
    assert config["trust_remote_code"] is True
    assert config["use_flash_attention"] is True

def test_create_model_config(provider):
    """Test model configuration creation."""
    config = provider._create_model_config()
    
    assert isinstance(config, ModelConfig)
    assert config.architecture == ModelArchitecture.DECODER_ONLY
    assert config.name == "microsoft/phi-2"
    
    # Check base config
    assert config.base["trust_remote_code"] is True
    assert config.base["device_map"] == "auto"
    
    # Check generation config
    assert config.generation["max_new_tokens"] == 2048
    assert config.generation["temperature"] == 0.7
    assert config.generation["do_sample"] is True

@patch("platform.system")
def test_create_model_config_macos(mock_system, provider):
    """Test model configuration creation on macOS."""
    mock_system.return_value = "Darwin"
    config = provider._create_model_config()
    
    assert config.base["use_flash_attention"] is False

@patch("huggingface_hub.HfApi")
def test_detect_model_architecture(mock_hf_api, provider):
    """Test model architecture detection."""
    # Test with HF API
    mock_hf_api.model_info.return_value = Mock(pipeline_tag="text-generation")
    arch = provider._detect_model_architecture("gpt2")
    assert arch == ModelArchitecture.DECODER_ONLY
    
    # Test name-based detection
    mock_hf_api.model_info.side_effect = Exception()
    
    assert provider._detect_model_architecture("llama") == ModelArchitecture.DECODER_ONLY
    assert provider._detect_model_architecture("t5") == ModelArchitecture.ENCODER_DECODER
    assert provider._detect_model_architecture("bert") == ModelArchitecture.ENCODER_ONLY
    assert provider._detect_model_architecture("clip") == ModelArchitecture.VISION_ENCODER
    assert provider._detect_model_architecture("whisper") == ModelArchitecture.SPEECH

def test_create_pipeline(provider):
    """Test pipeline creation."""
    config = ModelConfig(architecture=ModelArchitecture.DECODER_ONLY)
    pipeline = provider._create_pipeline(config)
    assert pipeline.__class__.__name__ == "DecoderOnlyPipeline"
    
    # Test unsupported architecture
    with pytest.raises(UnsupportedFeatureError):
        provider._create_pipeline(Mock(architecture="unsupported"))

@patch("torch.cuda.is_available")
@patch("torch.cuda.get_device_properties")
@patch("psutil.virtual_memory")
def test_check_system_requirements(mock_memory, mock_gpu, mock_cuda, provider):
    """Test system requirements checking."""
    mock_cuda.return_value = True
    mock_gpu.return_value = Mock(total_memory=8 * 1024**3)  # 8GB
    mock_memory.return_value = Mock(available=16 * 1024**3)  # 16GB
    
    # Test valid config
    config = ModelConfig(
        architecture=ModelArchitecture.DECODER_ONLY,
        base=dict(
            device_type=DeviceType.CUDA,
            max_memory={
                "cuda:0": "4GiB",
                "cpu": "8GiB"
            }
        )
    )
    provider._check_system_requirements(config)
    
    # Test insufficient GPU memory
    config.base.max_memory["cuda:0"] = "16GiB"
    with pytest.raises(ResourceError) as exc_info:
        provider._check_system_requirements(config)
    assert "Insufficient GPU memory" in str(exc_info.value)
    
    # Test insufficient CPU memory
    config.base.max_memory = {"cpu": "32GiB"}
    with pytest.raises(ResourceError) as exc_info:
        provider._check_system_requirements(config)
    assert "Insufficient CPU memory" in str(exc_info.value)

def test_parse_memory_string(provider):
    """Test memory string parsing."""
    assert provider._parse_memory_string("1024") == 1024
    assert provider._parse_memory_string("1KB") == 1024
    assert provider._parse_memory_string("1MB") == 1024**2
    assert provider._parse_memory_string("1GiB") == 1024**3
    assert provider._parse_memory_string("1.5GB") == int(1.5 * 1024**3)
    
    with pytest.raises(ValueError):
        provider._parse_memory_string("invalid")

def test_model_recommendations(provider):
    """Test model recommendation system."""
    # Test getting recommendations
    recs = provider.get_model_recommendations("text-generation")
    assert len(recs) == 3
    assert all(isinstance(r, dict) for r in recs)
    assert all("model" in r and "description" in r for r in recs)
    
    # Test updating recommendations
    new_recs = [("new-model", "New description")]
    provider.update_model_recommendations("test", new_recs)
    assert provider.get_model_recommendations("test") == [
        {"model": "new-model", "description": "New description"}
    ]
    
    # Test non-existent task
    assert provider.get_model_recommendations("nonexistent") == []

@patch("abstractllm.providers.huggingface.provider.HuggingFaceProvider._create_pipeline")
def test_load_model(mock_create_pipeline, provider):
    """Test model loading."""
    mock_pipeline = Mock()
    mock_create_pipeline.return_value = mock_pipeline
    
    provider.load_model()
    
    assert provider._model_config is not None
    assert provider._pipeline is mock_pipeline
    mock_pipeline.load.assert_called_once()
    
    # Test error handling
    mock_pipeline.load.side_effect = Exception("Load error")
    with pytest.raises(ModelLoadingError):
        provider.load_model()
    assert provider._pipeline is None

def test_generate(provider):
    """Test text generation."""
    mock_pipeline = Mock()
    mock_pipeline.process.return_value = "Generated text"
    provider._pipeline = mock_pipeline
    
    # Test basic generation
    result = provider.generate("Hello")
    assert result == "Generated text"
    
    # Test with system prompt
    result = provider.generate("Hello", system_prompt="Be helpful")
    assert result == "Generated text"
    mock_pipeline.process.assert_called()
    
    # Test with files
    result = provider.generate("Describe", files=["test.jpg"])
    assert result == "Generated text"
    
    # Test error handling
    mock_pipeline.process.side_effect = Exception("Generation error")
    with pytest.raises(GenerationError):
        provider.generate("Hello")

@pytest.mark.asyncio
async def test_generate_async(provider):
    """Test asynchronous generation."""
    mock_pipeline = Mock()
    mock_pipeline.process.return_value = "Generated text"
    provider._pipeline = mock_pipeline
    
    result = await provider.generate_async("Hello")
    assert result == "Generated text"

def test_get_capabilities(provider):
    """Test capability retrieval."""
    mock_pipeline = Mock()
    mock_pipeline.capabilities.get_all_capabilities.return_value = {
        "test": ModelCapability(
            name="test",
            confidence=1.0,
            requires_finetuning=False
        )
    }
    provider._pipeline = mock_pipeline
    
    caps = provider.get_capabilities()
    assert "test" in caps
    assert isinstance(caps["test"], ModelCapability)

def test_cleanup(provider):
    """Test resource cleanup."""
    mock_pipeline = Mock()
    provider._pipeline = mock_pipeline
    
    provider.cleanup()
    mock_pipeline.cleanup.assert_called_once()
    assert provider._pipeline is None 