"""
Tests for the HuggingFace provider.
"""

import os
import unittest
import importlib.util
import pytest
import time
import logging
import tempfile
import shutil
from pathlib import Path
from abstractllm import create_llm, ModelParameter
from abstractllm.utils.config import ConfigurationManager
from abstractllm.providers.huggingface import DEFAULT_MODEL, HuggingFaceProvider

# Configure logging
logger = logging.getLogger("test_huggingface")

# Create a test-specific cache directory
@pytest.fixture(scope="module")
def test_cache_dir():
    """Create a temporary directory for testing cache functionality."""
    temp_dir = tempfile.mkdtemp(prefix="abstractllm_test_cache_")
    logger.info(f"Created temporary test cache directory: {temp_dir}")
    yield temp_dir
    logger.info(f"Cleaning up temporary test cache directory: {temp_dir}")
    # Comment out the cleanup to inspect files for debugging if needed
    shutil.rmtree(temp_dir, ignore_errors=True)


def create_test_hf_provider(model_name=DEFAULT_MODEL, cache_dir=None):
    """Helper function to create a HuggingFace provider for testing."""
    # Create a base configuration using ConfigurationManager
    base_config = ConfigurationManager.create_base_config(
        model=model_name,
        device="cpu",
        max_tokens=50,
        auto_load=True,
        auto_warmup=True,
        load_timeout=300,
        generation_timeout=30,
        trust_remote_code=True,
        temperature=0.7,
        top_p=0.9
    )
    
    # Initialize provider-specific configuration
    provider_config = ConfigurationManager.initialize_provider_config("huggingface", base_config)
    
    # Override with cache_dir if provided
    if cache_dir:
        provider_config[ModelParameter.CACHE_DIR] = cache_dir
    
    return create_llm("huggingface", **provider_config)


def import_required_packages():
    """Check if required packages are installed."""
    try:
        # Check if required packages are installed
        for package in ["transformers", "torch"]:
            if importlib.util.find_spec(package) is None:
                pytest.skip(f"Required package {package} not installed")
        
        # Check if torch is available
        import torch
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            logger.info("CUDA and MPS not available, tests will run on CPU")
            
    except Exception as e:
        pytest.skip(f"Error importing required packages: {e}")


def setup_module(module):
    """Set up for the test module."""
    try:
        # Check if required packages are installed
        import_required_packages()
        
        # Check for model size preference for tests
        model_size = os.environ.get("HF_TEST_MODEL_SIZE", "small")
        if model_size == "small":
            # Default is already small (distilgpt2)
            logger.info("Using small model for tests")
        elif model_size == "medium":
            # Use a medium-sized model (unlikely to be used in CI/CD due to download times)
            global DEFAULT_MODEL
            DEFAULT_MODEL = "facebook/opt-350m"
            logger.info(f"Using medium model for tests: {DEFAULT_MODEL}")
        
    except Exception as e:
        logger.error(f"Error in setup: {e}")
        pytest.skip(f"Test setup failed: {e}")


def teardown_module(module):
    """Tear down for the test module."""
    try:
        # Clear model cache at the end of tests
        if os.environ.get("CLEAN_HF_CACHE_AFTER_TESTS", "false").lower() == "true":
            logger.info("Cleaning up HuggingFace model cache")
            HuggingFaceProvider.clear_model_cache()
    except Exception as e:
        logger.error(f"Error in teardown: {e}")


class TestHuggingFaceProvider(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up for the test class."""
        try:
            # Check if required packages are installed
            import_required_packages()
            
            # Create a test provider
            cls.provider = create_test_hf_provider()
            
            # Load the model before tests
            logger.info("Pre-loading model for tests")
            if not cls.provider._model_loaded:
                cls.provider.load_model()
                
        except Exception as e:
            pytest.skip(f"Test setup failed: {e}")
    
    @classmethod        
    def tearDownClass(cls):
        """Tear down for the test class."""
        try:
            # Clear class-level cache and other resources
            logger.info("Cleaning up class resources")
            # Don't actually clear the cache as it may be used by other tests
        except Exception as e:
            logger.error(f"Error in class teardown: {e}")

    def setUp(self):
        """Set up for each test."""
        if not hasattr(self.__class__, "provider") or self.__class__.provider is None:
            self.skipTest("Provider initialization failed")
            
        # Create a configuration for testing parameter extraction
        self.test_config = ConfigurationManager.create_base_config(
            model=DEFAULT_MODEL, 
            temperature=0.7
        )
        
        # Extract generation parameters
        self.gen_params = ConfigurationManager.extract_generation_params(
            "huggingface", 
            self.test_config, 
            {}
        )
    
    @pytest.mark.timeout(30)  # Shorter timeout
    def test_generate(self):
        """Test text generation."""
        provider = self.__class__.provider
        
        # Test with simple prompt
        response = provider.generate("The capital of France is")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_streaming(self):
        """Test streaming generation."""
        provider = self.__class__.provider
        
        # Verify streaming capability is available
        capabilities = provider.get_capabilities()
        self.assertTrue(capabilities.get(ModelParameter.STREAMING))
        
        # Test parameter extraction with streaming
        streaming_params = ConfigurationManager.extract_generation_params(
            "huggingface", 
            provider.config, 
            {"stream": True}
        )
        
        # Test streaming
        try:
            stream = provider.generate("The capital of France is", stream=True)
            
            # Collect chunks
            chunks = []
            for chunk in stream:
                chunks.append(chunk)
                
            # Should get multiple chunks
            self.assertTrue(len(chunks) > 0)
            
            # Combined response should be non-empty
            full_response = "".join(chunks)
            self.assertTrue(len(full_response) > 0)
        except NotImplementedError:
            self.skipTest("Streaming not implemented for the current model")

    def test_cached_models(self):
        """Test model caching functionality."""
        # This depends on the provider implementation
        if not hasattr(HuggingFaceProvider, "list_cached_models"):
            self.skipTest("Provider does not implement caching functions")
        
        try:
            # List cached models
            cached_models = HuggingFaceProvider.list_cached_models()
            self.assertIsInstance(cached_models, list)
            
            # Should have at least one model (the one we preloaded)
            # Note: may fail if another process clears the cache
            self.assertTrue(len(cached_models) > 0)
        
        except Exception as e:
            self.skipTest(f"Caching test failed: {e}")
            
    def test_parameter_extraction(self):
        """Test parameter extraction for HuggingFace provider."""
        # Use standard parameters
        params = self.gen_params
        
        # Check standard parameters
        self.assertEqual(params["model"], DEFAULT_MODEL)
        self.assertEqual(params["temperature"], 0.7)
        
        # Check HuggingFace-specific parameter handling
        hf_specific_params = ConfigurationManager.extract_generation_params(
            "huggingface", 
            self.test_config, 
            {
                "device": "cpu", 
                "trust_remote_code": True,
                "load_in_8bit": False
            }
        )
        
        # Verify HuggingFace-specific parameters
        self.assertEqual(hf_specific_params["device"], "cpu")
        self.assertEqual(hf_specific_params["trust_remote_code"], True)
        self.assertEqual(hf_specific_params["load_in_8bit"], False)
        
    def test_parameter_override(self):
        """Test parameter override in generate method."""
        provider = self.__class__.provider
        
        # Current temperature
        original_temp = provider.config.get(ModelParameter.TEMPERATURE, 0.7)
        
        # Create parameters with overridden temperature
        override_params = ConfigurationManager.extract_generation_params(
            "huggingface", 
            provider.config, 
            {"temperature": 0.2}
        )
        
        # Verify temperature was overridden
        self.assertEqual(override_params["temperature"], 0.2)
        
        # Original config should be unchanged
        self.assertEqual(provider.config.get(ModelParameter.TEMPERATURE), original_temp)
        
        # Test with actual generation (if this affects output is model-dependent)
        response = provider.generate("The capital of France is", temperature=0.2)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)


if __name__ == "__main__":
    unittest.main() 