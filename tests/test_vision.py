"""
Tests for vision capabilities in AbstractLLM providers.
"""

import os
import sys
import pytest
import requests
import shutil
from io import BytesIO
from typing import Dict, Any, List, Union, Generator
import unittest.mock
from pathlib import Path

from abstractllm import create_llm, ModelParameter, ModelCapability
from abstractllm.providers.openai import OpenAIProvider, VISION_CAPABLE_MODELS as OPENAI_VISION_MODELS
from abstractllm.providers.anthropic import AnthropicProvider, VISION_CAPABLE_MODELS as ANTHROPIC_VISION_MODELS
from abstractllm.providers.ollama import OllamaProvider, VISION_CAPABLE_MODELS as OLLAMA_VISION_MODELS
from abstractllm.providers.huggingface import HuggingFaceProvider, VISION_CAPABLE_MODELS as HF_VISION_MODELS
from abstractllm.utils.image import format_image_for_provider
from tests.utils import validate_response, validate_not_contains, has_capability

# Define test resources directory and examples directory
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")
os.makedirs(RESOURCES_DIR, exist_ok=True)

# Define test image paths and their keywords
TEST_IMAGES = {
    "test_image_1.jpg": {
        "path": os.path.join(RESOURCES_DIR, "test_image_1.jpg"),
        "source": os.path.join(EXAMPLES_DIR, "test_image_1.jpg"),
        "keywords": [
            "mountain", "mountains", "range", "dirt", "path", "trail", "wooden", "fence", 
            "sunlight", "sunny", "blue sky", "grass", "meadow", "hiking", 
            "countryside", "rural", "landscape", "horizon"
        ],
        "prompt": "Describe what you see in this image in detail."
    },
    "test_image_2.jpg": {
        "path": os.path.join(RESOURCES_DIR, "test_image_2.jpg"),
        "source": os.path.join(EXAMPLES_DIR, "test_image_2.jpg"),
        "keywords": [
            "lamppost", "street light", "sunset", "dusk", "pink", "orange", "sky",
            "pathway", "walkway", "park", "urban", "trees", "buildings", "benches",
            "garden", "evening"
        ],
        "prompt": "What's shown in this image? Give a detailed description."
    },
    "test_image_3.jpg": {
        "path": os.path.join(RESOURCES_DIR, "test_image_3.jpg"),
        "source": os.path.join(EXAMPLES_DIR, "test_image_3.jpg"),
        "keywords": [
            "whale", "humpback", "ocean", "sea", "breaching", "jumping", "splash",
            "marine", "mammal", "fins", "flipper", "gray", "waves", "wildlife",
            "water"
        ],
        "prompt": "Describe the creature in this image and what it's doing."
    },
    "test_image_4.jpg": {
        "path": os.path.join(RESOURCES_DIR, "test_image_4.jpg"),
        "source": os.path.join(EXAMPLES_DIR, "test_image_4.jpg"),
        "keywords": [
            "cat", "pet", "carrier", "transport", "dome", "window", "plastic",
            "orange", "tabby", "fur", "eyes", "round", "opening", "white", "base",
            "ventilation", "air holes"
        ],
        "prompt": "What animal is shown in this image and where is it located?"
    }
}

