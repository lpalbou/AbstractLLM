"""
Template Management

Unified template management combining HuggingFace chat templates and harvested templates
from representative models for each architecture.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from .detection import detect_architecture, normalize_model_name

logger = logging.getLogger(__name__)

@dataclass 
class TemplateInfo:
    """Information about a chat template."""
    model_name: str
    template: str
    template_type: str = "default"
    supports_tools: bool = False
    supports_system: bool = False
    cached_at: float = 0.0
    source: str = "unknown"  # "hf_direct", "hf_harvested", "static"
    architecture: Optional[str] = None


class TemplateManager:
    """
    Unified template manager that uses multiple sources:
    1. Direct HuggingFace tokenizer access (primary)
    2. Harvested templates from representative models (fallback)
    3. Static templates (last resort)
    """
    
    def __init__(self, cache_dir: Optional[str] = None, cache_ttl: int = 86400):
        """Initialize the template manager."""
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/abstractllm/templates")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl
        
        # Template caches
        self.hf_templates: Dict[str, TemplateInfo] = {}
        self.harvested_templates: Dict[str, TemplateInfo] = {}
        self.static_templates: Dict[str, TemplateInfo] = {}
        
        # Load existing caches
        self._load_caches()
    
    def _load_caches(self):
        """Load all template caches."""
        # Load HF templates cache
        hf_cache_file = self.cache_dir / "hf_templates.json"
        if hf_cache_file.exists():
            try:
                with open(hf_cache_file, 'r') as f:
                    data = json.load(f)
                    for key, template_data in data.items():
                        self.hf_templates[key] = TemplateInfo(**template_data)
                logger.debug(f"Loaded {len(self.hf_templates)} HF templates")
            except Exception as e:
                logger.warning(f"Failed to load HF templates cache: {e}")
        
        # Load harvested templates cache
        harvested_cache_file = self.cache_dir / "harvested_templates.json"
        if harvested_cache_file.exists():
            try:
                with open(harvested_cache_file, 'r') as f:
                    data = json.load(f)
                    for key, template_data in data.items():
                        self.harvested_templates[key] = TemplateInfo(**template_data)
                logger.debug(f"Loaded {len(self.harvested_templates)} harvested templates")
            except Exception as e:
                logger.warning(f"Failed to load harvested templates cache: {e}")
        
        # Load static templates
        self._load_static_templates()
    
    def _load_static_templates(self):
        """Load static fallback templates."""
        # Basic static templates for common architectures
        static_data = {
            "granite": {
                "template": """{% for message in messages %}{% if message['role'] == 'system' %}<|system|>
{{ message['content'] }}<|end|>
{% elif message['role'] == 'user' %}<|user|>
{{ message['content'] }}<|end|>
{% elif message['role'] == 'assistant' %}<|assistant|>
{{ message['content'] }}<|end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>
{% endif %}""",
                "supports_tools": True,
                "supports_system": True,
                "architecture": "granite"
            },
            "qwen": {
                "template": """{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}""",
                "supports_tools": True,
                "supports_system": True,
                "architecture": "qwen"
            },
            "llama": {
                "template": """{% for message in messages %}{% if message['role'] == 'system' %}<s>[INST] <<SYS>>
{{ message['content'] }}
<</SYS>>

{% elif message['role'] == 'user' %}{{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %} {{ message['content'] }}</s><s>[INST] {% endif %}{% endfor %}""",
                "supports_tools": True,
                "supports_system": True,
                "architecture": "llama"
            }
        }
        
        for arch, template_data in static_data.items():
            template_info = TemplateInfo(
                model_name=f"static_{arch}",
                template=template_data["template"],
                supports_tools=template_data["supports_tools"],
                supports_system=template_data["supports_system"],
                architecture=template_data["architecture"],
                source="static",
                cached_at=time.time()
            )
            self.static_templates[arch] = template_info
    
    def _save_cache(self, cache_type: str):
        """Save specific cache to disk."""
        try:
            if cache_type == "hf":
                cache_file = self.cache_dir / "hf_templates.json"
                cache_data = {k: self._template_to_dict(v) for k, v in self.hf_templates.items()}
            elif cache_type == "harvested":
                cache_file = self.cache_dir / "harvested_templates.json"
                cache_data = {k: self._template_to_dict(v) for k, v in self.harvested_templates.items()}
            else:
                return
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.debug(f"Saved {cache_type} templates cache")
        except Exception as e:
            logger.warning(f"Failed to save {cache_type} templates cache: {e}")
    
    def _template_to_dict(self, template: TemplateInfo) -> Dict[str, Any]:
        """Convert TemplateInfo to dict for JSON serialization."""
        return {
            "model_name": template.model_name,
            "template": template.template,
            "template_type": template.template_type,
            "supports_tools": template.supports_tools,
            "supports_system": template.supports_system,
            "cached_at": template.cached_at,
            "source": template.source,
            "architecture": template.architecture
        }
    
    def get_template(self, model_name: str, template_type: str = "default") -> Optional[TemplateInfo]:
        """
        Get template for a model using multiple fallback strategies.
        
        Args:
            model_name: Model name (can be provider-specific)
            template_type: Template type
            
        Returns:
            TemplateInfo or None if not found
        """
        cache_key = f"{model_name}:{template_type}"
        
        # Strategy 1: Try direct HF access for HF model names
        if self._looks_like_hf_model(model_name):
            # Check cache first
            if cache_key in self.hf_templates:
                template = self.hf_templates[cache_key]
                if time.time() - template.cached_at < self.cache_ttl:
                    logger.debug(f"Using cached HF template for {model_name}")
                    return template
            
            # Try to fetch from HF
            template = self._fetch_hf_template(model_name, template_type)
            if template:
                self.hf_templates[cache_key] = template
                self._save_cache("hf")
                return template
        
        # Strategy 2: Try harvested templates by architecture
        architecture = detect_architecture(model_name)
        if architecture:
            if architecture in self.harvested_templates:
                template = self.harvested_templates[architecture]
                if time.time() - template.cached_at < self.cache_ttl:
                    logger.debug(f"Using harvested template for {architecture} architecture")
                    return template
        
        # Strategy 3: Try static templates as last resort
        if architecture and architecture in self.static_templates:
            logger.debug(f"Using static template for {architecture} architecture")
            return self.static_templates[architecture]
        
        return None
    
    def _looks_like_hf_model(self, model_name: str) -> bool:
        """Check if model name looks like a HuggingFace model."""
        return "/" in model_name and not model_name.startswith("mlx-community/")
    
    def _fetch_hf_template(self, model_name: str, template_type: str = "default") -> Optional[TemplateInfo]:
        """Fetch template directly from HuggingFace."""
        try:
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=os.environ.get('HF_HUB_CACHE')
            )
            
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                template = tokenizer.chat_template
                if isinstance(template, dict):
                    template = template.get(template_type, template.get("default", str(template)))
                
                # Test capabilities
                supports_tools = self._test_tool_support(tokenizer)
                supports_system = self._test_system_support(tokenizer)
                architecture = detect_architecture(model_name)
                
                return TemplateInfo(
                    model_name=model_name,
                    template=template,
                    template_type=template_type,
                    supports_tools=supports_tools,
                    supports_system=supports_system,
                    cached_at=time.time(),
                    source="hf_direct",
                    architecture=architecture
                )
        except Exception as e:
            logger.debug(f"Failed to fetch HF template for {model_name}: {e}")
        
        return None
    
    def _test_tool_support(self, tokenizer: Any) -> bool:
        """Test if tokenizer supports tool calling."""
        try:
            def dummy_tool():
                """A dummy tool for testing."""
                return "test"
            
            messages = [{"role": "user", "content": "test"}]
            tokenizer.apply_chat_template(
                messages,
                tools=[dummy_tool],
                tokenize=False,
                add_generation_prompt=True
            )
            return True
        except Exception:
            return False
    
    def _test_system_support(self, tokenizer: Any) -> bool:
        """Test if tokenizer supports system messages."""
        try:
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "test"}
            ]
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return True
        except Exception:
            return False
    
    def harvest_template(self, architecture: str, representative_model: str) -> Optional[TemplateInfo]:
        """
        Harvest template from a representative model.
        
        Args:
            architecture: Architecture name
            representative_model: HF model to harvest from
            
        Returns:
            TemplateInfo or None if failed
        """
        try:
            template = self._fetch_hf_template(representative_model)
            if template:
                template.source = "hf_harvested"
                template.architecture = architecture
                
                # Cache it under architecture name
                self.harvested_templates[architecture] = template
                self._save_cache("harvested")
                
                logger.info(f"Successfully harvested template for {architecture} from {representative_model}")
                return template
        except Exception as e:
            logger.warning(f"Failed to harvest template for {architecture} from {representative_model}: {e}")
        
        return None
    
    def get_supported_models(self) -> List[str]:
        """Get list of models with cached templates."""
        models = []
        models.extend(self.hf_templates.keys())
        models.extend([f"{arch}_architecture" for arch in self.harvested_templates.keys()])
        models.extend([f"{arch}_static" for arch in self.static_templates.keys()])
        return models


# Global template manager instance
_template_manager = None

def get_template_manager() -> TemplateManager:
    """Get global template manager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager


def get_template(model_name: str, template_type: str = "default") -> Optional[TemplateInfo]:
    """
    Get template for a model.
    
    Args:
        model_name: Model name
        template_type: Template type
        
    Returns:
        TemplateInfo or None if not found
    """
    return get_template_manager().get_template(model_name, template_type)


def apply_chat_template(messages: List[Dict[str, Any]], model_name: str, template_type: str = "default") -> str:
    """
    Apply chat template to messages using architecture-specific formatting.
    
    Args:
        messages: List of message dictionaries with roles and content
        model_name: Model name for architecture detection
        template_type: Template type to use
        
    Returns:
        Formatted prompt string
    """
    from .detection import detect_architecture
    
    # Get architecture and apply appropriate template
    architecture = detect_architecture(model_name)
    
    if architecture == "gemma":
        return apply_gemma_template(messages)
    elif architecture == "llama":
        return apply_llama_template(messages)
    elif architecture == "qwen":
        return apply_qwen_template(messages)
    elif architecture == "phi":
        return apply_phi_template(messages)
    elif architecture == "mistral":
        return apply_mistral_template(messages)
    else:
        # Fallback to simple template
        return apply_simple_fallback_template(messages, model_name)


def apply_gemma_template(messages: List[Dict[str, Any]]) -> str:
    """Apply Gemma-specific chat template."""
    prompt_parts = []
    system_content = None
    
    # Extract system content first
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
            break
    
    # Build conversation
    for msg in messages:
        if msg["role"] == "system":
            continue  # Skip system messages - will be integrated into first user message
        elif msg["role"] == "user":
            prompt_parts.append(f"<start_of_turn>user\n{msg['content']}<end_of_turn>")
        elif msg["role"] == "assistant":
            prompt_parts.append(f"<start_of_turn>model\n{msg['content']}<end_of_turn>")
    
    # Add system content to first user message if present
    if system_content and prompt_parts:
        first_user_msg = prompt_parts[0]
        if first_user_msg.startswith("<start_of_turn>user\n"):
            user_content = first_user_msg[len("<start_of_turn>user\n"):-len("<end_of_turn>")]
            enhanced_user_content = f"System: {system_content}\n\nUser: {user_content}"
            prompt_parts[0] = f"<start_of_turn>user\n{enhanced_user_content}<end_of_turn>"
    
    prompt_parts.append("<start_of_turn>model\n")
    return "\n".join(prompt_parts)


def apply_llama_template(messages: List[Dict[str, Any]]) -> str:
    """Apply Llama-specific chat template."""
    prompt_parts = []
    system_content = None
    
    # Extract system content
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
            break
    
    # Build conversation in Llama format
    if system_content:
        prompt_parts.append(f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n")
    else:
        prompt_parts.append("<s>[INST] ")
    
    user_messages = []
    assistant_messages = []
    
    for msg in messages:
        if msg["role"] == "user":
            user_messages.append(msg["content"])
        elif msg["role"] == "assistant":
            assistant_messages.append(msg["content"])
    
    # Interleave user and assistant messages
    for i, user_msg in enumerate(user_messages):
        if i == 0 and system_content:
            prompt_parts.append(f"{user_msg} [/INST]")
        else:
            prompt_parts.append(f"<s>[INST] {user_msg} [/INST]")
        
        if i < len(assistant_messages):
            prompt_parts.append(f" {assistant_messages[i]} </s>")
    
    # If we have more user messages than assistant messages, end with generation prompt
    if len(user_messages) > len(assistant_messages):
        prompt_parts.append(" ")
    
    return "".join(prompt_parts)


def apply_qwen_template(messages: List[Dict[str, Any]]) -> str:
    """Apply Qwen-specific chat template."""
    prompt_parts = []
    
    for msg in messages:
        if msg["role"] == "system":
            prompt_parts.append(f"<|im_start|>system\n{msg['content']}<|im_end|>")
        elif msg["role"] == "user":
            prompt_parts.append(f"<|im_start|>user\n{msg['content']}<|im_end|>")
        elif msg["role"] == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{msg['content']}<|im_end|>")
    
    prompt_parts.append("<|im_start|>assistant\n")
    return "\n".join(prompt_parts)


def apply_phi_template(messages: List[Dict[str, Any]]) -> str:
    """Apply Phi-specific chat template."""
    prompt_parts = []
    
    for msg in messages:
        if msg["role"] == "system":
            prompt_parts.append(f"<|system|>\n{msg['content']}<|end|>")
        elif msg["role"] == "user":
            prompt_parts.append(f"<|user|>\n{msg['content']}<|end|>")
        elif msg["role"] == "assistant":
            prompt_parts.append(f"<|assistant|>\n{msg['content']}<|end|>")
    
    prompt_parts.append("<|assistant|>\n")
    return "\n".join(prompt_parts)


def apply_mistral_template(messages: List[Dict[str, Any]]) -> str:
    """Apply Mistral-specific chat template."""
    prompt_parts = []
    system_content = None
    
    # Extract system content
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
            break
    
    # Build conversation in Mistral format
    for msg in messages:
        if msg["role"] == "system":
            continue  # System message handled separately
        elif msg["role"] == "user":
            if system_content and len(prompt_parts) == 0:
                # Integrate system prompt into first user message
                prompt_parts.append(f"<s>[INST] {system_content}\n\n{msg['content']} [/INST]")
                system_content = None  # Don't repeat it
            else:
                prompt_parts.append(f"<s>[INST] {msg['content']} [/INST]")
        elif msg["role"] == "assistant":
            prompt_parts.append(f" {msg['content']} </s>")
    
    # End with generation prompt if needed
    if not prompt_parts or not prompt_parts[-1].endswith("</s>"):
        prompt_parts.append(" ")
    
    return "".join(prompt_parts)


def apply_simple_fallback_template(messages: List[Dict[str, Any]], model_name: str) -> str:
    """Apply simple fallback template for unknown architectures."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Applying simple concatenation fallback for {model_name}")
    
    prompt_parts = []
    for msg in messages:
        if msg["role"] == "system":
            prompt_parts.append(f"System: {msg['content']}")
        elif msg["role"] == "user":
            prompt_parts.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            prompt_parts.append(f"Assistant: {msg['content']}")
    
    prompt_parts.append("Assistant:")
    return "\n\n".join(prompt_parts) 