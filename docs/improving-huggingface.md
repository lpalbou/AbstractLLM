# Improving HuggingFace Provider Implementation

## Current Issues

1. **Architectural Issues**:
   - Monolithic implementation in `huggingface.py`
   - Task-centric instead of model-centric design
   - Mixed responsibilities between provider and pipelines
   - Complex configuration management
   - Unclear relationship between models and their capabilities

2. **Provider-Specific Challenges**:
   - Multiple model architectures requiring different interaction patterns
   - Various model capabilities and specializations
   - Model-specific parameters and configurations
   - Need for flexible capability handling

## Goals

1. **Model-Centric Architecture**:
   - Models as primary entities with defined capabilities
   - Clear mapping of architectures to interaction patterns
   - Flexible capability handling without strict task constraints
   - Simple, maintainable pipeline system

2. **Clean Configuration**:
   - Tree-structured configuration system
   - Clear separation of concerns:
     - Base parameters (shared across all models)
     - Architecture-specific parameters
     - Capability-specific parameters
     - Model-specific overrides

3. **Provider Consistency**:
   - Unified interface across providers
   - Common parameter handling
   - Consistent capability checking
   - Clear resource boundaries

## Implementation Plan

### Phase 1: Model Architecture System

1. **Architecture Definition**:
```python
class ModelArchitecture(str, Enum):
    """Core model architectures determining interaction patterns."""
    
    ENCODER_ONLY = "encoder_only"      # BERT, RoBERTa
    DECODER_ONLY = "decoder_only"      # GPT, LLaMA
    ENCODER_DECODER = "enc_dec"        # T5, BART
    VISION_ENCODER = "vision_enc"      # ViT, CLIP
    MULTIMODAL = "multimodal"         # LLaVA, BLIP
    SPEECH = "speech"                 # Whisper, SpeechT5
```

2. **Capability System**:
```python
@dataclass
class ModelCapability:
    """Defines what a model can do."""
    
    name: str                      # Capability identifier
    confidence: float              # How well it handles this capability
    requires_finetuning: bool      # Whether specialized training is needed
    parameters: Dict[str, Any]     # Capability-specific parameters
    
class ModelCapabilities:
    """Manages model capabilities."""
    
    def __init__(self, architecture: ModelArchitecture):
        self._architecture = architecture
        self._capabilities: Dict[str, ModelCapability] = {}
        self._load_architecture_capabilities()
    
    def can_handle(self, capability: str) -> bool:
        """Check if model can handle a capability."""
        return capability in self._capabilities
    
    def get_config(self, capability: str) -> Dict[str, Any]:
        """Get capability-specific configuration."""
        if not self.can_handle(capability):
            logger.info(f"Model not specifically trained for {capability}")
        return self._get_default_config(capability)
```

### Phase 2: Configuration System

1. **Tree-Structured Config**:
```python
@dataclass
class ModelConfig:
    """Tree-structured configuration system."""
    
    # Base configuration (always present)
    base: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 1.0,
        "top_k": 50,
    })
    
    # Architecture-specific configuration
    architecture: Dict[str, Any] = field(default_factory=dict)
    
    # Capability-specific configurations
    capabilities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Model-specific overrides
    model_specific: Dict[str, Any] = field(default_factory=dict)
    
    def get_merged_config(self, capability: str) -> Dict[str, Any]:
        """Get merged configuration for a specific capability."""
        config = self.base.copy()
        config.update(self.architecture)
        if capability in self.capabilities:
            config.update(self.capabilities[capability])
        config.update(self.model_specific)
        return config
```

### Phase 3: Pipeline System

1. **Base Pipeline**:
```python
class BasePipeline:
    """Simplified pipeline focusing on core functionality."""
    
    def __init__(self, architecture: ModelArchitecture):
        self.architecture = architecture
        self.capabilities = ModelCapabilities(architecture)
        self._model = None
        self._processor = None
    
    def load(self, model_name: str, config: ModelConfig) -> None:
        """Load model based on architecture."""
        loader = self._get_architecture_loader()
        self._model = loader.load_model(model_name, config)
        self._processor = loader.load_processor(model_name, config)
    
    def process(self, 
                inputs: List[MediaInput],
                capability: str,
                config: Optional[Dict[str, Any]] = None,
                **kwargs) -> Any:
        """Process inputs using specified capability."""
        if not self.capabilities.can_handle(capability):
            logger.info(f"Using general processing for {capability}")
        
        processor = self._get_capability_processor(capability)
        return processor.process(inputs, config, **kwargs)
```

2. **Architecture-Specific Implementations**:
```python
class EncoderDecoderPipeline(BasePipeline):
    """Pipeline for encoder-decoder models."""
    
    def _get_architecture_loader(self) -> ModelLoader:
        return EncoderDecoderLoader()
    
    def _get_capability_processor(self, capability: str) -> CapabilityProcessor:
        if capability == "summarization":
            return SummarizationProcessor(self._model, self._processor)
        elif capability == "translation":
            return TranslationProcessor(self._model, self._processor)
        return GeneralProcessor(self._model, self._processor)
```

### Phase 4: Provider Implementation

1. **HuggingFace Provider**:
```python
class HuggingFaceProvider(AbstractLLMInterface):
    """Model-centric HuggingFace provider."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._pipeline = None
        self._model_config = None
        
        # Default configuration focusing on model capabilities
        default_config = {
            "model": "microsoft/phi-2",
            "architecture": ModelArchitecture.DECODER_ONLY,
            "capabilities": {
                "text-generation": {"max_tokens": 2048},
                "summarization": {"min_length": 50, "max_length": 200},
                "translation": {"src_lang": "en", "tgt_lang": "fr"}
            }
        }
        self.config_manager.merge_with_defaults(default_config)
    
    def generate(self, 
                prompt: str,
                capability: Optional[str] = None,
                **kwargs) -> str:
        """Generate output using model capabilities."""
        if not self._pipeline:
            self._setup_pipeline()
        
        # Determine capability from input and model
        capability = capability or self._detect_capability(prompt, kwargs)
        
        # Get merged configuration
        config = self._model_config.get_merged_config(capability)
        config.update(kwargs)
        
        # Process using appropriate capability
        return self._pipeline.process(
            self._prepare_inputs(prompt),
            capability=capability,
            config=config
        )
```

### Phase 5: Capability Detection and Recommendations

1. **Capability Registry**:
```python
@dataclass
class ModelRecommendation:
    """Model recommendation for a specific size tier."""
    model_name: str
    size_gb: float
    capabilities: List[str]
    requirements: Optional[Dict[str, Any]] = None  # e.g., flash attention, etc.

class CapabilityRegistry:
    """Registry of capabilities and recommended models."""
    
    # Size tiers in billions of parameters
    SIZE_TIERS = {
        "small": 4,    # <= 4B parameters
        "medium": 14,  # <= 14B parameters
        "large": 32,   # <= 32B parameters
        "xlarge": 80,  # <= 80B parameters
    }
    
    def __init__(self):
        self._recommendations = self._get_default_recommendations()
        
    def _get_default_recommendations(self) -> Dict[str, Dict[str, ModelRecommendation]]:
        """Get default model recommendations."""
        return {
            # Text Generation
            "text-generation": {
                "small": ModelRecommendation(
                    model_name="microsoft/phi-4-mini",
                    size_gb=3.8,
                    capabilities=["text-generation", "function-calling"]
                ),
                "medium": ModelRecommendation(
                    model_name="microsoft/phi-4",
                    size_gb=14.0,
                    capabilities=["text-generation", "function-calling"]
                ),
                "large": ModelRecommendation(
                    model_name="QwQ/QwQ-32B",
                    size_gb=32.0,
                    capabilities=["text-generation", "function-calling", "reasoning"]
                ),
                "xlarge": ModelRecommendation(
                    model_name="meta-llama/llama3.3-70b",
                    size_gb=70.0,
                    capabilities=["text-generation", "function-calling"]
                )
            },
            # ... other default recommendations ...
        }
    
    def update_recommendations(self, 
                             updates: Dict[str, Dict[str, ModelRecommendation]],
                             merge: bool = True) -> None:
        """Update model recommendations.
        
        Args:
            updates: New recommendations to apply
            merge: If True, merge with existing recommendations
                  If False, replace existing recommendations entirely
        """
        if merge:
            for capability, tiers in updates.items():
                if capability not in self._recommendations:
                    self._recommendations[capability] = {}
                for tier, rec in tiers.items():
                    self._recommendations[capability][tier] = rec
        else:
            self._recommendations = updates
            
    def update_model(self,
                    capability: str,
                    size_tier: str,
                    model: ModelRecommendation) -> None:
        """Update a single model recommendation."""
        if capability not in self._recommendations:
            self._recommendations[capability] = {}
        self._recommendations[capability][size_tier] = model
        
    def remove_recommendation(self,
                            capability: str,
                            size_tier: Optional[str] = None) -> None:
        """Remove a recommendation for a capability or specific tier."""
        if size_tier:
            if capability in self._recommendations:
                self._recommendations[capability].pop(size_tier, None)
        else:
            self._recommendations.pop(capability, None)
    
    def get_recommendation(self, 
                         capability: str,
                         size_tier: str = "medium",
                         system_info: Optional[Dict[str, Any]] = None) -> Optional[ModelRecommendation]:
        """Get recommended model for capability and size tier."""
        if capability not in self._recommendations:
            return None
            
        tier_recs = self._recommendations[capability]
        if size_tier not in tier_recs:
            return None
            
        rec = tier_recs[size_tier]
        
        # Check system requirements if provided
        if system_info and rec.requirements:
            if not self._check_requirements(rec.requirements, system_info):
                return self._find_alternative(capability, size_tier, system_info)
                
        return rec
    
    def _check_requirements(self,
                          requirements: Dict[str, Any],
                          system_info: Dict[str, Any]) -> bool:
        """Check if system meets model requirements."""
        for req, value in requirements.items():
            if req not in system_info or system_info[req] != value:
                return False
        return True
    
    def _find_alternative(self, 
                         capability: str,
                         size_tier: str,
                         system_info: Dict[str, Any]) -> Optional[ModelRecommendation]:
        """Find alternative model when primary recommendation isn't suitable."""
        # Try same tier models with different capabilities
        for cap, tiers in self._recommendations.items():
            if cap == capability:
                continue
            if size_tier in tiers:
                rec = tiers[size_tier]
                if capability in rec.capabilities:
                    if not rec.requirements or self._check_requirements(rec.requirements, system_info):
                        return rec
        
        # Try smaller models if available
        tiers = list(self.SIZE_TIERS.keys())
        current_idx = tiers.index(size_tier)
        for tier in tiers[current_idx-1::-1]:  # Try progressively smaller models
            if tier in self._recommendations[capability]:
                rec = self._recommendations[capability][tier]
                if not rec.requirements or self._check_requirements(rec.requirements, system_info):
                    return rec
        
        return None
```

2. **Usage Examples**:
```python
# Initialize with defaults
registry = CapabilityRegistry()

# Update a single model recommendation
registry.update_model(
    "text-generation",
    "medium",
    ModelRecommendation(
        model_name="mistral-7b",
        size_gb=7.0,
        capabilities=["text-generation", "function-calling"]
    )
)

# Update multiple recommendations at once
new_recommendations = {
    "text-generation": {
        "small": ModelRecommendation(
            model_name="phi-2",
            size_gb=2.7,
            capabilities=["text-generation"]
        )
    },
    "code-generation": {
        "medium": ModelRecommendation(
            model_name="codellama-13b",
            size_gb=13.0,
            capabilities=["code-generation", "text-generation"]
        )
    }
}
registry.update_recommendations(new_recommendations, merge=True)

# Remove a specific recommendation
registry.remove_recommendation("text-generation", "small")

# Remove all recommendations for a capability
registry.remove_recommendation("code-generation")
```

3. **Provider Integration**:
```python
class HuggingFaceProvider(AbstractLLMInterface):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._registry = CapabilityRegistry()
        
        # Update recommendations if provided in config
        if config and "model_recommendations" in config:
            self._registry.update_recommendations(
                config["model_recommendations"],
                merge=config.get("merge_recommendations", True)
            )
        
        # Get system info for requirements checking
        self._system_info = {
            "supports_flash_attention": self._check_flash_attention_support(),
            "platform": platform.system().lower(),
            "gpu_memory": self._get_available_gpu_memory()
        }
        
        # Get size tier from config or default
        self._size_tier = config.get("size_tier", "medium")
    
    def update_model_recommendations(self,
                                   updates: Dict[str, Dict[str, ModelRecommendation]],
                                   merge: bool = True) -> None:
        """Update model recommendations.
        
        This allows external programs to update model recommendations
        without needing to access configuration files.
        """
        self._registry.update_recommendations(updates, merge)
    
    def get_model_for_capability(self, capability: str) -> str:
        """Get appropriate model for capability."""
        rec = self._registry.get_recommendation(
            capability,
            self._size_tier,
            self._system_info
        )
        
        if rec is None:
            raise UnsupportedCapabilityError(
                f"No suitable model found for {capability} at {self._size_tier} tier"
            )
            
        return rec.model_name
```

## Remaining Questions

1. **Caching Strategy**:
   - Should we extend HuggingFace's caching system?
   - Do we need additional caching for capability-specific resources?
   - How to handle cache invalidation for model updates?

2. **Capability Detection**:
   - How to best detect required capabilities from input?
   - Should we provide capability suggestions?
   - How to handle capability conflicts?

3. **Configuration Management**:
   - How to validate capability-specific configurations?
   - Should we allow dynamic capability addition?
   - How to handle configuration inheritance between capabilities?

## Migration Steps

1. **Phase 1: Core Architecture** (Week 1)
   - Implement model architecture system
   - Create capability management
   - Set up configuration structure

2. **Phase 2: Pipeline System** (Week 2)
   - Create base pipeline
   - Implement architecture-specific pipelines
   - Add capability processors

3. **Phase 3: Provider Updates** (Week 3)
   - Update HuggingFace provider
   - Implement capability detection
   - Add configuration management

4. **Phase 4: Testing & Documentation** (Week 4)
   - Comprehensive testing
   - Documentation updates
   - Migration guides

5. **Phase 5: Capability Detection and Recommendations** (Week 5)
   - Implement capability registry
   - Add model recommendation logic
   - Update provider to use recommendations

## Success Metrics

1. **Code Quality**:
   - Clear separation of concerns
   - Reduced complexity
   - Better maintainability

2. **Functionality**:
   - All current features preserved
   - Improved capability handling
   - Better error messages

3. **User Experience**:
   - Simpler API
   - Clear capability documentation
   - Better error handling
``` 