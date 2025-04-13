"""
Text2Text pipeline implementation for HuggingFace provider.

This pipeline handles sequence-to-sequence models like T5, BART, etc.
Used for tasks like translation, summarization, and other text-to-text tasks.
"""

import logging
from typing import Optional, Dict, Any, Union, List, Generator, Set, Tuple
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TextIteratorStreamer,
    MBartTokenizer,
    T5Tokenizer,
    BartTokenizer
)
from threading import Thread
import numpy as np

from abstractllm.media.interface import MediaInput
from abstractllm.exceptions import ModelLoadingError, GenerationError
from .model_types import BasePipeline, ModelConfig, ModelCapabilities, ModelArchitecture

# Configure logger
logger = logging.getLogger(__name__)

class Text2TextPipeline(BasePipeline):
    """Pipeline for text-to-text models (translation, summarization, etc.)."""
    
    # Model-specific language codes
    MBART_LANG_CODES = {
        "ar": "ar_AR", "cs": "cs_CZ", "de": "de_DE", "en": "en_XX", "es": "es_XX",
        "et": "et_EE", "fi": "fi_FI", "fr": "fr_XX", "gu": "gu_IN", "hi": "hi_IN",
        "it": "it_IT", "ja": "ja_XX", "kk": "kk_KZ", "ko": "ko_KR", "lt": "lt_LT",
        "lv": "lv_LV", "my": "my_MM", "ne": "ne_NP", "nl": "nl_XX", "ro": "ro_RO",
        "ru": "ru_RU", "si": "si_LK", "tr": "tr_TR", "vi": "vi_VN", "zh": "zh_CN"
    }
    
    def load(self, model_name: str, config: ModelConfig) -> None:
        """Load the text-to-text model."""
        try:
            # Load configuration
            model_config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=config.trust_remote_code
            )
            
            # Prepare loading kwargs
            load_kwargs = {
                "device_map": config.device_map,
                "torch_dtype": config.torch_dtype,
                "use_safetensors": config.use_safetensors,
                "trust_remote_code": config.trust_remote_code,
                "use_flash_attention_2": config.use_flash_attention
            }
            
            # Add quantization if specified
            if config.quantization == "4bit":
                from transformers import BitsAndBytesConfig
                load_kwargs.update({
                    "load_in_4bit": True,
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                })
            elif config.quantization == "8bit":
                load_kwargs.update({"load_in_8bit": True})
            
            # Load model and tokenizer
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                config=model_config,
                **load_kwargs
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=config.trust_remote_code
            )
            
            # Apply model-specific optimizations
            self._optimize_model(model_name)
            
            # Ensure we have required tokens
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Set up generation config
            self._generation_config = self._model.generation_config
            
            # Store task type and model type
            self._task_type = self._detect_task_type(model_name, model_config)
            self._model_type = self._detect_model_type(model_name)
            
            self._is_loaded = True
            
        except Exception as e:
            self.cleanup()
            raise ModelLoadingError(f"Failed to load model: {e}")
    
    def _optimize_model(self, model_name: str) -> None:
        """Apply model-specific optimizations."""
        try:
            # T5-specific optimizations
            if "t5" in model_name.lower():
                # Enable model parallelism if available
                if hasattr(self._model, "parallelize") and torch.cuda.device_count() > 1:
                    self._model.parallelize()
                # Use memory efficient attention if available
                if hasattr(self._model, "enable_memory_efficient_attention"):
                    self._model.enable_memory_efficient_attention()
            
            # BART-specific optimizations
            elif "bart" in model_name.lower():
                # Set optimal beam search parameters
                self._generation_config.num_beams = 4
                self._generation_config.length_penalty = 2.0
                
            # mBART-specific optimizations
            elif "mbart" in model_name.lower():
                # Enable token type embeddings
                self._model.config.output_hidden_states = True
                # Set language codes if available
                if isinstance(self._tokenizer, MBartTokenizer):
                    self._tokenizer.src_lang = "en_XX"
                    self._tokenizer.tgt_lang = "fr_XX"
            
            logger.debug(f"Applied optimizations for model type: {self._model_type}")
            
        except Exception as e:
            logger.warning(f"Failed to apply some optimizations: {e}")
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of input text.
        
        Returns:
            ISO 639-1 language code
        """
        try:
            from langdetect import detect
            return detect(text)
        except ImportError:
            logger.warning("langdetect not installed, defaulting to 'en'")
            return "en"
        except Exception as e:
            logger.warning(f"Language detection failed: {e}, defaulting to 'en'")
            return "en"
    
    def _get_quality_metrics(self, source: str, generated: str) -> Dict[str, float]:
        """
        Calculate quality metrics for the generated text.
        
        Returns:
            Dictionary of metric names and scores
        """
        metrics = {}
        
        try:
            # Basic length ratio
            metrics["length_ratio"] = len(generated.split()) / len(source.split())
            
            if self._task_type == "translation":
                # Translation-specific metrics
                try:
                    from sacrebleu.metrics import BLEU
                    bleu = BLEU()
                    metrics["bleu"] = bleu.corpus_score(
                        [generated], [[source]]).score
                except ImportError:
                    logger.debug("sacrebleu not installed, skipping BLEU score")
                
            elif self._task_type == "summarization":
                # Summarization-specific metrics
                try:
                    from rouge_score import rouge_scorer
                    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
                    scores = scorer.score(source, generated)
                    metrics.update({
                        "rouge1": scores['rouge1'].fmeasure,
                        "rouge2": scores['rouge2'].fmeasure,
                        "rougeL": scores['rougeL'].fmeasure
                    })
                except ImportError:
                    logger.debug("rouge_score not installed, skipping ROUGE scores")
            
            # Add semantic similarity if available
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                embeddings = model.encode([source, generated])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                metrics["semantic_similarity"] = float(similarity)
            except ImportError:
                logger.debug("sentence-transformers not installed, skipping similarity")
            
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
        
        return metrics
    
    def process(self, 
               inputs: List[MediaInput], 
               generation_config: Optional[Dict[str, Any]] = None,
               stream: bool = False,
               **kwargs) -> Union[str, Generator[str, None, None]]:
        """Process inputs and generate text."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Process text inputs
            source_text = self._combine_text_inputs(inputs)
            
            # Detect language if needed
            source_lang = self._detect_language(source_text)
            
            # Add task-specific prefixes and handle languages
            prompt = self._prepare_input(source_text, source_lang)
            
            # Generate
            result = self._generate(prompt, generation_config, stream)
            
            # Calculate quality metrics if not streaming
            if not stream:
                metrics = self._get_quality_metrics(source_text, result)
                logger.debug(f"Quality metrics: {metrics}")
            
            return result
            
        except Exception as e:
            raise GenerationError(f"Generation failed: {e}")
    
    def _prepare_input(self, text: str, source_lang: str) -> str:
        """Prepare input text with appropriate prefixes and language codes."""
        # Handle mBART language codes
        if isinstance(self._tokenizer, MBartTokenizer):
            src_code = self.MBART_LANG_CODES.get(source_lang, "en_XX")
            self._tokenizer.src_lang = src_code
        
        # Add task-specific prefixes for T5 models
        if isinstance(self._tokenizer, T5Tokenizer):
            if self._task_type == "translation":
                return f"translate {source_lang} to en: {text}"
            elif self._task_type == "summarization":
                return f"summarize: {text}"
        
        return text
    
    def _combine_text_inputs(self, inputs: List[MediaInput]) -> str:
        """Combine text inputs into a single prompt."""
        text_parts = []
        for input_obj in inputs:
            if input_obj.media_type == "text":
                formatted = input_obj.to_provider_format("huggingface")
                text_parts.append(formatted["content"])
        return "\n".join(text_parts)
    
    def _detect_task_type(self, model_name: str, model_config: Any) -> str:
        """Detect the specific task type for this model."""
        name_lower = model_name.lower()
        
        # Check model name patterns
        if any(x in name_lower for x in ["translate", "translation", "mt5", "mbart"]):
            return "translation"
        elif any(x in name_lower for x in ["summarize", "summarization", "bart"]):
            return "summarization"
        elif "t5" in name_lower:
            # T5 models can do multiple tasks, default to translation
            return "translation"
            
        # Default to translation
        return "translation"
    
    def _detect_model_type(self, model_name: str) -> str:
        """Detect the specific model type for this model."""
        name_lower = model_name.lower()
        
        # Check model name patterns
        if "t5" in name_lower:
            return "T5"
        elif "bart" in name_lower:
            return "BART"
        elif "mbart" in name_lower:
            return "mBART"
            
        # Default to T5
        return "T5"
    
    def _generate(self,
                 prompt: str,
                 generation_config: Optional[Dict[str, Any]] = None,
                 stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate text using the model."""
        # Prepare inputs
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Update generation config
        gen_config = self._generation_config.copy()
        if generation_config:
            gen_config.update(generation_config)
        
        if stream:
            # Set up streaming
            streamer = TextIteratorStreamer(self._tokenizer)
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                **gen_config
            )
            
            # Run generation in a thread
            thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Return generator
            def stream_generator():
                for text in streamer:
                    yield text
            return stream_generator()
        else:
            # Generate without streaming
            outputs = self._model.generate(**inputs, **gen_config)
            return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        return ModelCapabilities(
            input_types={"text"},
            output_types={"text"},
            supports_streaming=True,
            supports_system_prompt=False,
            context_window=self._get_context_window()
        )
    
    def _get_context_window(self) -> Optional[int]:
        """Get the model's context window size."""
        if hasattr(self._model, "config"):
            if hasattr(self._model.config, "max_position_embeddings"):
                return self._model.config.max_position_embeddings
            elif hasattr(self._model.config, "max_length"):
                return self._model.config.max_length
        return None 