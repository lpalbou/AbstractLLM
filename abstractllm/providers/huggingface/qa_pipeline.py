"""
Question Answering pipeline implementation for HuggingFace provider.

This pipeline handles both extractive QA models (like BERT, RoBERTa, DeBERTa)
and generative QA models (like T5, BART). For extractive QA, the model finds
the answer span within the context. For generative QA, the model generates
a free-form answer.
"""

import logging
from typing import Optional, Dict, Any, Union, List, Generator, Tuple
import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

from abstractllm.media.interface import MediaInput
from abstractllm.exceptions import ModelLoadError, InvalidInputError, GenerationError
from .model_types import BasePipeline, ModelConfig, ModelCapabilities

# Configure logger
logger = logging.getLogger(__name__)

class QuestionAnsweringPipeline(BasePipeline):
    """Pipeline for question answering tasks using HuggingFace models.
    
    This pipeline supports both extractive QA models (like BERT) and 
    generative QA models (like T5). For extractive models, it finds the answer
    span in the context. For generative models, it generates a free-form answer.
    """
    
    def __init__(self) -> None:
        """Initialize the QA pipeline."""
        super().__init__()
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model_config: Optional[ModelConfig] = None
    
    def load(self, model_name: str, config: ModelConfig) -> None:
        """Load the QA model and tokenizer.
        
        Args:
            model_name: Name or path of the model
            config: Model configuration
            
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=config.trust_remote_code
            )
            
            # Try loading as extractive QA model first
            try:
                self.model = AutoModelForQuestionAnswering.from_pretrained(
                    model_name,
                    trust_remote_code=config.trust_remote_code,
                    device_map=config.device_map,
                    torch_dtype=config.torch_dtype
                )
            except Exception:
                # If fails, try loading as generative model
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    trust_remote_code=config.trust_remote_code,
                    device_map=config.device_map,
                    torch_dtype=config.torch_dtype
                )
            
            # Move model to device if not using device_map="auto"
            if config.device_map != "auto":
                self.model.to(self.device)
            
            self.model_config = config
            self._is_loaded = True
            
        except Exception as e:
            self.cleanup()
            raise ModelLoadError(f"Failed to load model {model_name}: {str(e)}")
    
    def _is_extractive_model(self, model: PreTrainedModel) -> bool:
        """Check if the model is an extractive QA model.
        
        Args:
            model: The loaded model to check.
            
        Returns:
            bool: True if extractive QA model, False if generative.
        """
        extractive_classes = [
            "BertForQuestionAnswering",
            "RobertaForQuestionAnswering",
            "DistilBertForQuestionAnswering",
            "AlbertForQuestionAnswering",
            "DeBertaForQuestionAnswering"
        ]
        return model.__class__.__name__ in extractive_classes
    
    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Tuple[str, str]:
        """Prepare and validate the input question and context.
        
        Args:
            inputs: Dictionary containing 'question' and 'context' keys.
            
        Returns:
            Tuple containing the question and context strings.
            
        Raises:
            InvalidInputError: If required inputs are missing or invalid.
        """
        if not isinstance(inputs, dict):
            raise InvalidInputError("Inputs must be a dictionary")
            
        question = inputs.get("question")
        context = inputs.get("context")
        
        if not question or not isinstance(question, str):
            raise InvalidInputError("Question must be a non-empty string")
        if not context or not isinstance(context, str):
            raise InvalidInputError("Context must be a non-empty string")
            
        return question, context
    
    def _extract_answer(self, question: str, context: str) -> str:
        """Extract answer span from context using extractive QA model.
        
        Args:
            question: The question to answer.
            context: The context containing the answer.
            
        Returns:
            str: The extracted answer span.
        """
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        start_idx = start_logits.argmax().item()
        end_idx = end_logits.argmax().item()
        
        # Ensure valid span (start before end)
        if end_idx < start_idx:
            end_idx = start_idx
            
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        answer_tokens = tokens[start_idx:end_idx + 1]
        
        # Clean up answer tokens
        answer = self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(answer_tokens),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return answer.strip()
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using a generative QA model.
        
        Args:
            question: The question to answer.
            context: The context containing information.
            
        Returns:
            str: The generated answer.
        """
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                min_length=1,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
        answer = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        return answer.strip()
    
    def process(self, 
                inputs: List[MediaInput], 
                generation_config: Optional[Dict[str, Any]] = None,
                stream: bool = False,
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """Process question and context to generate answer."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Extract question and context from inputs
            text_inputs = [inp for inp in inputs if inp.media_type == "text"]
            if len(text_inputs) != 2:
                raise InvalidInputError("QA requires exactly two text inputs: question and context")
            
            # First input is question, second is context
            question = text_inputs[0].to_provider_format("huggingface")["content"]
            context = text_inputs[1].to_provider_format("huggingface")["content"]
            
            # Process based on model type
            if self._is_extractive_model(self.model):
                return self._extract_answer(question, context)
            else:
                return self._generate_answer(question, context)
            
        except Exception as e:
            raise GenerationError(f"Generation failed: {e}")
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        return ModelCapabilities(
            input_types={"text"},
            output_types={"text"},
            supports_streaming=False,
            supports_system_prompt=False,
            context_window=self._get_context_window()
        )
    
    def _get_context_window(self) -> Optional[int]:
        """Get the model's context window size."""
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "max_position_embeddings"):
                return self.model.config.max_position_embeddings
            elif hasattr(self.model.config, "max_length"):
                return self.model.config.max_length
        return None 