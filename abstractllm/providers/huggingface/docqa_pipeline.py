"""
Document Question Answering pipeline implementation for HuggingFace provider.

This pipeline handles document-based question answering tasks including:
- PDF document understanding
- Document layout analysis
- Table question answering
- Form understanding and extraction
"""

import logging
from typing import Optional, Dict, Any, Union, List, Generator, Tuple
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
    LayoutLMv3Processor,
    LayoutLMv3ForQuestionAnswering,
    LayoutLMv3TokenizerFast,
    LayoutLMv3FeatureExtractor,
    AutoModelForDocumentQuestionAnswering,
    PreTrainedModel,
    PreTrainedTokenizer
)
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

from abstractllm.media.interface import MediaInput
from abstractllm.exceptions import ModelLoadError, InvalidInputError, GenerationError
from .model_types import BasePipeline, ModelConfig, ModelCapabilities

# Configure logger
logger = logging.getLogger(__name__)

class DocumentQuestionAnsweringPipeline(BasePipeline):
    """Pipeline for document question answering tasks.
    
    This pipeline supports various document QA tasks:
    - PDF document understanding
    - Document layout analysis
    - Table question answering
    - Form understanding and extraction
    
    It handles both text extraction and generative answers,
    with proper handling of document structure and layout.
    """
    
    # Model architecture mapping
    ARCHITECTURES = {
        "layoutlm": (LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering),
        "donut": (AutoProcessor, AutoModelForDocumentQuestionAnswering)
    }
    
    def __init__(self) -> None:
        """Initialize the Document QA pipeline."""
        super().__init__()
        self.model: Optional[PreTrainedModel] = None
        self.processor = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.feature_extractor = None
        self.model_config: Optional[ModelConfig] = None
        self._model_type: str = ""
        self._max_seq_length: int = 512
        self._supports_tables: bool = False
    
    def load(self, model_name: str, config: ModelConfig) -> None:
        """Load the Document QA model and processor.
        
        Args:
            model_name: Name or path of the model
            config: Model configuration
            
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            # Detect model architecture
            self._model_type = self._detect_architecture(model_name)
            processor_class, model_class = self.ARCHITECTURES.get(
                self._model_type, 
                (AutoProcessor, AutoModelForDocumentQuestionAnswering)
            )
            
            # Load processor/tokenizer
            if self._model_type == "layoutlm":
                self.processor = LayoutLMv3Processor(
                    LayoutLMv3FeatureExtractor(),
                    LayoutLMv3TokenizerFast.from_pretrained(model_name)
                )
                self.tokenizer = self.processor.tokenizer
                self.feature_extractor = self.processor.feature_extractor
            else:
                self.processor = processor_class.from_pretrained(
                    model_name,
                    trust_remote_code=config.trust_remote_code
                )
            
            # Prepare loading kwargs
            load_kwargs = {
                "trust_remote_code": config.trust_remote_code,
                "device_map": config.device_map,
                "torch_dtype": config.torch_dtype
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
            
            # Load model
            self.model = model_class.from_pretrained(
                model_name,
                **load_kwargs
            )
            
            # Move model to device if not using device_map="auto"
            if config.device_map != "auto":
                self.model.to(self.device)
            
            # Get model capabilities
            model_config = self.model.config
            self._max_seq_length = getattr(model_config, "max_position_embeddings", 512)
            self._supports_tables = hasattr(model_config, "has_table_encoder")
            
            self.model_config = config
            self._is_loaded = True
            
            logger.info(f"Loaded Document QA model {model_name} ({self._model_type})")
            logger.debug(f"Max sequence length: {self._max_seq_length}")
            logger.debug(f"Supports tables: {self._supports_tables}")
            
        except Exception as e:
            self.cleanup()
            raise ModelLoadError(f"Failed to load model {model_name}: {str(e)}")
    
    def _detect_architecture(self, model_name: str) -> str:
        """Detect the model architecture from the model name."""
        name_lower = model_name.lower()
        if "layoutlm" in name_lower:
            return "layoutlm"
        elif "donut" in name_lower:
            return "donut"
        else:
            # Default to LayoutLM for unknown architectures
            logger.warning(f"Unknown model architecture for {model_name}, defaulting to LayoutLM")
            return "layoutlm"
    
    def _preprocess_document(self, document_input: MediaInput) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """Preprocess document input (convert PDF to image, perform OCR).
        
        Args:
            document_input: Document input (PDF or image)
            
        Returns:
            Tuple of (processed image, OCR results)
        """
        # Convert document to image if needed
        if document_input.mime_type == "application/pdf":
            # Convert first page of PDF to image
            document_path = document_input.to_provider_format("huggingface")["content"]
            images = convert_from_path(document_path, first_page=1, last_page=1)
            image = images[0]
        else:
            # Use image directly
            image_data = document_input.to_provider_format("huggingface")
            image = Image.open(image_data["content"]) if isinstance(image_data["content"], str) else image_data["content"]
        
        # Perform OCR to get text and bounding boxes
        ocr_results = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Convert OCR results to required format
        words = []
        boxes = []
        for i in range(len(ocr_results["text"])):
            word = ocr_results["text"][i].strip()
            if word:
                words.append(word)
                boxes.append([
                    ocr_results["left"][i],
                    ocr_results["top"][i],
                    ocr_results["left"][i] + ocr_results["width"][i],
                    ocr_results["top"][i] + ocr_results["height"][i]
                ])
        
        return image, {"words": words, "boxes": boxes}
    
    def process(self, 
                inputs: List[MediaInput], 
                generation_config: Optional[Dict[str, Any]] = None,
                stream: bool = False,
                **kwargs) -> Dict[str, Any]:
        """Process document and question to generate answer.
        
        Args:
            inputs: List containing document and question inputs
            generation_config: Optional generation parameters
            stream: Not used for Document QA
            **kwargs: Additional arguments including:
                - max_answer_length: Maximum length of generated answer
                - num_beams: Number of beams for answer generation
                - return_logits: Return raw logits instead of processed answers
                
        Returns:
            Dictionary containing:
            - answer: Generated answer text
            - confidence: Confidence score
            - context: Relevant document context
            - boxes: Bounding boxes for answer in document
            - logits: Optional raw logits if requested
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Extract document and question from inputs
            document_input = None
            question_input = None
            
            for inp in inputs:
                if inp.media_type in ["image", "application/pdf"]:
                    if document_input is not None:
                        raise InvalidInputError("Multiple documents provided, only one is supported")
                    document_input = inp
                elif inp.media_type == "text":
                    if question_input is not None:
                        raise InvalidInputError("Multiple questions provided, only one is supported")
                    question_input = inp
            
            if not document_input or not question_input:
                raise InvalidInputError("Both document and question inputs are required")
            
            # Get question text
            question = question_input.to_provider_format("huggingface")["content"]
            
            # Preprocess document
            image, ocr_results = self._preprocess_document(document_input)
            
            # Get processing parameters
            max_answer_length = kwargs.get("max_answer_length", 50)
            num_beams = kwargs.get("num_beams", 3)
            return_logits = kwargs.get("return_logits", False)
            
            # Process based on model type
            if self._model_type == "layoutlm":
                # Prepare inputs for LayoutLM
                encoding = self.processor(
                    image,
                    ocr_results["words"],
                    ocr_results["boxes"],
                    question,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self._max_seq_length
                )
                
                # Move inputs to device
                encoding = {k: v.to(self.model.device) for k, v in encoding.items()}
                
                # Generate answer
                with torch.no_grad():
                    outputs = self.model(**encoding)
                
                # Get answer span
                start_logits = outputs.start_logits[0]
                end_logits = outputs.end_logits[0]
                
                # Get most likely answer span
                start_idx = torch.argmax(start_logits)
                end_idx = torch.argmax(end_logits[start_idx:]) + start_idx
                
                # Get answer text and boxes
                answer_tokens = encoding["input_ids"][0][start_idx:end_idx + 1]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                answer_boxes = [ocr_results["boxes"][i] for i in range(start_idx, end_idx + 1)]
                
                # Calculate confidence
                confidence = torch.softmax(start_logits, dim=0)[start_idx].item() * \
                           torch.softmax(end_logits, dim=0)[end_idx].item()
                
            else:  # Donut and others
                # Standard document processing
                encoding = self.processor(
                    image,
                    question,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self._max_seq_length
                ).to(self.model.device)
                
                # Generate answer
                with torch.no_grad():
                    outputs = self.model.generate(
                        **encoding,
                        max_length=max_answer_length,
                        num_beams=num_beams,
                        early_stopping=True
                    )
                    answer = self.processor.decode(outputs[0], skip_special_tokens=True)
                    confidence = 1.0  # Donut doesn't provide confidence scores
                    answer_boxes = []  # Donut doesn't provide bounding boxes
            
            # Prepare result
            result = {
                "answer": answer,
                "confidence": confidence,
                "context": " ".join(ocr_results["words"]),
                "boxes": answer_boxes
            }
            
            # Add logits if requested
            if return_logits:
                result["logits"] = {
                    "start_logits": outputs.start_logits.cpu().numpy(),
                    "end_logits": outputs.end_logits.cpu().numpy()
                }
            
            return result
            
        except Exception as e:
            raise GenerationError(f"Document QA failed: {e}")
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        input_types = {"text"}
        if self._model_type == "layoutlm":
            input_types.add("image")
            input_types.add("application/pdf")
        
        return ModelCapabilities(
            input_types=input_types,
            output_types={"text"},
            supports_streaming=False,
            supports_system_prompt=False,
            context_window=self._max_seq_length
        )
    
    def _get_context_window(self) -> Optional[int]:
        """Get the model's context window size."""
        return self._max_seq_length 