import os
import re
from typing import Literal, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import logging
from functools import lru_cache
from peft import PeftModel, PeftConfig
from langchain.output_parsers import PydanticOutputParser
import json
from core.config import get_settings

logger = logging.getLogger(__name__)

# Prompt templates
INFERENCE_PROMPT = """<s>[INST]
    - KEEP YOUR ANSWER SIMPLE AND OUTPUT ONLY WHAT IS ASKED.
    - DO NOT HALLUCINATE OR TELL MORE THAN ASKED.

    You are a Multi-Label Classifier. Analyze the following abstract. 
    Identify whether it is about 'Cancer' and/or 'Non-Cancer'.
    You also need to prove your result with probabilities of all classes.
    Since it is a multi-label classification, the sum of probabilities my not sum to 1

    Abstract: {abstract}

    The output should be formatted as a JSON instance that conforms to the JSON schema below.

    {json_format}

    Classification Labels: (JSON Output)[/INST]
    </s>"""

CANCER_TYPE_PROMPT = """<s>[INST]
    - KEEP YOUR ANSWER SIMPLE AND OUTPUT ONLY WHAT IS ASKED.
    - DO NOT HALLUCINATE OR TELL MORE THAN ASKED.

    Analyze the following abstract. 
    Identify whether it is about 'Cancer' and/or 'Non-Cancer'. If the abstract is about cancer output the types of cancer discussed.

    Abstract: {abstract}

    The output should be formatted as a JSON instance that conforms to the JSON schema below.

    {json_format}

    Classification Labels <json>JSON Output</json>: [/INST]
    </s>"""

class CancerClassificationResults(BaseModel):
    predicted_labels: list[Literal["Cancer", "Non-Cancer"]] = Field(
        ...,
        description="List of applicable labels identified. Should include 'Cancer' and/or 'Non-Cancer'."
    )
    scores: dict[Literal["Cancer", "Non-Cancer"], float] = Field(
        ...,
        description="Dictionary containing confidence scores (0.0-1.0) for 'Cancer' and/or 'Non-Cancer' labels."
    )

class CancerTypes(BaseModel):
    abstract_id: int = Field(default=None, description="Abstract id passed along with prompt defaults to None")
    extracted_diseases: list[str] = Field(
        ...,
        description="List of types of cancers discussed in the abstract example Lung Cancer, Breast Cancer, Blood Cancer, Kidney Cancer etc."
    )

CLASSES = ["No Cancer", "Cancer"]

class ModelService:
    """Singleton class for model inference."""
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance is created."""
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
        return cls._instance
    
    
    def __init__(self):
        """Initialize the model service if not already initialized."""
        if not self._initialized:
            settings = get_settings()
            self.model_path = settings.MODEL_PATH
            if not self.model_path:
                raise ValueError("MODEL_PATH environment variable is not set")
            
            # Initialize model here
            self._load_model()
            self._initialized = True

    def _load_model(self) -> None:
        """Load the model and tokenizer from a specific path."""
        try:
            logger.info(f"Loading model from {model_path}")

            # Check if model path exists
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model path {self.model_path} does not exist")

            # Load tokenizer
            peft_config = PeftConfig.from_pretrained(self.model_path)
            base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map="auto", trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()

            logger.info("Model and tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    @lru_cache(maxsize=1000)
    def _tokenize(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Tokenize input text with caching."""
        return self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.model.device)

    def predict(self, prompt: str) -> Dict[str, Any]:
        """Generate prediction from the model."""
        inputs = self._tokenize(prompt)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,  # Max tokens for the JSON output (e.g., ["Cancer"])
            tokenizer=self.tokenizer,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,  # Set pad_token_id
            do_sample=False,
            stop_strings=["}}", "}\n}", "[/INST]"],
        )

        # Decode the generated tokens, skipping the prompt part
        generated_ids = outputs[0, inputs['input_ids'].shape[1]:]  # Get only generated tokens
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        generated_text = generated_text.replace("[/INST]", "")
        return json.loads(generated_text)

    @torch.no_grad()
    def classify_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Make a prediction for the input text.

        Args:
            prompt: Input text to classify

        Returns:
            Dict containing predicted labels and scores
        """
        try:
            # Ensure model is loaded
            if not self.model or not self.tokenizer:
                raise RuntimeError("Model not loaded")

            pydantic_parser = PydanticOutputParser(pydantic_object=CancerClassificationResults)
            format_instructions = pydantic_parser.get_format_instructions()

            prompt = INFERENCE_PROMPT.format(abstract=prompt, json_format=format_instructions)

            return self.predict(prompt)

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    @torch.no_grad()
    def predict_cancer_type(self, prompt: str) -> Dict[str, Any]:
        """
        Make a prediction for the input text.

        Args:
            prompt: Input text to classify

        Returns:
            Dict containing extracted diseases
        """
        try:
            # Ensure model is loaded
            if not self.model or not self.tokenizer:
                raise RuntimeError("Model not loaded")

            pydantic_parser = PydanticOutputParser(pydantic_object=CancerTypes)
            format_instructions = pydantic_parser.get_format_instructions()

            prompt = CANCER_TYPE_PROMPT.format(abstract=prompt, json_format=format_instructions)
            return self.predict(prompt)

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise