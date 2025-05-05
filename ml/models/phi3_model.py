from typing import Literal, Dict, Any, Optional
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig, PreTrainedTokenizer,
)
import json
import re

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from peft import PeftModel, PeftConfig

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

class ClassificationResults(BaseModel):
    predicted_labels: list[Literal["Cancer", "Non-Cancer"]] = Field(
        ...,
        description="List of applicable labels identified. Should include 'Cancer' and/or 'Non-Cancer'."
    )
    scores: dict[Literal["Cancer", "Non-Cancer"], float] = Field(
        ...,
        description="Dictionary containing confidence scores (0.0-1.0) for 'Cancer' and/or 'Non-Cancer' labels."
    )

class CancerTypes(BaseModel):
    abstract_id: int = Field(..., description="Abstract id passed along with prompt defaults to None")
    extracted_diseases: list[str] = Field(
        ...,
        description="List of types of cancers discussed in the abstract example Lung Cancer, Breast Cancer, Blood Cancer, Kidney Cancer etc."
    )

class PHI3Model:
    MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

    def __init__(self, cfg, train: bool = False, tokenizer: PreTrainedTokenizer = None):
        self.cfg = cfg

        if self.cfg.LORA_ADAPTER is not None:
            peft_config = PeftConfig.from_pretrained(self.cfg.LORA_ADAPTER)
            base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map="auto", trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = PeftModel.from_pretrained(base_model, self.cfg.LORA_ADAPTER)
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            )
            if tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
            else:
                self.tokenizer = tokenizer

            self.tokenizer.pad_token = self.tokenizer.eos_token  # Use EOS token for padding
            self.tokenizer.padding_side = "right"  # Pad right for Causal LM

            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",  # Automatically places layers across devices (mainly GPU)
                trust_remote_code=True,
            )

        if train:
            self.model.train()
        else:
            self.model.eval()


    @torch.no_grad()
    def _predict(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a response from the model based on the input prompt.

        Args:
            prompt: Input text to classify

        Returns:
            str: Generated text
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,  # Max tokens for the JSON output (e.g., ["Cancer"])
            tokenizer=self.tokenizer,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            stop_strings=["}}", "}\n}", "[/INST]", "</s>"],
            do_sample=False,
        )

        # Decode the generated tokens, skipping the prompt part
        generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        return json.loads(generated_text)


    def classify_article(self, prompt: str) -> Dict[str, Any]:
        """
        Classify the input article.

        Args:
            prompt: Input text to classify

        Returns:
            Dict containing predicted labels and scores
        """
        pydantic_parser = PydanticOutputParser(pydantic_object=ClassificationResults)
        format_instructions = pydantic_parser.get_format_instructions()

        prompt = INFERENCE_PROMPT.format(abstract=prompt, json_format=format_instructions)

        return self._predict(prompt)

    def predict_cancer(self, prompt: str) -> Dict[str, Any]:
        """
        Predict cancer types from the input article.

        Args:
            prompt: Input text to classify

        Returns:
            Dict containing extracted diseases
        """
        pydantic_parser = PydanticOutputParser(pydantic_object=CancerTypes)
        format_instructions = pydantic_parser.get_format_instructions()

        prompt = CANCER_TYPE_PROMPT.format(abstract=prompt, json_format=format_instructions)

        return self._predict(prompt)

