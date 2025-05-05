import torch
import numpy as np

from transformers import TrainingArguments
from transformers import AutoTokenizer, PreTrainedTokenizer
import random

from ml.data.dataset import TextDataset
from ml.models import get_model
from ml.utils.log_utils import logger

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer # Supervised Fine-tuning Trainer - simplifies instruction tuning
from ml.utils.evaluate_utils import compute_metrics

# Configuration constants
TOKENIZER_KWARGS = {
    "bert-base-uncased": {"padding": True, "truncation": True, "max_length": 512},
    "microsoft/Phi-3-mini-4k-instruct" : {"return_attention_mask":True, "truncation": True, "padding": True}
}
CLASSES = ["Non-Cancer", "Cancer"]
label2id = {clas:i for i, clas in enumerate(CLASSES)}

class Engine:
    """Class to handle finetuning and evaluation execution and tracking"""
    
    def __init__(self, cfg):
        """
        Initialize the finetuning runner.
        
        Args:
            cfg: config for the finetuning
        """
        self.cfg = cfg
        self.setup_experiment()
        self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.MODEL.NAME, trust_remote_code=True
            )

        self.dataset = TextDataset(
            path=self.cfg.DATA.PATH,
            cfg=self.cfg,
            tokenizer=self.tokenizer
        )

        # Create model
        self.model = get_model(cfg=self.cfg.MODEL, tokenizer=self.tokenizer, train=cfg.PROJECT.TRAIN)
        self.output_dir = f"checkpoints/{self.cfg.PROJECT.RUN_NAME}"

    def setup_experiment(self) -> None:
        """Set up the experiment environment."""
        # Set random seeds
        random.seed(self.cfg.PROJECT.SEED)
        np.random.seed(self.cfg.PROJECT.SEED)
        torch.manual_seed(self.cfg.PROJECT.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.PROJECT.SEED)
        

    def create_trainer(self, dataset: TextDataset) -> SFTTrainer:
        """Create the trainer instance."""
        self.model = self.model.model

        prepare_model_for_kbit_training(self.model)
        tokenized_dataset, data_collator = dataset.get_dataset()
        self.model.config.use_cache = False

        args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="steps",
            do_eval=True,
            optim="adamw_torch",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=8,
            log_level="debug",
            logging_steps=10,
            learning_rate=1e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            eval_steps=10,
            num_train_epochs=3,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            report_to="wandb",
            seed=42,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_strategy="epoch",
        )

        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
        )

        """Apply LoRA to the model if enabled."""

        self.model = get_peft_model(self.model, peft_config)
        logger.info("Successfully applied LoRA to the model")
        logger.info(f"Trainable parameters: {self.model.print_trainable_parameters()}")  # See how few parameters are being trained

        return SFTTrainer(
        model=self.model,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=self.tokenizer,
        args=args,
)

    
    def finetune(self) -> None:
        """Run the complete finetuning process."""
        try:
            
            # Create trainer
            trainer = self.create_trainer(self.dataset)
            
            # Train the model
            logger.info("Starting model training...")
            trainer.train()
            trainer.save_model(self.output_dir)
            
            # Evaluate the model
            eval_results = trainer.evaluate()
            logger.info(f"Evaluation results: {eval_results}")

        except Exception as e:
            logger.error(f"Finetuning failed: {str(e)}")
            raise e

    def predict(self, prompt:str=None):
        """Make a prediction using the model."""
        return self.model.classify_article(prompt=prompt)


    def evaluate(self, inputs:list[str]=None) -> None:
        """Evaluate the model on the provided inputs or test dataset."""
        if inputs is None:
            logger.info("No input provided, Choosing from test dataset")

            # Ensure get_test_dataset returns list of dicts like [{"text": "...", "label": "..."}]
            inputs = self.dataset.get_test_dataset()
            if not inputs:
                logger.error("Test dataset is empty. Cannot evaluate.")
                return

        pred = []
        gt = []
        for _input in tqdm(inputs, desc="Evaluating", total=len(inputs)):
            json_output = self.predict(_input["text"])
            gt.append(_input['label'])
            pred.append(json_output['predicted_labels'][0])

        acc = accuracy_score(gt, pred)
        cancer_f1 = f1_score(gt, pred, pos_label="Cancer")
        not_cancer_f1 = f1_score(gt, pred, pos_label="Non-Cancer")
        cm = confusion_matrix(gt, pred)  # Use defined labels for order
        logger.info(f"Computed : {acc}, {cancer_f1}, {not_cancer_f1}")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title('Confusion Matrix')

        # Save before showing
        plt.savefig("confusion_matrix.png")
        plt.show()
