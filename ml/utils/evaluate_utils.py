import evaluate
from typing import Dict, Tuple
import numpy as np
from ml.utils.log_utils import logger


# Initialize metrics
f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")
confusion_metric = evaluate.load("confusion_matrix")
perplexity = evaluate.load("perplexity")

def compute_metrics(eval_pred) -> Dict[str, float]:
    logger.info(eval_pred)
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "f1": f1.compute(predictions=predictions, references=labels)["f1"],
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
    }

def confusion_matrix(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "confusion_matrix": confusion_metric.compute(predictions=predictions, references=labels)["confusion_matrix"],
    }
