from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class TextInput(BaseModel):
    """Input model for text-based predictions."""
    text: str

class PredictionResponse(BaseModel):
    """Response model for prediction requests."""
    prediction_id: str
    message: str

class CancerClassificationResponse(BaseModel):
    """Response model for cancer classification predictions."""
    prediction_id: str
    predicted_labels: List[str]
    scores: Dict[str, float]

class CancerTypesResponse(BaseModel):
    """Response model for cancer type predictions."""
    prediction_id: str
    abstract_id: int
    extracted_diseases: List[str] 