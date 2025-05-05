from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, Any
import uuid
from asyncio import to_thread
import logging

from ..models.schemas import TextInput, PredictionResponse, CancerClassificationResponse, CancerTypesResponse
from ..services.model_service import ModelService
from ..services.redis_service import RedisService
from ..core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

def get_model_service() -> ModelService:
    """Dependency to get the model service instance."""
    return ModelService()

def get_redis_service() -> RedisService:
    """Dependency to get the Redis service instance."""
    redis_service = RedisService()
    if not redis_service.connect():
        logger.warning("Failed to connect to Redis service")
    return redis_service

async def process_prediction(
    text: str,
    prediction_id: str,
    prediction_type: str,
    model_service: ModelService,
    redis_service: RedisService
) -> None:
    """Process prediction asynchronously and publish results.
    
    Args:
        text (str): Input text to process
        prediction_id (str): Unique identifier for the prediction
        prediction_type (str): Type of prediction to make
        model_service (ModelService): Model service instance
        redis_service (RedisService): Redis service instance
    """
    try:
        response = None
        if prediction_type == 'classify_abstract':
            result = await to_thread(model_service.classify_prompt, text)
            response = CancerClassificationResponse(
                prediction_id=prediction_id,
                predicted_labels=result["predicted_labels"],
                scores=result["scores"]
            )
        elif prediction_type == 'predict_cancer_type':
            result = await to_thread(model_service.predict_cancer_type, text)
            response = CancerTypesResponse(
                prediction_id=prediction_id,
                abstract_id=0,
                extracted_diseases=result["extracted_diseases"]
            )

        if not redis_service.publish_prediction(prediction_id, response.model_dump()):
            raise Exception("Failed to publish prediction result")

    except Exception as e:
        print(f"Prediction processing failed: {str(e)}")
        error_result = {
            "error": str(e),
            "prediction_id": prediction_id
        }
        redis_service.publish_prediction(prediction_id, error_result)

@router.post("/predict_cancer_type", response_model=PredictionResponse)
async def predict_cancer_type(
    input_data: TextInput,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(get_model_service),
    redis_service: RedisService = Depends(get_redis_service)
) -> PredictionResponse:
    """Make a prediction for cancer type.
    
    Args:
        input_data (TextInput): Input text to classify
        background_tasks (BackgroundTasks): FastAPI background tasks
        model_service (ModelService): Model service instance
        redis_service (RedisService): Redis service instance
        
    Returns:
        PredictionResponse: Initial response with prediction ID
    """
    if not redis_service.client:
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable: Redis connection failed"
        )

    try:
        prediction_id = str(uuid.uuid4())
        background_tasks.add_task(
            process_prediction,
            input_data.text,
            prediction_id,
            "predict_cancer_type",
            model_service,
            redis_service
        )

        return PredictionResponse(
            prediction_id=prediction_id,
            message="LLM request submitted successfully. Processing in background."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify_abstract", response_model=PredictionResponse)
async def classify_abstract(
    input_data: TextInput,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(get_model_service),
    redis_service: RedisService = Depends(get_redis_service)
) -> PredictionResponse:
    """Make a prediction for abstract classification.
    
    Args:
        input_data (TextInput): Input text to classify
        background_tasks (BackgroundTasks): FastAPI background tasks
        model_service (ModelService): Model service instance
        redis_service (RedisService): Redis service instance
        
    Returns:
        PredictionResponse: Initial response with prediction ID
    """
    if not redis_service.client:
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable: Redis connection failed"
        )

    try:
        prediction_id = str(uuid.uuid4())
        background_tasks.add_task(
            process_prediction,
            input_data.text,
            prediction_id,
            "classify_abstract",
            model_service,
            redis_service
        )

        return PredictionResponse(
            prediction_id=prediction_id,
            message="LLM request submitted successfully. Processing in background."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check(redis_service: RedisService = Depends(get_redis_service)) -> Dict[str, str]:
    """Health check endpoint.
    
    Args:
        redis_service (RedisService): Redis service instance
        
    Returns:
        Dict[str, str]: Health status
    """
    if not redis_service.client:
        return {"status": "unhealthy", "reason": "Redis connection failed"}
    return {"status": "healthy"} 