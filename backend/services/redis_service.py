import redis
from typing import Optional, Dict, Any
import json
from core.config import get_settings

class RedisService:
    """Service for handling Redis operations."""
    
    def __init__(self):
        """Initialize Redis service."""
        settings = get_settings()
        self.host = settings.REDIS_HOST
        self.port = settings.REDIS_PORT
        self.db = settings.REDIS_DB
        self.client = None
        self.stream_key = settings.REDIS_STREAM_KEY
        self.max_stream_length = settings.REDIS_MAX_STREAM_LENGTH

    def connect(self) -> bool:
        """Connect to Redis server.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.client = redis.Redis(host=self.host, port=self.port, db=self.db)
            self.client.ping()  # Test connection
            return True
        except redis.exceptions.ConnectionError as e:
            print(f"Cannot connect to Redis: {str(e)}")
            return False

    def publish_prediction(self, prediction_id: str, result: Dict[str, Any]) -> bool:
        """Publish prediction result to Redis stream.
        
        Args:
            prediction_id (str): Unique identifier for the prediction
            result (Dict[str, Any]): Prediction result to publish
            
        Returns:
            bool: True if publish successful, False otherwise
        """
        if not self.client:
            print("Cannot publish prediction: Redis not connected")
            return False

        try:
            # Add message to stream with max length constraint
            self.client.xadd(
                self.stream_key,
                {
                    "prediction_id": prediction_id,
                    "result": json.dumps(result),
                    "status": "completed"
                },
                maxlen=self.max_stream_length,
                approximate=True
            )
            print(f"Published prediction {prediction_id} to Redis stream")
            return True
        except Exception as e:
            print(f"Failed to publish prediction: {str(e)}")
            return False

    def get_prediction(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Get prediction result from Redis stream.
        
        Args:
            prediction_id (str): Unique identifier for the prediction
            
        Returns:
            Optional[Dict[str, Any]]: Prediction result if found, None otherwise
        """
        if not self.client:
            return None

        try:
            # Get all messages from stream
            messages = self.client.xrange(self.stream_key)
            for msg_id, msg_data in messages:
                if msg_data.get(b"prediction_id", b"").decode() == prediction_id:
                    return json.loads(msg_data.get(b"result", b"{}").decode())
            return None
        except Exception as e:
            print(f"Failed to get prediction: {str(e)}")
            return None 