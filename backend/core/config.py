from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings."""
    
    # API settings
    API_TITLE: str = "Text Classification API"
    API_VERSION: str = "1.0.0"
    
    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_STREAM_KEY: str = "predictions"
    REDIS_MAX_STREAM_LENGTH: int = 1000
    
    # Model settings
    MODEL_PATH: Optional[str] = os.getenv("MODEL_PATH")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings() 