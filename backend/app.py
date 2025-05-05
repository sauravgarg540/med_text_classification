from fastapi import FastAPI
from api.endpoints import router
from services.redis_service import RedisService

def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application
    """
    app = FastAPI(title="Text Classification API")
    
    # Initialize Redis service
    redis_service = RedisService()
    if not redis_service.connect():
        print("Warning: Redis connection failed. Some features may be unavailable.")
    
    # Include API routes
    app.include_router(router)
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
