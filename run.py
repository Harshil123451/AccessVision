import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    # Start the ASGI server with configuration from app settings
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.APP_ENV == "development",
        log_level="info"
    )
