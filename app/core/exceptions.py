from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("accessvision")

class AppException(Exception):
    """Base exception for all AccessVision application errors."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

class ModelInferenceError(AppException):
    """Exception raised when an AI model fails to perform inference."""
    def __init__(self, message: str):
        super().__init__(message=message, status_code=502)

class InvalidImageError(AppException):
    """Exception raised when an uploaded image is invalid or corrupt."""
    def __init__(self, message: str):
        super().__init__(message=message, status_code=400)

class UnauthorizedError(AppException):
    """Exception raised for authorization failures."""
    def __init__(self, message: str = "Invalid API Key"):
        super().__init__(message=message, status_code=401)

def register_exception_handlers(app: FastAPI) -> None:
    """Registers global exception handlers to map custom exceptions to JSON responses."""
    
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        logger.error(f"App exception: {exc.message} on request {request.url.path}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.__class__.__name__,
                "message": exc.message
            }
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception on request {request.url.path}: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "InternalServerError",
                "message": "An unexpected error occurred on the server."
            }
        )
