from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.ai.registry import ModelRegistry
from app.api.router import api_router
import logging

logger = logging.getLogger("accessvision")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize logging configuration
    setup_logging()
    logger.info("Initializing FastAPI accessibility application...")
    
    # 2. Warm up/Pre-load models on startup to avoid latency on first API request
    try:
        ModelRegistry.load_all()
    except Exception as e:
        logger.error(f"Startup warning: Failed to pre-load some models: {str(e)}")
        
    yield
    
    # 3. Clean up and unload models to release system resources/VRAM
    logger.info("Shutting down FastAPI application, clearing registry...")
    ModelRegistry.unload_all()

# Create FastAPI app instance
app = FastAPI(
    title=settings.APP_NAME,
    description="Scalable FastAPI backend structure with modular service layers, "
                "reusable AI inference wrappers, and asynchronous processing.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Telemetry and Request Tracing Middleware
from fastapi import Request
import uuid
import time
from app.core.telemetry import (
    request_telemetry_var,
    RequestTelemetry,
    log_console_trace,
    log_structured,
    check_slow_request
)

@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    req_id = str(uuid.uuid4())[:4]
    telemetry = RequestTelemetry(request_id=req_id, endpoint=request.url.path)
    token = request_telemetry_var.set(telemetry)
    
    try:
        response = await call_next(request)
        
        # Calculate totals
        total_time_ms = (time.perf_counter() - telemetry.start_time) * 1000
        telemetry.record_timing("TOTAL", total_time_ms)
        
        # Add tracing headers
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Total-Duration-MS"] = f"{total_time_ms:.2f}"
        
        # Log console trace & structured logs
        log_console_trace(telemetry)
        log_structured(telemetry)
        check_slow_request(telemetry)
        
        return response
    except Exception as e:
        total_time_ms = (time.perf_counter() - telemetry.start_time) * 1000
        telemetry.record_timing("TOTAL", total_time_ms)
        
        log_console_trace(telemetry)
        log_structured(telemetry)
        
        logger.error(f"Request failed with exception: {str(e)}")
        raise e
    finally:
        request_telemetry_var.reset(token)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register custom global exceptions
register_exception_handlers(app)

# Include central API router with standard prefix
app.include_router(api_router, prefix="/api/v1")

@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirects root URL requests directly to the Swagger interactive documentation."""
    return RedirectResponse(url="/docs")
