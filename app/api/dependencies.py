from fastapi import Header
from app.core.config import settings
from app.core.exceptions import UnauthorizedError
import logging

logger = logging.getLogger("accessvision")

async def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")) -> str:
    """Dependency that validates requests contain a valid X-API-Key header.
    
    This is active when settings.API_KEY_SECRET is not empty or defaults are changed.
    """
    # In development / mock mode, if secret is default, we can bypass or print a warning
    if not settings.API_KEY_SECRET or settings.API_KEY_SECRET == "development-secret-key":
        return "bypassed"

    if x_api_key != settings.API_KEY_SECRET:
        logger.warning(f"Unauthorized API access attempt with key: '{x_api_key}'")
        raise UnauthorizedError("Invalid or missing API key in headers.")
        
    return x_api_key
