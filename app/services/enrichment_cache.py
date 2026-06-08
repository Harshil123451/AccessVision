import time
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("accessvision")

class EnrichmentCacheEntry:
    """Represents a cached enrichment item with creation timestamp and TTL."""
    def __init__(self, key: str, value: Any, ttl: float = 600.0):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl

class BackgroundEnrichmentCache:
    """Session-scoped enrichment cache.
    Stores Florence captions, OCR results, VQA context, and scene summaries.
    Provides automatic cleanup, session isolation, and telemetry metrics.
    """
    def __init__(self):
        # session_id -> { key -> EnrichmentCacheEntry }
        self._cache: Dict[str, Dict[str, EnrichmentCacheEntry]] = {}
        self.hits = 0
        self.misses = 0

    def set(self, session_id: Optional[str], key: str, value: Any, ttl: float = 600.0):
        """Sets a value in the cache for the specified session."""
        self.cleanup()
        sid = session_id or "global_default"
        if sid not in self._cache:
            self._cache[sid] = {}
        self._cache[sid][key] = EnrichmentCacheEntry(key, value, ttl)
        logger.debug(f"[CACHE] Set key '{key}' for session '{sid}' (TTL: {ttl}s)")

    def get(self, session_id: Optional[str], key: str) -> Optional[Any]:
        """Retrieves a cached value if present and not expired."""
        self.cleanup()
        sid = session_id or "global_default"
        if sid not in self._cache:
            self.misses += 1
            return None
        
        entry = self._cache[sid].get(key)
        if entry is None or entry.is_expired():
            if entry:
                self._cache[sid].pop(key, None) # Remove expired entry
            self.misses += 1
            return None
            
        self.hits += 1
        logger.info(f"[TELEMETRY] Enrichment cache HIT for session '{sid}', key '{key}'. (Hits: {self.hits}, Misses: {self.misses})")
        return entry.value

    def cleanup(self):
        """Removes expired entries from the cache to release memory."""
        now = time.time()
        expired_sessions = []
        
        for sid, entries in list(self._cache.items()):
            # Remove expired entries
            expired_keys = [k for k, entry in entries.items() if entry.is_expired()]
            for k in expired_keys:
                entries.pop(k, None)
                
            # Track empty sessions for complete deletion
            if not entries:
                expired_sessions.append(sid)
                
        for sid in expired_sessions:
            self._cache.pop(sid, None)

# Global singleton instance of the cache
enrichment_cache = BackgroundEnrichmentCache()
