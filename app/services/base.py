import time
from contextlib import contextmanager
from typing import Generator
import logging

logger = logging.getLogger("accessvision")

class BaseService:
    """Base class for service layers providing shared telemetry and utility functions."""
    
    @contextmanager
    def measure_latency(self) -> Generator[dict, None, None]:
        """A context manager to calculate execution latency of services.
        
        Usage:
            with self.measure_latency() as metrics:
                # perform task
            print(metrics["latency_ms"])
        """
        metrics = {"latency_ms": 0.0}
        start_time = time.perf_counter()
        try:
            yield metrics
        finally:
            end_time = time.perf_counter()
            metrics["latency_ms"] = round((end_time - start_time) * 1000, 2)
            logger.debug(f"Service completed operation in {metrics['latency_ms']} ms")
