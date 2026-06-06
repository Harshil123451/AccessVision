import asyncio
import functools
from typing import Callable, Any

async def run_in_thread(func: Callable[..., Any], *args: Any, executor: Any = None, **kwargs: Any) -> Any:
    """Backport of asyncio.to_thread for Python 3.8.
    
    Runs a synchronous/blocking function in an executor thread.
    """
    loop = asyncio.get_running_loop()
    if kwargs:
        # run_in_executor does not accept kwargs directly; wrap in partial
        func_wrapped = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, func_wrapped)
    return await loop.run_in_executor(executor, func, *args)
