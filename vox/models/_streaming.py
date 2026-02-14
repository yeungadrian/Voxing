"""Async bridge for synchronous iterators."""

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator

_SENTINEL = object()


async def sync_to_async_iter[T](
    fn: Callable[..., Iterator[T]],
    *args: object,
    **kwargs: object,
) -> AsyncIterator[T]:
    """Bridge a synchronous iterator into an async iterator via a thread executor.

    Runs fn(*args, **kwargs) in a thread, pushing each yielded item through an
    asyncio.Queue so the caller can consume them with ``async for``.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[T | BaseException | object] = asyncio.Queue()

    def _produce() -> None:
        """Run the sync iterator and push items onto the queue."""
        try:
            for item in fn(*args, **kwargs):
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    loop.run_in_executor(None, _produce)

    while True:
        item = await queue.get()
        if item is _SENTINEL:
            break
        if isinstance(item, BaseException):
            raise item
        yield item
