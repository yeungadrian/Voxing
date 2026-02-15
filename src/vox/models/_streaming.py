"""Async bridge for synchronous iterators."""

import asyncio
from collections.abc import AsyncGenerator, Callable, Iterator


async def sync_to_async_iter[T](
    fn: Callable[..., Iterator[T]],
    *args: object,
    **kwargs: object,
) -> AsyncGenerator[T]:
    """Bridge a synchronous iterator into an async generator via a thread executor."""
    sentinel = object()
    iterator = await asyncio.to_thread(fn, *args, **kwargs)
    while (item := await asyncio.to_thread(next, iterator, sentinel)) is not sentinel:
        yield item
