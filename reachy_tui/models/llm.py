"""LFM2.5 streaming LLM wrapper."""

import asyncio
from collections.abc import AsyncIterator

import mlx.nn as nn
from mlx_lm import stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper

from reachy_tui.config import settings

type ChatMessage = dict[str, str]


def _format_prompt(
    tokenizer: TokenizerWrapper,
    user_input: str,
    history: list[ChatMessage] | None = None,
) -> str:
    """Format user input into a chat prompt with optional history."""
    messages: list[ChatMessage] = [
        {"role": "system", "content": settings.system_prompt}
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


async def generate_streaming(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    user_input: str,
    max_tokens: int = settings.llm_max_tokens,
    history: list[ChatMessage] | None = None,
) -> AsyncIterator[str]:
    """Async generator that yields tokens as they are generated."""
    prompt = _format_prompt(tokenizer, user_input, history)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()

    def _produce() -> None:
        """Run sync stream_generate and push tokens to the queue."""
        try:
            for chunk in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, chunk.text)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        else:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, _produce)

    while True:
        token = await queue.get()
        if token is None:
            break
        if isinstance(token, Exception):
            raise token
        yield token
