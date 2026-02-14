"""LFM2.5 streaming LLM wrapper."""

from collections.abc import AsyncGenerator

import mlx.nn as nn
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from vox.config import settings
from vox.models._streaming import sync_to_async_iter

type ChatMessage = dict[str, str]


def count_tokens(tokenizer: TokenizerWrapper, text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(tokenizer.encode(text))


def count_conversation_tokens(
    tokenizer: TokenizerWrapper,
    history: list[ChatMessage],
    system_prompt: str = settings.system_prompt,
) -> int:
    """Count total tokens in conversation history including system prompt."""
    messages: list[ChatMessage] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )
    return count_tokens(tokenizer, prompt)


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
) -> AsyncGenerator[str]:
    """Async generator that yields tokens as they are generated."""
    prompt = _format_prompt(tokenizer, user_input, history)
    sampler = make_sampler(temp=0.1, top_k=50, top_p=0.1)
    logits_processors = make_logits_processors(repetition_penalty=1.05)

    async for chunk in sync_to_async_iter(
        stream_generate,
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
    ):
        yield chunk.text
