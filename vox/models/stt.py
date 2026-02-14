"""Speech-to-text wrapper."""

import asyncio
from collections.abc import AsyncGenerator
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import numpy as np


async def transcribe(stt_model: nn.Module, audio: np.ndarray) -> str:
    """Transcribe audio using STT model."""
    audio_mx = mx.array(audio)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, partial(stt_model.generate, audio_mx))
    return result.text.strip()


async def transcribe_streaming(
    stt_model: nn.Module, audio: np.ndarray
) -> AsyncGenerator[str]:
    """Transcribe audio with streaming output, yielding text chunks."""
    audio_mx = mx.array(audio)
    loop = asyncio.get_running_loop()
    chunks = await loop.run_in_executor(
        None, partial(stt_model.stream_generate, audio_mx)
    )
    for chunk in chunks:
        if chunk.text:
            yield chunk.text
