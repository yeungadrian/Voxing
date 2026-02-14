"""Whisper speech-to-text wrapper."""

import asyncio
from functools import partial

import mlx.nn as nn
import numpy as np


async def transcribe(stt_model: nn.Module, audio: np.ndarray) -> str:
    """Transcribe audio using Whisper model."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, partial(stt_model.generate, audio))
    return result.text.strip()
