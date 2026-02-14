"""Model loading and initialization."""

import logging
import warnings
from dataclasses import dataclass

import mlx.nn as nn
from mlx_audio.stt import load_model as load_stt_model
from mlx_audio.tts.utils import load_model as load_tts_model
from mlx_lm import load
from mlx_lm.tokenizer_utils import TokenizerWrapper

logging.getLogger("mlx").setLevel(logging.ERROR)
logging.getLogger("mlx_audio").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*deprecated.*")


@dataclass(slots=True)
class Models:
    """Container for all loaded models."""

    stt: nn.Module
    llm: nn.Module
    tts: nn.Module
    tokenizer: TokenizerWrapper


def load_models() -> Models:
    """Load and warm up all models."""
    stt_model = load_stt_model("mlx-community/parakeet-tdt-0.6b-v3")
    llm_model, tokenizer = load("LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit")  # ty:ignore[invalid-assignment]
    tts_model = load_tts_model("mlx-community/Kokoro-82M-bf16")  # ty:ignore[invalid-argument-type]

    return Models(stt=stt_model, llm=llm_model, tts=tts_model, tokenizer=tokenizer)
