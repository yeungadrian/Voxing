"""Model loading and initialization."""

import contextlib
import logging
import os
import sys
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import mlx.nn as nn
from mlx_audio.stt import load_model as load_stt_model
from mlx_audio.tts.utils import load_model as load_tts_model
from mlx_lm import load
from mlx_lm.tokenizer_utils import TokenizerWrapper

from vox.config import settings

for _name in ("mlx", "mlx_lm", "mlx_audio"):
    logging.getLogger(_name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*deprecated.*")


@contextlib.contextmanager
def suppress_output() -> Iterator[None]:
    """Redirect stdout/stderr to devnull."""
    devnull = Path(os.devnull).open("w")  # noqa: SIM115
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull, devnull
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        devnull.close()


@dataclass(slots=True)
class Models:
    """Container for all loaded models."""

    stt: nn.Module
    llm: nn.Module
    tts: nn.Module
    tokenizer: TokenizerWrapper


def load_stt(model_name: str | None = None) -> nn.Module:
    """Load the speech-to-text model."""
    with suppress_output():
        return load_stt_model(model_name or settings.stt_model)


def load_llm(model_name: str | None = None) -> tuple[nn.Module, TokenizerWrapper]:
    """Load the LLM model and tokenizer."""
    with suppress_output():
        model, tokenizer = load(model_name or settings.llm_model)  # ty:ignore[invalid-assignment]
    return model, tokenizer


def load_tts(model_name: str | None = None) -> nn.Module:
    """Load the text-to-speech model."""
    with suppress_output():
        return load_tts_model(model_name or settings.tts_model)  # ty:ignore[invalid-argument-type]
