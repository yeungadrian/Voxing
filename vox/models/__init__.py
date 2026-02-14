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

logging.getLogger("mlx").setLevel(logging.ERROR)
logging.getLogger("mlx_audio").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*deprecated.*")


@contextlib.contextmanager
def _suppress_output() -> Iterator[None]:
    """Redirect stdout/stderr to devnull and disable tqdm during model loading."""
    os.environ["TQDM_DISABLE"] = "1"
    devnull = Path(os.devnull).open("w")  # noqa: SIM115
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull, devnull
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        devnull.close()
        os.environ.pop("TQDM_DISABLE", None)


@dataclass(slots=True)
class Models:
    """Container for all loaded models."""

    stt: nn.Module
    llm: nn.Module
    tts: nn.Module
    tokenizer: TokenizerWrapper


def load_stt(model_name: str | None = None) -> nn.Module:
    """Load the speech-to-text model."""
    with _suppress_output():
        return load_stt_model(model_name or settings.stt_model)


def load_llm(model_name: str | None = None) -> tuple[nn.Module, TokenizerWrapper]:
    """Load the LLM model and tokenizer."""
    with _suppress_output():
        model, tokenizer = load(model_name or settings.llm_model)  # ty:ignore[invalid-assignment]
    return model, tokenizer


def load_tts(model_name: str | None = None) -> nn.Module:
    """Load the text-to-speech model."""
    with _suppress_output():
        return load_tts_model(model_name or settings.tts_model)  # ty:ignore[invalid-argument-type]
