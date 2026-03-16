# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from __future__ import annotations

import json
import logging
import re
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from voxing._download import _resolve_model_path
from voxing.chatterbox._base import GenerationResult
from voxing.chatterbox.models.s3gen import S3GEN_SIL, S3GEN_SR, S3Gen
from voxing.chatterbox.models.t3 import T3, T3Cond, T3Config

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def punc_norm(text: str) -> str:
    """Normalize punctuation for TTS input."""
    if len(text) == 0:
        return "You need to add some text for me to talk."

    if text[0].islower():
        text = text[0].upper() + text[1:]

    text = " ".join(text.split())

    punc_to_replace = [
        ("\u2026", ", "),
        (":", ","),
        ("\u2014", "-"),
        ("\u2013", "-"),
        (" ,", ","),
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2018", "'"),
        ("\u2019", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """Conditionals for T3 and S3Gen."""

    t3: T3Cond
    gen: dict[str, mx.array]


class ChatterboxTurboTTS(nn.Module):
    """MLX implementation of Chatterbox Turbo TTS."""

    def __init__(
        self,
        config: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.sr = S3GEN_SR
        self.config: dict[str, object] = config or {}
        hp = T3Config.turbo()
        self.t3 = T3(hp)
        self.s3gen = S3Gen(meanflow=True)
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self._conds: Conditionals | None = None

    @property
    def sample_rate(self) -> int:
        """Output sample rate."""
        return self.sr

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize PyTorch weights for MLX."""
        new_weights: dict[str, mx.array] = {}

        t3_weights: dict[str, mx.array] = {}
        s3gen_weights: dict[str, mx.array] = {}

        for key, value in weights.items():
            if key.startswith("ve."):
                continue
            if key.startswith("t3."):
                t3_weights[key[3:]] = value
            elif key.startswith("s3gen."):
                s3gen_weights[key[6:]] = value

        if t3_weights:
            for k, v in t3_weights.items():
                new_weights[f"t3.{k}"] = v

        if s3gen_weights:
            if hasattr(self.s3gen, "sanitize"):
                s3gen_weights = self.s3gen.sanitize(s3gen_weights)
            for k, v in s3gen_weights.items():
                new_weights[f"s3gen.{k}"] = v

        return new_weights

    def _load_weights(
        self,
        weights: list[tuple[str, mx.array]] | dict[str, mx.array],
        strict: bool = True,
    ) -> None:
        """Load weights into the model."""
        if isinstance(weights, dict):
            weights = list(weights.items())

        t3_weights: list[tuple[str, mx.array]] = []
        s3gen_weights: list[tuple[str, mx.array]] = []

        for k, v in weights:
            if k.startswith("t3."):
                t3_weights.append((k[3:], v))
            elif k.startswith("s3gen."):
                s3gen_weights.append((k[6:], v))

        if t3_weights:
            self.t3.load_weights(t3_weights, strict=False)
        if s3gen_weights:
            self.s3gen.load_weights(s3gen_weights, strict=False)

        mx.eval(
            self.t3.parameters(),
            self.s3gen.parameters(),
        )

    def generate(
        self,
        text: str,
        repetition_penalty: float = 1.2,
        top_p: float = 0.95,
        temperature: float = 0.8,
        top_k: int = 1000,
        max_tokens: int = 800,
    ) -> Generator[GenerationResult]:
        """Generate speech from text using pre-loaded conditionals."""
        assert self._conds is not None, "No conditionals loaded"

        text = punc_norm(text)

        max_chars_per_chunk = (max_tokens // 8) * 4
        split_pattern = r"(?<=[.!?])\s+"

        sentences = re.split(split_pattern, text)
        chunks: list[str] = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if (
                current_chunk
                and len(current_chunk) + len(sentence) + 1 > max_chars_per_chunk
            ):
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = (
                    f"{current_chunk} {sentence}" if current_chunk else sentence
                )

        if current_chunk:
            chunks.append(current_chunk.strip())

        chunks = [c for c in chunks if c.strip()]
        if not chunks:
            chunks = [text]

        mx.clear_cache()

        for chunk in chunks:
            if self.tokenizer is not None:
                text_tokens = self.tokenizer(
                    chunk, return_tensors="np", padding=True, truncation=True
                )
                text_tokens = mx.array(text_tokens.input_ids)
            else:
                text_tokens = mx.array([[ord(c) for c in chunk[:512]]])

            speech_tokens = self.t3.inference_turbo(
                t3_cond=self._conds.t3,
                text_tokens=text_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_gen_len=max_tokens,
            )

            mx.clear_cache()

            speech_tokens = speech_tokens.reshape(-1)
            mask = np.where(np.array(speech_tokens) < 6561)[0].tolist()
            speech_tokens = speech_tokens[mask]
            silence = mx.array([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL], dtype=mx.int32)
            speech_tokens = mx.concatenate([speech_tokens, silence])
            speech_tokens = speech_tokens[None, :]

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self._conds.gen,
                n_cfm_timesteps=2,
            )

            if wav.ndim == 2:
                wav = wav.squeeze(0)

            yield GenerationResult(
                audio=wav,
                sample_rate=self.sample_rate,
            )

            mx.clear_cache()


def load_model(model_id: str) -> ChatterboxTurboTTS:
    """Download (if needed) and load a Chatterbox Turbo TTS model."""
    model_path = _resolve_model_path(
        model_id,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
        ],
    )

    config = json.loads((model_path / "config.json").read_text())
    model = ChatterboxTurboTTS(config)

    weight_files = list(model_path.glob("*.safetensors"))
    weights: dict[str, mx.array] = {}
    for wf in weight_files:
        if wf.name == "conds.safetensors":
            continue
        loaded = cast(dict[str, mx.array], mx.load(str(wf)))
        weights.update(loaded)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    model._load_weights(weights)
    model.eval()

    # Load text tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer is None:
        raise RuntimeError(f"Failed to load tokenizer from {model_path}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.tokenizer = tokenizer

    # Load pre-computed conditionals
    conds_path = model_path / "conds.safetensors"
    if not conds_path.exists():
        raise FileNotFoundError("conds.safetensors not found in model directory")

    conds_data = cast(dict[str, mx.array], mx.load(str(conds_path)))

    speaker_emb = conds_data.get("t3.speaker_emb")
    if speaker_emb is None:
        speaker_emb = mx.zeros((1, 256))

    cond_tokens = conds_data.get("t3.cond_prompt_speech_tokens")

    t3_cond = T3Cond(
        speaker_emb=speaker_emb,
        cond_prompt_speech_tokens=cond_tokens,
    )

    gen_mlx: dict[str, mx.array] = {}
    for k, v in conds_data.items():
        if k.startswith("gen."):
            gen_mlx[k.replace("gen.", "")] = v

    model._conds = Conditionals(t3_cond, gen_mlx)

    return model
