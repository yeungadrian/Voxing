# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import json
import logging
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import (
    apply_min_p,
    apply_top_k,
    apply_top_p,
    categorical_sampling,
)

from voxing.parakeet._dsp import mel_filters, stft
from voxing.qwen3_tts._base import GenerationResult
from voxing.qwen3_tts.config import (
    ModelConfig,
    Qwen3TTSTalkerConfig,
    Qwen3TTSTokenizerConfig,
    Qwen3TTSTokenizerDecoderConfig,
    Qwen3TTSTokenizerEncoderConfig,
)
from voxing.qwen3_tts.speaker_encoder import Qwen3TTSSpeakerEncoder
from voxing.qwen3_tts.speech_tokenizer import Qwen3TTSSpeechTokenizer
from voxing.qwen3_tts.talker import Qwen3TTSTalkerForConditionalGeneration

logger = logging.getLogger(__name__)


def mel_spectrogram(
    audio: mx.array,
    n_fft: int = 1024,
    num_mels: int = 128,
    sample_rate: int = 24000,
    hop_size: int = 256,
    win_size: int = 1024,
    fmin: float = 0.0,
    fmax: float = 12000.0,
) -> mx.array:
    """Compute mel spectrogram from audio waveform."""
    if audio.ndim == 1:
        audio = audio[None, :]

    batch_size, _ = audio.shape

    # Get mel filterbank from shared DSP module (cached)
    mel_basis = mel_filters(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=num_mels,
        f_min=fmin,
        f_max=fmax,
        norm="slaney",
        mel_scale="slaney",
    )

    # Compute STFT for each sample in batch
    mels = []
    padding = (n_fft - hop_size) // 2
    for i in range(batch_size):
        # Manual reflect padding to match the PyTorch reference
        # (center=False with manual pad).
        sample = audio[i]
        left_pad = sample[1 : padding + 1][::-1]
        right_pad = sample[-(padding + 1) : -1][::-1]
        sample = mx.concatenate([left_pad, sample, right_pad])

        spec = stft(
            sample,
            n_fft=n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window="hann",
            center=False,
            pad_mode="reflect",
        )
        # Get magnitude spectrum (with epsilon for numerical stability)
        spec_mag = mx.sqrt(mx.abs(spec) ** 2 + 1e-9)

        # Apply mel filterbank: spec_mag is [frames, n_fft//2+1]
        # and mel_basis is [n_mels, n_fft//2+1].
        mel = mx.matmul(spec_mag, mel_basis.T)

        # Log scale
        mel = mx.log(mx.clip(mel, 1e-5, None))
        mels.append(mel)

    return mx.stack(mels, axis=0)  # [batch, frames, n_mels]


def check_array_shape_qwen3(arr: mx.array) -> bool:
    """Check if Conv1d weights are already in MLX format.

    MLX format: (out_channels, kernel_size, in_channels)
    PyTorch format: (out_channels, in_channels, kernel_size)
    """
    shape = arr.shape
    if len(shape) != 3:
        return False

    out_channels, dim2, dim3 = shape

    if dim2 == 1:
        # Pattern: (out, 1, dim3)
        # dim3 is large, likely in_channels -> MLX format (out, kernel=1, in)
        # dim3 is small, likely kernel -> PyTorch format (out, in=1, kernel)
        return dim3 > 64
    if dim3 == 1:
        # Pattern: (out, dim2, 1)
        # dim2 is large, likely in_channels -> PyTorch format (out, in, kernel=1)
        # dim2 is small, likely kernel -> MLX format (out, kernel, in=1)
        return dim2 <= 64

    # General heuristic: kernel_size < in_channels is more common
    # So if middle dimension is smaller, it's likely already MLX format
    return dim2 < dim3


def format_duration(seconds: float) -> str:
    """Format duration as HH:MM:SS.mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._sample_rate = config.sample_rate
        talker_config = config.talker_config

        # Main talker model
        self.talker = Qwen3TTSTalkerForConditionalGeneration(talker_config)

        # Speaker encoder (only for base models that support voice cloning)
        if config.tts_model_type == "base":
            self.speaker_encoder = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)
        else:
            self.speaker_encoder = None

        # Speech tokenizer (loaded separately)
        self.speech_tokenizer: Qwen3TTSSpeechTokenizer | None = None

        # Text tokenizer (loaded in post_load_hook)
        self.tokenizer: Any | None = None

        # Generation config
        self.generate_config: dict[str, Any] | None = None

        # Supported speakers and languages from config
        self.supported_speakers = (
            list(talker_config.spk_id.keys()) if talker_config.spk_id else []
        )
        self.supported_languages = ["auto"]
        if talker_config.codec_language_id:
            for lang_id in talker_config.codec_language_id:
                if "dialect" not in lang_id:
                    self.supported_languages.append(lang_id)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def model_type(self) -> str:
        return "qwen3_tts"

    def load_speech_tokenizer(self, speech_tokenizer: Qwen3TTSSpeechTokenizer):
        """Load the speech tokenizer model."""
        self.speech_tokenizer = speech_tokenizer

    @property
    def _speech_tokenizer(self) -> Qwen3TTSSpeechTokenizer:
        if self.speech_tokenizer is None:
            raise ValueError("Speech tokenizer not loaded")
        return self.speech_tokenizer

    @property
    def _tokenizer(self) -> Any:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call post_load_hook first.")
        return self.tokenizer

    def load_generate_config(self, generate_config: dict):
        """Load generation configuration."""
        self.generate_config = generate_config

    def get_supported_speakers(self) -> list[str]:
        """Get list of supported speaker names."""
        return self.supported_speakers

    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes."""
        return self.supported_languages

    def model_quant_predicate(self, path: str, module) -> bool:

        skip_patterns = [
            "codec_embedding",
            "text_embedding",
            "speech_tokenizer",
            "speaker_encoder",
        ]
        return not any(pattern in path for pattern in skip_patterns)

    def extract_speaker_embedding(
        self,
        audio: mx.array,
        sr: int = 24000,
    ) -> mx.array:
        """Extract speaker embedding from reference audio.

        Args:
            audio: Audio waveform [samples]
            sr: Sample rate (must be 24000)

        Returns:
            Speaker embedding [1, enc_dim]
        """
        if sr != 24000:
            raise ValueError(
                "Only 24kHz audio is supported for speaker embedding extraction"
            )

        if self.speaker_encoder is None:
            raise ValueError("Speaker encoder not available for this model type")

        # Compute mel spectrogram
        mels = mel_spectrogram(
            audio,
            n_fft=1024,
            num_mels=128,
            sample_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        )  # [batch, time, mels]
        mx.eval(mels)

        # Extract embedding
        speaker_embedding = self.speaker_encoder(mels)
        mx.eval(speaker_embedding)

        return speaker_embedding

    def _prepare_generation_inputs(
        self,
        text: str,
        language: str = "auto",
        speaker: str | None = None,
        ref_audio: mx.array | None = None,
        ref_text: str | None = None,
        instruct: str | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Prepare inputs for generation.

        Args:
            text: Text to synthesize
            language: Language code
            speaker: Speaker name (for CustomVoice/Base models)
            ref_audio: Reference audio for voice cloning
            ref_text: Reference text for voice cloning
            instruct: Instruction text for voice style
                (for VoiceDesign/CustomVoice models)

        Returns:
            input_embeds: Input embeddings for the talker
            trailing_text_hidden: Remaining text embeddings
            tts_pad_embed: Padding embedding
        """
        tokenizer = self._tokenizer
        config = self.config.talker_config

        # Tokenize text with chat template
        chat_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = mx.array(tokenizer.encode(chat_text))[None, :]

        # Get text embeddings (computed once, sliced later for efficiency)
        text_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(input_ids)
        )

        # TTS special tokens
        tts_tokens = mx.array(
            [
                [
                    self.config.tts_bos_token_id,
                    self.config.tts_eos_token_id,
                    self.config.tts_pad_token_id,
                ]
            ]
        )
        tts_embeds = self.talker.text_projection(
            self.talker.get_text_embeddings()(tts_tokens)
        )
        tts_bos_embed = tts_embeds[:, 0:1, :]
        tts_eos_embed = tts_embeds[:, 1:2, :]
        tts_pad_embed = tts_embeds[:, 2:3, :]

        # Speaker embedding
        speaker_embed = None
        spk_id_map = config.spk_id or {}
        if ref_audio is not None and self.speaker_encoder is not None:
            speaker_embed = self.extract_speaker_embedding(ref_audio)
        elif speaker and speaker.lower() in spk_id_map:
            spk_ids = mx.array([[spk_id_map[speaker.lower()]]])  # [1, 1]
            speaker_embed = self.talker.get_input_embeddings()(
                spk_ids
            )  # [1, 1, hidden]

        # Language ID
        language_id = None
        if language.lower() != "auto" and config.codec_language_id:
            language_key = language.lower()
            if language_key in config.codec_language_id:
                language_id = config.codec_language_id[language_key]

        # Check for dialect override
        dialect_map = config.spk_is_dialect or {}
        if (
            language.lower() in ["chinese", "auto"]
            and speaker
            and speaker.lower() in dialect_map
            and isinstance(dialect_map[speaker.lower()], str)
        ):
            dialect = cast(str, dialect_map[speaker.lower()])
            if (
                config.codec_language_id is not None
                and dialect in config.codec_language_id
            ):
                language_id = config.codec_language_id[dialect]

        # Build codec prefix
        if language_id is None:
            codec_prefill = [
                config.codec_nothink_id,
                config.codec_think_bos_id,
                config.codec_think_eos_id,
            ]
        else:
            codec_prefill = [
                config.codec_think_id,
                config.codec_think_bos_id,
                language_id,
                config.codec_think_eos_id,
            ]

        codec_embed = self.talker.get_input_embeddings()(mx.array([codec_prefill]))

        codec_embed_suffix = self.talker.get_input_embeddings()(
            mx.array([[config.codec_pad_id, config.codec_bos_id]])
        )

        if speaker_embed is not None:
            codec_embed = mx.concatenate(
                [
                    codec_embed,
                    speaker_embed.reshape(1, 1, -1),
                    codec_embed_suffix,
                ],
                axis=1,
            )
        else:
            codec_embed = mx.concatenate([codec_embed, codec_embed_suffix], axis=1)

        # Instruct embedding (for VoiceDesign/CustomVoice models)
        instruct_embed = None
        if instruct:
            instruct_text = f"<|im_start|>user\n{instruct}<|im_end|>\n"
            instruct_ids = mx.array(tokenizer.encode(instruct_text))[None, :]
            instruct_embed = self.talker.text_projection(
                self.talker.get_text_embeddings()(instruct_ids)
            )

        # Role embedding (first 3 tokens: <|im_start|>assistant\n)
        role_embed = text_embed[:, :3, :]

        # Combine embeddings
        # tts_pad * (codec_len - 2) + tts_bos
        pad_count = codec_embed.shape[1] - 2
        pad_embeds = mx.broadcast_to(
            tts_pad_embed, (1, pad_count, tts_pad_embed.shape[-1])
        )
        combined_embed = mx.concatenate([pad_embeds, tts_bos_embed], axis=1)
        combined_embed = combined_embed + codec_embed[:, :-1, :]

        # Full input embedding
        # If instruct is provided, prepend it
        if instruct_embed is not None:
            input_embeds = mx.concatenate(
                [instruct_embed, role_embed, combined_embed], axis=1
            )
        else:
            input_embeds = mx.concatenate([role_embed, combined_embed], axis=1)

        # Add first text token (token index 3)
        first_text_embed = text_embed[:, 3:4, :] + codec_embed[:, -1:, :]
        input_embeds = mx.concatenate([input_embeds, first_text_embed], axis=1)

        # Trailing text (tokens 4 to -5, plus EOS)
        trailing_text_hidden = mx.concatenate(
            [text_embed[:, 4:-5, :], tts_eos_embed],
            axis=1,
        )

        return input_embeds, trailing_text_hidden, tts_pad_embed

    def _sample_token(
        self,
        logits: mx.array,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        generated_tokens: list[int] | None = None,
        suppress_tokens: list[int] | None = None,
        eos_token_id: int | None = None,
        min_p: float = 0.0,
    ) -> mx.array:

        logits = logits[:, -1, :]  # Get last position [1, vocab_size]

        # Suppress invalid tokens (set to -inf) - pure MLX
        if suppress_tokens:
            suppress_idx = mx.array(suppress_tokens, dtype=mx.int32)
            logits = mx.put_along_axis(
                logits,
                suppress_idx[None, :],
                mx.array(float("-inf"), logits.dtype),
                axis=-1,
            )

        # Apply repetition penalty
        if generated_tokens and repetition_penalty != 1.0:
            unique_tokens = list(set(generated_tokens))
            valid_tokens = [t for t in unique_tokens if t < logits.shape[-1]]
            if valid_tokens:
                token_ids = mx.array(valid_tokens, dtype=mx.int32)

                selected_logits = mx.take(logits, token_ids, axis=-1)
                penalized = mx.where(
                    selected_logits < 0,
                    selected_logits * repetition_penalty,
                    selected_logits / repetition_penalty,
                )

                logits = mx.put_along_axis(
                    logits, token_ids[None, :], penalized, axis=-1
                )

        # Greedy decoding if temperature is 0
        if temperature <= 0:
            return mx.argmax(logits, axis=-1, keepdims=True)

        eos_logit = None
        if eos_token_id is not None and eos_token_id < logits.shape[-1]:
            eos_logit = logits[:, eos_token_id : eos_token_id + 1]

        if top_k > 0 and top_k < logits.shape[-1]:
            logits = apply_top_k(logits, top_k)

        if 0.0 < top_p < 1.0:
            logits = apply_top_p(logits, top_p)

        if min_p > 0.0:
            logits = apply_min_p(logits, min_p)

        if eos_logit is not None:
            eos_idx = mx.array([[eos_token_id]], dtype=mx.int32)
            logits = mx.put_along_axis(logits, eos_idx, eos_logit, axis=-1)

        token = categorical_sampling(logits, temperature)
        return token[:, None]

    def _decode_chunk(self, codes: mx.array, chunk_tokens: int = 300) -> mx.array:
        """Decode a chunk of codes to audio using the vocoder.

        Uses streaming_decode with chunk_tokens (default 300, matching the
        reference implementation's chunk_size=300) so that short inputs
        are decoded in a single pass while long inputs are properly chunked
        with left_context_size=25 for quality.

        Args:
            codes: [1, time, num_code_groups] codes to decode
            chunk_tokens: number of tokens per decode chunk (default 300)

        Returns:
            audio: [samples] decoded audio waveform
        """
        audio_chunks = []
        speech_tokenizer = self.speech_tokenizer
        if speech_tokenizer is None:
            raise ValueError("Speech tokenizer not loaded")
        for chunk in speech_tokenizer.streaming_decode(
            codes, chunk_tokens=chunk_tokens
        ):
            audio_chunks.append(chunk)

        audio = mx.concatenate(audio_chunks, axis=-1)[0]

        # Trim to valid length
        valid_len = int(
            (codes[..., 0] > 0).sum() * speech_tokenizer.decode_upsample_rate
        )
        if valid_len > 0 and valid_len < audio.shape[0]:
            audio = audio[:valid_len]

        mx.eval(audio)
        return audio

    def generate(
        self,
        text: str,
        voice: str | None = None,
        instruct: str | None = None,
        temperature: float = 0.9,
        speed: float = 1.0,
        lang_code: str = "auto",
        split_pattern: str = "\n",
        max_tokens: int = 4096,
        stream: bool = False,
        streaming_interval: float = 2.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        **kwargs,
    ) -> Generator[GenerationResult]:
        """Generate audio from text using the custom_voice pipeline."""
        if not voice:
            raise ValueError(
                "CustomVoice model requires 'voice' (speaker name) "
                "(e.g., 'Chelsie', 'Ethan', 'Vivian')"
            )
        yield from self.generate_custom_voice(
            text=text,
            speaker=voice,
            language=lang_code,
            instruct=instruct,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=stream,
            streaming_interval=streaming_interval,
        )

    def generate_custom_voice(
        self,
        text: str,
        speaker: str,
        language: str = "auto",
        instruct: str | None = None,
        temperature: float = 0.9,
        max_tokens: int = 4096,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        stream: bool = False,
        streaming_interval: float = 2.0,
    ) -> Generator[GenerationResult]:
        """Generate speech with the CustomVoice model using a predefined speaker."""
        if self.config.tts_model_type != "custom_voice":
            msg = (
                f"Model type '{self.config.tts_model_type}' does not "
                "support generate_custom_voice. Please use a "
                "CustomVoice model (Qwen/Qwen3-TTS-12Hz-*-CustomVoice)."
            )
            raise ValueError(msg)

        # Validate speaker
        if speaker.lower() not in [s.lower() for s in self.supported_speakers]:
            msg = (
                f"Speaker '{speaker}' not supported. "
                f"Available: {self.supported_speakers}"
            )
            raise ValueError(msg)

        # For 0.6B models, instruct is not supported
        if (
            self.config.tts_model_size == "0b6"
            and self.config.tts_model_type != "custom_voice"
        ):
            instruct = None

        yield from self._generate_with_instruct(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=stream,
            streaming_interval=streaming_interval,
        )

    def _generate_with_instruct(
        self,
        text: str,
        speaker: str | None,
        language: str,
        instruct: str | None,
        temperature: float,
        max_tokens: int,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        stream: bool = False,
        streaming_interval: float = 2.0,
        streaming_context_size: int = 25,
    ) -> Generator[GenerationResult]:
        """Internal method for generation with instruct support."""
        if self.speech_tokenizer is None:
            raise ValueError("Speech tokenizer not loaded")
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call post_load_hook first.")

        start_time = time.time()

        # Prepare inputs with instruct
        input_embeds, trailing_text_hidden, tts_pad_embed = (
            self._prepare_generation_inputs(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
            )
        )

        # Cap max_tokens based on target text length to prevent runaway generation
        # when EOS logit doesn't become dominant (seen especially with 0.6B model).
        # At 12.5 Hz codec rate, ~3-5 codec tokens per text token is typical speech.
        # Factor of 6 gives ~50% margin for slow speech / pauses.
        target_token_count = len(self.tokenizer.encode(text))
        effective_max_tokens = min(max_tokens, max(75, target_token_count * 6))

        # Initialize cache
        cache = self.talker.make_cache()
        code_cache = self.talker.code_predictor.make_cache()
        generated_codes = []
        generated_token_ids = []
        config = cast(Qwen3TTSTalkerConfig, self.config.talker_config)
        eos_token_id = config.codec_eos_token_id
        suppress_tokens = [
            i
            for i in range(config.vocab_size - 1024, config.vocab_size)
            if i != eos_token_id
        ]
        trailing_idx = 0

        # Initialize streaming state
        if stream:
            streaming_chunk_size = max(1, int(streaming_interval * 12.5))
            decoded_tokens = 0
            chunk_start_time = time.time()
            self._speech_tokenizer.decoder.reset_streaming_state()

        for step in range(effective_max_tokens):
            # Forward pass through talker
            logits, hidden = self.talker(input_embeds, cache=cache)

            # Sample first codebook token
            next_token = self._sample_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=(generated_token_ids if generated_token_ids else None),
                suppress_tokens=suppress_tokens,
                eos_token_id=eos_token_id,
            )

            # Lazy EOS check — defer sync to batch with input_embeds eval
            is_eos = mx.equal(next_token[0, 0], eos_token_id)

            # Generate remaining codebook tokens with code predictor
            code_tokens = [next_token]
            code_hidden = hidden[:, -1:, :]

            # Reset code cache (reuse allocation instead of make_cache/del)
            for c in code_cache:
                c.keys = None
                c.values = None
                c.offset = 0

            for code_idx in range(config.num_code_groups - 1):
                if code_idx == 0:
                    code_0_embed = self.talker.get_input_embeddings()(next_token)
                    code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
                else:
                    code_embed = self.talker.code_predictor.codec_embedding[
                        code_idx - 1
                    ](code_tokens[-1])
                    code_input = code_embed

                code_logits, code_cache, _ = self.talker.code_predictor(
                    code_input,
                    cache=code_cache,
                    generation_step=code_idx,
                )

                next_code = self._sample_token(
                    code_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                code_tokens.append(next_code)

            # Stack all codebook tokens
            all_codes = mx.concatenate(code_tokens, axis=1)

            # Prepare next input
            if trailing_idx < trailing_text_hidden.shape[1]:
                text_embed = trailing_text_hidden[:, trailing_idx : trailing_idx + 1, :]
                trailing_idx += 1
            else:
                text_embed = tts_pad_embed

            codec_embed = self.talker.get_input_embeddings()(next_token)
            for i, code in enumerate(code_tokens[1:]):
                codec_embed = codec_embed + self.talker.code_predictor.codec_embedding[
                    i
                ](code)

            input_embeds = text_embed + codec_embed

            # Single sync point — evaluate input_embeds and EOS check together
            mx.eval(input_embeds, is_eos)

            if is_eos.item():
                break

            generated_token_ids.append(int(next_token[0, 0]))
            generated_codes.append(all_codes)

            # Periodically clear cache to prevent memory buildup during long generation
            if step > 0 and step % 50 == 0:
                mx.clear_cache()

            # Streaming: incrementally decode only new tokens
            if stream and len(generated_codes) - decoded_tokens >= streaming_chunk_size:
                new_tokens = len(generated_codes) - decoded_tokens
                codes_chunk = mx.stack(generated_codes[decoded_tokens:], axis=1)
                codes_for_decoder = mx.transpose(codes_chunk, (0, 2, 1))
                mx.eval(codes_for_decoder)

                wav = self._speech_tokenizer.decoder.streaming_step(codes_for_decoder)
                audio_chunk = wav.squeeze(1)[0]
                mx.eval(audio_chunk)

                decoded_tokens = len(generated_codes)

                chunk_elapsed = time.time() - chunk_start_time
                chunk_audio_dur = audio_chunk.shape[0] / self.sample_rate
                chunk_rtf = chunk_audio_dur / chunk_elapsed if chunk_elapsed > 0 else 0

                yield GenerationResult(
                    audio=audio_chunk,
                    samples=audio_chunk.shape[0],
                    sample_rate=self.sample_rate,
                    segment_idx=0,
                    token_count=new_tokens,
                    audio_duration=format_duration(chunk_audio_dur),
                    real_time_factor=chunk_rtf,
                    prompt={
                        "tokens": new_tokens,
                        "tokens-per-sec": (
                            new_tokens / chunk_elapsed if chunk_elapsed > 0 else 0
                        ),
                    },
                    audio_samples={
                        "samples": audio_chunk.shape[0],
                        "samples-per-sec": (
                            audio_chunk.shape[0] / chunk_elapsed
                            if chunk_elapsed > 0
                            else 0
                        ),
                    },
                    processing_time_seconds=chunk_elapsed,
                    peak_memory_usage=mx.get_peak_memory() / 1e9,
                    is_streaming_chunk=True,
                )

                chunk_start_time = time.time()
                mx.clear_cache()

        # Yield any remaining tokens and clean up streaming state
        if stream:
            if len(generated_codes) > decoded_tokens:
                codes_chunk = mx.stack(generated_codes[decoded_tokens:], axis=1)
                codes_for_decoder = mx.transpose(codes_chunk, (0, 2, 1))
                mx.eval(codes_for_decoder)

                wav = self._speech_tokenizer.decoder.streaming_step(codes_for_decoder)
                audio_chunk = wav.squeeze(1)[0]
                mx.eval(audio_chunk)

                new_tokens = len(generated_codes) - decoded_tokens

                chunk_elapsed = time.time() - chunk_start_time
                chunk_audio_dur = audio_chunk.shape[0] / self.sample_rate
                chunk_rtf = chunk_audio_dur / chunk_elapsed if chunk_elapsed > 0 else 0

                yield GenerationResult(
                    audio=audio_chunk,
                    samples=audio_chunk.shape[0],
                    sample_rate=self.sample_rate,
                    segment_idx=0,
                    token_count=new_tokens,
                    audio_duration=format_duration(chunk_audio_dur),
                    real_time_factor=chunk_rtf,
                    prompt={
                        "tokens": new_tokens,
                        "tokens-per-sec": (
                            new_tokens / chunk_elapsed if chunk_elapsed > 0 else 0
                        ),
                    },
                    audio_samples={
                        "samples": audio_chunk.shape[0],
                        "samples-per-sec": (
                            audio_chunk.shape[0] / chunk_elapsed
                            if chunk_elapsed > 0
                            else 0
                        ),
                    },
                    processing_time_seconds=chunk_elapsed,
                    peak_memory_usage=mx.get_peak_memory() / 1e9,
                    is_streaming_chunk=True,
                    is_final_chunk=True,
                )
            self._speech_tokenizer.decoder.reset_streaming_state()
            return

        if not generated_codes:
            return

        # Stack all generated codes
        codes = mx.stack(generated_codes, axis=1)

        # Non-streaming: decode all at once
        audio, audio_lengths = self._speech_tokenizer.decode(codes)
        audio = audio[0]  # Remove batch dim

        # Trim to valid length
        valid_len = int(audio_lengths[0])
        if valid_len > 0 and valid_len < audio.shape[0]:
            audio = audio[:valid_len]

        mx.eval(audio)

        elapsed_time = time.time() - start_time
        samples = audio.shape[0]
        token_count = len(generated_codes)

        duration_seconds = samples / self.sample_rate
        rtf = duration_seconds / elapsed_time if elapsed_time > 0 else 0

        yield GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=format_duration(duration_seconds),
            real_time_factor=rtf,
            prompt={
                "tokens": token_count,
                "tokens-per-sec": token_count / elapsed_time if elapsed_time > 0 else 0,
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": samples / elapsed_time if elapsed_time > 0 else 0,
            },
            processing_time_seconds=elapsed_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

        mx.clear_cache()

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "Model":
        """Load model from pretrained weights.

        Args:
            path: Local path or Hugging Face repo ID (e.g., 'Qwen/Qwen3-TTS-0.6B-Base')
        """

        from voxing.qwen3_tts._model import load_model as load

        logger.warning(
            "Loading model from pretrained weights is deprecated."
            " Use voxing.qwen3_tts._model.load_model instead."
        )
        return load(str(path))

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """Initialize tokenizer and other resources after weight loading."""
        try:
            from transformers import AutoTokenizer

            model.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        except Exception as e:
            logger.warning("Could not load tokenizer: %s", e)

        # Load speech tokenizer if available
        speech_tokenizer_path = model_path / "speech_tokenizer"
        if speech_tokenizer_path.exists():
            try:
                with (speech_tokenizer_path / "config.json").open() as f:
                    tokenizer_config_dict = json.load(f)

                # Build tokenizer config (filter unknown fields)
                from voxing.qwen3_tts.config import filter_dict_for_dataclass

                tokenizer_kwargs: dict[str, Any] = {}

                if "decoder_config" in tokenizer_config_dict:
                    filtered = filter_dict_for_dataclass(
                        Qwen3TTSTokenizerDecoderConfig,
                        tokenizer_config_dict["decoder_config"],
                    )
                    tokenizer_kwargs["decoder_config"] = Qwen3TTSTokenizerDecoderConfig(
                        **filtered
                    )
                if "encoder_config" in tokenizer_config_dict:
                    filtered = filter_dict_for_dataclass(
                        Qwen3TTSTokenizerEncoderConfig,
                        tokenizer_config_dict["encoder_config"],
                    )
                    tokenizer_kwargs["encoder_config"] = Qwen3TTSTokenizerEncoderConfig(
                        **filtered
                    )

                tokenizer_config = Qwen3TTSTokenizerConfig(**tokenizer_kwargs)

                # Copy top-level config values
                for k, v in tokenizer_config_dict.items():
                    if k not in ("decoder_config", "encoder_config") and hasattr(
                        tokenizer_config, k
                    ):
                        setattr(tokenizer_config, k, v)

                speech_tokenizer = Qwen3TTSSpeechTokenizer(tokenizer_config)

                # Load speech tokenizer weights

                tokenizer_weights: dict[str, Any] = {}
                for wf in speech_tokenizer_path.glob("*.safetensors"):
                    loaded = mx.load(str(wf))
                    tokenizer_weights.update(cast(dict[str, Any], loaded))

                if tokenizer_weights:
                    tokenizer_weights = Qwen3TTSSpeechTokenizer.sanitize(
                        tokenizer_weights
                    )
                    speech_tokenizer.load_weights(
                        list(tokenizer_weights.items()), strict=False
                    )
                    mx.eval(speech_tokenizer.parameters())
                    speech_tokenizer.eval()

                    # Initialize encoder codebooks (compute _embedding and _c2)
                    if speech_tokenizer.encoder_model is not None:
                        quantizer = speech_tokenizer.encoder_model.quantizer
                        for layer in quantizer.rvq_first.vq.layers:
                            layer.codebook.update_in_place()
                        for layer in quantizer.rvq_rest.vq.layers:
                            layer.codebook.update_in_place()
                        logger.debug("Initialized encoder codebooks")

                model.load_speech_tokenizer(speech_tokenizer)

                # Compile the vocoder decoder
                speech_tokenizer.decoder = mx.compile(speech_tokenizer.decoder)
                logger.debug("Loaded speech tokenizer from %s", speech_tokenizer_path)
            except Exception as e:
                logger.warning("Could not load speech tokenizer: %s", e, exc_info=True)

        # Load generation config
        gen_config_path = model_path / "generation_config.json"
        if gen_config_path.exists():
            with gen_config_path.open() as f:
                model.load_generate_config(json.load(f))

        return model

    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights from PyTorch to MLX format."""
        sanitized = {}

        for k, v in weights.items():
            new_key = k

            # Skip position_ids (not used in inference)
            if "position_ids" in k:
                continue

            # Handle Conv1d weights: PyTorch [out, in, kernel] -> MLX [out, kernel, in]
            # This covers:
            # - All conv patterns: .conv.weight, conv1.weight, conv2.weight, etc.
            # - speaker_encoder.fc.weight (which is also a Conv1d)
            # - speech_tokenizer decoder convolutions
            is_conv_weight = (
                "conv" in k or "speaker_encoder.fc" in k
            ) and "weight" in k
            if is_conv_weight and len(v.shape) == 3:
                v = v if check_array_shape_qwen3(v) else mx.transpose(v, (0, 2, 1))
            sanitized[new_key] = v

        return sanitized
