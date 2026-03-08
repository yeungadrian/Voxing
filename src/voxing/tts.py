import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from voxing.qwen3_tts import Model
from voxing.qwen3_tts import load_model as load_qwen3_model
from voxing.qwen3_tts._base import GenerationResult


@dataclass
class TTSChunk:
    audio: np.ndarray
    sample_rate: int
    segment_idx: int
    token_count: int
    audio_duration: str
    real_time_factor: float
    processing_time_seconds: float
    peak_memory_usage: float


def load_model(
    model_id: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
) -> Model:
    """Download (if needed) and load a Qwen3-TTS model from HuggingFace Hub."""
    return load_qwen3_model(model_id)


def _to_chunk(result: GenerationResult) -> TTSChunk:
    """Convert Qwen3 generation output into a playback-ready chunk."""
    audio = np.asarray(result.audio, dtype=np.float32)
    if audio.ndim != 1:
        audio = np.ravel(audio)
    return TTSChunk(
        audio=audio,
        sample_rate=result.sample_rate,
        segment_idx=result.segment_idx,
        token_count=result.token_count,
        audio_duration=result.audio_duration,
        real_time_factor=result.real_time_factor,
        processing_time_seconds=result.processing_time_seconds,
        peak_memory_usage=result.peak_memory_usage,
    )


def stream_tts(
    model: Model,
    text: str,
    *,
    voice: str = "Ryan",
    speed: float = 1.0,
    lang_code: str = "english",
    instruct: str = "",
    split_pattern: str = "\n",
    temperature: float = 0.9,
    stream: bool = True,
    streaming_interval: float = 0.32,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
) -> Iterator[TTSChunk]:
    for result in model.generate(
        text,
        voice=voice,
        speed=speed,
        lang_code=lang_code,
        instruct=instruct,
        split_pattern=split_pattern,
        temperature=temperature,
        stream=stream,
        streaming_interval=streaming_interval,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    ):
        yield _to_chunk(result)


def _enqueue_chunks(
    chunks: Iterator[TTSChunk],
    chunk_queue: queue.Queue[TTSChunk | None],
    errors: list[BaseException],
) -> None:
    """Read synthesized chunks and push them into the playback queue."""
    try:
        for chunk in chunks:
            chunk_queue.put(chunk)
    except BaseException as exc:
        errors.append(exc)
    finally:
        chunk_queue.put(None)


def play_tts_stream(
    chunks: Iterator[TTSChunk],
    *,
    max_queue_chunks: int = 8,
) -> Iterator[TTSChunk]:
    """Play chunked audio as it is generated, yielding each chunk after playback."""
    chunk_queue: queue.Queue[TTSChunk | None] = queue.Queue(maxsize=max_queue_chunks)
    errors: list[BaseException] = []

    producer = threading.Thread(
        target=_enqueue_chunks,
        args=(chunks, chunk_queue, errors),
        daemon=True,
    )
    producer.start()

    stream: sd.OutputStream | None = None
    try:
        while True:
            chunk = chunk_queue.get()
            if chunk is None:
                break

            if stream is None:
                stream = sd.OutputStream(
                    samplerate=chunk.sample_rate,
                    channels=1,
                    dtype=np.float32,
                )
                stream.start()
            elif int(stream.samplerate) != chunk.sample_rate:
                raise ValueError(
                    "All chunks must use the same sample rate: "
                    f"expected {int(stream.samplerate)}, got {chunk.sample_rate}"
                )

            stream.write(chunk.audio[:, None])
            yield chunk
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        producer.join(timeout=2.0)

    if errors:
        raise errors[0]


def speak_text(
    model: Model,
    text: str,
    *,
    voice: str = "Ryan",
    speed: float = 1.0,
    lang_code: str = "english",
    instruct: str = "",
    split_pattern: str = "\n",
    temperature: float = 0.9,
    stream: bool = True,
    streaming_interval: float = 0.32,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    max_queue_chunks: int = 8,
) -> Iterator[TTSChunk]:
    """Synthesize and play text incrementally, yielding played chunks."""
    yield from play_tts_stream(
        stream_tts(
            model,
            text,
            voice=voice,
            speed=speed,
            lang_code=lang_code,
            instruct=instruct,
            split_pattern=split_pattern,
            temperature=temperature,
            stream=stream,
            streaming_interval=streaming_interval,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ),
        max_queue_chunks=max_queue_chunks,
    )
