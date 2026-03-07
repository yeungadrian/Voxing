"""Minimal STFT/iSTFT and window functions for Kokoro TTS.

Vendored from mlx-audio (https://github.com/Blaizzy/mlx-audio).
"""

import math
from functools import cache

import mlx.core as mx


@cache
def hanning(size: int, periodic: bool = False) -> mx.array:
    """Hanning (Hann) window."""
    denom = size if periodic else size - 1
    return mx.array(
        [0.5 * (1 - math.cos(2 * math.pi * n / denom)) for n in range(size)]
    )


STR_TO_WINDOW_FN = {
    "hann": hanning,
    "hanning": hanning,
}


def stft(
    x: mx.array,
    n_fft: int = 800,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: mx.array | str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
) -> mx.array:
    """Short-Time Fourier Transform."""
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    if isinstance(window, str):
        window_fn = STR_TO_WINDOW_FN.get(window.lower())
        if window_fn is None:
            raise ValueError(f"Unknown window function: {window}")
        w = window_fn(win_length)
    else:
        w = window

    if w.shape[0] < n_fft:
        pad_size = n_fft - w.shape[0]
        w = mx.concatenate([w, mx.zeros((pad_size,))], axis=0)

    def _pad(x: mx.array, padding: int, pad_mode: str = "reflect") -> mx.array:
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        if pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        raise ValueError(f"Invalid pad_mode {pad_mode}")

    if center:
        x = _pad(x, n_fft // 2, pad_mode)

    num_frames = 1 + (x.shape[0] - n_fft) // hop_length
    if num_frames <= 0:
        raise ValueError(
            f"Input is too short (length={x.shape[0]}) for n_fft={n_fft} with "
            f"hop_length={hop_length} and center={center}."
        )

    shape = (num_frames, n_fft)
    strides = (hop_length, 1)
    frames = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(frames * w)


def istft(
    x: mx.array,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str = "hann",
    center: bool = True,
    length: int | None = None,
    normalized: bool = False,
) -> mx.array:
    """Inverse Short-Time Fourier Transform."""
    if win_length is None:
        win_length = (x.shape[1] - 1) * 2
    if hop_length is None:
        hop_length = win_length // 4

    if isinstance(window, str):
        window_fn = STR_TO_WINDOW_FN.get(window.lower())
        if window_fn is None:
            raise ValueError(f"Unknown window function: {window}")
        w = window_fn(win_length + 1)[:-1]
    else:
        w = window

    if w.shape[0] < win_length:
        w = mx.concatenate([w, mx.zeros((win_length - w.shape[0],))], axis=0)

    num_frames = x.shape[1]
    t = (num_frames - 1) * hop_length + win_length

    reconstructed = mx.zeros(t)
    window_sum = mx.zeros(t)

    frames_time = mx.fft.irfft(x, axis=0).transpose(1, 0)

    frame_offsets = mx.arange(num_frames) * hop_length
    indices = frame_offsets[:, None] + mx.arange(win_length)
    indices_flat = indices.flatten()

    updates_reconstructed = (frames_time * w).flatten()
    window_norm = (w * w) if normalized else w
    updates_window = mx.tile(window_norm, (num_frames,)).flatten()

    reconstructed = reconstructed.at[indices_flat].add(updates_reconstructed)
    window_sum = window_sum.at[indices_flat].add(updates_window)

    reconstructed = mx.where(
        window_sum > 1e-10, reconstructed / window_sum, reconstructed
    )

    if center and length is None:
        reconstructed = reconstructed[win_length // 2 : -win_length // 2]

    if length is not None:
        reconstructed = reconstructed[:length]

    return reconstructed
