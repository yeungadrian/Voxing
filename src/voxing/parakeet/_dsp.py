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


@cache
def hamming(size: int, periodic: bool = False) -> mx.array:
    """Hamming window."""
    denom = size if periodic else size - 1
    return mx.array(
        [0.54 - 0.46 * math.cos(2 * math.pi * n / denom) for n in range(size)]
    )


@cache
def blackman(size: int, periodic: bool = False) -> mx.array:
    """Blackman window."""
    denom = size if periodic else size - 1
    return mx.array(
        [
            0.42
            - 0.5 * math.cos(2 * math.pi * n / denom)
            + 0.08 * math.cos(4 * math.pi * n / denom)
            for n in range(size)
        ]
    )


@cache
def bartlett(size: int, periodic: bool = False) -> mx.array:
    """Bartlett (triangular) window."""
    denom = size if periodic else size - 1
    return mx.array([1 - 2 * abs(n - denom / 2) / denom for n in range(size)])


STR_TO_WINDOW_FN = {
    "hann": hanning,
    "hanning": hanning,
    "hamming": hamming,
    "blackman": blackman,
    "bartlett": bartlett,
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
    """Short-time Fourier transform."""
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


@cache
def mel_filters(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0,
    f_max: float | None = None,
    norm: str | None = None,
    mel_scale: str = "htk",
) -> mx.array:
    """Mel filterbank matrix."""

    def hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
        if mel_scale == "htk":
            return 2595.0 * math.log10(1.0 + freq / 700.0)
        f_min, f_sp = 0.0, 200.0 / 3
        mels = (freq - f_min) / f_sp
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz) / logstep
        return mels

    def mel_to_hz(mels: mx.array, mel_scale: str = "htk") -> mx.array:
        if mel_scale == "htk":
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        f_min, f_sp = 0.0, 200.0 / 3
        freqs = f_min + f_sp * mels
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        return mx.where(
            mels >= min_log_mel,
            min_log_hz * mx.exp(logstep * (mels - min_log_mel)),
            freqs,
        )

    f_max = f_max or sample_rate / 2

    n_freqs = n_fft // 2 + 1
    all_freqs = mx.linspace(0, sample_rate // 2, n_freqs)

    m_min = hz_to_mel(f_min, mel_scale)
    m_max = hz_to_mel(f_max, mel_scale)
    m_pts = mx.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts, mel_scale)

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = mx.expand_dims(f_pts, 0) - mx.expand_dims(all_freqs, 1)

    down_slopes = (-slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    filterbank = mx.maximum(
        mx.zeros_like(down_slopes), mx.minimum(down_slopes, up_slopes)
    )

    if norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        filterbank *= mx.expand_dims(enorm, 0)

    return filterbank.moveaxis(0, 1)
