"""iSTFTNet decoder and generator for Kokoro TTS.

Vendored from mlx-audio (https://github.com/Blaizzy/mlx-audio).
"""

import math
from collections.abc import Callable
from typing import cast

import mlx.core as mx
import mlx.nn as nn

from voxing.kokoro._base import check_array_shape
from voxing.kokoro._dsp import istft, stft
from voxing.kokoro._interpolate import interpolate

type ConvFunction = Callable[..., mx.array]


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def compute_norm(
    x: mx.array,
    p: int,
    dim: int | list[int] | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> mx.array:
    """Compute the p-norm of a tensor along specified dimensions."""
    if p not in [1, 2]:
        raise ValueError("Only p-norms with p of 1 or 2 are supported")

    if dim is None:
        dim = tuple(range(x.ndim))
    elif isinstance(dim, int):
        dim = (dim,)

    if p == 1:
        return mx.sum(mx.abs(x), axis=dim, keepdims=keepdim)
    return mx.sqrt(mx.sum(x * x, axis=dim, keepdims=keepdim))


def weight_norm(
    weight_v: mx.array, weight_g: mx.array, dim: int | None = None
) -> mx.array:
    """Apply weight normalization: w = g * v/||v||."""
    rank = len(weight_v.shape)

    if dim is not None:
        if dim < -1:
            dim += rank
        axes = list(range(rank))
        if dim != -1:
            axes.remove(dim)
    else:
        axes = list(range(rank))

    norm_v = compute_norm(weight_v, p=2, dim=axes, keepdim=True)
    normalized_weight = weight_v / (norm_v + 1e-7)
    return normalized_weight * weight_g


class ConvWeighted(nn.Module):
    """Conv1d with weight normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        encode: bool = False,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight_g = mx.ones((out_channels, 1, 1))
        self.weight_v = mx.ones((out_channels, kernel_size, in_channels))
        self.bias = mx.zeros(in_channels if encode else out_channels) if bias else None

    def __call__(self, x: mx.array, conv: ConvFunction) -> mx.array:
        weight = weight_norm(self.weight_v, self.weight_g, dim=0)
        bias = self.bias.reshape(1, 1, -1) if self.bias is not None else None

        def apply_conv(x: mx.array, weight_to_use: mx.array) -> mx.array:
            result = conv(
                x,
                weight_to_use,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            if bias is not None:
                return result + bias
            return result

        if x.shape[-1] == weight.shape[-1] or self.groups > 1:
            return apply_conv(x, weight)
        return apply_conv(x, weight.T)


class _InstanceNorm(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.running_mean = mx.zeros((num_features,))
            self.running_var = mx.ones((num_features,))
        else:
            self.running_mean = None
            self.running_var = None

    def _check_input_dim(self, input: mx.array) -> None:
        raise NotImplementedError

    def _get_no_batch_dim(self) -> int:
        raise NotImplementedError

    def _handle_no_batch_input(self, input: mx.array) -> mx.array:
        expanded = mx.expand_dims(input, axis=0)
        result = self._apply_instance_norm(expanded)
        return mx.squeeze(result, axis=0)

    def _apply_instance_norm(self, input: mx.array) -> mx.array:
        dims = list(range(input.ndim))
        feature_dim = dims[-self._get_no_batch_dim()]
        reduce_dims = [d for d in dims if d != 0 and d != feature_dim]

        mean = mx.mean(input, axis=reduce_dims, keepdims=True)
        var = mx.var(input, axis=reduce_dims, keepdims=True)

        x_norm = (input - mean) / mx.sqrt(var + self.eps)

        if self.affine:
            assert self.weight is not None
            assert self.bias is not None
            weight_shape = [1] * input.ndim
            weight_shape[feature_dim] = self.num_features
            bias_shape = weight_shape.copy()
            w = mx.reshape(self.weight, weight_shape)
            b = mx.reshape(self.bias, bias_shape)
            return x_norm * w + b
        return x_norm

    def __call__(self, input: mx.array) -> mx.array:
        self._check_input_dim(input)
        if input.ndim == self._get_no_batch_dim():
            return self._handle_no_batch_input(input)
        return self._apply_instance_norm(input)


class InstanceNorm1d(_InstanceNorm):
    def _get_no_batch_dim(self) -> int:
        return 2

    def _check_input_dim(self, input: mx.array) -> None:
        if input.ndim not in (2, 3):
            raise ValueError(f"expected 2D or 3D input (got {input.ndim}D input)")


class AdaIN1d(nn.Module):
    def __init__(self, style_dim: int, num_features: int):
        super().__init__()
        self.norm = InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def __call__(self, x: mx.array, s: mx.array) -> mx.array:
        h = self.fc(s)
        h = mx.expand_dims(h, axis=2)
        gamma, beta = mx.split(h, 2, axis=1)
        return (1 + gamma) * self.norm(x) + beta


class AdaINResBlock1(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
        style_dim: int = 64,
    ):
        super().__init__()
        self.convs1 = [
            ConvWeighted(
                channels,
                channels,
                kernel_size,
                stride=1,
                padding=get_padding(kernel_size, dilation[i]),
                dilation=dilation[i],
            )
            for i in range(3)
        ]
        self.convs2 = [
            ConvWeighted(
                channels,
                channels,
                kernel_size,
                stride=1,
                padding=get_padding(kernel_size, 1),
                dilation=1,
            )
            for _ in range(3)
        ]
        self.adain1 = [AdaIN1d(style_dim, channels) for _ in range(3)]
        self.adain2 = [AdaIN1d(style_dim, channels) for _ in range(3)]
        self.alpha1 = [mx.ones((1, channels, 1)) for _ in range(len(self.convs1))]
        self.alpha2 = [mx.ones((1, channels, 1)) for _ in range(len(self.convs2))]

    def __call__(self, x: mx.array, s: mx.array) -> mx.array:
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1,
            self.convs2,
            self.adain1,
            self.adain2,
            self.alpha1,
            self.alpha2,
            strict=False,
        ):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (mx.sin(a1 * xt) ** 2)

            xt = xt.swapaxes(2, 1)
            xt = c1(xt, mx.conv1d)
            xt = xt.swapaxes(2, 1)

            xt = n2(xt, s)
            xt = xt + (1 / a2) * (mx.sin(a2 * xt) ** 2)

            xt = xt.swapaxes(2, 1)
            xt = c2(xt, mx.conv1d)
            xt = xt.swapaxes(2, 1)

            x = xt + x
        return x


def mlx_angle(z: mx.array, deg: bool = False) -> mx.array:
    z = mx.array(z)

    if z.dtype == mx.complex64:
        zimag = mx.imag(z)
        zreal = mx.real(z)
    else:
        zimag = mx.zeros_like(z)
        zreal = z

    a = mx.arctan2(zimag, zreal)
    if deg:
        a = a * (180.0 / math.pi)
    return a


def mlx_unwrap(
    p: mx.array,
    discont: float | None = None,
    axis: int = -1,
    period: float = 2 * math.pi,
) -> mx.array:
    if discont is None:
        discont = period / 2
    discont = max(discont, period / 2)

    slice_indices = [slice(None)] * p.ndim
    slice_indices[axis] = slice(1, None)
    after_slice = tuple(slice_indices)
    slice_indices[axis] = slice(None, -1)
    before_slice = tuple(slice_indices)

    dd = p[after_slice] - p[before_slice]

    interval_high = period / 2
    interval_low = -interval_high

    ddmod = dd - period * mx.floor((dd - interval_low) / period)
    ddmod = mx.where(
        (mx.abs(dd - interval_high) < 1e-10) & (dd > 0), interval_high, ddmod
    )

    ph_correct = ddmod - dd
    ph_correct = mx.where(mx.abs(dd) < discont, 0, ph_correct)

    padding_shape = list(ph_correct.shape)
    padding_shape[axis] = 1
    zero_padding = mx.zeros(padding_shape)
    padded_corrections = mx.concatenate([zero_padding, ph_correct], axis=axis)
    cumulative_corrections = mx.cumsum(padded_corrections, axis=axis)

    return p + cumulative_corrections


class MLXSTFT:
    def __init__(
        self,
        filter_length: int = 800,
        hop_length: int = 200,
        win_length: int = 800,
        window: str = "hann",
    ):
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window

    def transform(self, input_data: mx.array) -> tuple[mx.array, mx.array]:
        if input_data.ndim == 1:
            input_data = input_data[None, :]

        magnitudes = []
        phases = []

        for batch_idx in range(input_data.shape[0]):
            x_stft = stft(
                input_data[batch_idx],
                n_fft=self.filter_length,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                pad_mode="reflect",
            ).transpose(1, 0)

            magnitude = mx.abs(x_stft)
            phase = mlx_angle(x_stft)

            magnitudes.append(magnitude)
            phases.append(phase)

        return mx.stack(magnitudes, axis=0), mx.stack(phases, axis=0)

    def inverse(self, magnitude: mx.array, phase: mx.array) -> mx.array:
        reconstructed = []

        for batch_idx in range(magnitude.shape[0]):
            phase_cont = mlx_unwrap(phase[batch_idx], axis=1)

            real_part = magnitude[batch_idx] * mx.cos(phase_cont)
            imag_part = magnitude[batch_idx] * mx.sin(phase_cont)
            x_stft = real_part + 1j * imag_part

            audio = istft(
                x_stft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                length=None,
            )

            reconstructed.append(audio)

        return mx.stack(reconstructed, axis=0)[:, None, :]

    def __call__(self, input_data: mx.array) -> mx.array:
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return mx.expand_dims(reconstruction, axis=-2)


class SineGen:
    def __init__(
        self,
        samp_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
        flag_for_pulse: bool = False,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0: mx.array) -> mx.array:
        return mx.array(f0 > self.voiced_threshold, dtype=mx.float32)

    def _f02sine(self, f0_values: mx.array) -> mx.array:
        """f0_values: (batchsize, length, dim)."""
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = mx.random.normal((f0_values.shape[0], f0_values.shape[2]))
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        if not self.flag_for_pulse:
            rad_values = interpolate(
                rad_values.transpose(0, 2, 1),
                scale_factor=1 / float(self.upsample_scale),
                mode="linear",
            ).transpose(0, 2, 1)
            phase = mx.cumsum(rad_values, axis=1) * 2 * mx.pi
            phase = interpolate(
                phase.transpose(0, 2, 1) * self.upsample_scale,
                scale_factor=float(self.upsample_scale),
                mode="linear",
            ).transpose(0, 2, 1)
            sines = mx.sin(phase)
        else:
            uv = self._f02uv(f0_values)
            uv_1 = mx.roll(uv, -1, 1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            tmp_cumsum = mx.cumsum(rad_values, axis=1)
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            i_phase = mx.cumsum(rad_values - tmp_cumsum, axis=1)
            sines = mx.cos(i_phase * 2 * mx.pi)
        return sines

    def __call__(self, f0: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        fn = f0 * mx.arange(1, self.harmonic_num + 2)[None, None, :]
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(sine_waves.shape)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshod: float = 0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshod,
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = mx.tanh(self.l_linear(sine_wavs))
        noise = mx.random.normal(uv.shape) * self.sine_amp / 3
        return sine_merge, noise, uv


class ReflectionPad1d(nn.Module):
    def __init__(self, padding: tuple[int, int]):
        super().__init__()
        self.padding = padding

    def __call__(self, x: mx.array) -> mx.array:
        pad_width: list[tuple[int, int]] = [
            (0, 0),
            (0, 0),
            (self.padding[0], self.padding[1]),
        ]
        return mx.pad(x, pad_width)


def leaky_relu(x: mx.array, negative_slope: float = 0.01) -> mx.array:
    return mx.where(x > 0, x, x * negative_slope)


class Generator(nn.Module):
    def __init__(
        self,
        style_dim: int,
        resblock_kernel_sizes: list[int],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        resblock_dilation_sizes: list[list[int]],
        upsample_kernel_sizes: list[int],
        gen_istft_n_fft: int,
        gen_istft_hop_size: int,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        upsample_rates_arr = mx.array(upsample_rates)
        upsample_scale = int(mx.prod(upsample_rates_arr)) * gen_istft_hop_size
        self.m_source = SourceModuleHnNSF(
            sampling_rate=24000,
            upsample_scale=upsample_scale,
            harmonic_num=8,
            voiced_threshod=10,
        )
        self.f0_upsamp = nn.Upsample(scale_factor=upsample_scale)
        self.noise_convs: list[nn.Conv1d] = []
        self.noise_res: list[AdaINResBlock1] = []
        self.ups: list[ConvWeighted] = []
        for i, (u, k) in enumerate(
            zip(upsample_rates, upsample_kernel_sizes, strict=False)
        ):
            self.ups.append(
                ConvWeighted(
                    upsample_initial_channel // (2 ** (i + 1)),
                    upsample_initial_channel // (2**i),
                    int(k),
                    int(u),
                    padding=int((k - u) // 2),
                    encode=True,
                )
            )
        self.resblocks: list[AdaINResBlock1] = []
        if not self.ups:
            raise ValueError("upsample_rates must not be empty")
        final_channels = upsample_initial_channel // 2
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            final_channels = ch
            for _j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=False)
            ):
                dilation = cast(tuple[int, int, int], tuple(d))
                self.resblocks.append(AdaINResBlock1(ch, k, dilation, style_dim))
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = int(mx.prod(upsample_rates_arr[i + 1 :]))
                self.noise_convs.append(
                    nn.Conv1d(
                        gen_istft_n_fft + 2,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=(stride_f0 + 1) // 2,
                    )
                )
                self.noise_res.append(AdaINResBlock1(c_cur, 7, (1, 3, 5), style_dim))
            else:
                self.noise_convs.append(
                    nn.Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1)
                )
                self.noise_res.append(AdaINResBlock1(c_cur, 11, (1, 3, 5), style_dim))
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = ConvWeighted(
            final_channels, self.post_n_fft + 2, 7, 1, padding=3
        )
        self.reflection_pad = ReflectionPad1d((1, 0))
        self.stft = MLXSTFT(
            filter_length=gen_istft_n_fft,
            hop_length=gen_istft_hop_size,
            win_length=gen_istft_n_fft,
        )

    def __call__(self, x: mx.array, s: mx.array, f0: mx.array) -> mx.array:
        f0 = self.f0_upsamp(f0[:, None].transpose(0, 2, 1))
        har_source, noi_source, uv = self.m_source(f0)
        har_source = mx.squeeze(har_source.transpose(0, 2, 1), axis=1)
        har_spec, har_phase = self.stft.transform(har_source)
        har = mx.concatenate([har_spec, har_phase], axis=1)
        har = har.swapaxes(2, 1)
        for i in range(self.num_upsamples):
            x = leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_convs[i](har)
            x_source = x_source.swapaxes(2, 1)
            x_source = self.noise_res[i](x_source, s)

            x = x.swapaxes(2, 1)
            x = self.ups[i](x, mx.conv_transpose1d)
            x = x.swapaxes(2, 1)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source

            xs = self.resblocks[i * self.num_kernels](x, s)
            for j in range(1, self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels

        x = leaky_relu(x, negative_slope=0.01)
        x = x.swapaxes(2, 1)
        x = self.conv_post(x, mx.conv1d)
        x = x.swapaxes(2, 1)

        spec = mx.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = mx.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        return self.stft.inverse(spec, phase)


class UpSample1d(nn.Module):
    def __init__(self, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.interpolate = nn.Upsample(
            scale_factor=2, mode="nearest", align_corners=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.layer_type == "none":
            return x
        return self.interpolate(x)


class AdainResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        style_dim: int = 64,
        actv: nn.Module | None = None,
        upsample: str | bool = "none",
        dropout_p: float = 0.0,
        bias: bool = False,
        conv_type: object = None,
    ):
        super().__init__()
        self.actv = nn.LeakyReLU(0.2) if actv is None else actv
        self.dim_in = dim_in
        self.conv_type = conv_type
        if isinstance(upsample, bool):
            upsample = "upsample" if upsample else "none"
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.pool: ConvWeighted | None = None
        if upsample != "none":
            self.pool = ConvWeighted(
                1, dim_in, kernel_size=3, stride=2, padding=1, groups=dim_in
            )

    def _build_weights(self, dim_in: int, dim_out: int, style_dim: int) -> None:
        self.conv1 = ConvWeighted(dim_in, dim_out, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvWeighted(dim_out, dim_out, kernel_size=3, stride=1, padding=1)
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = ConvWeighted(
                dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False
            )

    def _shortcut(self, x: mx.array) -> mx.array:
        x = x.swapaxes(2, 1)
        x = self.upsample(x)
        x = x.swapaxes(2, 1)
        if self.learned_sc:
            x = x.swapaxes(2, 1)
            x = self.conv1x1(x, mx.conv1d)
            x = x.swapaxes(2, 1)
        return x

    def _residual(self, x: mx.array, s: mx.array) -> mx.array:
        x = self.norm1(x, s)
        x = self.actv(x)
        x = x.swapaxes(2, 1)
        if self.upsample_type != "none":
            assert self.pool is not None
            x = self.pool(x, mx.conv_transpose1d)
            pad_width: list[tuple[int, int]] = [(0, 0), (1, 0), (0, 0)]
            x = mx.pad(x, pad_width)
        x = x.swapaxes(2, 1)

        x = x.swapaxes(2, 1)
        x = self.conv1(self.dropout(x), mx.conv1d)
        x = x.swapaxes(2, 1)

        x = self.norm2(x, s)
        x = self.actv(x)
        x = x.swapaxes(2, 1)
        return self.conv2(x, mx.conv1d).swapaxes(2, 1)

    def __call__(self, x: mx.array, s: mx.array) -> mx.array:
        out = self._residual(x, s)
        return (out + self._shortcut(x)) / mx.sqrt(mx.array(2.0))


class Decoder(nn.Module):
    def __init__(
        self,
        dim_in: int,
        style_dim: int,
        dim_out: int,
        resblock_kernel_sizes: list[int],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        resblock_dilation_sizes: list[list[int]],
        upsample_kernel_sizes: list[int],
        gen_istft_n_fft: int,
        gen_istft_hop_size: int,
    ):
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim, conv_type=mx.conv1d)
        self.decode = [
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim, conv_type=mx.conv1d),
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim, conv_type=mx.conv1d),
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim, conv_type=mx.conv1d),
            AdainResBlk1d(
                1024 + 2 + 64, 512, style_dim, upsample=True, conv_type=mx.conv1d
            ),
        ]
        self.F0_conv = ConvWeighted(1, 1, kernel_size=3, stride=2, padding=1, groups=1)
        self.N_conv = ConvWeighted(1, 1, kernel_size=3, stride=2, padding=1, groups=1)
        self.asr_res = [ConvWeighted(512, 64, kernel_size=1, padding=0)]
        self.generator = Generator(
            style_dim,
            resblock_kernel_sizes,
            upsample_rates,
            upsample_initial_channel,
            resblock_dilation_sizes,
            upsample_kernel_sizes,
            gen_istft_n_fft,
            gen_istft_hop_size,
        )

    def __call__(
        self,
        asr: mx.array,
        F0_curve: mx.array,
        N: mx.array,
        s: mx.array,
    ) -> mx.array:
        s = mx.array(s)
        F0 = self.F0_conv(F0_curve[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        N = self.N_conv(N[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        x = mx.concatenate([asr, F0, N], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res[0](asr.swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        res = True
        for block in self.decode:
            if res:
                x = mx.concatenate([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if hasattr(block, "upsample_type") and block.upsample_type != "none":
                res = False
        return self.generator(x, s, F0_curve)

    def sanitize(self, key: str, weights: mx.array) -> mx.array:
        if "noise_convs" in key and key.endswith(".weight"):
            return weights.transpose(0, 2, 1)
        if "weight_v" in key:
            if check_array_shape(weights):
                return weights
            return weights.transpose(0, 2, 1)
        return weights
