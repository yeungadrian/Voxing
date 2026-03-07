"""1D interpolation for MLX arrays.

Vendored from mlx-audio (https://github.com/Blaizzy/mlx-audio).
"""

from typing import Optional, Union

import mlx.core as mx


def interpolate(
    input: mx.array,
    size: Optional[Union[int, tuple[int, ...], list[int]]] = None,
    scale_factor: Optional[Union[float, list[float], tuple[float, ...]]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
) -> mx.array:
    """Interpolate array with correct shape handling."""
    ndim = input.ndim
    if ndim < 3:
        raise ValueError(f"Expected at least 3D input (N, C, D1), got {ndim}D")

    spatial_dims = ndim - 2

    if size is not None and scale_factor is not None:
        raise ValueError("Only one of size or scale_factor should be defined")
    elif size is None and scale_factor is None:
        raise ValueError("One of size or scale_factor must be defined")

    if size is not None and not isinstance(size, (list, tuple)):
        size = [size] * spatial_dims
    if scale_factor is not None and not isinstance(scale_factor, (list, tuple)):
        scale_factor = [scale_factor] * spatial_dims

    if size is None:
        assert scale_factor is not None
        size = []
        for i in range(spatial_dims):
            curr_size = max(1, int(mx.ceil(input.shape[i + 2] * scale_factor[i])))  # ty: ignore
            size.append(curr_size)

    if spatial_dims == 1:
        return interpolate1d(input, size[0], mode, align_corners)
    else:
        raise ValueError(
            f"Only 1D interpolation currently supported, got {spatial_dims}D"
        )


def interpolate1d(
    input: mx.array,
    size: int,
    mode: str = "linear",
    align_corners: Optional[bool] = None,
) -> mx.array:
    """1D interpolation implementation."""
    batch_size, channels, in_width = input.shape

    if size < 1:
        size = 1
    if in_width < 1:
        in_width = 1

    if mode == "nearest":
        if size == 1:
            indices = mx.array([0])
        else:
            scale = in_width / size
            indices = mx.floor(mx.arange(size) * scale).astype(mx.int32)
            indices = mx.clip(indices, 0, in_width - 1)
        return input[:, :, indices]

    if align_corners and size > 1:
        x = mx.arange(size) * ((in_width - 1) / (size - 1))
    else:
        if size == 1:
            x = mx.array([0.0])
        else:
            x = mx.arange(size) * (in_width / size)
            if not align_corners:
                x = x + 0.5 * (in_width / size) - 0.5

    if in_width == 1:
        return mx.broadcast_to(input, (batch_size, channels, size))

    x_low = mx.floor(x).astype(mx.int32)
    x_high = mx.minimum(x_low + 1, in_width - 1)
    x_frac = x - x_low

    y_low = input[:, :, x_low]
    y_high = input[:, :, x_high]

    return y_low * (1 - x_frac)[None, None, :] + y_high * x_frac[None, None, :]
