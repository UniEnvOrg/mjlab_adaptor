from __future__ import annotations

from functools import singledispatch
from typing import Any, Optional, Union

import numpy as np
import torch
from mjlab.utils import spaces as mjlab_spaces

from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace, DictSpace, Space

__all__ = ["from_mjlab_space", "mjlab_dtype_to_torch"]


_MJLAB_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}


def mjlab_dtype_to_torch(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in _MJLAB_DTYPE_MAP:
        raise ValueError(f"Unsupported mjlab dtype: {dtype}")
    return _MJLAB_DTYPE_MAP[dtype]


def _bound_to_torch(
    bound: Any,
    *,
    dtype: torch.dtype,
    device: Optional[torch.device | str] = None,
) -> Any:
    if isinstance(bound, torch.Tensor):
        return bound.to(dtype=dtype, device=device)
    if isinstance(bound, (tuple, list, np.ndarray)):
        return torch.as_tensor(bound, dtype=dtype, device=device)
    return bound


@singledispatch
def from_mjlab_space(
    space: mjlab_spaces.Space,
    device: Optional[torch.device | str] = None,
) -> Space:
    raise TypeError(
        f"The space provided to `from_mjlab_space` is not supported, type: {type(space)}, {space}"
    )


@from_mjlab_space.register(mjlab_spaces.Box)
def _from_mjlab_space_box(
    space: mjlab_spaces.Box,
    device: Optional[torch.device | str] = None,
) -> BoxSpace:
    dtype = mjlab_dtype_to_torch(space.dtype)
    low = _bound_to_torch(space.low, dtype=dtype, device=device)
    high = _bound_to_torch(space.high, dtype=dtype, device=device)
    return BoxSpace(
        PyTorchComputeBackend,
        low=low,
        high=high,
        dtype=dtype,
        device=device,
        shape=space.shape,
    )


@from_mjlab_space.register(mjlab_spaces.Dict)
def _from_mjlab_space_dict(
    space: mjlab_spaces.Dict,
    device: Optional[torch.device | str] = None,
) -> DictSpace:
    return DictSpace(
        PyTorchComputeBackend,
        {
            key: from_mjlab_space(subspace, device=device)
            for key, subspace in space.spaces.items()
        },
        device=device,
    )
