from __future__ import annotations

from typing import Tuple

import torch


UINT16_MAX = 65535
INT32_SCALE = 2**31


def quantize_uniform(x: torch.Tensor, bits: int, x_min: float = -1.0, x_max: float = 1.0) -> torch.Tensor:
    if bits <= 0:
        raise ValueError("bits must be positive")
    levels = 2**bits - 1
    x = x.clamp(x_min, x_max)
    x_scaled = (x - x_min) / (x_max - x_min)
    x_quant = torch.round(x_scaled * levels) / levels
    return x_quant * (x_max - x_min) + x_min


def encode_fixed32(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode y in [-1, 1) to signed 32-bit fixed point and return:
      hi_word in [0, 1]
      lo_word in [0, 1]
      full signed integer representation (int64 tensor for safety)
    """
    y = y.clamp(-0.999999, 0.999999)
    y_int = torch.round(y * INT32_SCALE).to(torch.int64)
    y_int = torch.clamp(y_int, -(2**31), 2**31 - 1)
    y_u32 = y_int & 0xFFFFFFFF

    hi = ((y_u32 >> 16) & 0xFFFF).float() / UINT16_MAX
    lo = (y_u32 & 0xFFFF).float() / UINT16_MAX
    return hi, lo, y_int


def decode_fixed32_from_words(hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
    hi_i = torch.round(hi.clamp(0.0, 1.0) * UINT16_MAX).to(torch.int64)
    lo_i = torch.round(lo.clamp(0.0, 1.0) * UINT16_MAX).to(torch.int64)
    y_u32 = (hi_i << 16) | lo_i
    y_i32 = torch.where(y_u32 >= 2**31, y_u32 - 2**32, y_u32)
    return y_i32.float() / float(INT32_SCALE)
