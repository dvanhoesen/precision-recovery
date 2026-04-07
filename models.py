from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from encoding import decode_fixed32_from_words


class MLPBackbone(nn.Module):
    def __init__(self, d_in: int, width: int = 128, depth: int = 3) -> None:
        super().__init__()
        layers = []
        current = d_in
        for _ in range(depth):
            layers.extend([nn.Linear(current, width), nn.GELU()])
            current = width
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScalarModel(nn.Module):
    def __init__(self, d_in: int, width: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.backbone = MLPBackbone(d_in=d_in, width=width, depth=depth)
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        y = torch.tanh(self.head(h))
        return {"y": y}


class TwoWordModel(nn.Module):
    def __init__(self, d_in: int, width: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.backbone = MLPBackbone(d_in=d_in, width=width, depth=depth)
        self.hi_head = nn.Linear(width, 1)
        self.lo_head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        hi = torch.sigmoid(self.hi_head(h))
        lo = torch.sigmoid(self.lo_head(h))
        y = decode_fixed32_from_words(hi, lo)
        return {"y": y, "hi": hi, "lo": lo}


class CoarseResidualModel(nn.Module):
    def __init__(self, d_in: int, width: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.backbone = MLPBackbone(d_in=d_in, width=width, depth=depth)
        self.coarse_head = nn.Linear(width, 1)
        self.resid_head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        hi = torch.sigmoid(self.coarse_head(h))
        lo = torch.sigmoid(self.resid_head(h))
        y = decode_fixed32_from_words(hi, lo)
        return {"y": y, "hi": hi, "lo": lo}


class SequentialWordModel(nn.Module):
    def __init__(self, d_in: int, width: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.backbone = MLPBackbone(d_in=d_in, width=width, depth=depth)
        self.hi_head = nn.Linear(width, 1)
        self.lo_head = nn.Sequential(
            nn.Linear(width + 1, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        hi = torch.sigmoid(self.hi_head(h))
        lo = torch.sigmoid(self.lo_head(torch.cat([h, hi], dim=1)))
        y = decode_fixed32_from_words(hi, lo)
        return {"y": y, "hi": hi, "lo": lo}


class BitwiseModel(nn.Module):
    def __init__(self, d_in: int, width: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.backbone = MLPBackbone(d_in=d_in, width=width, depth=depth)
        self.head = nn.Linear(width, 32)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        bits = torch.sigmoid(self.head(h))
        hi_bits = bits[:, :16]
        lo_bits = bits[:, 16:]

        powers = (2 ** torch.arange(15, -1, -1, device=x.device)).float().unsqueeze(0)
        hi = (hi_bits.round() * powers).sum(dim=1, keepdim=True) / 65535.0
        lo = (lo_bits.round() * powers).sum(dim=1, keepdim=True) / 65535.0
        y = decode_fixed32_from_words(hi, lo)
        return {"y": y, "hi": hi, "lo": lo, "bits": bits}


def build_model(model_name: str, d_in: int, width: int = 128, depth: int = 3) -> nn.Module:
    if model_name == "scalar":
        return ScalarModel(d_in=d_in, width=width, depth=depth)
    if model_name == "two_word":
        return TwoWordModel(d_in=d_in, width=width, depth=depth)
    if model_name == "coarse_residual":
        return CoarseResidualModel(d_in=d_in, width=width, depth=depth)
    if model_name == "sequential":
        return SequentialWordModel(d_in=d_in, width=width, depth=depth)
    if model_name == "bitwise":
        return BitwiseModel(d_in=d_in, width=width, depth=depth)
    raise ValueError(f"Unknown model name: {model_name}")
