from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def scalar_loss(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    return F.mse_loss(outputs["y"], targets["y"])


def chunked_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    lambda_hi: float = 0.2,
    lambda_lo: float = 0.2,
) -> torch.Tensor:
    loss_full = F.mse_loss(outputs["y"], targets["y"])
    loss_hi = F.mse_loss(outputs["hi"], targets["hi"])
    loss_lo = F.mse_loss(outputs["lo"], targets["lo"])
    return loss_full + lambda_hi * loss_hi + lambda_lo * loss_lo


def bitwise_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    lambda_bits: float = 0.1,
) -> torch.Tensor:
    loss_full = F.mse_loss(outputs["y"], targets["y"])
    loss_hi = F.mse_loss(outputs["hi"], targets["hi"])
    loss_lo = F.mse_loss(outputs["lo"], targets["lo"])
    loss_bits = F.binary_cross_entropy(outputs["bits"], targets["bits"])
    return loss_full + 0.1 * loss_hi + 0.1 * loss_lo + lambda_bits * loss_bits


def build_loss(loss_name: str, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], cfg: Dict) -> torch.Tensor:
    if loss_name == "scalar_mse":
        return scalar_loss(outputs, targets)
    if loss_name == "chunked":
        return chunked_loss(
            outputs,
            targets,
            lambda_hi=cfg.get("lambda_hi", 0.2),
            lambda_lo=cfg.get("lambda_lo", 0.2),
        )
    if loss_name == "bitwise":
        return bitwise_loss(outputs, targets, lambda_bits=cfg.get("lambda_bits", 0.1))
    raise ValueError(f"Unknown loss name: {loss_name}")
