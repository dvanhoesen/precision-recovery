from __future__ import annotations

from typing import Dict

import torch


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def word_accuracy(pred_word: torch.Tensor, true_word: torch.Tensor) -> float:
    pred_i = torch.round(pred_word * 65535.0)
    true_i = torch.round(true_word * 65535.0)
    return (pred_i == true_i).float().mean().item()


def summarize_metrics(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    result = {
        "rmse": rmse(outputs["y"], targets["y"]),
        "mae": mae(outputs["y"], targets["y"]),
    }
    if "hi" in outputs and "lo" in outputs:
        result["hi_word_acc"] = word_accuracy(outputs["hi"], targets["hi"])
        result["lo_word_acc"] = word_accuracy(outputs["lo"], targets["lo"])
    return result
