import torch
from encoding import decode_words_to_scalar, decode_bits_to_scalar


def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()


def word_accuracy(pred_word, true_word):
    pred_int = torch.round(pred_word * 65535.0)
    true_int = torch.round(true_word * 65535.0)
    return (pred_int == true_int).float().mean().item()


def compute_metrics(outputs, batch, model_name):
    if model_name == "scalar":
        pred_y = outputs["y"]
        result = {
            "rmse": rmse(pred_y, batch["y"]),
            "mae": mae(pred_y, batch["y"]),
        }
        return result

    if model_name in {"two_word", "coarse_residual", "sequential"}:
        pred_y = decode_words_to_scalar(
            outputs["hi"], outputs["lo"], word_bits=batch["word_bits"]
        )
        result = {
            "rmse": rmse(pred_y, batch["y"]),
            "mae": mae(pred_y, batch["y"]),
            "hi_word_accuracy": word_accuracy(outputs["hi"], batch["y_hi"]),
            "lo_word_accuracy": word_accuracy(outputs["lo"], batch["y_lo"]),
        }
        return result

    if model_name == "bitwise":
        pred_y = decode_bits_to_scalar(outputs["bits"])
        result = {
            "rmse": rmse(pred_y, batch["y"]),
            "mae": mae(pred_y, batch["y"]),
        }
        return result

    raise ValueError(f"Unsupported model for metrics: {model_name}")